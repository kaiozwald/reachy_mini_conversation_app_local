"""Audio stream handler for Reachy Mini.

Pipeline: VAD → Groq Whisper STT → Stuart AI RAG (+ LLM fallback) → Groq Orpheus TTS

When RAG returns no useful answer, falls back to a direct Groq LLaMA chat call
so the user always gets a reasonable in-character response.
"""

import re
import io
import wave
import base64
import asyncio
import logging
from typing import Final, Tuple, Literal, Optional

import aiohttp
import numpy as np
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample

from reachy_mini_conversation_app.local_audio import GroqASR, GroqTTS, LocalVAD
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


logger = logging.getLogger(__name__)

SAMPLE_RATE: Final[Literal[24000]] = 24000

# ---------------------------------------------------------------------------
# Whisper prompt — seeds domain vocabulary so STT stops mishearing NOI terms
# ---------------------------------------------------------------------------
WHISPER_PROMPT = (
    "NOI Techpark, Reachy, laser cutter, laser engraver, 3D printer, CNC, "
    "Fab Lab, FabLab, soldering, oscilloscope, multimeter, filament, resin, "
    "G-code, CAD, CAM, PLA, ABS, PETG, power, wattage, bed temperature, "
    "nozzle, extruder, stepper motor, enclosure, ventilation, safety, "
    "software, firmware, Ultimaker, Prusa, Bambu, Snapmaker, Glowforge, "
    "Trotec, xTool, access control, booking, training, certification."
)

# ---------------------------------------------------------------------------
# Phrases that mean the RAG found nothing — triggers LLM fallback
# ---------------------------------------------------------------------------
RAG_EMPTY_RESPONSES = {
    "i don't know", "i do not know",
    "no information available", "no information found",
    "i cannot answer", "i can't answer", "",
}

# ---------------------------------------------------------------------------
# Fallback LLM — Groq chat model (fast, cheap)
# ---------------------------------------------------------------------------
FALLBACK_CHAT_MODEL = "llama-3.1-8b-instant"

FALLBACK_SYSTEM_PROMPT = (
    "You are Reachy Mini, a helpful robotic assistant at NOI Techpark's Fab Lab. "
    "Your personality is friendly, professional, and slightly high-tech. "
    "RULES:\n"
    "1. The internal knowledge base had no answer for this question, so answer "
    "   from your general knowledge.\n"
    "2. Always respond in the SAME LANGUAGE as the user's question.\n"
    "3. Keep answers SHORT — 2-3 sentences max.\n"
    "4. End every factual answer with: 'If you would like more detail, just let me know!'\n"
    "5. NEVER invent specific NOI Techpark machine specs or policies — "
    "   if unsure, say the knowledge base didn't have it and suggest asking staff.\n"
    "6. Use occasional tech metaphors but stay concise."
)


def _log(msg: str) -> None:
    print(msg, flush=True)
    logger.info(msg)


def _err(msg: str) -> None:
    print(f"[ERROR] {msg}", flush=True)
    logger.error(msg)


def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


class StuartRealtimeHandler(AsyncStreamHandler):

    def __init__(self, deps: ToolDependencies, gradio_mode: bool = False, instance_path: Optional[str] = None):
        super().__init__(expected_layout="mono", output_sample_rate=SAMPLE_RATE, input_sample_rate=SAMPLE_RATE)

        self.output_sample_rate: Literal[24000] = SAMPLE_RATE
        self.input_sample_rate:  Literal[24000] = SAMPLE_RATE
        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path

        self.output_queue: asyncio.Queue = asyncio.Queue()
        self._shutdown_requested = False
        self._connected_event = asyncio.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # VAD
        self._local_vad = LocalVAD(
            energy_threshold=config.VAD_ENERGY_THRESHOLD,
            silence_duration=config.VAD_SILENCE_DURATION,
            min_speech_duration=config.VAD_MIN_SPEECH_DURATION,
            sample_rate=self.input_sample_rate,
        )
        self._audio_buffer: list[bytes] = []
        self._is_speech_active = False
        self._vad_processing = False
        self._tts_playing = False
        self._frame_count = 0

        # STT
        self._asr = GroqASR(api_key=config.GROQ_API_KEY, model=config.GROQ_STT_MODEL, language=config.WHISPER_LANGUAGE)

        # TTS
        self._tts = GroqTTS(api_key=config.GROQ_API_KEY, model=config.GROQ_TTS_MODEL, voice=config.GROQ_TTS_VOICE, output_sample_rate=self.output_sample_rate)

        _log(f"STT     : Groq Whisper  ({config.GROQ_STT_MODEL})")
        _log(f"TTS     : Groq Orpheus  ({config.GROQ_TTS_MODEL} / {config.GROQ_TTS_VOICE})")
        _log(f"RAG     : Stuart AI     {config.STUART_ENDPOINT}")
        _log(f"FALLBACK: Groq chat     {FALLBACK_CHAT_MODEL}")
        print("=" * 60, flush=True)
        print("Pipeline: VAD → STT → RAG → (LLM fallback if needed) → TTS", flush=True)
        print("=" * 60, flush=True)

    def copy(self):
        return StuartRealtimeHandler(self.deps, self.gradio_mode, self.instance_path)

    async def start_up(self) -> None:
        self._loop = asyncio.get_event_loop()
        _log("Session started — listening for speech")
        self._connected_event.set()
        while not self._shutdown_requested:
            await asyncio.sleep(0.1)

    async def shutdown(self) -> None:
        self._shutdown_requested = True
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    # ------------------------------------------------------------------
    # STT
    # ------------------------------------------------------------------

    async def _transcribe(self, audio_bytes: bytes) -> Optional[str]:
        _log(f"[STT] Transcribing {len(audio_bytes)} bytes...")
        wav = _pcm_to_wav(audio_bytes, self.input_sample_rate)
        try:
            form = aiohttp.FormData()
            form.add_field("file", wav, filename="audio.wav", content_type="audio/wav")
            form.add_field("model", config.GROQ_STT_MODEL)
            form.add_field("language", config.WHISPER_LANGUAGE)
            form.add_field("response_format", "json")
            form.add_field("prompt", WHISPER_PROMPT)

            async with aiohttp.ClientSession() as s:
                async with s.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
                    data=form,
                    timeout=aiohttp.ClientTimeout(total=30.0),
                ) as resp:
                    if resp.status != 200:
                        _err(f"Groq STT HTTP {resp.status} — using GroqASR fallback")
                        return await self._asr.transcribe(audio_bytes, self.input_sample_rate)
                    result = await resp.json()

            text = result.get("text", "").strip()
            if not text:
                _log("[STT] Empty transcript")
                return None
            _log(f"[STT] Transcript: {text}")
            return text
        except Exception as exc:
            _err(f"STT error ({exc}) — using GroqASR fallback")
            try:
                return await self._asr.transcribe(audio_bytes, self.input_sample_rate)
            except Exception as exc2:
                _err(f"STT fallback failed: {exc2}")
                return None

    # ------------------------------------------------------------------
    # RAG
    # ------------------------------------------------------------------

    async def _query_rag(self, question: str) -> Optional[str]:
        """Returns answer string if RAG found something, None if not."""
        try:
            _log(f"[RAG] Querying: {question[:120]}")
            form = aiohttp.FormData()
            form.add_field("question", question)

            async with aiohttp.ClientSession() as s:
                async with s.post(
                    config.STUART_ENDPOINT,
                    data=form,
                    timeout=aiohttp.ClientTimeout(total=60.0),
                ) as resp:
                    _log(f"[RAG] HTTP {resp.status}")
                    if resp.status != 200:
                        _err(f"Stuart AI HTTP {resp.status}")
                        return None
                    result = await resp.json()

            raw = result.get("manswer", "").strip()
            _log(f"[RAG] Raw: {raw[:120]}")

            # Check if it's an empty/useless answer
            if raw.lower().rstrip(" .!") in RAG_EMPTY_RESPONSES:
                _log("[RAG] No useful answer in knowledge base → will use LLM fallback")
                return None

            # Strip markdown for clean TTS
            answer = re.sub(r"\*+", "", raw)
            answer = re.sub(r"^#+\s*", "", answer, flags=re.MULTILINE)
            return answer.strip()

        except asyncio.TimeoutError:
            _err("RAG timed out")
            return None
        except Exception as exc:
            _err(f"RAG error: {exc}")
            return None

    # ------------------------------------------------------------------
    # LLM fallback
    # ------------------------------------------------------------------

    async def _query_llm_fallback(self, question: str) -> str:
        """
        Called when RAG has no answer.
        Asks Groq chat API directly — keeps the robot in character and
        avoids hallucinating NOI-specific facts.
        """
        _log(f"[FALLBACK] Asking LLM: {question[:120]}")
        try:
            payload = {
                "model": FALLBACK_CHAT_MODEL,
                "messages": [
                    {"role": "system", "content": FALLBACK_SYSTEM_PROMPT},
                    {"role": "user",   "content": question},
                ],
                "max_tokens": 200,
                "temperature": 0.4,
            }

            async with aiohttp.ClientSession() as s:
                async with s.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {config.GROQ_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30.0),
                ) as resp:
                    if resp.status != 200:
                        err = await resp.text()
                        _err(f"Fallback LLM HTTP {resp.status}: {err[:200]}")
                        return "My knowledge base did not have that. Please ask a staff member for help!"
                    result = await resp.json()

            answer = result["choices"][0]["message"]["content"].strip()
            answer = re.sub(r"\*+", "", answer)
            answer = re.sub(r"^#+\s*", "", answer, flags=re.MULTILINE)
            _log(f"[FALLBACK] Answer: {answer[:120]}")
            return answer.strip()

        except asyncio.TimeoutError:
            _err("Fallback LLM timed out")
            return "My knowledge base did not have that. Please ask a staff member!"
        except Exception as exc:
            _err(f"Fallback LLM error: {exc}")
            return "My knowledge base did not have that. Please ask a staff member!"

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    async def _speak(self, text: str) -> None:
        if not text:
            return
        try:
            self._tts_playing = True
            _log(f"[TTS] Synthesizing: {text[:80]}")
            audio_data = await self._tts.synthesize(text)
            if audio_data is None:
                _err("TTS returned no audio")
                return
            _log(f"[TTS] {len(audio_data)} samples — queuing")
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.feed(base64.b64encode(audio_data.tobytes()).decode())
            for i in range(0, len(audio_data), 4800):
                chunk = audio_data[i:i + 4800]
                await self.output_queue.put((self.output_sample_rate, chunk.reshape(1, -1)))
            _log("[TTS] Playback queued ✓")
        except Exception as exc:
            _err(f"TTS failed: {exc}")
        finally:
            self._tts_playing = False

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    async def _process_speech(self, audio_bytes: bytes) -> None:
        """
        PCM → STT → RAG → [LLM fallback if RAG empty] → TTS

            RAG has answer  →  speak RAG answer        [source: RAG]
            RAG has nothing →  ask Groq LLM directly   [source: FALLBACK]
        """
        try:
            # 1. STT
            transcript = await self._transcribe(audio_bytes)
            if not transcript:
                return

            await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))

            # 2. RAG
            answer = await self._query_rag(transcript)
            source = "RAG"

            # 3. Fallback if RAG had nothing
            if answer is None:
                answer = await self._query_llm_fallback(transcript)
                source = "FALLBACK"

            _log(f"[PIPELINE] Answering via {source}: {answer[:80]}")

            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": f"[{source}] {answer}"}))

            # 4. TTS
            await self._speak(answer)

        except Exception as exc:
            _err(f"_process_speech unhandled: {exc}")
        finally:
            self._vad_processing = False

    # ------------------------------------------------------------------
    # Audio receive
    # ------------------------------------------------------------------

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        input_sr, audio_frame = frame

        if audio_frame.ndim == 2:
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]
        audio_frame = audio_frame.flatten()

        if input_sr != self.input_sample_rate:
            audio_frame = resample(audio_frame, int(len(audio_frame) * self.input_sample_rate / input_sr))

        audio_frame = audio_to_int16(audio_frame)

        self._frame_count += 1
        if self._frame_count % 100 == 0:
            rms = np.sqrt(np.mean((audio_frame.astype(np.float32) / 32768.0) ** 2))
            print(f"[VAD] RMS={rms:.5f}  threshold={config.VAD_ENERGY_THRESHOLD:.5f}", flush=True)

        if self._tts_playing:
            return

        speech_started, speech_ended = self._local_vad.process(audio_frame)

        if speech_started:
            self._is_speech_active = True
            self._audio_buffer.clear()
            self.deps.movement_manager.set_listening(True)
            print("[VAD] Speech started ▶", flush=True)

        if self._is_speech_active:
            self._audio_buffer.append(audio_frame.tobytes())

        if speech_ended and not self._vad_processing:
            self._vad_processing = True
            self._is_speech_active = False
            self.deps.movement_manager.set_listening(False)

            audio_bytes = b"".join(self._audio_buffer)
            self._audio_buffer.clear()
            print(f"[VAD] Speech ended ◼  {len(audio_bytes)} bytes captured", flush=True)

            coro = self._process_speech(audio_bytes)
            if self._loop is not None and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(coro, self._loop)
            else:
                asyncio.ensure_future(coro)

    # ------------------------------------------------------------------
    # Audio emit
    # ------------------------------------------------------------------

    async def emit(self):
        return await wait_for_item(self.output_queue)
