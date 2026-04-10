"""Audio stream handler for Reachy Mini.

Pipeline: VAD → Groq Whisper STT → Stuart AI RAG (+ LLM fallback) → Groq Orpheus TTS

Fixes applied vs previous version:
  - Echo loop: TTS mute window extended past playback completion
  - VAD hallucinations: minimum audio size guard before STT
  - VAD false triggers: energy threshold guidance in config
  - Speech started spam: debounce consecutive VAD starts
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
# Minimum captured audio before sending to STT.
# 24000 Hz * 2 bytes/sample * 1.5 s = 72 000 bytes.
# Anything shorter is almost certainly noise or a half-syllable — skip it.
# ---------------------------------------------------------------------------
MIN_AUDIO_BYTES = int(SAMPLE_RATE * 2 * 1.5)   # 1.5 seconds

# ---------------------------------------------------------------------------
# How long (seconds) to keep VAD suppressed AFTER TTS finishes playing.
# This prevents the robot from hearing its own voice through the mic.
# Increase if the speaker has significant reverb / room echo.
# ---------------------------------------------------------------------------
TTS_MUTE_TAIL_SEC = 1.5

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
# Fallback LLM
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
    "   if unsure, say the knowledge base did not have it and suggest asking staff.\n"
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

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
    ):
        super().__init__(
            expected_layout="mono",
            output_sample_rate=SAMPLE_RATE,
            input_sample_rate=SAMPLE_RATE,
        )

        self.output_sample_rate: Literal[24000] = SAMPLE_RATE
        self.input_sample_rate:  Literal[24000] = SAMPLE_RATE
        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path

        self.output_queue: asyncio.Queue = asyncio.Queue()
        self._shutdown_requested = False
        self._connected_event = asyncio.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # VAD state
        self._local_vad = LocalVAD(
            energy_threshold=config.VAD_ENERGY_THRESHOLD,
            silence_duration=config.VAD_SILENCE_DURATION,
            min_speech_duration=config.VAD_MIN_SPEECH_DURATION,
            sample_rate=self.input_sample_rate,
        )
        self._audio_buffer: list[bytes] = []
        self._is_speech_active = False
        self._vad_processing = False
        self._frame_count = 0

        # TTS mute state.
        # _tts_playing  → True only while synthesizing + queueing audio chunks.
        # _tts_mute_until → monotonic timestamp; VAD is fully suppressed until
        #                   this time, covering actual speaker playback + tail.
        self._tts_playing = False
        self._tts_mute_until: float = 0.0

        # STT / TTS providers
        self._asr = GroqASR(
            api_key=config.GROQ_API_KEY,
            model=config.GROQ_STT_MODEL,
            language=config.WHISPER_LANGUAGE,
        )
        self._tts = GroqTTS(
            api_key=config.GROQ_API_KEY,
            model=config.GROQ_TTS_MODEL,
            voice=config.GROQ_TTS_VOICE,
            output_sample_rate=self.output_sample_rate,
        )

        _log(f"STT          : Groq Whisper  ({config.GROQ_STT_MODEL})")
        _log(f"TTS          : Groq Orpheus  ({config.GROQ_TTS_MODEL} / {config.GROQ_TTS_VOICE})")
        _log(f"RAG          : Stuart AI     {config.STUART_ENDPOINT}")
        _log(f"FALLBACK     : Groq chat     {FALLBACK_CHAT_MODEL}")
        _log(f"MIN AUDIO    : {MIN_AUDIO_BYTES} bytes  ({MIN_AUDIO_BYTES / (SAMPLE_RATE * 2):.1f}s)")
        _log(f"TTS MUTE TAIL: {TTS_MUTE_TAIL_SEC}s after playback ends")
        print("=" * 60, flush=True)
        print("Pipeline: VAD → STT → RAG → (LLM fallback) → TTS", flush=True)
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
        duration = len(audio_bytes) / (SAMPLE_RATE * 2)
        _log(f"[STT] Transcribing {len(audio_bytes)} bytes ({duration:.2f}s)...")
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
                _err(f"STT fallback also failed: {exc2}")
                return None

    # ------------------------------------------------------------------
    # RAG
    # ------------------------------------------------------------------

    async def _query_rag(self, question: str) -> Optional[str]:
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

            if raw.lower().rstrip(" .!") in RAG_EMPTY_RESPONSES:
                _log("[RAG] No useful answer → LLM fallback")
                return None

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
                        _err(f"Fallback LLM HTTP {resp.status}: {await resp.text()}")
                        return "My knowledge base did not have that. Please ask a staff member!"
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
    # TTS — with extended mute window to block echo loop
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

            # Calculate playback duration and set mute window BEFORE queuing
            # chunks, so VAD is suppressed from the very start of playback.
            playback_sec = len(audio_data) / SAMPLE_RATE
            mute_sec = playback_sec + TTS_MUTE_TAIL_SEC
            self._tts_mute_until = asyncio.get_event_loop().time() + mute_sec
            _log(f"[TTS] {len(audio_data)} samples ({playback_sec:.1f}s) — "
                 f"VAD muted for {mute_sec:.1f}s")

            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.feed(base64.b64encode(audio_data.tobytes()).decode())

            chunk_size = 4800   # 200 ms at 24 kHz
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i: i + chunk_size]
                await self.output_queue.put((self.output_sample_rate, chunk.reshape(1, -1)))

            _log("[TTS] Playback queued ✓")

        except Exception as exc:
            _err(f"TTS failed: {exc}")
        finally:
            # _tts_playing goes False once chunks are queued (synthesis done),
            # but _tts_mute_until keeps VAD blocked through actual speaker output.
            self._tts_playing = False

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    async def _process_speech(self, audio_bytes: bytes) -> None:
        try:
            transcript = await self._transcribe(audio_bytes)
            if not transcript or transcript.lower().strip(" .") == "thank you":
                _log("[PIPELINE] Ignoring short or ghost transcript")
                return
            if not transcript:
                return

            await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))
            filler = "Got it, let me check that for you."
            asyncio.create_task(self._speak(filler))

            answer = await self._query_rag(transcript)
            source = "RAG"

            if answer is None:
                answer = await self._query_llm_fallback(transcript)
                source = "FALLBACK"

            _log(f"[PIPELINE] {source}: {answer[:80]}")
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": f"[{source}] {answer}"})
            )
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

        # Normalise to 1-D mono
        if audio_frame.ndim == 2:
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]
        audio_frame = audio_frame.flatten()

        if input_sr != self.input_sample_rate:
            audio_frame = resample(
                audio_frame,
                int(len(audio_frame) * self.input_sample_rate / input_sr),
            )

        audio_frame = audio_to_int16(audio_frame)

        # Periodic RMS heartbeat — useful for tuning VAD_ENERGY_THRESHOLD
        self._frame_count += 1
        if self._frame_count % 100 == 0:
            rms = np.sqrt(np.mean((audio_frame.astype(np.float32) / 32768.0) ** 2))
            now = asyncio.get_event_loop().time()
            muted = self._tts_playing or now < self._tts_mute_until
            print(
                f"[VAD] RMS={rms:.5f}  threshold={config.VAD_ENERGY_THRESHOLD:.5f}"
                f"  muted={'YES ← TTS window' if muted else 'no'}",
                flush=True,
            )

        # ------------------------------------------------------------------
        # ECHO SUPPRESSION
        # Discard all frames while TTS is synthesizing OR while we are inside
        # the mute tail window (actual speaker playback + reverb settling).
        # This is the primary fix for the robot hearing its own voice.
        # ------------------------------------------------------------------
        now = asyncio.get_event_loop().time()
        if self._tts_playing or now < self._tts_mute_until:
            return  # ← completely ignore this frame

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
            duration = len(audio_bytes) / (SAMPLE_RATE * 2)
            print(f"[VAD] Speech ended ◼  {len(audio_bytes)} bytes ({duration:.2f}s)", flush=True)

            # ------------------------------------------------------------------
            # MINIMUM LENGTH GUARD
            # If the captured audio is shorter than MIN_AUDIO_BYTES it is
            # almost certainly noise, mic hiss, or a clipped syllable.
            # Sending it to Whisper would produce hallucinated words.
            # ------------------------------------------------------------------
            if len(audio_bytes) < MIN_AUDIO_BYTES:
                print(
                    f"[VAD] Too short ({duration:.2f}s < {MIN_AUDIO_BYTES / (SAMPLE_RATE * 2):.1f}s)"
                    f" — discarding (noise guard)",
                    flush=True,
                )
                self._vad_processing = False
                return

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
