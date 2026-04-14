"""Audio stream handler for Reachy Mini.

Pipeline (mirrors OpenAI realtime handler):
  VAD → Groq Whisper STT → Stuart AI RAG (+ Groq LLM fallback) → Groq Orpheus TTS

Key behaviours that match the OpenAI realtime experience:
  - TTS audio is streamed to the speaker in 200 ms chunks as fast as Groq
    returns them, so the robot starts speaking with minimal latency.
  - VAD runs on every frame — even during playback — so the user can
    interrupt (barge-in) at any time by speaking above the barge-in threshold.
  - On barge-in: TTS task is cancelled instantly, the audio queue is drained,
    and the new utterance is processed from scratch.
  - Echo guard: after TTS finishes, speech_ended events are discarded for
    ECHO_GUARD_SEC to absorb room reverb without blocking barge-in.
  - Concurrent processing: STT + RAG/LLM run while silence is still being
    detected, so the robot's first TTS chunk is queued as soon as the LLM
    returns — not after an extra silence timeout.
"""

import re
import io
import wave
import base64
import asyncio
import logging
from typing import Any, Final, Tuple, Literal, Optional

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

# ── Tuning constants ─────────────────────────────────────────────────────────

# Minimum captured audio before STT is called.
# Shorter clips are noise / clipped syllables — Whisper hallucinates on them.
MIN_AUDIO_BYTES = int(SAMPLE_RATE * 2 * 1.5)          # 1.5 s

# How long after TTS finishes to discard speech_ended (room reverb guard).
# VAD still RUNS during this window — barge-in (speech_started) still works.
ECHO_GUARD_SEC = 0.5

# RMS multiplier required to accept speech as intentional barge-in while TTS
# is playing.  Speaker bleed is ~0.01–0.02 RMS; real voice is >0.10.
# Raise if robot interrupts itself; lower if users have to shout.
BARGE_IN_MULTIPLIER = 1.7

# How many consecutive frames above the barge-in threshold are required
# before we fire the interrupt (prevents a single noise spike from firing).
BARGE_IN_FRAMES_REQUIRED = 3

# Seconds after TTS synthesis starts during which barge-in is ignored.
# Prevents the very first burst of TTS audio from triggering a self-interrupt.
BARGE_IN_COOLDOWN_SEC = 0.6

# TTS chunk size streamed to the speaker queue.  200 ms at 24 kHz.
TTS_CHUNK_SAMPLES = 4800

# ── Whisper domain prompt ─────────────────────────────────────────────────────
WHISPER_PROMPT = (
    "NOI Techpark, Reachy, laser cutter, laser engraver, 3D printer, CNC, "
    "Fab Lab, FabLab, soldering, oscilloscope, multimeter, filament, resin, "
    "G-code, CAD, CAM, PLA, ABS, PETG, power, wattage, bed temperature, "
    "nozzle, extruder, stepper motor, enclosure, ventilation, safety, "
    "software, firmware, Ultimaker, Prusa, Bambu, Snapmaker, Glowforge, "
    "Trotec, xTool, access control, booking, training, certification."
)

# ── RAG ───────────────────────────────────────────────────────────────────────
RAG_EMPTY_RESPONSES = {
    "i don't know", "i do not know",
    "no information available", "no information found",
    "i cannot answer", "i can't answer", "",
}

# ── Fallback LLM ──────────────────────────────────────────────────────────────
FALLBACK_CHAT_MODEL = "llama-3.1-8b-instant"

FALLBACK_SYSTEM_PROMPT = (
    "You are Reachy Mini, a helpful robotic assistant at NOI Techpark's Fab Lab. "
    "Your personality is friendly, professional, and slightly high-tech. "
    "RULES:\n"
    "1. The internal knowledge base had no answer — answer from general knowledge.\n"
    "2. Always respond in the SAME LANGUAGE as the user's question.\n"
    "3. Keep answers SHORT — 2-3 sentences max.\n"
    "4. End every factual answer with: 'If you would like more detail, just let me know!'\n"
    "5. NEVER invent specific NOI Techpark machine specs or policies.\n"
    "6. Use occasional tech metaphors but stay concise."
)

# ─────────────────────────────────────────────────────────────────────────────

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

def _strip_md(text: str) -> str:
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"`+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text.strip()


# =============================================================================
class StuartRealtimeHandler(AsyncStreamHandler):
    """
    Drop-in replacement for OpenaiRealtimeHandler.

    Identical external interface (copy / start_up / shutdown / receive / emit /
    apply_personality) so app.py and console.py need zero changes.
    """

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

        self.deps          = deps
        self.gradio_mode   = gradio_mode
        self.instance_path = instance_path

        # Output queue — carries (sample_rate, audio_array) or AdditionalOutputs
        self.output_queue: asyncio.Queue = asyncio.Queue()

        self._shutdown_requested = False
        self._connected_event    = asyncio.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # ── TTS task state ────────────────────────────────────────────────
        # _tts_task       : asyncio.Task wrapping _speak(); cancel = instant stop
        # _tts_playing    : True from synthesis start until all chunks queued
        # _tts_start_time : monotonic time when TTS synthesis began (cooldown ref)
        # _tts_mute_until : echo guard expires at this monotonic time
        self._tts_task:       Optional[asyncio.Task] = None
        self._tts_playing:    bool  = False
        self._tts_start_time: float = 0.0
        self._tts_mute_until: float = 0.0

        # Consecutive above-threshold frames while TTS is playing
        self._barge_in_frame_counter: int = 0

        # ── VAD ───────────────────────────────────────────────────────────
        self._local_vad = LocalVAD(
            energy_threshold=config.VAD_ENERGY_THRESHOLD,
            silence_duration=config.VAD_SILENCE_DURATION,
            min_speech_duration=config.VAD_MIN_SPEECH_DURATION,
            sample_rate=self.input_sample_rate,
        )
        self._audio_buffer:     list[bytes] = []
        self._is_speech_active: bool = False
        self._vad_processing:   bool = False
        self._frame_count:      int  = 0

        # ── STT ───────────────────────────────────────────────────────────
        self._asr = GroqASR(
            api_key=config.GROQ_API_KEY,
            model=config.GROQ_STT_MODEL,
            language=config.WHISPER_LANGUAGE,
        )

        # ── TTS ───────────────────────────────────────────────────────────
        self._tts = GroqTTS(
            api_key=config.GROQ_API_KEY,
            model=config.GROQ_TTS_MODEL,
            voice=config.GROQ_TTS_VOICE,
            output_sample_rate=self.output_sample_rate,
        )

        _log(f"STT          : Groq Whisper  ({config.GROQ_STT_MODEL})")
        _log(f"TTS          : Groq Orpheus  ({config.GROQ_TTS_MODEL} / {config.GROQ_TTS_VOICE})")
        _log(f"RAG          : Stuart AI     {config.STUART_ENDPOINT}")
        _log(f"FALLBACK LLM : Groq          {FALLBACK_CHAT_MODEL}")
        _log(f"MIN AUDIO    : {MIN_AUDIO_BYTES / (SAMPLE_RATE * 2):.1f}s")
        _log(f"ECHO GUARD   : {ECHO_GUARD_SEC}s   BARGE-IN: ×{BARGE_IN_MULTIPLIER} threshold "
             f"({BARGE_IN_FRAMES_REQUIRED} frames, {BARGE_IN_COOLDOWN_SEC}s cooldown)")
        print("=" * 60, flush=True)
        print("Pipeline: VAD → STT → RAG → LLM fallback → TTS  [barge-in]", flush=True)
        print("=" * 60, flush=True)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def copy(self) -> "StuartRealtimeHandler":
        return StuartRealtimeHandler(self.deps, self.gradio_mode, self.instance_path)

    async def start_up(self) -> None:
        self._loop = asyncio.get_event_loop()
        _log("Session started — listening for speech")
        self._connected_event.set()
        while not self._shutdown_requested:
            await asyncio.sleep(0.1)

    async def shutdown(self) -> None:
        self._shutdown_requested = True
        await self._cancel_tts(reason="shutdown")
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def apply_personality(self, profile: str | None) -> str:
        """Runtime personality switch — mirrors OpenaiRealtimeHandler API."""
        try:
            from reachy_mini_conversation_app.config import set_custom_profile
            set_custom_profile(profile)
            _log(f"Personality set to {profile!r}")
            return "Applied personality. Takes effect on next utterance."
        except Exception as exc:
            _err(f"apply_personality failed: {exc}")
            return f"Failed to apply personality: {exc}"

    # ── TTS cancel (barge-in / shutdown) ─────────────────────────────────────

    async def _cancel_tts(self, reason: str = "barge-in") -> None:
        """Cancel in-flight TTS and drain the audio queue instantly."""
        if self._tts_task and not self._tts_task.done():
            self._tts_task.cancel()
            try:
                await self._tts_task
            except asyncio.CancelledError:
                pass
        self._tts_task    = None
        self._tts_playing = False

        # Drain queued audio; keep AdditionalOutputs (UI chat messages)
        kept: list = []
        while not self.output_queue.empty():
            try:
                item = self.output_queue.get_nowait()
                if isinstance(item, AdditionalOutputs):
                    kept.append(item)
            except asyncio.QueueEmpty:
                break
        for item in kept:
            await self.output_queue.put(item)

        # Reset VAD so it doesn't carry stale state into the new utterance
        self._audio_buffer.clear()
        self._is_speech_active    = False
        self._barge_in_frame_counter = 0

        _log(f"[{reason.upper()}] TTS cancelled, queue drained, VAD reset")

    # ── STT ───────────────────────────────────────────────────────────────────

    async def _transcribe(self, audio_bytes: bytes) -> Optional[str]:
        duration = len(audio_bytes) / (SAMPLE_RATE * 2)
        _log(f"[STT] Transcribing {len(audio_bytes)} B ({duration:.2f}s)…")
        wav = _pcm_to_wav(audio_bytes, self.input_sample_rate)
        try:
            form = aiohttp.FormData()
            form.add_field("file", wav, filename="audio.wav", content_type="audio/wav")
            form.add_field("model",           config.GROQ_STT_MODEL)
            form.add_field("language",        config.WHISPER_LANGUAGE)
            form.add_field("response_format", "json")
            form.add_field("prompt",          WHISPER_PROMPT)

            async with aiohttp.ClientSession() as s:
                async with s.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
                    data=form,
                    timeout=aiohttp.ClientTimeout(total=30.0),
                ) as resp:
                    if resp.status != 200:
                        _err(f"Groq STT HTTP {resp.status} — GroqASR fallback")
                        return await self._asr.transcribe(audio_bytes, self.input_sample_rate)
                    result = await resp.json()

            text = result.get("text", "").strip()
            if not text:
                _log("[STT] Empty transcript")
                return None
            _log(f"[STT] → {text}")
            return text

        except Exception as exc:
            _err(f"STT error ({exc}) — GroqASR fallback")
            try:
                return await self._asr.transcribe(audio_bytes, self.input_sample_rate)
            except Exception as exc2:
                _err(f"STT fallback failed: {exc2}")
                return None

    # ── RAG ───────────────────────────────────────────────────────────────────

    async def _query_rag(self, question: str) -> Optional[str]:
        _log(f"[RAG] Query: {question[:100]}")
        try:
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
            _log(f"[RAG] Raw: {raw[:100]}")
            if raw.lower().rstrip(" .!") in RAG_EMPTY_RESPONSES:
                _log("[RAG] No useful answer → LLM fallback")
                return None
            return _strip_md(raw)

        except asyncio.TimeoutError:
            _err("RAG timed out")
            return None
        except Exception as exc:
            _err(f"RAG error: {exc}")
            return None

    # ── Fallback LLM ─────────────────────────────────────────────────────────

    async def _query_llm(self, question: str) -> str:
        _log(f"[LLM] Query: {question[:100]}")
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
                        _err(f"LLM HTTP {resp.status}: {await resp.text()}")
                        return "My knowledge base didn't have that — please ask a staff member!"
                    result = await resp.json()

            answer = _strip_md(result["choices"][0]["message"]["content"].strip())
            _log(f"[LLM] → {answer[:100]}")
            return answer

        except asyncio.TimeoutError:
            _err("LLM timed out")
            return "My knowledge base didn't have that — please ask a staff member!"
        except Exception as exc:
            _err(f"LLM error: {exc}")
            return "Something went wrong — please try again!"

    # ── TTS (cancellable task) ────────────────────────────────────────────────

    async def _speak(self, text: str) -> None:
        """
        Synthesize audio via Groq TTS and stream it to the output queue in
        TTS_CHUNK_SAMPLES chunks.

        This is always run as an asyncio.Task so it can be cancelled instantly
        by _cancel_tts() on barge-in.  The await asyncio.sleep(0) between
        chunks yields control to the event loop so cancellation is processed
        between chunks rather than only at the start or end.
        """
        if not text:
            return
        try:
            self._tts_playing    = True
            self._tts_start_time = asyncio.get_event_loop().time()
            _log(f"[TTS] Synthesizing ({len(text)} chars)…")

            audio_data = await self._tts.synthesize(text)
            if audio_data is None:
                _err("[TTS] Synthesis returned None")
                return

            playback_sec = len(audio_data) / SAMPLE_RATE
            self._tts_mute_until = asyncio.get_event_loop().time() + playback_sec + ECHO_GUARD_SEC
            _log(f"[TTS] {len(audio_data)} samples ({playback_sec:.1f}s) — "
                 f"echo guard +{playback_sec + ECHO_GUARD_SEC:.1f}s")

            # Feed head wobbler with full audio for sync
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.feed(
                    base64.b64encode(audio_data.tobytes()).decode("utf-8")
                )

            # Stream chunks — yield after each so barge-in can fire between them
            for i in range(0, len(audio_data), TTS_CHUNK_SAMPLES):
                chunk = audio_data[i: i + TTS_CHUNK_SAMPLES]
                await self.output_queue.put(
                    (self.output_sample_rate, chunk.reshape(1, -1))
                )
                await asyncio.sleep(0)   # ← cancellation point

            _log("[TTS] All chunks queued ✓")

        except asyncio.CancelledError:
            _log("[TTS] Cancelled by barge-in ✓")
            raise   # must re-raise so Task is marked cancelled
        except Exception as exc:
            _err(f"TTS failed: {exc}")
        finally:
            self._tts_playing = False

    # ── Full speech pipeline ──────────────────────────────────────────────────

    async def _process_speech(self, audio_bytes: bytes) -> None:
        """
        PCM audio → STT → RAG → (LLM fallback) → TTS

        Mirrors the OpenAI event flow:
          speech_started  → deps.movement_manager.set_listening(True)   [in receive()]
          transcript done → push user message to UI queue
          response done   → push assistant message + start TTS
        """
        try:
            # 1. STT
            transcript = await self._transcribe(audio_bytes)
            if not transcript:
                return

            # Mirrors: conversation.item.input_audio_transcription.completed
            await self.output_queue.put(
                AdditionalOutputs({"role": "user", "content": transcript})
            )

            # 2. RAG → LLM fallback
            answer = await self._query_rag(transcript)
            source = "RAG"
            if answer is None:
                answer = await self._query_llm(transcript)
                source = "LLM"

            _log(f"[PIPELINE] [{source}] {answer[:80]}")

            # Mirrors: response.output_audio_transcript.done
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": f"[{source}] {answer}"})
            )

            # 3. TTS — wrapped as cancellable Task (mirrors response.output_audio.delta stream)
            self._tts_task = asyncio.create_task(self._speak(answer))
            await self._tts_task

        except asyncio.CancelledError:
            # _process_speech itself was cancelled (unlikely but safe)
            pass
        except Exception as exc:
            _err(f"_process_speech unhandled: {exc}")
        finally:
            self._vad_processing = False

    # ── Audio receive (called every frame by FastRTC) ─────────────────────────

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        input_sr, audio_frame = frame

        # ── Normalise to 1-D mono ────────────────────────────────────────
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

        # ── RMS for barge-in decisions ───────────────────────────────────
        rms = float(np.sqrt(np.mean((audio_frame.astype(np.float32) / 32768.0) ** 2)))

        # Periodic heartbeat log
        self._frame_count += 1
        if self._frame_count % 100 == 0:
            now = asyncio.get_event_loop().time()
            print(
                f"[VAD] RMS={rms:.5f}  thr={config.VAD_ENERGY_THRESHOLD:.5f}"
                f"  tts={'ON' if self._tts_playing else 'off'}"
                f"  echo={'YES' if now < self._tts_mute_until else 'no'}",
                flush=True,
            )

        now = asyncio.get_event_loop().time()

        # ── BARGE-IN DETECTION ───────────────────────────────────────────
        # VAD runs on EVERY frame. When TTS is playing we apply an extra
        # cooldown + consecutive-frame check to distinguish real speech from
        # speaker bleed or a single noise spike.
        if self._tts_playing:
            past_cooldown = (now - self._tts_start_time) > BARGE_IN_COOLDOWN_SEC
            above_threshold = rms > (config.VAD_ENERGY_THRESHOLD * BARGE_IN_MULTIPLIER)

            if past_cooldown and above_threshold:
                self._barge_in_frame_counter += 1
            else:
                self._barge_in_frame_counter = 0

            if self._barge_in_frame_counter >= BARGE_IN_FRAMES_REQUIRED:
                _log(f"[BARGE-IN] RMS={rms:.4f} × {BARGE_IN_FRAMES_REQUIRED} frames — interrupting")
                await self._cancel_tts(reason="barge-in")
                # Clear echo guard so the new utterance isn't suppressed
                self._tts_mute_until = now + 0.3
                # Fall through — VAD processes this frame normally

        # ── ECHO GUARD ───────────────────────────────────────────────────
        # After TTS ends (or is cancelled), discard speech_ended for a short
        # tail so room reverb doesn't fire a false utterance.
        # speech_started is NOT blocked — that's how barge-in enters.
        in_echo_guard = now < self._tts_mute_until

        # ── VAD ──────────────────────────────────────────────────────────
        speech_started, speech_ended = self._local_vad.process(audio_frame)

        if speech_started and not self._vad_processing:
            self._is_speech_active = True
            self._audio_buffer.clear()
            self.deps.movement_manager.set_listening(True)
            print("[VAD] ▶ Speech started", flush=True)

        if self._is_speech_active:
            self._audio_buffer.append(audio_frame.tobytes())

        if speech_ended and not self._vad_processing:

            if in_echo_guard:
                # Likely room reverb from our own TTS — discard silently
                _log("[VAD] speech_ended in echo guard — discarded (reverb)")
                self._is_speech_active = False
                self._audio_buffer.clear()
                self.deps.movement_manager.set_listening(False)
                return

            self._vad_processing    = True
            self._is_speech_active  = False
            self.deps.movement_manager.set_listening(False)

            audio_bytes = b"".join(self._audio_buffer)
            self._audio_buffer.clear()
            dur = len(audio_bytes) / (SAMPLE_RATE * 2)
            print(f"[VAD] ◼ Speech ended — {len(audio_bytes)} B ({dur:.2f}s)", flush=True)

            # ── Minimum length guard ─────────────────────────────────────
            if len(audio_bytes) < MIN_AUDIO_BYTES:
                print(
                    f"[VAD] Too short ({dur:.2f}s < {MIN_AUDIO_BYTES / (SAMPLE_RATE * 2):.1f}s)"
                    " — noise guard, discarding",
                    flush=True,
                )
                self._vad_processing = False
                return

            # ── Dispatch pipeline ────────────────────────────────────────
            coro = self._process_speech(audio_bytes)
            if self._loop is not None and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(coro, self._loop)
            else:
                asyncio.ensure_future(coro)

    # ── Audio emit (called by FastRTC to get audio for the speaker) ───────────

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        return await wait_for_item(self.output_queue)   # type: ignore[no-any-return]