import json
import base64
import random
import asyncio
import logging
from typing import Any, Final, Tuple, Literal, Optional
from pathlib import Path
from datetime import datetime
import os

import aiohttp
import numpy as np
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample
from websockets.exceptions import ConnectionClosedError

from reachy_mini_conversation_app.local_audio import GroqASR, GroqTTS, LocalVAD, LocalASR, LocalTTS
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.prompts import get_session_voice, get_session_instructions
from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    get_tool_specs,
    dispatch_tool_call,
)


logger = logging.getLogger(__name__)

OPEN_AI_INPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000
OPEN_AI_OUTPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000


class StuartRealtimeHandler(AsyncStreamHandler):
    """Audio stream handler — VAD → Groq STT → Stuart AI → Groq TTS."""

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
    ):
        super().__init__(
            expected_layout="mono",
            output_sample_rate=OPEN_AI_OUTPUT_SAMPLE_RATE,
            input_sample_rate=OPEN_AI_INPUT_SAMPLE_RATE,
        )

        self.output_sample_rate: Literal[24000] = OPEN_AI_OUTPUT_SAMPLE_RATE
        self.input_sample_rate: Literal[24000] = OPEN_AI_INPUT_SAMPLE_RATE

        self.deps = deps
        self.connection: Any = None
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()

        self.start_time = asyncio.get_event_loop().time()
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path

        self._key_source: Literal["env", "textbox"] = "env"
        self._provided_api_key: str | None = None

        # Debouncing
        self.partial_transcript_task: asyncio.Task[None] | None = None
        self.partial_transcript_sequence: int = 0
        self.partial_debounce_delay = 0.5

        # Lifecycle
        self._shutdown_requested: bool = False
        self._connected_event: asyncio.Event = asyncio.Event()

        # =====================================================================
        # VAD
        # =====================================================================
        self._local_vad = LocalVAD(
            energy_threshold=config.VAD_ENERGY_THRESHOLD,
            silence_duration=config.VAD_SILENCE_DURATION,
            min_speech_duration=config.VAD_MIN_SPEECH_DURATION,
            sample_rate=self.input_sample_rate,
        )
        self._audio_buffer: list[bytes] = []
        self._is_speech_active: bool = False
        self._vad_processing: bool = False
        self._tts_playing: bool = False
        self._frame_count: int = 0

        self._local_vad_endpoint: str | None = config.LOCAL_VAD_ENDPOINT
        if self._local_vad_endpoint:
            logger.info("External VAD enabled at %s", self._local_vad_endpoint)

        # =====================================================================
        # STT — Groq Whisper
        # =====================================================================
        self._local_asr: GroqASR | None = None
        if config.STT_PROVIDER == "groq":
            self._local_asr = GroqASR(
                api_key=config.GROQ_API_KEY,
                model=config.GROQ_STT_MODEL,
                language=config.WHISPER_LANGUAGE,
            )
            logger.info("STT: Groq (%s)", config.GROQ_STT_MODEL)
        else:
            logger.warning("STT_PROVIDER '%s' not supported. Only 'groq' is available.", config.STT_PROVIDER)

        # =====================================================================
        # TTS — Groq Orpheus
        # =====================================================================
        self._local_tts: GroqTTS | None = None
        if config.TTS_PROVIDER == "groq":
            self._local_tts = GroqTTS(
                api_key=config.GROQ_API_KEY,
                model=config.GROQ_TTS_MODEL,
                voice=config.GROQ_TTS_VOICE,
                output_sample_rate=self.output_sample_rate,
            )
            logger.info("TTS: Groq (%s / %s)", config.GROQ_TTS_MODEL, config.GROQ_TTS_VOICE)
        else:
            logger.warning("TTS_PROVIDER '%s' not supported. Only 'groq' is available.", config.TTS_PROVIDER)

        # =====================================================================
        # LLM — Stuart AI
        # =====================================================================
        self._stuart_endpoint: str | None = config.STUART_ENDPOINT or None
        if self._stuart_endpoint:
            logger.info("LLM: Stuart AI at %s", self._stuart_endpoint)
        else:
            logger.error("STUART_ENDPOINT is not set — LLM will not work.")

        self._conversation_history: list[dict[str, Any]] = []

        logger.info("=" * 60)
        logger.info("FULL LOCAL MODE: VAD+STT+TTS on device, LLM via Stuart AI")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def _is_full_local_mode(self) -> bool:
        return True

    def copy(self) -> "StuartRealtimeHandler":
        return StuartRealtimeHandler(self.deps, self.gradio_mode, self.instance_path)

    # ------------------------------------------------------------------
    # Stuart AI — LLM
    # ------------------------------------------------------------------

    async def _generate_stuart_response(self, user_message: str) -> None:
        """POST question to Stuart AI, read manswer, pipe to TTS."""
        if not self._stuart_endpoint:
            logger.warning("Stuart AI endpoint not configured")
            return

        try:
            logger.info("Stuart AI request: %s", user_message[:100])

            async with aiohttp.ClientSession() as session:
                form_data = aiohttp.FormData()
                form_data.add_field("question", user_message)

                async with session.post(
                    self._stuart_endpoint,
                    data=form_data,
                    timeout=aiohttp.ClientTimeout(total=60.0),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(
                            "Stuart AI failed (HTTP %d): %s", resp.status, error_text[:200]
                        )
                        await self.output_queue.put(
                            AdditionalOutputs(
                                {"role": "assistant", "content": f"[error] Stuart AI HTTP {resp.status}"}
                            )
                        )
                        return

                    result = await resp.json()

            text_response = result.get("manswer", "").strip()

            if not text_response:
                logger.warning("Stuart AI returned empty response")
                return

            # Strip markdown formatting
            import re
            text_response = re.sub(r"\*+", "", text_response)
            text_response = re.sub(r"^#+\s*", "", text_response, flags=re.MULTILINE)
            text_response = text_response.strip()

            logger.info("Stuart AI response: %s", text_response[:100])

            # Update conversation history
            self._conversation_history.append({"role": "user", "content": user_message})
            self._conversation_history.append({"role": "assistant", "content": text_response})

            # Show in UI
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": text_response})
            )

            # Speak
            await self._synthesize_locally(text_response)

        except asyncio.TimeoutError:
            logger.error("Stuart AI request timed out")
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": "[error] Stuart AI timed out"})
            )
        except Exception as e:
            logger.error("Stuart AI failed: %s", e)
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": f"[error] Stuart AI: {e}"})
            )

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    async def _synthesize_locally(self, text: str) -> None:
        """Send text to Groq TTS and queue audio for playback."""
        if not text or not text.strip():
            return

        if not self._local_tts:
            logger.warning("No TTS provider available")
            return

        try:
            self._tts_playing = True
            audio_data = await self._local_tts.synthesize(text)
            if audio_data is not None:
                if self.deps.head_wobbler is not None:
                    self.deps.head_wobbler.feed(
                        base64.b64encode(audio_data.tobytes()).decode("utf-8")
                    )
                chunk_size = 4800  # 200ms at 24kHz
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i: i + chunk_size]
                    await self.output_queue.put(
                        (self.output_sample_rate, chunk.reshape(1, -1))
                    )
                logger.debug("TTS complete")
            else:
                logger.warning("TTS returned no audio")
        except Exception as e:
            logger.error("TTS failed: %s", e)
        finally:
            self._tts_playing = False

    # ------------------------------------------------------------------
    # ASR
    # ------------------------------------------------------------------

    async def _transcribe_with_local_asr(self, audio_data: bytes) -> str | None:
        """Send buffered audio to Groq Whisper and return transcript."""
        if not self._local_asr:
            logger.warning("No ASR provider available")
            return None
        try:
            transcript = await self._local_asr.transcribe(audio_data, self.input_sample_rate)
            if transcript:
                return transcript
            logger.warning("ASR returned empty result")
            return None
        except Exception as e:
            logger.error("ASR transcription failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Speech processing pipeline
    # ------------------------------------------------------------------

    async def _process_local_speech(self, audio_data: bytes) -> None:
        """Full pipeline: audio bytes → STT → Stuart AI → TTS."""
        try:
            transcript = await self._transcribe_with_local_asr(audio_data)
            if not transcript:
                logger.warning("No transcription — skipping")
                return

            logger.info("Transcript: %s", transcript)
            await self.output_queue.put(
                AdditionalOutputs({"role": "user", "content": transcript})
            )

            await self._generate_stuart_response(transcript)

        finally:
            self._vad_processing = False

    # ------------------------------------------------------------------
    # VAD turn check (external endpoint, optional)
    # ------------------------------------------------------------------

    async def _check_turn_complete(self, audio_data: bytes) -> bool:
        if not self._local_vad_endpoint:
            return True
        try:
            import tempfile
            import wave

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                with wave.open(f, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(self.input_sample_rate)
                    wav.writeframes(audio_data)

            with open(temp_path, "rb") as f:
                audio_bytes = f.read()
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

            try:
                os.unlink(temp_path)
            except Exception:
                pass

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._local_vad_endpoint}/predict",
                    json={"audio_base64": audio_b64},
                    timeout=aiohttp.ClientTimeout(total=5.0),
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result.get("prediction", 1) == 1
                    return True
        except Exception as e:
            logger.error("VAD check failed: %s", e)
            return True

    # ------------------------------------------------------------------
    # Personality (runtime profile switch)
    # ------------------------------------------------------------------

    async def apply_personality(self, profile: str | None) -> str:
        try:
            from reachy_mini_conversation_app.config import set_custom_profile
            set_custom_profile(profile)
            logger.info("Profile set to %r", profile)
            return "Applied personality. Will take effect on next utterance."
        except Exception as e:
            logger.error("Error applying personality '%s': %s", profile, e)
            return f"Failed to apply personality: {e}"

    # ------------------------------------------------------------------
    # Startup / session loop
    # ------------------------------------------------------------------

    async def start_up(self) -> None:
        logger.info(
            "Startup: FULL_LOCAL_MODE=%s LLM_PROVIDER=%s",
            config.FULL_LOCAL_MODE,
            config.LLM_PROVIDER,
        )
        logger.info("Starting in FULL LOCAL MODE — Stuart AI for LLM")
        await self._run_local_only_session()

    async def _run_local_only_session(self) -> None:
        """Keep the session alive; audio processing happens in receive()."""
        logger.info("Local session started — waiting for audio")
        self._connected_event.set()
        while not self._shutdown_requested:
            await asyncio.sleep(0.1)
        logger.info("Local session ended")

    # ------------------------------------------------------------------
    # Audio receive (called by FastRTC per frame)
    # ------------------------------------------------------------------

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        input_sample_rate, audio_frame = frame

        # Reshape to mono
        if audio_frame.ndim == 2:
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        # Resample if needed
        if self.input_sample_rate != input_sample_rate:
            audio_frame = resample(
                audio_frame,
                int(len(audio_frame) * self.input_sample_rate / input_sample_rate),
            )

        audio_frame = audio_to_int16(audio_frame)

        # Periodic energy heartbeat for VAD tuning
        self._frame_count += 1
        if self._frame_count % 50 == 0:
            audio_float = audio_frame.astype(np.float32) / 32768.0
            rms = np.sqrt(np.mean(audio_float ** 2))
            logger.debug("Mic RMS: %.5f (threshold: %.5f)", rms, config.VAD_ENERGY_THRESHOLD)

        # Mute VAD while TTS is playing (prevents echo loop)
        if self._tts_playing:
            return

        speech_started, speech_ended = self._local_vad.process(audio_frame)

        if speech_started:
            self._is_speech_active = True
            self._audio_buffer.clear()
            self.deps.movement_manager.set_listening(True)
            logger.info("VAD: speech started")

        if self._is_speech_active:
            self._audio_buffer.append(audio_frame.tobytes())

        if speech_ended and not self._vad_processing:
            self._vad_processing = True
            self._is_speech_active = False
            self.deps.movement_manager.set_listening(False)

            audio_data = b"".join(self._audio_buffer)
            self._audio_buffer.clear()
            logger.info("VAD: speech ended (%d bytes)", len(audio_data))

            asyncio.create_task(self._process_local_speech(audio_data))

    # ------------------------------------------------------------------
    # Audio emit (called by FastRTC to get audio to play)
    # ------------------------------------------------------------------

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        self._shutdown_requested = True

        if self.partial_transcript_task and not self.partial_transcript_task.done():
            self.partial_transcript_task.cancel()
            try:
                await self.partial_transcript_task
            except asyncio.CancelledError:
                pass

        if self.connection:
            try:
                await self.connection.close()
            except Exception as e:
                logger.debug("connection.close() ignored: %s", e)
            finally:
                self.connection = None

        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def format_timestamp(self) -> str:
        loop_time = asyncio.get_event_loop().time()
        elapsed = loop_time - self.start_time
        from datetime import datetime
        dt = datetime.now()
        return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed:.1f}s]"