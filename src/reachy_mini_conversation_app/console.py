"""Bidirectional local audio stream for Reachy Mini conversation app.

Headless mode only — no OpenAI key required.
LLM is handled by Stuart AI, STT/TTS by Groq.
"""

from __future__ import annotations

import os
import asyncio
import logging
from typing import List, Optional
from pathlib import Path

from fastrtc import AdditionalOutputs, audio_to_float32
from scipy.signal import resample

from reachy_mini import ReachyMini
from reachy_mini.media.media_manager import MediaBackend
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.stuart_realtime import StuartRealtimeHandler
from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes


try:
    from fastapi import FastAPI, Response
    from pydantic import BaseModel
    from fastapi.responses import FileResponse, JSONResponse
    from starlette.staticfiles import StaticFiles
except Exception:
    FastAPI = object  # type: ignore
    FileResponse = object  # type: ignore
    JSONResponse = object  # type: ignore
    StaticFiles = object  # type: ignore
    BaseModel = object  # type: ignore


logger = logging.getLogger(__name__)


class LocalStream:
    """LocalStream using Reachy Mini's recorder/player."""

    def __init__(
        self,
        handler: StuartRealtimeHandler,
        robot: ReachyMini,
        *,
        settings_app: Optional[FastAPI] = None,
        instance_path: Optional[str] = None,
    ):
        self.handler = handler
        self._robot = robot
        self._stop_event = asyncio.Event()
        self._tasks: List[asyncio.Task[None]] = []
        self.handler._clear_queue = self.clear_audio_queue
        self._settings_app: Optional[FastAPI] = settings_app
        self._instance_path: Optional[str] = instance_path
        self._settings_initialized = False
        self._asyncio_loop = None

    # ------------------------------------------------------------------
    # Personality persistence helpers
    # ------------------------------------------------------------------

    def _read_env_lines(self, env_path: Path) -> list[str]:
        """Load env file contents or template as a list of lines."""
        inst = env_path.parent
        try:
            if env_path.exists():
                try:
                    return env_path.read_text(encoding="utf-8").splitlines()
                except Exception:
                    return []
            for candidate in [
                inst / ".env.example",
                Path.cwd() / ".env.example",
                Path(__file__).parent / ".env.example",
            ]:
                if candidate.exists():
                    try:
                        return candidate.read_text(encoding="utf-8").splitlines()
                    except Exception:
                        pass
            return []
        except Exception:
            return []

    def _persist_personality(self, profile: Optional[str]) -> None:
        """Persist the startup personality to the instance .env and config."""
        selection = (profile or "").strip() or None
        try:
            from reachy_mini_conversation_app.config import set_custom_profile
            set_custom_profile(selection)
        except Exception:
            pass

        if not self._instance_path:
            return
        try:
            env_path = Path(self._instance_path) / ".env"
            lines = self._read_env_lines(env_path)
            replaced = False
            for i, ln in enumerate(list(lines)):
                if ln.strip().startswith("REACHY_MINI_CUSTOM_PROFILE="):
                    if selection:
                        lines[i] = f"REACHY_MINI_CUSTOM_PROFILE={selection}"
                    else:
                        lines.pop(i)
                    replaced = True
                    break
            if selection and not replaced:
                lines.append(f"REACHY_MINI_CUSTOM_PROFILE={selection}")
            if selection is None and not env_path.exists():
                return
            env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            logger.info("Persisted startup personality to %s", env_path)
            try:
                from dotenv import load_dotenv
                load_dotenv(dotenv_path=str(env_path), override=True)
            except Exception:
                pass
        except Exception as e:
            logger.warning("Failed to persist REACHY_MINI_CUSTOM_PROFILE: %s", e)

    def _read_persisted_personality(self) -> Optional[str]:
        """Read persisted startup personality from instance .env (if any)."""
        if not self._instance_path:
            return None
        env_path = Path(self._instance_path) / ".env"
        try:
            if env_path.exists():
                for ln in env_path.read_text(encoding="utf-8").splitlines():
                    if ln.strip().startswith("REACHY_MINI_CUSTOM_PROFILE="):
                        _, _, val = ln.partition("=")
                        v = val.strip()
                        return v or None
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Settings UI (personality only — no API key)
    # ------------------------------------------------------------------

    def _init_settings_ui_if_needed(self) -> None:
        """Attach minimal settings UI to the settings app."""
        if self._settings_initialized or self._settings_app is None:
            return

        static_dir = Path(__file__).parent / "static"
        index_file = static_dir / "index.html"

        if hasattr(self._settings_app, "mount"):
            try:
                self._settings_app.mount(
                    "/static", StaticFiles(directory=str(static_dir)), name="static"
                )
            except Exception:
                pass

        @self._settings_app.get("/")
        def _root() -> FileResponse:
            return FileResponse(str(index_file))

        @self._settings_app.get("/favicon.ico")
        def _favicon() -> Response:
            return Response(status_code=204)

        @self._settings_app.get("/status")
        def _status() -> "JSONResponse":
            return JSONResponse({"ok": True, "llm": "stuart", "stt": "groq", "tts": "groq"})

        @self._settings_app.get("/ready")
        def _ready() -> "JSONResponse":
            import sys
            try:
                mod = sys.modules.get("reachy_mini_conversation_app.tools.core_tools")
                ready = bool(getattr(mod, "_TOOLS_INITIALIZED", False)) if mod else False
            except Exception:
                ready = False
            return JSONResponse({"ready": ready})

        self._settings_initialized = True

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    def launch(self) -> None:
        """Start recorder/player and run async processing loops."""
        self._stop_event.clear()

        # Load persisted personality if available
        if self._instance_path:
            try:
                from dotenv import load_dotenv
                from reachy_mini_conversation_app.config import set_custom_profile

                env_path = Path(self._instance_path) / ".env"
                if env_path.exists():
                    load_dotenv(dotenv_path=str(env_path), override=True)
                    new_profile = os.getenv("REACHY_MINI_CUSTOM_PROFILE")
                    if new_profile is not None:
                        try:
                            set_custom_profile(new_profile.strip() or None)
                        except Exception:
                            pass
            except Exception:
                pass

        self._init_settings_ui_if_needed()

        # Start media
        self._robot.media.start_recording()
        self._robot.media.start_playing()

        import time
        time.sleep(1)  # give pipelines time to start

        async def runner() -> None:
            loop = asyncio.get_running_loop()
            self._asyncio_loop = loop

            try:
                if self._settings_app is not None:
                    mount_personality_routes(
                        self._settings_app,
                        self.handler,
                        lambda: self._asyncio_loop,
                        persist_personality=self._persist_personality,
                        get_persisted_personality=self._read_persisted_personality,
                    )
            except Exception:
                pass

            self._tasks = [
                asyncio.create_task(self.handler.start_up(), name="stuart-handler"),
                asyncio.create_task(self.record_loop(), name="stream-record-loop"),
                asyncio.create_task(self.play_loop(), name="stream-play-loop"),
            ]
            try:
                await asyncio.gather(*self._tasks)
            except asyncio.CancelledError:
                logger.info("Tasks cancelled during shutdown")
            finally:
                await self.handler.shutdown()

        asyncio.run(runner())

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Stop the stream and underlying media pipelines."""
        logger.info("Stopping LocalStream...")

        try:
            self._robot.media.stop_recording()
        except Exception as e:
            logger.debug("Error stopping recording: %s", e)

        try:
            self._robot.media.stop_playing()
        except Exception as e:
            logger.debug("Error stopping playback: %s", e)

        self._stop_event.set()

        for task in self._tasks:
            if not task.done():
                task.cancel()

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    def clear_audio_queue(self) -> None:
        """Flush the player's queue to drop queued audio immediately."""
        logger.info("Flushing player queue")
        if self._robot.media.backend == MediaBackend.GSTREAMER:
            self._robot.media.audio.clear_player()
        self.handler.output_queue = asyncio.Queue()

    async def record_loop(self) -> None:
        """Read mic frames from the recorder and forward to the handler."""
        input_sample_rate = self._robot.media.get_input_audio_samplerate()
        logger.debug("Audio recording started at %d Hz", input_sample_rate)

        while not self._stop_event.is_set():
            audio_frame = self._robot.media.get_audio_sample()
            if audio_frame is not None:
                await self.handler.receive((input_sample_rate, audio_frame))
            await asyncio.sleep(0)

    async def play_loop(self) -> None:
        """Fetch outputs from handler — log text, play audio."""
        while not self._stop_event.is_set():
            handler_output = await self.handler.emit()

            if isinstance(handler_output, AdditionalOutputs):
                for msg in handler_output.args:
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        logger.info(
                            "role=%s content=%s",
                            msg.get("role"),
                            content if len(content) < 500 else content[:500] + "…",
                        )

            elif isinstance(handler_output, tuple):
                input_sample_rate, audio_data = handler_output
                output_sample_rate = self._robot.media.get_output_audio_samplerate()

                if audio_data.ndim == 2:
                    if audio_data.shape[1] > audio_data.shape[0]:
                        audio_data = audio_data.T
                    if audio_data.shape[1] > 1:
                        audio_data = audio_data[:, 0]

                audio_frame = audio_to_float32(audio_data)

                if input_sample_rate != output_sample_rate:
                    audio_frame = resample(
                        audio_frame,
                        int(len(audio_frame) * output_sample_rate / input_sample_rate),
                    )

                self._robot.media.push_audio_sample(audio_frame)

            else:
                logger.debug("Ignoring output type=%s", type(handler_output).__name__)

            await asyncio.sleep(0)