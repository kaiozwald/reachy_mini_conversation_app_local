import os
import logging

from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)

dotenv_path = find_dotenv(usecwd=True)
if dotenv_path:
    load_dotenv(dotenv_path=dotenv_path, override=True)
    logger.info(f"Config loaded from {dotenv_path}")
else:
    logger.warning("No .env file found — using environment variables only")


class Config:
    """Runtime configuration for the Reachy Mini conversation app."""

    # ── Mode ──────────────────────────────────────────────────────────────────
    FULL_LOCAL_MODE = os.getenv("FULL_LOCAL_MODE", "true").lower().strip() in ("true", "1", "yes")
    OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "not-needed")
    MODEL_NAME      = os.getenv("MODEL_NAME", "gpt-4o-realtime-preview-2024-10-01")

    # ── Hardware ───────────────────────────────────────────────────────────────
    JETSON_OPTIMIZE = os.getenv("JETSON_OPTIMIZE", "false").lower().strip() in ("true", "1", "yes")

    # ── Vision ────────────────────────────────────────────────────────────────
    HF_HOME            = os.getenv("HF_HOME", "./cache")
    LOCAL_VISION_MODEL = os.getenv("LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    HF_TOKEN           = os.getenv("HF_TOKEN")
    REACHY_MINI_CUSTOM_PROFILE = os.getenv("REACHY_MINI_CUSTOM_PROFILE")

    # ── LLM — Stuart AI RAG endpoint ─────────────────────────────────────────
    # POSTs form-data field "question", expects JSON {"manswer": "..."} back.
    LLM_PROVIDER    = "stuart"
    STUART_ENDPOINT = os.getenv("STUART_ENDPOINT", "")

    # ── STT — Groq Whisper ────────────────────────────────────────────────────
    STT_PROVIDER     = os.getenv("STT_PROVIDER",     "groq")
    GROQ_API_KEY     = os.getenv("GROQ_API_KEY",     "")
    GROQ_STT_MODEL   = os.getenv("GROQ_STT_MODEL",   "whisper-large-v3-turbo")
    WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")

    # ── TTS — Groq Orpheus ────────────────────────────────────────────────────
    TTS_PROVIDER   = os.getenv("TTS_PROVIDER",   "groq")
    GROQ_TTS_MODEL = os.getenv("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english")
    GROQ_TTS_VOICE = os.getenv("GROQ_TTS_VOICE", "daniel")

    # Kept so nothing raises AttributeError if referenced elsewhere
    KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_sarah")
    KOKORO_SPEED = float(os.getenv("KOKORO_SPEED", "1.0"))

    # ── VAD ───────────────────────────────────────────────────────────────────
    # VAD_ENERGY_THRESHOLD
    #   RMS level above which audio is classified as speech.
    #   Your mic noise floor measured ~0.003-0.006 RMS; real speech ~0.18.
    #   Default 0.05 sits cleanly between them.
    #   Raise if false triggers persist; lower for quiet speakers.
    #
    # VAD_SILENCE_DURATION
    #   Seconds of silence after last speech frame before the utterance is
    #   declared complete and sent to STT.  1.2 s prevents mid-sentence cutoff.
    #
    # VAD_MIN_SPEECH_DURATION
    #   Minimum seconds of detected speech before VAD fires speech_started.
    #   0.4 s filters out click/pop false starts.

    VAD_ENERGY_THRESHOLD    = float(os.getenv("VAD_ENERGY_THRESHOLD",    "0.05"))
    VAD_SILENCE_DURATION    = float(os.getenv("VAD_SILENCE_DURATION",    "1.2"))
    VAD_MIN_SPEECH_DURATION = float(os.getenv("VAD_MIN_SPEECH_DURATION", "0.4"))
    LOCAL_VAD_ENDPOINT      = os.getenv("LOCAL_VAD_ENDPOINT")

    # ── Misc ──────────────────────────────────────────────────────────────────
    DISTIL_WHISPER_MODEL = os.getenv("DISTIL_WHISPER_MODEL", "distil-whisper/distil-small.en")
    ONNX_PROVIDERS       = os.getenv("ONNX_PROVIDERS",       "CPUExecutionProvider")

    # ── Startup log ───────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"RAG (Stuart AI) : {STUART_ENDPOINT or '(STUART_ENDPOINT not set!)'}")
    logger.info(f"STT             : Groq / {GROQ_STT_MODEL}")
    logger.info(f"TTS             : Groq / {GROQ_TTS_MODEL} / {GROQ_TTS_VOICE}")
    logger.info(f"VAD             : threshold={VAD_ENERGY_THRESHOLD}  "
                f"silence={VAD_SILENCE_DURATION}s  min_speech={VAD_MIN_SPEECH_DURATION}s")
    logger.info("=" * 60)

config = Config()

def set_custom_profile(profile: str | None) -> None:
    """Update the active personality profile at runtime."""
    try:
        config.REACHY_MINI_CUSTOM_PROFILE = profile
    except Exception:
        pass
    try:
        if profile:
            os.environ["REACHY_MINI_CUSTOM_PROFILE"] = profile
        else:
            os.environ.pop("REACHY_MINI_CUSTOM_PROFILE", None)
    except Exception:
        pass