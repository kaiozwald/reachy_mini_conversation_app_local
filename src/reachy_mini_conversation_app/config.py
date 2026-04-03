import os
import logging

from dotenv import find_dotenv, load_dotenv


logger = logging.getLogger(__name__)

dotenv_path = find_dotenv(usecwd=True)

if dotenv_path:
    load_dotenv(dotenv_path=dotenv_path, override=True)
    logger.info(f"Configuration loaded from {dotenv_path}")
else:
    logger.warning("No .env file found, using environment variables")


class Config:
    """Configuration class for the conversation app."""

    # =========================================================================
    # MODE
    # =========================================================================
    FULL_LOCAL_MODE = os.getenv("FULL_LOCAL_MODE", "true").lower().strip() in ("true", "1", "yes")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "not-needed")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-realtime-preview-2024-10-01")

    # =========================================================================
    # JETSON OPTIMIZATION
    # =========================================================================
    JETSON_OPTIMIZE = os.getenv("JETSON_OPTIMIZE", "false").lower().strip() in ("true", "1", "yes")

    # =========================================================================
    # VISION CONFIGURATION
    # =========================================================================
    HF_HOME = os.getenv("HF_HOME", "./cache")
    LOCAL_VISION_MODEL = os.getenv("LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    HF_TOKEN = os.getenv("HF_TOKEN")

    REACHY_MINI_CUSTOM_PROFILE = os.getenv("REACHY_MINI_CUSTOM_PROFILE")

    # =========================================================================
    # LLM PROVIDER — only "stuart" is supported
    # =========================================================================
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "stuart").lower().strip()

    # Stuart AI endpoint (POST form-data, returns {"manswer": "..."})
    STUART_ENDPOINT = os.getenv("STUART_ENDPOINT", "http://52.214.229.167:9871/stuart-reachy")

    if LLM_PROVIDER == "stuart":
        logger.info(f"Stuart AI enabled at {STUART_ENDPOINT}")
    else:
        logger.warning(
            f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Only 'stuart' is supported. "
            f"Defaulting to Stuart AI at {STUART_ENDPOINT}."
        )
        LLM_PROVIDER = "stuart"

    # =========================================================================
    # STT — Groq Whisper
    # =========================================================================
    STT_PROVIDER = os.getenv("STT_PROVIDER", "groq")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
    WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")

    # =========================================================================
    # TTS — Groq Orpheus
    # =========================================================================
    TTS_PROVIDER = os.getenv("TTS_PROVIDER", "groq")
    GROQ_TTS_MODEL = os.getenv("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english")
    GROQ_TTS_VOICE = os.getenv("GROQ_TTS_VOICE", "diana")

    # Kokoro fallback vars — kept so stuart_realtime.py never raises AttributeError
    KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_sarah")
    KOKORO_SPEED = float(os.getenv("KOKORO_SPEED", "1.0"))

    # =========================================================================
    # VAD
    # =========================================================================
    VAD_ENERGY_THRESHOLD = float(os.getenv("VAD_ENERGY_THRESHOLD", "0.01"))
    VAD_SILENCE_DURATION = float(os.getenv("VAD_SILENCE_DURATION", "0.8"))
    VAD_MIN_SPEECH_DURATION = float(os.getenv("VAD_MIN_SPEECH_DURATION", "0.3"))
    LOCAL_VAD_ENDPOINT = os.getenv("LOCAL_VAD_ENDPOINT")
    if LOCAL_VAD_ENDPOINT:
        logger.info(f"External VAD enabled at {LOCAL_VAD_ENDPOINT}")

    # =========================================================================
    # DISTIL-WHISPER (kept for LocalASR fallback reference, not actively used)
    # =========================================================================
    DISTIL_WHISPER_MODEL = os.getenv("DISTIL_WHISPER_MODEL", "distil-whisper/distil-small.en")

    # =========================================================================
    # ONNX
    # =========================================================================
    ONNX_PROVIDERS = os.getenv("ONNX_PROVIDERS", "CPUExecutionProvider")

    # =========================================================================
    # STATUS LOG
    # =========================================================================
    logger.info("=" * 60)
    logger.info(f"LLM  : Stuart AI  → {STUART_ENDPOINT}")
    logger.info(f"STT  : Groq       → {GROQ_STT_MODEL}")
    logger.info(f"TTS  : Groq       → {GROQ_TTS_MODEL} / {GROQ_TTS_VOICE}")
    logger.info("=" * 60)


config = Config()


def set_custom_profile(profile: str | None) -> None:
    """Update the selected custom profile at runtime and expose it via env."""
    try:
        config.REACHY_MINI_CUSTOM_PROFILE = profile
    except Exception:
        pass
    try:
        import os as _os
        if profile:
            _os.environ["REACHY_MINI_CUSTOM_PROFILE"] = profile
        else:
            _os.environ.pop("REACHY_MINI_CUSTOM_PROFILE", None)
    except Exception:
        pass