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
    """Configuration for the Reachy Mini conversation app.

    Pipeline: VAD → Groq Whisper STT → Stuart AI RAG → Groq Orpheus TTS
    """

    # =========================================================================
    # RAG / LLM — Stuart AI
    # =========================================================================
    STUART_ENDPOINT = os.getenv(
        "STUART_ENDPOINT", "http://52.214.229.167:9871/stuart-reachy"
    )

    # =========================================================================
    # STT — Groq Whisper
    # =========================================================================
    GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
    GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
    WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")

    # =========================================================================
    # TTS — Groq Orpheus
    # =========================================================================
    GROQ_TTS_MODEL = os.getenv("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english")
    GROQ_TTS_VOICE = os.getenv("GROQ_TTS_VOICE", "diana")

    # =========================================================================
    # VAD
    # =========================================================================
    VAD_ENERGY_THRESHOLD  = float(os.getenv("VAD_ENERGY_THRESHOLD",  "0.01"))
    VAD_SILENCE_DURATION  = float(os.getenv("VAD_SILENCE_DURATION",  "0.8"))
    VAD_MIN_SPEECH_DURATION = float(os.getenv("VAD_MIN_SPEECH_DURATION", "0.3"))
    LOCAL_VAD_ENDPOINT    = os.getenv("LOCAL_VAD_ENDPOINT")

    # =========================================================================
    # STATUS LOG
    # =========================================================================
    logger.info("=" * 60)
    logger.info(f"LLM  : Stuart AI RAG  → {STUART_ENDPOINT}")
    logger.info(f"STT  : Groq Whisper   → {GROQ_STT_MODEL} [{WHISPER_LANGUAGE}]")
    logger.info(f"TTS  : Groq Orpheus   → {GROQ_TTS_MODEL} / {GROQ_TTS_VOICE}")
    if LOCAL_VAD_ENDPOINT:
        logger.info(f"VAD  : External       → {LOCAL_VAD_ENDPOINT}")
    else:
        logger.info("VAD  : Local energy-based")
    logger.info("=" * 60)


config = Config()
