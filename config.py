"""
Configuration for the wearable voice translator pipeline.
Edit these values to match your hardware and preferences.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from parent directory (shared with v1) or current directory
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv()

# ── Inference backend selection 
# sarvam: cloud STT + translation
# local:  on-device STT + translation
# deepgram: Deepgram STT + local NLLB translation
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "sarvam").strip().lower()
if INFERENCE_BACKEND not in {"sarvam", "local", "deepgram"}:
    INFERENCE_BACKEND = "sarvam"

# Local STT configuration (used when INFERENCE_BACKEND=local)
# Common model values: tiny, base, small
LOCAL_STT_MODEL = os.getenv("LOCAL_STT_MODEL", "small")
LOCAL_STT_DEVICE = os.getenv("LOCAL_STT_DEVICE", "cpu")
LOCAL_STT_COMPUTE_TYPE = os.getenv("LOCAL_STT_COMPUTE_TYPE", "int8")

# NLLB-200 translation configuration (used when INFERENCE_BACKEND=local)
# CTranslate2-converted NLLB model for fast CPU inference
NLLB_MODEL = os.getenv("NLLB_MODEL", "JustFrederik/nllb-200-distilled-600M-ct2-int8")

# BCP-47 to NLLB language code mapping (FLORES-200 codes)
NLLB_LANG_MAP = {
    "en-IN": "eng_Latn",
    "hi-IN": "hin_Deva",
    "ta-IN": "tam_Taml",
    "te-IN": "tel_Telu",
    "kn-IN": "kan_Knda",
    "ml-IN": "mal_Mlym",
    "bn-IN": "ben_Beng",
    "mr-IN": "mar_Deva",
    "gu-IN": "guj_Gujr",
    "pa-IN": "pan_Guru",
}

# ── Sarvam API ──────────────────────────────────────────────────────────────
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_BASE_URL = "https://api.sarvam.ai"

# Model settings (centralized for tuning)
STT_MODEL = "saaras:v3"  # <--- Changed from saaras:v2.5 to saaras:v3
TRANSLATE_MODEL = "mayura:v1"
TRANSLATE_MODE = "formal"

# Latency optimization
# When True, use speech-to-text-translate with the target language
# and skip the separate translate call.
DIRECT_TRANSLATE = False  # <--- Changed back to False so we can use local translation
DIRECT_TRANSLATE_FALLBACK = True

# ── Deepgram API (STT-only backend) ───────────────────────────────────────
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "").strip()
DEEPGRAM_BASE_URL = os.getenv("DEEPGRAM_BASE_URL", "https://api.deepgram.com/v1").strip()
DEEPGRAM_STT_MODEL = os.getenv("DEEPGRAM_STT_MODEL", "nova-2").strip()
# Use ISO language (e.g. en, hi) or "auto" for detect_language.
DEEPGRAM_SOURCE_LANG = os.getenv("DEEPGRAM_SOURCE_LANG", "en").strip().lower()
try:
    DEEPGRAM_TIMEOUT_SECS = float(os.getenv("DEEPGRAM_TIMEOUT_SECS", "20"))
except ValueError:
    DEEPGRAM_TIMEOUT_SECS = 20.0
DEEPGRAM_TIMEOUT_SECS = max(5.0, DEEPGRAM_TIMEOUT_SECS)

# ── Language settings ───────────────────────────────────────────────────────
# BCP-47 codes used by Sarvam
LANG_MAP = {
    "english":  "en-IN",
    "hindi":    "hi-IN",
    "tamil":    "ta-IN",
    "telugu":   "te-IN",
    "kannada":  "kn-IN",
    "malayalam":"ml-IN",
    "bengali":  "bn-IN",
    "marathi":  "mr-IN",
    "gujarati": "gu-IN",
    "punjabi":  "pa-IN",
}

# Default output language (change at runtime via command-line arg)
DEFAULT_TARGET_LANG = "hindi"

# ── Audio capture ───────────────────────────────────────────────────────────
SAMPLE_RATE = 16000        # Hz — required by Sarvam STT
CHANNELS = 1               # Mono
BLOCK_DURATION_MS = 30     # Duration of each audio block from mic (ms)
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION_MS / 1000)  # samples per block

# ── Voice Activity Detection ───────────────────────────────────────────────
RMS_THRESHOLD = 300        # Minimum RMS to consider as speech
SILENCE_TIMEOUT = 0.4      # Seconds of silence before triggering processing
MAX_RECORD_SECS = 5        # Max recording length before forced processing (prevents huge clips)
MIN_SPEECH_RMS = 200       # Overall RMS must exceed this to avoid noise triggers
MAX_WAV_KB = 200           # Max WAV size in KB to send to API (skip if larger)

# ── Piper TTS ──────────────────────────────────────────────────────────────
PIPER_MODELS_DIR = Path(__file__).resolve().parent / "piper_models"

# Piper voice model filenames (without .onnx extension) per language code
# These are downloaded by download_models.py
# Fast English model (16kHz, ~50ms) for low-latency output
PIPER_VOICE_ENGLISH_FAST = "en_US-lessac-low"

PIPER_VOICES = {
    "en-IN": "en_US-lessac-low",
    "hi-IN": "hi_IN-pratham-medium",
    "ml-IN": "ml_IN-arjun-medium",
    "te-IN": "te_IN-maya-medium",
    # Tamil, Kannada etc. use IndicF5 or espeak-ng/edge-tts fallback
    "ta-IN": None,
    "kn-IN": None,
    "bn-IN": None,
    "mr-IN": None,
    "gu-IN": None,
    "pa-IN": None,
}

# IndicF5 TTS — supports: Assamese, Bengali, Gujarati, Hindi, Kannada,
# Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu
# Set to True to prefer IndicF5 over Piper for supported Indic languages
USE_INDICF5 = os.getenv("USE_INDICF5", "true").strip().lower() in ("1", "true", "yes")

# IndicF5 language code mapping (BCP-47 -> IndicF5 script identifier)
INDICF5_LANG_MAP = {
    "hi-IN": "hin",
    "ta-IN": "tam",
    "te-IN": "tel",
    "kn-IN": "kan",
    "ml-IN": "mal",
    "bn-IN": "ben",
    "mr-IN": "mar",
    "gu-IN": "guj",
    "pa-IN": "pan",
}

# ── Meta MMS TTS (Multilingual Offline Fallback) ───────────────────────────
MMS_TTS_VOICES = {
    "ta-IN": "facebook/mms-tts-tam",
}

# ── Fallback TTS ───────────────────────────────────────────────────────────
# edge-tts voice names (Microsoft) — used when Piper model is unavailable
EDGE_TTS_VOICES = {
    "en-IN": "en-IN-NeerjaNeural",
    "hi-IN": "hi-IN-SwaraNeural",
    "ta-IN": "ta-IN-PallaviNeural",
    "te-IN": "te-IN-ShrutiNeural",
    "kn-IN": "kn-IN-SapnaNeural",
    "ml-IN": "ml-IN-MidhunNeural",
    "bn-IN": "bn-IN-TanishaaNeural",
    "mr-IN": "mr-IN-AarohiNeural",
    "gu-IN": "gu-IN-DhwaniNeural",
    "pa-IN": "pa-IN-GurpreetNeural",
}

# ── Edge Server (Pi→Laptop architecture) ──────────────────────────────────
EDGE_SERVER_HOST = os.getenv("EDGE_SERVER_HOST", "0.0.0.0")
EDGE_SERVER_PORT = int(os.getenv("EDGE_SERVER_PORT", "5555"))

# ── Filler / noise phrases to ignore ───────────────────────────────────────
FILLER_PHRASES = {
    "thank you", "thanks", "thank you.", "thanks.",
    "thanks for watching", "thanks for watching.",
    "thank you for watching", "thank you for watching.",
    "bye", "bye.", "goodbye", "goodbye.",
    "you", "the", "i", "a", ".", "",
}
