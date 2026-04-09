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

# ── Sarvam API ──────────────────────────────────────────────────────────────
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_BASE_URL = "https://api.sarvam.ai"

# Model settings (centralized for tuning)
STT_MODEL = "saaras:v2.5"
TRANSLATE_MODEL = "mayura:v1"
TRANSLATE_MODE = "formal"

# Latency optimization
# When True, use speech-to-text-translate with the target language
# and skip the separate translate call.
DIRECT_TRANSLATE = False
DIRECT_TRANSLATE_FALLBACK = True

# ── Language settings ───────────────────────────────────────────────────────
# BCP-47 codes used by Sarvam
LANG_MAP = {
    "english": "en-IN",
    "hindi":   "hi-IN",
    "tamil":   "ta-IN",
    "telugu":  "te-IN",
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
PIPER_VOICES = {
    "en-IN": "en_US-lessac-medium",
    "hi-IN": "hi_IN-pratham-medium",
    "ta-IN": None,   # Not yet available in Piper — falls back to espeak-ng
    "te-IN": None,   # Not yet available in Piper — falls back to espeak-ng
}

# ── Fallback TTS ───────────────────────────────────────────────────────────
# edge-tts voice names (Microsoft) — used when Piper model is unavailable
EDGE_TTS_VOICES = {
    "en-IN": "en-IN-NeerjaNeural",
    "hi-IN": "hi-IN-SwaraNeural",
    "ta-IN": "ta-IN-PallaviNeural",
    "te-IN": "te-IN-ShrutiNeural",
}

# ── Filler / noise phrases to ignore ───────────────────────────────────────
FILLER_PHRASES = {
    "thank you", "thanks", "thank you.", "thanks.",
    "thanks for watching", "thanks for watching.",
    "thank you for watching", "thank you for watching.",
    "bye", "bye.", "goodbye", "goodbye.",
    "you", "the", "i", "a", ".", "",
}
