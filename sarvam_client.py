"""
Sarvam AI client — STT + Translation only (no TTS).
TTS is handled locally by tts_engine.py for lower latency.
"""

import time
import requests
import threading
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry  # For better connection pooling
from config import (
    SARVAM_API_KEY,
    SARVAM_BASE_URL,
    STT_MODEL,
    TRANSLATE_MODEL,
    TRANSLATE_MODE,
    DIRECT_TRANSLATE,
    DIRECT_TRANSLATE_FALLBACK,
)

# ── Persistent HTTP session with improved connection pooling ──────────────
_session = requests.Session()

# Configure retry strategy for better reliability
retry_strategy = Retry(
    total=1,
    backoff_factor=0.1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["POST", "GET", "HEAD"]
)
adapter = HTTPAdapter(
    pool_connections=2,
    pool_maxsize=4,
    max_retries=retry_strategy,
)
_session.mount("https://", adapter)

_AUTH = {"api-subscription-key": SARVAM_API_KEY}


def _warm():
    """Pre-warm the TCP+TLS connection in the background."""
    urls = [
        SARVAM_BASE_URL,
        f"{SARVAM_BASE_URL}/speech-to-text-translate",
        f"{SARVAM_BASE_URL}/translate",
    ]
    for url in urls:
        try:
            _session.head(url, headers=_AUTH, timeout=5)
        except Exception:
            continue

# Start warm-up immediately
threading.Thread(target=_warm, daemon=True).start()


def transcribe_and_translate(
    wav_bytes: bytes,
    tgt_lang: str = "en-IN",
    direct_translate: bool = DIRECT_TRANSLATE,
    fallback_to_two_step: bool = DIRECT_TRANSLATE_FALLBACK,
    audio_filename: str = "audio.wav",
    audio_content_type: str = "audio/wav",
) -> tuple:
    """
    Two-step cloud pipeline (optimized for latency):
      1. STT  (saaras:v2.5) — always returns English text
      2. Translate (mayura:v1) — English → target language (skipped if target is English)

    Returns (translated_text, english_text, stt_seconds, translate_seconds).
    """
    if not wav_bytes:
        return ("", "", 0.0, 0.0)

    # ── Direct STT+Translate (single call) for lower latency ─────────────
    if direct_translate and tgt_lang != "en-IN":
        t0 = time.time()
        try:
            resp = _session.post(
                f"{SARVAM_BASE_URL}/speech-to-text-translate",
                headers=_AUTH,
                files={"file": (audio_filename, wav_bytes, audio_content_type)},
                data={"model": STT_MODEL, "target_language_code": tgt_lang},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            translated = (data.get("transcript") or data.get("translated_text") or "").strip()
            t_stt = time.time() - t0
            if translated:
                return (translated, "", t_stt, 0.0)
        except Exception as e:
            t_stt = time.time() - t0
            print(f"  [STT direct error] {e}")
            if not fallback_to_two_step:
                return ("", "", t_stt, 0.0)

    # ── Step 1: Speech-to-Text (always English) ────────────────────────────
    t0 = time.time()
    try:
        resp = _session.post(
            f"{SARVAM_BASE_URL}/speech-to-text-translate",
            headers=_AUTH,
            files={"file": (audio_filename, wav_bytes, audio_content_type)},
            data={"model": STT_MODEL, "target_language_code": "en-IN"},
            timeout=15,  # Reduced from 20 for faster timeout on slow connections
        )
        resp.raise_for_status()
        english_text = resp.json().get("transcript", "").strip()
        t_stt = time.time() - t0
        if not english_text:
            return ("", "", t_stt, 0.0)
    except Exception as e:
        t_stt = time.time() - t0
        print(f"  [STT error] {e}")
        return ("", "", t_stt, 0.0)

    # ── Step 2: Translate English → target (skip if already English) ───────
    if tgt_lang == "en-IN":
        return (english_text, english_text, t_stt, 0.0)

    t1 = time.time()
    try:
        resp = _session.post(
            f"{SARVAM_BASE_URL}/translate",
            headers={**_AUTH, "Content-Type": "application/json"},
            json={
                "input": english_text,
                "source_language_code": "en-IN",
                "target_language_code": tgt_lang,
                "model": TRANSLATE_MODEL,
                "mode": TRANSLATE_MODE,
            },
            timeout=12,  # Reduced from 15 for faster timeout
        )
        resp.raise_for_status()
        translated = resp.json().get("translated_text", english_text).strip()
        t_tr = time.time() - t1
        return (translated, english_text, t_stt, t_tr)
    except Exception as e:
        t_tr = time.time() - t1
        print(f"  [Translate error] {e}")
        return (english_text, english_text, t_stt, t_tr)


def translate_text(
    text: str,
    source_lang: str = "en-IN",
    target_lang: str = "hi-IN",
    fallback_to_source: bool = True,
) -> tuple[str, float]:
    """
    Translate plain text using Sarvam Translate API.

    Returns (translated_text, translate_seconds).
    """
    clean_text = (text or "").strip()
    if not clean_text:
        return ("", 0.0)

    if source_lang == target_lang:
        return (clean_text, 0.0)

    t0 = time.time()
    try:
        resp = _session.post(
            f"{SARVAM_BASE_URL}/translate",
            headers={**_AUTH, "Content-Type": "application/json"},
            json={
                "input": clean_text,
                "source_language_code": source_lang,
                "target_language_code": target_lang,
                "model": TRANSLATE_MODEL,
                "mode": TRANSLATE_MODE,
            },
            timeout=12,
        )
        resp.raise_for_status()
        translated = (resp.json().get("translated_text") or "").strip()
        t_translate = time.time() - t0
        if translated:
            return (translated, t_translate)
        if fallback_to_source:
            return (clean_text, t_translate)
        return ("", t_translate)
    except Exception as e:
        t_translate = time.time() - t0
        print(f"  [Translate text error] {e}")
        if fallback_to_source:
            return (clean_text, t_translate)
        return ("", t_translate)
