"""
Deepgram client — STT via Deepgram Nova + local NLLB translation.

This backend keeps translation local (NLLB) and only swaps STT to Deepgram.
"""

import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    DEEPGRAM_API_KEY,
    DEEPGRAM_BASE_URL,
    DEEPGRAM_STT_MODEL,
    DEEPGRAM_SOURCE_LANG,
    DEEPGRAM_TIMEOUT_SECS,
    FILLER_PHRASES,
)

try:
    import local_inference_client as _local
except Exception as exc:
    _local = None
    _local_import_error = exc
else:
    _local_import_error = None

_session = requests.Session()
_adapter = HTTPAdapter(
    pool_connections=2,
    pool_maxsize=4,
    max_retries=Retry(
        total=1,
        backoff_factor=0.1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST", "GET", "HEAD"],
    ),
)
_session.mount("https://", _adapter)

_DEEPGRAM_TO_BCP47 = {
    "en": "en-IN",
    "hi": "hi-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "bn": "bn-IN",
    "mr": "mr-IN",
    "gu": "gu-IN",
    "pa": "pa-IN",
}


def _normalize_text(text: str) -> str:
    return (text or "").strip().lower().strip(".,!?;: ")


def _to_bcp47(lang: str | None) -> str:
    code = (lang or "").strip().lower().replace("_", "-")
    if not code:
        return "en-IN"
    primary = code.split("-", 1)[0]
    return _DEEPGRAM_TO_BCP47.get(primary, "en-IN")


def _extract_transcript(payload: dict) -> tuple[str, str | None]:
    results = payload.get("results") or {}
    channels = results.get("channels") or []
    if not channels:
        return ("", None)

    alternatives = channels[0].get("alternatives") or []
    if not alternatives:
        return ("", None)

    alt = alternatives[0]
    transcript = (alt.get("transcript") or "").strip()

    detected_lang = None
    langs = alt.get("languages")
    if isinstance(langs, list) and langs:
        detected_lang = str(langs[0]).strip().lower() or None

    if not detected_lang:
        detected_lang = (alt.get("detected_language") or "").strip().lower() or None

    if not detected_lang:
        detected_lang = (channels[0].get("detected_language") or "").strip().lower() or None

    return (transcript, detected_lang)


def _local_translate_ready() -> tuple[bool, str]:
    if _local is None:
        return (False, f"Local translation import failed: {_local_import_error}")

    if hasattr(_local, "is_translation_ready"):
        return _local.is_translation_ready()

    return (False, "Local translation readiness check is unavailable")


def is_ready() -> tuple[bool, str]:
    if not DEEPGRAM_API_KEY:
        return (False, "DEEPGRAM_API_KEY is not configured")

    tr_ready, tr_status = _local_translate_ready()
    if not tr_ready:
        return (False, tr_status)

    return (True, f"Deepgram STT ready ({DEEPGRAM_STT_MODEL}) + {tr_status}")


def preload_models() -> tuple[bool, str]:
    tr_ready, tr_status = _local_translate_ready()
    if not tr_ready:
        return (False, tr_status)

    if hasattr(_local, "preload_translation_model"):
        return _local.preload_translation_model()

    return (True, tr_status)


def _deepgram_transcribe(
    audio_bytes: bytes,
    audio_content_type: str,
) -> tuple[str, str, float]:
    t0 = time.time()
    params = {
        "model": DEEPGRAM_STT_MODEL,
        "smart_format": "true",
        "punctuate": "true",
    }

    source_lang = (DEEPGRAM_SOURCE_LANG or "en").strip().lower()
    if source_lang in {"", "auto"}:
        params["detect_language"] = "true"
    else:
        params["language"] = source_lang

    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": audio_content_type or "audio/wav",
    }

    resp = _session.post(
        f"{DEEPGRAM_BASE_URL}/listen",
        headers=headers,
        params=params,
        data=audio_bytes,
        timeout=DEEPGRAM_TIMEOUT_SECS,
    )
    resp.raise_for_status()

    transcript, detected = _extract_transcript(resp.json())
    t_stt = time.time() - t0

    clean = transcript.strip()
    if _normalize_text(clean) in {_normalize_text(x) for x in FILLER_PHRASES}:
        return ("", "en-IN", t_stt)

    if source_lang in {"", "auto"}:
        src_bcp47 = _to_bcp47(detected)
    else:
        src_bcp47 = _to_bcp47(source_lang)

    return (clean, src_bcp47, t_stt)


def transcribe_and_translate(
    wav_bytes: bytes,
    tgt_lang: str = "en-IN",
    direct_translate: bool = False,
    fallback_to_two_step: bool = True,
    audio_filename: str = "audio.wav",
    audio_content_type: str = "audio/wav",
) -> tuple:
    del direct_translate, fallback_to_two_step, audio_filename

    if not wav_bytes:
        return ("", "", 0.0, 0.0)

    try:
        source_text, source_lang, t_stt = _deepgram_transcribe(wav_bytes, audio_content_type)
    except Exception as exc:
        t_stt = 0.0
        raise RuntimeError(f"Deepgram STT failed: {exc}") from exc

    if not source_text:
        return ("", "", t_stt, 0.0)

    if tgt_lang == source_lang or tgt_lang == "en-IN":
        return (source_text, source_text, t_stt, 0.0)

    if _local is None:
        raise RuntimeError(f"Local translation import failed: {_local_import_error}")

    translated, t_translate = _local.translate_text(
        source_text,
        source_lang=source_lang,
        target_lang=tgt_lang,
        fallback_to_source=True,
    )
    return (translated or source_text, source_text, t_stt, t_translate)


def translate_text(
    text: str,
    source_lang: str = "en-IN",
    target_lang: str = "hi-IN",
    fallback_to_source: bool = True,
) -> tuple[str, float]:
    if _local is None:
        raise RuntimeError(f"Local translation import failed: {_local_import_error}")

    return _local.translate_text(
        text,
        source_lang=source_lang,
        target_lang=target_lang,
        fallback_to_source=fallback_to_source,
    )
