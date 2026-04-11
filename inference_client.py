"""
Inference backend router.

Routes STT/translation calls to either:
- Sarvam cloud client (sarvam_client.py)
- Local on-device client (local_inference_client.py)
"""

from config import INFERENCE_BACKEND, SARVAM_API_KEY
import sarvam_client as _sarvam

try:
    import local_inference_client as _local
except Exception as exc:
    _local = None
    _local_import_error = exc
else:
    _local_import_error = None

try:
    import deepgram_client as _deepgram
except Exception as exc:
    _deepgram = None
    _deepgram_import_error = exc
else:
    _deepgram_import_error = None

_VALID_BACKENDS = {"sarvam", "local", "deepgram"}
_active_backend = INFERENCE_BACKEND if INFERENCE_BACKEND in _VALID_BACKENDS else "sarvam"


def set_backend(name: str) -> None:
    """Set active inference backend at runtime."""
    global _active_backend
    candidate = (name or "").strip().lower()
    if candidate not in _VALID_BACKENDS:
        raise ValueError(f"Unsupported backend: {name}")
    _active_backend = candidate


def get_backend() -> str:
    """Get currently selected inference backend."""
    return _active_backend


def backend_ready() -> tuple[bool, str]:
    """Return (ready, status_message) for active backend."""
    if _active_backend == "sarvam":
        if not SARVAM_API_KEY:
            return (False, "SARVAM_API_KEY is not configured")
        return (True, "Sarvam backend ready")

    if _active_backend == "deepgram":
        if _deepgram is None:
            return (False, f"Deepgram backend import failed: {_deepgram_import_error}")
        return _deepgram.is_ready()

    if _local is None:
        return (False, f"Local backend import failed: {_local_import_error}")

    return _local.is_ready()


def backend_status() -> str:
    """Convenience status string for logs/UI."""
    return backend_ready()[1]


def preload_backend() -> tuple[bool, str]:
    """Warm up selected backend models/resources at startup."""
    if _active_backend == "local":
        if _local is None:
            return (False, f"Local backend import failed: {_local_import_error}")
        if hasattr(_local, "preload_models"):
            return _local.preload_models()
        return _local.is_ready()

    if _active_backend == "deepgram":
        if _deepgram is None:
            return (False, f"Deepgram backend import failed: {_deepgram_import_error}")
        if hasattr(_deepgram, "preload_models"):
            return _deepgram.preload_models()
        return _deepgram.is_ready()

    return backend_ready()


def transcribe_and_translate(
    wav_bytes: bytes,
    tgt_lang: str = "en-IN",
    direct_translate: bool = False,
    fallback_to_two_step: bool = True,
    audio_filename: str = "audio.wav",
    audio_content_type: str = "audio/wav",
) -> tuple:
    """Route speech translation request to active backend."""
    if _active_backend == "local":
        if _local is None:
            raise RuntimeError(f"Local backend import failed: {_local_import_error}")
        return _local.transcribe_and_translate(
            wav_bytes,
            tgt_lang=tgt_lang,
            direct_translate=direct_translate,
            fallback_to_two_step=fallback_to_two_step,
            audio_filename=audio_filename,
            audio_content_type=audio_content_type,
        )

    if _active_backend == "deepgram":
        if _deepgram is None:
            raise RuntimeError(f"Deepgram backend import failed: {_deepgram_import_error}")
        return _deepgram.transcribe_and_translate(
            wav_bytes,
            tgt_lang=tgt_lang,
            direct_translate=direct_translate,
            fallback_to_two_step=fallback_to_two_step,
            audio_filename=audio_filename,
            audio_content_type=audio_content_type,
        )

    return _sarvam.transcribe_and_translate(
        wav_bytes,
        tgt_lang=tgt_lang,
        direct_translate=direct_translate,
        fallback_to_two_step=fallback_to_two_step,
        audio_filename=audio_filename,
        audio_content_type=audio_content_type,
    )


def translate_text(
    text: str,
    source_lang: str = "en-IN",
    target_lang: str = "hi-IN",
    fallback_to_source: bool = True,
) -> tuple[str, float]:
    """Route text translation request to active backend."""
    if _active_backend == "local":
        if _local is None:
            raise RuntimeError(f"Local backend import failed: {_local_import_error}")
        return _local.translate_text(
            text,
            source_lang=source_lang,
            target_lang=target_lang,
            fallback_to_source=fallback_to_source,
        )

    if _active_backend == "deepgram":
        if _deepgram is None:
            raise RuntimeError(f"Deepgram backend import failed: {_deepgram_import_error}")
        return _deepgram.translate_text(
            text,
            source_lang=source_lang,
            target_lang=target_lang,
            fallback_to_source=fallback_to_source,
        )

    return _sarvam.translate_text(
        text,
        source_lang=source_lang,
        target_lang=target_lang,
        fallback_to_source=fallback_to_source,
    )
