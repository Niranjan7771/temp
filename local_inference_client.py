"""
Local inference client: on-device STT + on-device translation.

STT: faster-whisper
Translation: Argos Translate

This module mirrors the public API of sarvam_client.py so the caller can
switch backends without changing pipeline code.
"""

import io
import shutil
import subprocess
import threading
import time
import wave

import numpy as np

from config import LOCAL_STT_COMPUTE_TYPE, LOCAL_STT_DEVICE, LOCAL_STT_MODEL

try:
    from faster_whisper import WhisperModel
except Exception as exc:
    WhisperModel = None
    _whisper_import_error = exc
else:
    _whisper_import_error = None

try:
    from argostranslate import translate as argos_translate
except Exception as exc:
    argos_translate = None
    _argos_import_error = exc
else:
    _argos_import_error = None

_stt_lock = threading.Lock()
_stt_model = None


def _bcp47_to_iso(lang_code: str | None) -> str:
    """Convert BCP-47 style code (en-IN) to ISO639-1 style code (en)."""
    if not lang_code:
        return "en"
    return str(lang_code).split("-", 1)[0].lower()


def _resample_float32(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Simple nearest-neighbor resampling for low-latency compatibility."""
    if from_rate == to_rate or len(audio) == 0:
        return audio.astype(np.float32, copy=False)
    ratio = to_rate / from_rate
    n_out = max(1, int(len(audio) * ratio))
    indices = np.arange(n_out) / ratio
    indices = np.clip(indices, 0, len(audio) - 1).astype(np.int32)
    return audio[indices].astype(np.float32, copy=False)


def _decode_wav_bytes(audio_bytes: bytes) -> np.ndarray:
    """Decode WAV bytes to mono float32 at 16kHz."""
    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sample_width == 2:
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 1:
        samples = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    return _resample_float32(samples, sr, 16000)


def _decode_with_ffmpeg(audio_bytes: bytes) -> np.ndarray:
    """Decode arbitrary audio container to mono 16kHz float32 using ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "Audio is not WAV and ffmpeg is not installed. "
            "Install ffmpeg or send WAV audio."
        )

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "pipe:1",
    ]

    try:
        proc = subprocess.run(
            cmd,
            input=audio_bytes,
            capture_output=True,
            check=True,
            timeout=20,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg decode failed: {stderr or exc}") from exc

    pcm = proc.stdout
    if not pcm:
        raise RuntimeError("ffmpeg returned empty audio stream")

    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def _decode_audio_to_float32(audio_bytes: bytes) -> np.ndarray:
    """Decode input audio bytes to mono float32 samples at 16kHz."""
    if len(audio_bytes) >= 12 and audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        return _decode_wav_bytes(audio_bytes)
    return _decode_with_ffmpeg(audio_bytes)


def _get_stt_model() -> WhisperModel:
    """Load faster-whisper model lazily and cache it for reuse."""
    if WhisperModel is None:
        raise RuntimeError(
            "faster-whisper is not installed. Install dependencies from requirements.txt"
        )

    global _stt_model
    if _stt_model is not None:
        return _stt_model

    with _stt_lock:
        if _stt_model is None:
            _stt_model = WhisperModel(
                LOCAL_STT_MODEL,
                device=LOCAL_STT_DEVICE,
                compute_type=LOCAL_STT_COMPUTE_TYPE,
            )
    return _stt_model


def _transcribe_to_english(audio_samples: np.ndarray) -> str:
    """Transcribe or translate speech audio into English text."""
    model = _get_stt_model()

    # Prefer task=translate so non-English speech also becomes English text.
    try:
        segments, _ = model.transcribe(
            audio_samples,
            task="translate",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
        )
    except Exception:
        # Fallback for models/configs that do not support translate task.
        segments, _ = model.transcribe(
            audio_samples,
            task="transcribe",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
        )

    text_parts = []
    for seg in segments:
        piece = (seg.text or "").strip()
        if piece:
            text_parts.append(piece)

    return " ".join(text_parts).strip()


def _get_installed_language_map() -> dict[str, object]:
    """Return installed Argos language objects keyed by ISO code."""
    if argos_translate is None:
        raise RuntimeError(
            "argostranslate is not installed. Install dependencies from requirements.txt"
        )

    langs = argos_translate.get_installed_languages()
    lang_map = {}
    for lang in langs:
        code = str(getattr(lang, "code", "")).strip().lower()
        if code:
            lang_map[code] = lang
    return lang_map


def _translate_direct(text: str, source_iso: str, target_iso: str) -> str:
    """Run direct Argos translation for a specific pair."""
    lang_map = _get_installed_language_map()
    source_lang = lang_map.get(source_iso)
    target_lang = lang_map.get(target_iso)

    if source_lang is None or target_lang is None:
        raise RuntimeError(
            f"Missing Argos language package: {source_iso}->{target_iso}. "
            "Run python download_local_models.py"
        )

    try:
        translator = source_lang.get_translation(target_lang)
    except Exception as exc:
        raise RuntimeError(
            f"Missing Argos translation pair: {source_iso}->{target_iso}. "
            "Run python download_local_models.py"
        ) from exc

    return (translator.translate(text) or "").strip()


def _translate_with_pivot(text: str, source_iso: str, target_iso: str) -> str:
    """Translate text, pivoting through English when direct pair is unavailable."""
    if source_iso == target_iso:
        return text

    try:
        return _translate_direct(text, source_iso, target_iso)
    except RuntimeError:
        pass

    if source_iso != "en" and target_iso != "en":
        mid = _translate_direct(text, source_iso, "en")
        if not mid:
            return ""
        return _translate_direct(mid, "en", target_iso)

    raise RuntimeError(
        f"No local translation path for {source_iso}->{target_iso}. "
        "Install matching Argos packages with python download_local_models.py"
    )


def is_ready() -> tuple[bool, str]:
    """Best-effort readiness check for local backend dependencies."""
    if _whisper_import_error is not None:
        return (False, f"faster-whisper import failed: {_whisper_import_error}")
    if _argos_import_error is not None:
        return (False, f"argostranslate import failed: {_argos_import_error}")

    try:
        lang_map = _get_installed_language_map()
    except Exception as exc:
        return (False, str(exc))

    if not lang_map:
        return (False, "No Argos language packs installed. Run python download_local_models.py")

    if "en" not in lang_map:
        return (False, "Argos English package is missing. Run python download_local_models.py")

    missing_optional = [code for code in ("hi", "ta", "te") if code not in lang_map]
    if missing_optional:
        return (
            True,
            "Local backend ready (missing optional language packs: "
            + ", ".join(missing_optional)
            + ")",
        )

    return (True, "Local backend ready")


def transcribe_and_translate(
    wav_bytes: bytes,
    tgt_lang: str = "en-IN",
    direct_translate: bool = False,
    fallback_to_two_step: bool = True,
    audio_filename: str = "audio.wav",
    audio_content_type: str = "audio/wav",
) -> tuple:
    """
    Offline STT + translation path.

    Returns (translated_text, english_text, stt_seconds, translate_seconds)
    to match the cloud client's API.
    """
    del direct_translate
    del fallback_to_two_step
    del audio_filename
    del audio_content_type

    if not wav_bytes:
        return ("", "", 0.0, 0.0)

    t0 = time.time()
    try:
        samples = _decode_audio_to_float32(wav_bytes)
        english_text = _transcribe_to_english(samples)
        t_stt = time.time() - t0
    except Exception as exc:
        t_stt = time.time() - t0
        print(f"  [Local STT error] {exc}")
        return ("", "", t_stt, 0.0)

    if not english_text:
        return ("", "", t_stt, 0.0)

    if _bcp47_to_iso(tgt_lang) == "en":
        return (english_text, english_text, t_stt, 0.0)

    translated, t_translate = translate_text(
        english_text,
        source_lang="en-IN",
        target_lang=tgt_lang,
        fallback_to_source=False,
    )

    if not translated:
        return (english_text, english_text, t_stt, t_translate)

    return (translated, english_text, t_stt, t_translate)


def translate_text(
    text: str,
    source_lang: str = "en-IN",
    target_lang: str = "hi-IN",
    fallback_to_source: bool = True,
) -> tuple[str, float]:
    """Translate plain text with local Argos models."""
    clean_text = (text or "").strip()
    if not clean_text:
        return ("", 0.0)

    source_iso = _bcp47_to_iso(source_lang)
    target_iso = _bcp47_to_iso(target_lang)

    if source_iso == target_iso:
        return (clean_text, 0.0)

    t0 = time.time()
    try:
        translated = _translate_with_pivot(clean_text, source_iso, target_iso)
        t_translate = time.time() - t0
        if translated:
            return (translated, t_translate)
        if fallback_to_source:
            return (clean_text, t_translate)
        return ("", t_translate)
    except Exception as exc:
        t_translate = time.time() - t0
        print(f"  [Local translate error] {exc}")
        if fallback_to_source:
            return (clean_text, t_translate)
        return ("", t_translate)
