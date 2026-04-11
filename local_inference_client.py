"""
Local inference client: on-device STT + on-device translation.

STT: faster-whisper (CTranslate2 backend)
Translation: NLLB-200-distilled-600M via CTranslate2

This module mirrors the public API of sarvam_client.py so the caller can
switch backends without changing pipeline code.
"""

import io
import os
import re
import shutil
import subprocess
import threading
import time
import wave

import numpy as np

from config import (
    LOCAL_STT_COMPUTE_TYPE,
    LOCAL_STT_DEVICE,
    LOCAL_STT_MODEL,
    NLLB_MODEL,
    NLLB_LANG_MAP,
)

# ── faster-whisper (STT) ──────────────────────────────────────────────────
try:
    from faster_whisper import WhisperModel
except Exception as exc:
    WhisperModel = None
    _whisper_import_error = exc
else:
    _whisper_import_error = None

# ── CTranslate2 + SentencePiece (NLLB translation) ────────────────────────
try:
    import ctranslate2
    import sentencepiece as spm
except Exception as exc:
    ctranslate2 = None
    spm = None
    _nllb_import_error = exc
else:
    _nllb_import_error = None

_stt_lock = threading.Lock()
_stt_model = None
_stt_model_source = None
_nllb_lock = threading.Lock()
_nllb_translator = None
_nllb_tokenizer = None
_nllb_model_dir = None

_PROMPT_LEAK_SNIPPETS = {
    "a person is speaking clearly in a conversation",
    "common names niranjan rajan priya rahul kumar",
    "transcribe accurately",
}

_ENGLISH_HI_SHORTCUTS = {
    "that s it": "बस इतना ही।",
    "thats it": "बस इतना ही।",
    "that s all": "बस इतना ही।",
    "thats all": "बस इतना ही।",
}


def _normalize_prediction(text: str) -> str:
    lowered = (text or "").lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


# ── Helper functions ───────────────────────────────────────────────────────

def _bcp47_to_iso(lang_code: str | None) -> str:
    if not lang_code:
        return "en"
    return str(lang_code).split("-", 1)[0].lower()


def _bcp47_to_nllb(lang_code: str | None) -> str:
    if not lang_code:
        return "eng_Latn"
    return NLLB_LANG_MAP.get(lang_code, "eng_Latn")


def _resample_float32(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    if from_rate == to_rate or len(audio) == 0:
        return audio.astype(np.float32, copy=False)
    ratio = to_rate / from_rate
    n_out = max(1, int(len(audio) * ratio))
    indices = np.arange(n_out) / ratio
    indices = np.clip(indices, 0, len(audio) - 1).astype(np.int32)
    return audio[indices].astype(np.float32, copy=False)


def _decode_wav_bytes(audio_bytes: bytes) -> np.ndarray:
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
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Audio is not WAV and ffmpeg is not installed.")

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ac", "1", "-ar", "16000",
        "pipe:1",
    ]

    try:
        proc = subprocess.run(
            cmd, input=audio_bytes, capture_output=True, check=True, timeout=20,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg decode failed: {stderr or exc}") from exc

    pcm = proc.stdout
    if not pcm:
        raise RuntimeError("ffmpeg returned empty audio stream")

    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def _decode_audio_to_float32(audio_bytes: bytes) -> np.ndarray:
    if len(audio_bytes) >= 12 and audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        return _decode_wav_bytes(audio_bytes)
    return _decode_with_ffmpeg(audio_bytes)


def _normalize_whisper_model_ref(model_ref: str | None) -> str:
    value = (model_ref or "small").strip()
    return value or "small"


def _has_whisper_model_bin(model_dir: str) -> bool:
    return os.path.isfile(os.path.join(model_dir, "model.bin"))


def _resolve_whisper_model_source(download_if_missing: bool) -> str:
    global _stt_model_source

    if _stt_model_source is not None:
        return _stt_model_source

    model_ref = _normalize_whisper_model_ref(LOCAL_STT_MODEL)
    if os.path.isdir(model_ref):
        if not _has_whisper_model_bin(model_ref):
            raise RuntimeError(
                f"Whisper model directory '{model_ref}' is missing model.bin"
            )
        _stt_model_source = model_ref
        return _stt_model_source

    # For explicit local paths that do not exist, fail early with a clear message.
    if os.path.isabs(model_ref) or model_ref.startswith("."):
        raise RuntimeError(
            f"LOCAL_STT_MODEL points to '{model_ref}', but that directory does not exist"
        )

    repo_id = model_ref if "/" in model_ref else f"Systran/faster-whisper-{model_ref}"

    try:
        from huggingface_hub import snapshot_download
    except Exception:
        # Fallback to faster-whisper's default behavior if huggingface_hub
        # cannot be imported in this environment.
        if not download_if_missing:
            raise RuntimeError(
                "huggingface_hub is unavailable and LOCAL_STT_MODEL is not a local path"
            )
        _stt_model_source = model_ref
        return _stt_model_source

    cached_dir = None
    try:
        cached_dir = snapshot_download(repo_id=repo_id, local_files_only=True)
    except Exception:
        cached_dir = None

    if cached_dir and _has_whisper_model_bin(cached_dir):
        _stt_model_source = cached_dir
        return _stt_model_source

    if not download_if_missing:
        if cached_dir:
            raise RuntimeError(
                f"Incomplete Whisper cache for '{repo_id}' at '{cached_dir}' (missing model.bin)"
            )
        raise RuntimeError(
            f"Whisper model '{repo_id}' is not cached locally. Run: python download_local_models.py"
        )

    try:
        downloaded_dir = snapshot_download(repo_id=repo_id)
    except Exception as exc:
        if cached_dir:
            raise RuntimeError(
                f"Whisper cache for '{repo_id}' is incomplete at '{cached_dir}' and refresh failed: {exc}"
            ) from exc
        raise RuntimeError(
            f"Failed to download Whisper model '{repo_id}': {exc}"
        ) from exc

    if not _has_whisper_model_bin(downloaded_dir):
        raise RuntimeError(
            f"Whisper snapshot '{downloaded_dir}' is missing model.bin after download"
        )

    _stt_model_source = downloaded_dir
    return _stt_model_source


def _find_sentencepiece_model(model_dir: str) -> str | None:
    for sp_name in ("sentencepiece.bpe.model", "sentencepiece.model", "source.spm"):
        sp_model_path = os.path.join(model_dir, sp_name)
        if os.path.isfile(sp_model_path):
            return sp_model_path
    return None


def _is_valid_nllb_model_dir(model_dir: str) -> bool:
    model_bin = os.path.join(model_dir, "model.bin")
    return os.path.isfile(model_bin) and _find_sentencepiece_model(model_dir) is not None


def _resolve_nllb_model_dir(download_if_missing: bool) -> str:
    global _nllb_model_dir

    if _nllb_model_dir is not None:
        return _nllb_model_dir

    model_ref = (NLLB_MODEL or "").strip()
    if os.path.isdir(model_ref):
        if not _is_valid_nllb_model_dir(model_ref):
            raise RuntimeError(
                f"NLLB model directory '{model_ref}' is missing model.bin or sentencepiece model"
            )
        _nllb_model_dir = model_ref
        return _nllb_model_dir

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        if not download_if_missing:
            raise RuntimeError(f"huggingface_hub is unavailable: {exc}") from exc
        raise RuntimeError(
            "huggingface_hub is required to fetch NLLB model. Install: pip install huggingface_hub"
        ) from exc

    cached_dir = None
    try:
        cached_dir = snapshot_download(repo_id=model_ref, local_files_only=True)
    except Exception:
        cached_dir = None

    if cached_dir and _is_valid_nllb_model_dir(cached_dir):
        _nllb_model_dir = cached_dir
        return _nllb_model_dir

    if not download_if_missing:
        if cached_dir:
            raise RuntimeError(
                f"Incomplete NLLB cache for '{model_ref}' at '{cached_dir}'"
            )
        raise RuntimeError(
            f"NLLB model '{model_ref}' is not cached locally. Run: python download_local_models.py"
        )

    try:
        downloaded_dir = snapshot_download(repo_id=model_ref)
    except Exception as exc:
        if cached_dir:
            raise RuntimeError(
                f"NLLB cache for '{model_ref}' is incomplete at '{cached_dir}' and refresh failed: {exc}"
            ) from exc
        raise RuntimeError(f"Failed to download NLLB model '{model_ref}': {exc}") from exc

    if not _is_valid_nllb_model_dir(downloaded_dir):
        raise RuntimeError(
            f"NLLB snapshot '{downloaded_dir}' is incomplete after download"
        )

    _nllb_model_dir = downloaded_dir
    return _nllb_model_dir


# ── STT (faster-whisper) ──────────────────────────────────────────────────

def _get_stt_model():
    if WhisperModel is None:
        raise RuntimeError("faster-whisper is not installed.")

    global _stt_model
    if _stt_model is not None:
        return _stt_model

    with _stt_lock:
        if _stt_model is None:
            model_source = _resolve_whisper_model_source(download_if_missing=True)
            use_local_only = os.path.isdir(model_source)
            _stt_model = WhisperModel(
                model_source,
                device=LOCAL_STT_DEVICE,
                compute_type=LOCAL_STT_COMPUTE_TYPE,
                local_files_only=use_local_only,
            )
    return _stt_model


def _transcribe_to_english(audio_samples: np.ndarray) -> str:
    model = _get_stt_model()

    # task="translate" auto-detects source language and translates to English
    # Do NOT set language= so Whisper can handle English, Hindi, or mixed input
    try:
        segments, info = model.transcribe(
            audio_samples,
            task="translate",
            beam_size=5,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=200,
            ),
            no_speech_threshold=0.55,
            log_prob_threshold=-1.0,
        )
    except Exception:
        segments, info = model.transcribe(
            audio_samples,
            task="transcribe",
            beam_size=5,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
            no_speech_threshold=0.55,
            log_prob_threshold=-1.0,
        )

    audio_secs = len(audio_samples) / 16000.0 if len(audio_samples) else 0.0

    text_parts = []
    for seg in segments:
        piece = (seg.text or "").strip()
        if piece:
            text_parts.append(piece)

    result = " ".join(text_parts).strip()

    # Filter Whisper hallucinations on noise/silence
    if result:
        normalized = _normalize_prediction(result)
        if not normalized:
            return ""

        if any(snippet in normalized for snippet in _PROMPT_LEAK_SNIPPETS):
            return ""

        words = result.split()
        # Detect single-word repetition (e.g. "Allah Allah Allah...")
        if len(words) >= 6:
            unique = set(w.lower().strip(".,!?") for w in words)
            if len(unique) <= 2:
                return ""

        # Reject very short text when a long clip was sent (common silence/noise hallucination).
        if audio_secs >= 3.0 and len(words) <= 2:
            return ""

        # Detect common Whisper noise hallucinations (only obvious ones)
        _HALLUCINATIONS = {
            "thanks for watching", "the end", "subscribe",
            "please subscribe", "like and subscribe",
            "thank you", "thanks", "thank you very much",
        }
        if normalized in _HALLUCINATIONS:
            return ""

    return result


# ── Translation (NLLB-200 via CTranslate2) ────────────────────────────────

def _get_nllb():
    if ctranslate2 is None or spm is None:
        raise RuntimeError(
            "ctranslate2 or sentencepiece not installed. "
            "Install: pip install ctranslate2 sentencepiece"
        )

    global _nllb_translator, _nllb_tokenizer
    if _nllb_translator is not None:
        return _nllb_translator, _nllb_tokenizer

    with _nllb_lock:
        if _nllb_translator is None:
            model_dir = _resolve_nllb_model_dir(download_if_missing=True)
            _nllb_translator = ctranslate2.Translator(
                model_dir, device="cpu", compute_type="int8",
            )
            sp_model_path = _find_sentencepiece_model(model_dir)
            if not sp_model_path:
                raise RuntimeError(
                    f"NLLB model '{model_dir}' is missing sentencepiece model file"
                )
            _nllb_tokenizer = spm.SentencePieceProcessor()
            _nllb_tokenizer.Load(sp_model_path)

    return _nllb_translator, _nllb_tokenizer


def _download_nllb_model() -> str:
    return _resolve_nllb_model_dir(download_if_missing=True)


def _nllb_translate(text: str, src_lang: str, tgt_lang: str) -> str:
    translator, tokenizer = _get_nllb()

    tokens = tokenizer.Encode(text, out_type=str)
    source_tokens = [src_lang] + tokens + ["</s>"]

    # Cap output length relative to input to prevent degeneration
    max_len = min(max(len(tokens) * 3, 20), 200)

    results = translator.translate_batch(
        [source_tokens],
        target_prefix=[[tgt_lang]],
        beam_size=4,
        max_decoding_length=max_len,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
    )

    output_tokens = results[0].hypotheses[0]
    if output_tokens and output_tokens[0] == tgt_lang:
        output_tokens = output_tokens[1:]

    translated = tokenizer.Decode(output_tokens)
    return translated.strip()


# ── Public API ─────────────────────────────────────────────────────────────

def is_ready() -> tuple[bool, str]:
    if _whisper_import_error is not None:
        return (False, f"faster-whisper import failed: {_whisper_import_error}")
    if _nllb_import_error is not None:
        return (False, f"NLLB dependencies import failed: {_nllb_import_error}")

    try:
        _resolve_whisper_model_source(download_if_missing=False)
    except Exception as exc:
        return (False, f"Whisper model not ready: {exc}")

    try:
        _resolve_nllb_model_dir(download_if_missing=False)
    except Exception as exc:
        return (False, f"NLLB model not ready: {exc}")

    return (True, "Local backend ready (Whisper + NLLB-200)")


def is_translation_ready() -> tuple[bool, str]:
    """Readiness check for translation-only usage (no Whisper required)."""
    if _nllb_import_error is not None:
        return (False, f"NLLB dependencies import failed: {_nllb_import_error}")

    try:
        _resolve_nllb_model_dir(download_if_missing=False)
    except Exception as exc:
        return (False, f"NLLB model not ready: {exc}")

    return (True, "Local translation ready (NLLB-200)")


def preload_models() -> tuple[bool, str]:
    """Load STT + translation models once during server startup."""
    try:
        _get_stt_model()
    except Exception as exc:
        return (False, f"Whisper preload failed: {exc}")

    try:
        _get_nllb()
    except Exception as exc:
        return (False, f"NLLB preload failed: {exc}")

    return (True, "Local backend ready (Whisper + NLLB-200)")


def preload_translation_model() -> tuple[bool, str]:
    """Load NLLB translation model once during server startup."""
    try:
        _get_nllb()
    except Exception as exc:
        return (False, f"NLLB preload failed: {exc}")

    return (True, "Local translation ready (NLLB-200)")


def transcribe_and_translate(
    wav_bytes: bytes,
    tgt_lang: str = "en-IN",
    direct_translate: bool = False,
    fallback_to_two_step: bool = True,
    audio_filename: str = "audio.wav",
    audio_content_type: str = "audio/wav",
) -> tuple:
    del direct_translate, fallback_to_two_step, audio_filename, audio_content_type

    if not wav_bytes:
        return ("", "", 0.0, 0.0)

    t0 = time.time()
    try:
        samples = _decode_audio_to_float32(wav_bytes)
        english_text = _transcribe_to_english(samples)
        t_stt = time.time() - t0
    except Exception as exc:
        t_stt = time.time() - t0
        raise RuntimeError(f"Local STT failed: {exc}") from exc

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
    clean_text = (text or "").strip()
    if not clean_text:
        return ("", 0.0)

    src_iso = _bcp47_to_iso(source_lang)
    tgt_iso = _bcp47_to_iso(target_lang)

    if src_iso == "en" and tgt_iso == "hi":
        shortcut = _ENGLISH_HI_SHORTCUTS.get(_normalize_prediction(clean_text))
        if shortcut:
            return (shortcut, 0.0)

    src_nllb = _bcp47_to_nllb(source_lang)
    tgt_nllb = _bcp47_to_nllb(target_lang)

    if src_nllb == tgt_nllb:
        return (clean_text, 0.0)

    t0 = time.time()
    try:
        translated = _nllb_translate(clean_text, src_nllb, tgt_nllb)
        t_translate = time.time() - t0
        if translated:
            return (translated, t_translate)
        if fallback_to_source:
            return (clean_text, t_translate)
        return ("", t_translate)
    except Exception as exc:
        t_translate = time.time() - t0
        print(f"  [NLLB translate error] {exc}")
        if fallback_to_source:
            return (clean_text, t_translate)
        return ("", t_translate)
