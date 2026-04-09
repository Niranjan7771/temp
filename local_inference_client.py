"""
Local inference client: on-device STT + on-device translation.

STT: faster-whisper (CTranslate2 backend)
Translation: NLLB-200-distilled-600M via CTranslate2

This module mirrors the public API of sarvam_client.py so the caller can
switch backends without changing pipeline code.
"""

import io
import os
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
_nllb_lock = threading.Lock()
_nllb_translator = None
_nllb_tokenizer = None


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


# ── STT (faster-whisper) ──────────────────────────────────────────────────

def _get_stt_model():
    if WhisperModel is None:
        raise RuntimeError("faster-whisper is not installed.")

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
    model = _get_stt_model()

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
            model_dir = _download_nllb_model()
            _nllb_translator = ctranslate2.Translator(
                model_dir, device="cpu", compute_type="int8",
            )
            # CTranslate2 NLLB models store the sentencepiece model
            sp_model_path = os.path.join(model_dir, "sentencepiece.model")
            if not os.path.exists(sp_model_path):
                sp_model_path = os.path.join(model_dir, "source.spm")
            _nllb_tokenizer = spm.SentencePieceProcessor()
            _nllb_tokenizer.Load(sp_model_path)

    return _nllb_translator, _nllb_tokenizer


def _download_nllb_model() -> str:
    from huggingface_hub import snapshot_download
    model_dir = snapshot_download(NLLB_MODEL)
    return model_dir


def _nllb_translate(text: str, src_lang: str, tgt_lang: str) -> str:
    translator, tokenizer = _get_nllb()

    tokens = tokenizer.Encode(text, out_type=str)
    source_tokens = [src_lang] + tokens

    results = translator.translate_batch(
        [source_tokens],
        target_prefix=[[tgt_lang]],
        beam_size=4,
        max_decoding_length=256,
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
    return (True, "Local backend ready (Whisper + NLLB-200)")


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
    clean_text = (text or "").strip()
    if not clean_text:
        return ("", 0.0)

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
