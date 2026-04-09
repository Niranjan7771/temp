"""
TTS Engine — priority chain for local speech synthesis.

Priority order:
  1. IndicF5   (Indian languages, near-human quality, 11 Indic langs)
  2. Piper TTS  (local ONNX, ~200-400ms, natural sounding, offline)
  3. espeak-ng  (local, ~50ms, robotic but ultra-fast, offline)
  4. edge-tts   (cloud fallback, ~300-500ms, requires internet)

For English: Piper is primary.
For Indic languages: IndicF5 is primary when available.
"""

import io
import os
import wave
import shutil
import subprocess
import tempfile
import numpy as np
from pathlib import Path

try:
    import miniaudio
except ImportError:
    miniaudio = None

from config import PIPER_MODELS_DIR, PIPER_VOICES, EDGE_TTS_VOICES

# ── Detect available TTS backends ──────────────────────────────────────────
_piper_available = False
_espeak_available = shutil.which("espeak-ng") is not None or shutil.which("espeak") is not None
_edge_tts_available = False
_indicf5_available = False

try:
    from piper import PiperVoice
    _piper_available = True
except ImportError:
    pass

try:
    import edge_tts
    _edge_tts_available = miniaudio is not None
except ImportError:
    pass

# IndicF5 TTS — loaded on demand
try:
    from config import USE_INDICF5, INDICF5_LANG_MAP
    if USE_INDICF5:
        try:
            from transformers import AutoModel as _IndicF5AutoModel
            _indicf5_available = True
        except ImportError:
            pass
except ImportError:
    USE_INDICF5 = False
    INDICF5_LANG_MAP = {}

_indicf5_model = None
_indicf5_prompts_dir = None

# Cache loaded Piper voice models (load once, reuse)
_piper_cache: dict = {}

# espeak language codes (different from BCP-47)
_ESPEAK_LANG = {
    "en-IN": "en",
    "hi-IN": "hi",
    "ta-IN": "ta",
    "te-IN": "te",
    "kn-IN": "kn",
    "ml-IN": "ml",
    "bn-IN": "bn",
    "mr-IN": "mr",
    "gu-IN": "gu",
    "pa-IN": "pa",
}


def get_backend_name(lang_code: str) -> str:
    """Return which TTS backend will be used for this language."""
    if _indicf5_available and lang_code in INDICF5_LANG_MAP:
        return "indicf5"
    if _piper_available and _get_piper_model_path(lang_code):
        return "piper"
    if _espeak_available:
        return "espeak-ng"
    if _edge_tts_available:
        return "edge-tts"
    return "none"


def synthesize(text: str, lang_code: str = "en-IN", backend: str = "auto") -> bytes:
    """
    Synthesize text to WAV bytes using the specified or best available backend.
    backend: 'auto' (priority chain), 'indicf5', 'piper', 'espeak', 'edge'
    Returns raw WAV bytes, or empty bytes on failure.
    """
    if not text:
        return b""

    # If a specific backend is requested, try it directly
    if backend == "indicf5" and _indicf5_available:
        result = _indicf5_synthesize(text, lang_code)
        if result:
            return result
    elif backend == "espeak" and _espeak_available:
        result = _espeak_synthesize(text, lang_code)
        if result:
            return result
    elif backend == "piper" and _piper_available:
        model_path = _get_piper_model_path(lang_code)
        if model_path:
            result = _piper_synthesize(text, model_path)
            if result:
                return result
    elif backend == "edge" and _edge_tts_available:
        result = _edge_tts_synthesize(text, lang_code)
        if result:
            return result
    elif backend != "auto":
        # Requested backend not available, fall through to auto
        pass

    if backend != "auto":
        return b""  # specific backend requested but failed

    # Auto: priority chain
    # 1. Try IndicF5 (local, high-quality, Indian languages)
    if _indicf5_available and lang_code in INDICF5_LANG_MAP:
        result = _indicf5_synthesize(text, lang_code)
        if result:
            return result

    # 2. Try Piper (local, fast, natural — mainly English)
    if _piper_available:
        model_path = _get_piper_model_path(lang_code)
        if model_path:
            result = _piper_synthesize(text, model_path)
            if result:
                return result

    # 3. Try espeak-ng (local, ultra-fast, robotic)
    if _espeak_available:
        result = _espeak_synthesize(text, lang_code)
        if result:
            return result

    # 4. Try edge-tts (cloud fallback)
    if _edge_tts_available:
        result = _edge_tts_synthesize(text, lang_code)
        if result:
            return result

    print("  [TTS] No backend available!")
    return b""


def synthesize_stream(text: str, lang_code: str = "en-IN", backend: str = "auto"):
    """
    Synthesize text to raw PCM bytes (generator).
    Yields tuple of (pcm_bytes, sample_rate, sample_width).
    Falls back to whole-file synthesis if streaming is unavailable.
    """
    if not text:
        return

    # Try Piper streaming first (if method exists)
    if (backend == "auto" or backend == "piper") and _piper_available:
        model_path = _get_piper_model_path(lang_code)
        if model_path:
            streamed = False
            try:
                for chunk in _piper_synthesize_stream(text, model_path):
                    streamed = True
                    yield chunk
            except Exception:
                streamed = False
            if streamed:
                return

    # Fallback: generate entire WAV, then yield as a single chunk
    result = synthesize(text, lang_code, backend)
    if result and len(result) > 44:
        try:
            with wave.open(io.BytesIO(result), "rb") as wf:
                sr = wf.getframerate()
                sw = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())
            yield (frames, sr, sw)
        except Exception as e:
            print(f"  [Stream fallback error] {e}")


def preload_piper(lang_code: str):
    """Pre-load a Piper model into cache so the first TTS call is fast."""
    if not _piper_available:
        return
    model_path = _get_piper_model_path(lang_code)
    if model_path:
        key = str(model_path)
        if key not in _piper_cache:
            try:
                _piper_cache[key] = PiperVoice.load(str(model_path))
            except Exception:
                pass


# ── IndicF5 TTS ────────────────────────────────────────────────────────────

def _get_indicf5_model():
    """Lazily load IndicF5 model (cached)."""
    global _indicf5_model, _indicf5_prompts_dir
    if _indicf5_model is not None:
        return _indicf5_model

    try:
        model = _IndicF5AutoModel.from_pretrained(
            "ai4bharat/IndicF5", trust_remote_code=True
        )
        _indicf5_model = model
        # Check for bundled prompt audios
        try:
            from huggingface_hub import snapshot_download
            repo_path = snapshot_download("ai4bharat/IndicF5")
            prompts_path = os.path.join(repo_path, "prompts")
            if os.path.isdir(prompts_path):
                _indicf5_prompts_dir = prompts_path
        except Exception:
            pass
        return _indicf5_model
    except Exception as e:
        print(f"  [IndicF5 load error] {e}")
        return None


def _indicf5_synthesize(text: str, lang_code: str) -> bytes:
    """Synthesize using IndicF5 model. Returns WAV bytes."""
    if not _indicf5_available:
        return b""

    model = _get_indicf5_model()
    if model is None:
        return b""

    try:
        import soundfile as sf

        # Find a reference prompt audio for the language
        ref_audio, ref_text = _get_indicf5_ref_prompt(lang_code)

        if ref_audio and ref_text:
            audio = model(text, ref_audio_path=ref_audio, ref_text=ref_text)
        else:
            # Try without reference (some versions support this)
            audio = model(text)

        if audio is None:
            return b""

        # Convert to WAV bytes
        audio_np = np.array(audio, dtype=np.float32)
        if audio_np.dtype == np.int16:
            audio_np = audio_np.astype(np.float32) / 32768.0

        wav_io = io.BytesIO()
        sf.write(wav_io, audio_np, 24000, format="WAV", subtype="PCM_16")
        return wav_io.getvalue()
    except Exception as e:
        print(f"  [IndicF5 TTS error] {e}")
        return b""


def _get_indicf5_ref_prompt(lang_code: str) -> tuple:
    """Get reference prompt audio path and text for IndicF5."""
    if _indicf5_prompts_dir is None:
        return (None, None)

    # Look for prompt files matching the language
    lang_prefix_map = {
        "hi-IN": "HIN",
        "ta-IN": "TAM",
        "te-IN": "TEL",
        "kn-IN": "KAN",
        "ml-IN": "MAL",
        "bn-IN": "BEN",
        "mr-IN": "MAR",
        "gu-IN": "GUJ",
        "pa-IN": "PAN",
    }

    prefix = lang_prefix_map.get(lang_code)
    if not prefix:
        return (None, None)

    # Find first matching prompt file
    for fname in sorted(os.listdir(_indicf5_prompts_dir)):
        if fname.startswith(prefix) and fname.endswith(".wav"):
            ref_audio = os.path.join(_indicf5_prompts_dir, fname)
            # Default reference text — IndicF5 works best with a ref but can work without
            ref_text = ""
            return (ref_audio, ref_text)

    return (None, None)


# ── Piper TTS ──────────────────────────────────────────────────────────────

def _get_piper_model_path(lang_code: str) -> Path | None:
    """Return the .onnx model path if it exists on disk."""
    voice_name = PIPER_VOICES.get(lang_code)
    if not voice_name:
        return None
    model_path = PIPER_MODELS_DIR / f"{voice_name}.onnx"
    if model_path.exists():
        return model_path
    return None


def _piper_synthesize(text: str, model_path: Path) -> bytes:
    """Synthesize using Piper voice model (optimized). Returns WAV bytes."""
    try:
        key = str(model_path)
        # Use cached model if available, load only once
        if key not in _piper_cache:
            _piper_cache[key] = PiperVoice.load(str(model_path))
        voice = _piper_cache[key]

        # Synthesize directly to WAV bytes without intermediate buffer
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wf:
            voice.synthesize_wav(text, wf)
        return wav_io.getvalue()
    except Exception as e:
        print(f"  [Piper TTS error] {e}")
        return b""


def _piper_synthesize_stream(text: str, model_path: Path):
    """Stream raw audio bytes from Piper TTS if available, else fall back."""
    key = str(model_path)
    if key not in _piper_cache:
        _piper_cache[key] = PiperVoice.load(str(model_path))
    voice = _piper_cache[key]
    sr = voice.config.sample_rate

    # Try streaming methods in order of preference
    if hasattr(voice, 'synthesize_stream_raw'):
        for audio_bytes in voice.synthesize_stream_raw(text):
            yield (audio_bytes, sr, 2)
        return

    # Fall back to full synthesis, yield as single chunk
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wf:
        voice.synthesize_wav(text, wf)
    wav_io.seek(0)
    with wave.open(wav_io, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
    yield (frames, sr, 2)


# ── espeak-ng ──────────────────────────────────────────────────────────────

def _espeak_synthesize(text: str, lang_code: str) -> bytes:
    """Synthesize using espeak-ng. Returns WAV bytes."""
    espeak_lang = _ESPEAK_LANG.get(lang_code, "en")
    espeak_cmd = "espeak-ng" if shutil.which("espeak-ng") else "espeak"

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        subprocess.run(
            [espeak_cmd, "-v", espeak_lang, "-w", tmp_path, "--", text],
            capture_output=True, timeout=5, check=True,
        )

        with open(tmp_path, "rb") as f:
            wav_bytes = f.read()
        return wav_bytes
    except Exception as e:
        print(f"  [espeak error] {e}")
        return b""
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── edge-tts (cloud fallback) ─────────────────────────────────────────────

def _edge_tts_synthesize(text: str, lang_code: str) -> bytes:
    """Synthesize using Microsoft Edge TTS (requires internet)."""
    if miniaudio is None:
        print("  [edge-tts error] miniaudio is not installed")
        return b""

    voice = EDGE_TTS_VOICES.get(lang_code, "en-IN-NeerjaNeural")

    try:
        # Use sync streaming API — avoids asyncio event loop overhead
        comm = edge_tts.Communicate(text, voice)
        mp3_buf = io.BytesIO()
        for chunk in comm.stream_sync():
            if chunk["type"] == "audio":
                mp3_buf.write(chunk.get("data", b""))

        mp3_bytes = mp3_buf.getvalue()
        if not mp3_bytes:
            return b""

        # Decode MP3 -> raw PCM using miniaudio, then wrap as WAV
        decoded = miniaudio.decode(mp3_bytes, output_format=miniaudio.SampleFormat.SIGNED16)
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(decoded.nchannels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(decoded.sample_rate)
            wf.writeframes(decoded.samples)
        return wav_io.getvalue()
    except Exception as e:
        print(f"  [edge-tts error] {e}")
        return b""


def print_status():
    """Print which TTS backends are available."""
    print(f"  Piper TTS:   {'✓ installed' if _piper_available else '✗ not installed (pip install piper-tts)'}")
    print(f"  espeak-ng:   {'✓ found' if _espeak_available else '✗ not found (apt install espeak-ng)'}")
    edge_msg = "✓ installed" if _edge_tts_available else "✗ not available (requires edge-tts + miniaudio)"
    print(f"  edge-tts:    {edge_msg}")
    print()
    for lang_code in ["en-IN", "hi-IN", "ta-IN", "te-IN"]:
        backend = get_backend_name(lang_code)
        print(f"  {lang_code}: will use {backend}")
