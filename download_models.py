"""
Download Piper TTS voice models for supported languages.

Run once before first use:
    python download_models.py

Downloads models to ./piper_models/
"""

import os
import sys
import urllib.request
from pathlib import Path

from config import PIPER_MODELS_DIR, PIPER_VOICES

# Piper models are hosted on HuggingFace
_HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# Model download paths (voice_name → relative path on HuggingFace)
_MODEL_PATHS = {
    "en_US-lessac-low": "en/en_US/lessac/low/en_US-lessac-low.onnx",
    "en_US-lessac-medium": "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    "hi_IN-pratham-medium": "hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx",
    "te_IN-maya-medium": "te/te_IN/maya/medium/te_IN-maya-medium.onnx",
}

# Each .onnx model also needs a .onnx.json config file
_CONFIG_SUFFIX = ".json"


def download_file(url: str, dest: Path):
    """Download a file with progress indication."""
    print(f"  Downloading: {url}")
    print(f"  To:          {dest}")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "voice-translator/2.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 // total
                        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                        print(f"\r  [{bar}] {pct}%  ({downloaded // 1024}KB / {total // 1024}KB)", end="", flush=True)
            print()
    except Exception as e:
        print(f"\n  ERROR: {e}")
        if dest.exists():
            dest.unlink()
        return False
    return True


def main():
    PIPER_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("  Piper TTS Model Downloader")
    print("=" * 50)
    print(f"  Models directory: {PIPER_MODELS_DIR}")
    print()

    to_download = []
    for lang_code, voice_name in PIPER_VOICES.items():
        if voice_name is None:
            print(f"  {lang_code}: No Piper model available (will use fallback TTS)")
            continue

        onnx_path = PIPER_MODELS_DIR / f"{voice_name}.onnx"
        json_path = PIPER_MODELS_DIR / f"{voice_name}.onnx.json"

        if onnx_path.exists() and json_path.exists():
            print(f"  {lang_code}: {voice_name} — already downloaded ✓")
            continue

        hf_rel = _MODEL_PATHS.get(voice_name)
        if not hf_rel:
            print(f"  {lang_code}: {voice_name} — download path unknown, skipping")
            continue

        to_download.append((lang_code, voice_name, hf_rel))

    if not to_download:
        print("\n  All available models already downloaded!")
        return

    print(f"\n  Will download {len(to_download)} model(s):\n")

    for lang_code, voice_name, hf_rel in to_download:
        print(f"── {lang_code}: {voice_name} ──")

        onnx_url = f"{_HF_BASE}/{hf_rel}"
        json_url = f"{onnx_url}.json"

        onnx_dest = PIPER_MODELS_DIR / f"{voice_name}.onnx"
        json_dest = PIPER_MODELS_DIR / f"{voice_name}.onnx.json"

        # Download config first (small file)
        if not json_dest.exists():
            if not download_file(json_url, json_dest):
                print(f"  Skipping {voice_name} — config download failed")
                continue

        # Download model (larger file, ~60-100MB)
        if not onnx_dest.exists():
            if not download_file(onnx_url, onnx_dest):
                print(f"  Skipping {voice_name} — model download failed")
                continue

        print(f"  ✓ {voice_name} ready\n")

    print("\nDone! Run `python main.py` to start the translator.")


if __name__ == "__main__":
    main()
