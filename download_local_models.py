"""
Download one-time local inference assets:
1) faster-whisper STT model  (default: small)
2) NLLB-200 translation model (CTranslate2 int8)

Run:
    python download_local_models.py
"""

import os
from config import LOCAL_STT_MODEL, NLLB_MODEL

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None


def _ensure_whisper_model() -> None:
    print("=" * 56)
    print("1/2  faster-whisper STT model")
    print("=" * 56)

    if WhisperModel is None:
        print("faster-whisper is not installed.")
        print("Install dependencies first: pip install -r requirements.txt")
        return

    print(f"Downloading/validating faster-whisper model: {LOCAL_STT_MODEL}")
    try:
        WhisperModel(LOCAL_STT_MODEL, device="cpu", compute_type="int8")
        print("STT model ready.\n")
    except Exception as exc:
        print(f"Failed to prepare STT model: {exc}\n")


def _ensure_nllb_model() -> None:
    print("=" * 56)
    print("2/2  NLLB-200 translation model (CTranslate2 int8)")
    print("=" * 56)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is not installed.")
        print("Run: pip install huggingface_hub")
        return

    try:
        import ctranslate2  # noqa: F401
    except ImportError:
        print("ctranslate2 is not installed.")
        print("Run: pip install ctranslate2")
        return

    try:
        import sentencepiece  # noqa: F401
    except ImportError:
        print("sentencepiece is not installed.")
        print("Run: pip install sentencepiece")
        return

    print(f"Downloading/validating NLLB model: {NLLB_MODEL}")
    try:
        path = snapshot_download(NLLB_MODEL)
        print(f"NLLB model cached at: {path}")

        # Verify model loads
        model = ctranslate2.Translator(path, device="cpu", compute_type="int8")
        del model
        print("NLLB model ready.\n")
    except Exception as exc:
        print(f"Failed to prepare NLLB model: {exc}\n")


def main() -> None:
    _ensure_whisper_model()
    _ensure_nllb_model()
    print("Done. You can now run with --inference-backend local")


if __name__ == "__main__":
    main()
