"""
Download one-time local inference assets:
1) faster-whisper STT model
2) Argos Translate language packs

Run:
    python download_local_models.py
"""

from config import LOCAL_STT_MODEL

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

try:
    from argostranslate import package as argos_package
    from argostranslate import translate as argos_translate
except Exception:
    argos_package = None
    argos_translate = None


_TARGET_PAIRS = [
    ("en", "hi"),
    ("hi", "en"),
    ("en", "ta"),
    ("ta", "en"),
    ("en", "te"),
    ("te", "en"),
]


def _installed_pair(src: str, tgt: str) -> bool:
    if argos_translate is None:
        return False
    langs = argos_translate.get_installed_languages()
    lang_map = {str(lang.code).lower(): lang for lang in langs}
    src_lang = lang_map.get(src)
    tgt_lang = lang_map.get(tgt)
    if src_lang is None or tgt_lang is None:
        return False
    try:
        src_lang.get_translation(tgt_lang)
        return True
    except Exception:
        return False


def _ensure_whisper_model() -> None:
    print("=" * 56)
    print("Local STT model")
    print("=" * 56)

    if WhisperModel is None:
        print("faster-whisper is not installed.")
        print("Install dependencies first: pip install -r requirements.txt")
        return

    print(f"Downloading/validating faster-whisper model: {LOCAL_STT_MODEL}")
    try:
        WhisperModel(LOCAL_STT_MODEL, device="cpu", compute_type="int8")
        print("STT model ready.")
    except Exception as exc:
        print(f"Failed to prepare STT model: {exc}")


def _ensure_argos_packages() -> None:
    print("\n" + "=" * 56)
    print("Local translation packages")
    print("=" * 56)

    if argos_package is None or argos_translate is None:
        print("argostranslate is not installed.")
        print("Install dependencies first: pip install -r requirements.txt")
        return

    try:
        argos_package.update_package_index()
        available = argos_package.get_available_packages()
    except Exception as exc:
        print(f"Could not fetch Argos package index: {exc}")
        return

    available_map = {(pkg.from_code, pkg.to_code): pkg for pkg in available}

    for src, tgt in _TARGET_PAIRS:
        if _installed_pair(src, tgt):
            print(f"{src}->{tgt}: already installed")
            continue

        pkg = available_map.get((src, tgt))
        if pkg is None:
            print(f"{src}->{tgt}: package not available in current index")
            continue

        print(f"{src}->{tgt}: downloading...")
        try:
            pkg_path = pkg.download()
            argos_package.install_from_path(pkg_path)
            if _installed_pair(src, tgt):
                print(f"{src}->{tgt}: installed")
            else:
                print(f"{src}->{tgt}: install finished but pair still unavailable")
        except Exception as exc:
            print(f"{src}->{tgt}: install failed ({exc})")


def main() -> None:
    _ensure_whisper_model()
    _ensure_argos_packages()
    print("\nDone. You can now run with --inference-backend local")


if __name__ == "__main__":
    main()
