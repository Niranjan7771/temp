"""
Web prototype server for the Wearable Voice Translator.

This serves the frontend from local_demo/ and exposes working API endpoints:
  - GET  /api/health
  - POST /api/text-translate
  - POST /api/voice-translate

Usage:
  python serve_demo.py --port 8080
"""

import argparse
import base64
import io
import socket
import threading
import time
import wave
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
import numpy as np
import sounddevice as sd

from config import LANG_MAP, SARVAM_API_KEY
from sarvam_client import transcribe_and_translate, translate_text
from tts_engine import get_backend_name, synthesize


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "local_demo"

LANG_LABELS = {
    "english": "English",
    "hindi": "Hindi",
    "tamil": "Tamil",
    "telugu": "Telugu",
}

_play_lock = threading.Lock()


def _safe_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _play_wav_bytes(wav_bytes: bytes, output_device: int | None, playback_gain: float) -> None:
    if not wav_bytes or len(wav_bytes) < 44:
        return

    with _play_lock:
        try:
            buf = io.BytesIO(wav_bytes)
            with wave.open(buf, "rb") as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                sw = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())

            if sw == 2:
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
            elif sw == 1:
                audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
            else:
                return

            if ch > 1:
                audio = audio.reshape(-1, ch)[:, 0]

            if playback_gain != 1.0:
                audio = np.clip(audio * playback_gain, -1.0, 1.0)

            sd.stop()
            sd.play(audio, samplerate=sr, blocking=True, device=output_device)
            sd.wait()
        except Exception:
            return


def _get_primary_ip() -> str | None:
    """Best-effort LAN IP detection (IPv4)."""
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return None
    finally:
        if sock is not None:
            sock.close()


def _get_host_ips() -> list[str]:
    """Collect non-loopback IPv4 addresses for the host."""
    ips: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            addr = info[4][0]
            if ":" in addr:
                continue
            if addr.startswith("127."):
                continue
            ips.add(addr)
    except OSError:
        pass
    return sorted(ips)


def _normalize_lang(raw: str | None, fallback: str = "hindi") -> tuple[str, str]:
    """Return (lang_key, lang_code) from either key (hindi) or code (hi-IN)."""
    if not raw:
        return fallback, LANG_MAP[fallback]

    text = raw.strip()
    if not text:
        return fallback, LANG_MAP[fallback]

    lowered = text.lower()
    if lowered in LANG_MAP:
        return lowered, LANG_MAP[lowered]

    for key, code in LANG_MAP.items():
        if code.lower() == lowered:
            return key, code

    return fallback, LANG_MAP[fallback]


def _build_app() -> Flask:
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/edge")
    def edge_page():
        return send_from_directory(STATIC_DIR, "edge.html")

    @app.get("/api/health")
    def health():
        target_code = LANG_MAP["hindi"]
        return jsonify(
            {
                "status": "ok",
                "apiKeyConfigured": bool(SARVAM_API_KEY),
                "defaultLanguage": "hindi",
                "languages": [
                    {
                        "key": key,
                        "label": LANG_LABELS.get(key, key.title()),
                        "code": code,
                    }
                    for key, code in LANG_MAP.items()
                ],
                "ttsPreviewBackend": get_backend_name(target_code),
            }
        )

    @app.post("/api/text-translate")
    def text_translate():
        payload = request.get_json(silent=True) or {}

        text = (payload.get("text") or "").strip()
        if not text:
            return jsonify({"error": "text is required"}), 400

        target_key, target_code = _normalize_lang(payload.get("targetLang"), fallback="hindi")
        source_key, source_code = _normalize_lang(payload.get("sourceLang"), fallback="english")
        tts_backend = (payload.get("ttsBackend") or "auto").strip().lower() or "auto"
        include_audio = bool(payload.get("includeAudio", True))

        if not SARVAM_API_KEY:
            return jsonify({"error": "SARVAM_API_KEY is not configured"}), 503

        t_total_start = time.time()
        translated, t_translate = translate_text(
            text,
            source_lang=source_code,
            target_lang=target_code,
            fallback_to_source=True,
        )

        t_tts = 0.0
        audio_b64 = ""
        selected_backend = get_backend_name(target_code)
        if include_audio and translated:
            t_tts_start = time.time()
            wav_bytes = synthesize(translated, target_code, backend=tts_backend)
            t_tts = time.time() - t_tts_start
            if wav_bytes:
                audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

        total = time.time() - t_total_start
        return jsonify(
            {
                "sourceText": text,
                "englishText": text if source_code == "en-IN" else "",
                "translatedText": translated,
                "targetLang": target_key,
                "targetCode": target_code,
                "metrics": {
                    "vadSeconds": 0.0,
                    "sttSeconds": 0.0,
                    "translateSeconds": t_translate,
                    "ttsSeconds": t_tts,
                    "totalSeconds": total,
                },
                "tts": {
                    "requested": tts_backend,
                    "selected": selected_backend,
                    "audioWavBase64": audio_b64,
                },
            }
        )

    @app.post("/api/voice-translate")
    def voice_translate():
        if "audio" not in request.files:
            return jsonify({"error": "audio file is required"}), 400

        if not SARVAM_API_KEY:
            return jsonify({"error": "SARVAM_API_KEY is not configured"}), 503

        target_key, target_code = _normalize_lang(request.form.get("targetLang"), fallback="hindi")
        direct_translate = str(request.form.get("directTranslate", "true")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        tts_backend = (request.form.get("ttsBackend") or "auto").strip().lower() or "auto"

        audio_file = request.files["audio"]
        wav_bytes = audio_file.read()
        if not wav_bytes:
            return jsonify({"error": "empty audio payload"}), 400

        t_total_start = time.time()
        result = transcribe_and_translate(
            wav_bytes,
            tgt_lang=target_code,
            direct_translate=direct_translate,
            audio_filename=audio_file.filename or "audio.wav",
            audio_content_type=audio_file.mimetype or "audio/wav",
        )

        if len(result) == 4:
            translated_text, english_text, t_stt, t_translate = result
        else:
            translated_text, t_stt, t_translate = result
            english_text = ""

        translated_text = (translated_text or "").strip()
        if not translated_text:
            return jsonify({"error": "no speech detected or translation failed"}), 422

        t_tts_start = time.time()
        out_wav = synthesize(translated_text, target_code, backend=tts_backend)
        t_tts = time.time() - t_tts_start
        audio_b64 = base64.b64encode(out_wav).decode("ascii") if out_wav else ""

        total = time.time() - t_total_start
        return jsonify(
            {
                "englishText": english_text,
                "translatedText": translated_text,
                "targetLang": target_key,
                "targetCode": target_code,
                "metrics": {
                    "vadSeconds": 0.0,
                    "sttSeconds": t_stt,
                    "translateSeconds": t_translate,
                    "ttsSeconds": t_tts,
                    "totalSeconds": total,
                },
                "tts": {
                    "requested": tts_backend,
                    "selected": get_backend_name(target_code),
                    "audioWavBase64": audio_b64,
                },
            }
        )

    @app.post("/api/speak")
    def speak_on_pi():
        payload = request.get_json(silent=True) or {}

        text = (payload.get("text") or "").strip()
        if not text:
            return jsonify({"error": "text is required"}), 400

        target_key, target_code = _normalize_lang(payload.get("targetLang"), fallback="hindi")
        source_key, source_code = _normalize_lang(payload.get("sourceLang"), fallback="english")
        tts_backend = (payload.get("ttsBackend") or "auto").strip().lower() or "auto"

        output_device = _safe_int(payload.get("outputDevice"), default=None)
        playback_gain = float(payload.get("playbackGain") or 1.0)
        playback_gain = max(0.1, min(4.0, playback_gain))

        if source_code != target_code:
            if not SARVAM_API_KEY:
                return jsonify({"error": "SARVAM_API_KEY is not configured"}), 503
            translated, t_translate = translate_text(
                text,
                source_lang=source_code,
                target_lang=target_code,
                fallback_to_source=False,
            )
            if not translated:
                return jsonify({"error": "translation failed"}), 422
        else:
            translated = text
            t_translate = 0.0

        wav_bytes = synthesize(translated, target_code, backend=tts_backend)
        if not wav_bytes:
            return jsonify({"error": "tts failed"}), 500

        threading.Thread(
            target=_play_wav_bytes,
            args=(wav_bytes, output_device, playback_gain),
            daemon=True,
        ).start()

        return jsonify(
            {
                "status": "queued",
                "sourceText": text,
                "translatedText": translated,
                "targetLang": target_key,
                "targetCode": target_code,
                "translateSeconds": t_translate,
                "ttsBackend": get_backend_name(target_code),
            }
        )

    @app.get("/<path:file_path>")
    def static_files(file_path: str):
        candidate = (STATIC_DIR / file_path).resolve()
        try:
            candidate.relative_to(STATIC_DIR.resolve())
        except ValueError:
            return jsonify({"error": "invalid path"}), 400

        if not candidate.exists() or not candidate.is_file():
            return jsonify({"error": "not found"}), 404
        return send_from_directory(STATIC_DIR, file_path)

    return app


def main():
    parser = argparse.ArgumentParser(description="Serve the web prototype over the network")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on (default: 8080)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    args = parser.parse_args()

    if not STATIC_DIR.exists():
        print(f"ERROR: Directory not found: {STATIC_DIR}")
        return

    if not (STATIC_DIR / "index.html").exists():
        print(f"WARNING: index.html not found in {STATIC_DIR}")

    app = _build_app()

    print("=" * 56)
    print("Wearable Voice Translator web prototype is running")
    print("=" * 56)
    print(f"Serving static files: {STATIC_DIR}")
    print(f"Bind host:            {args.host}")
    print(f"Port:                 {args.port}")
    print(f"API key configured:   {'yes' if SARVAM_API_KEY else 'no'}")
    print()
    print(f"Open on this device:  http://localhost:{args.port}")

    candidates = []
    primary = _get_primary_ip()
    if primary:
        candidates.append(primary)
    candidates.extend(_get_host_ips())

    seen = set()
    for ip in candidates:
        if ip in seen:
            continue
        seen.add(ip)
        print(f"Open on phone:        http://{ip}:{args.port}")

    print("\nPress Ctrl+C to stop.")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
