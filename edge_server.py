"""
Edge Server — runs on the laptop to process audio from the Raspberry Pi.

Architecture:
  Pi (mic+speaker) → sends audio over HTTP → Laptop (STT + translate + TTS) → returns WAV

Endpoints:
  POST /api/process-audio
    - Accepts: multipart/form-data with 'audio' file and 'target_lang' field
    - Returns: JSON with translated_text, english_text, tts_audio (base64 WAV), timings

  GET /api/health
    - Returns: server status and backend info

Usage:
  python edge_server.py --port 5555 --lang hindi
  python edge_server.py --port 5555 --lang tamil --tts indicf5
"""

import argparse
import base64
import io
import socket
import time
import wave
from pathlib import Path

from flask import Flask, jsonify, request
import numpy as np

from config import (
    LANG_MAP,
    EDGE_SERVER_HOST,
    EDGE_SERVER_PORT,
    INFERENCE_BACKEND,
)
from inference_client import (
    set_backend,
    get_backend,
    backend_ready,
    backend_status,
    transcribe_and_translate,
    translate_text,
)
from tts_engine import synthesize, get_backend_name


app = Flask(__name__)

# Default target language (overridable via CLI)
_default_tgt_lang = "hi-IN"


@app.route("/api/health", methods=["GET"])
def health():
    ready, status_msg = backend_ready()
    return jsonify({
        "status": "ok" if ready else "not_ready",
        "inference_backend": get_backend(),
        "message": status_msg,
        "default_target_lang": _default_tgt_lang,
        "supported_languages": list(LANG_MAP.keys()),
    })


@app.route("/api/process-audio", methods=["POST"])
def process_audio():
    """
    Full pipeline: receive audio → STT → translate → TTS → return everything.
    
    Expected form fields:
      audio: WAV file (required)
      target_lang: BCP-47 code like hi-IN (optional, uses server default)
    """
    t_start = time.time()

    # Get audio file
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()

    if not audio_bytes or len(audio_bytes) < 44:
        return jsonify({"error": "Audio file too small or empty"}), 400

    # Get target language
    tgt_lang = request.form.get("target_lang", _default_tgt_lang)

    # Step 1+2: STT + Translation
    try:
        translated_text, english_text, t_stt, t_translate = transcribe_and_translate(
            audio_bytes,
            tgt_lang=tgt_lang,
        )
    except Exception as e:
        return jsonify({"error": f"STT/Translation failed: {e}"}), 500

    if not english_text and not translated_text:
        return jsonify({
            "english_text": "",
            "translated_text": "",
            "tts_audio": "",
            "timings": {
                "stt_ms": 0,
                "translate_ms": 0,
                "tts_ms": 0,
                "total_ms": int((time.time() - t_start) * 1000),
            },
        })

    # Step 3: TTS on translated text
    tts_text = translated_text or english_text
    t_tts_start = time.time()
    try:
        tts_wav = synthesize(tts_text, lang_code=tgt_lang, backend="auto")
    except Exception as e:
        print(f"  [Edge TTS error] {e}")
        tts_wav = b""
    t_tts = time.time() - t_tts_start

    # Encode TTS audio as base64 for JSON transport
    tts_b64 = base64.b64encode(tts_wav).decode("ascii") if tts_wav else ""

    t_total = time.time() - t_start

    return jsonify({
        "english_text": english_text or "",
        "translated_text": translated_text or "",
        "tts_audio": tts_b64,
        "tts_backend": get_backend_name(tgt_lang),
        "timings": {
            "stt_ms": int(t_stt * 1000),
            "translate_ms": int(t_translate * 1000),
            "tts_ms": int(t_tts * 1000),
            "total_ms": int(t_total * 1000),
        },
    })


@app.route("/api/translate-text", methods=["POST"])
def translate_text_endpoint():
    """Text-only translation (no audio)."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    source_lang = data.get("source_lang", "en-IN")
    target_lang = data.get("target_lang", _default_tgt_lang)

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        translated, t_ms = translate_text(
            text, source_lang=source_lang, target_lang=target_lang,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "translated_text": translated,
        "translate_ms": int(t_ms * 1000),
    })


def _get_primary_ip() -> str | None:
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return None
    finally:
        if sock:
            sock.close()


def main():
    global _default_tgt_lang

    parser = argparse.ArgumentParser(description="Edge Server for Pi Wearable Translator")
    parser.add_argument("--host", default=EDGE_SERVER_HOST, help="Bind address")
    parser.add_argument("--port", type=int, default=EDGE_SERVER_PORT, help="Port number")
    parser.add_argument(
        "--lang", default="hindi",
        choices=list(LANG_MAP.keys()),
        help="Default target language",
    )
    parser.add_argument(
        "--inference-backend", default="local",
        choices=["sarvam", "local"],
        help="Inference backend",
    )
    args = parser.parse_args()

    _default_tgt_lang = LANG_MAP.get(args.lang, "hi-IN")
    set_backend(args.inference_backend)

    ready, msg = backend_ready()
    if not ready:
        print(f"WARNING: Backend not ready — {msg}")

    lan_ip = _get_primary_ip()
    print("=" * 60)
    print("  EDGE SERVER — Wearable Voice Translator")
    print("=" * 60)
    print(f"  Backend:   {get_backend()}")
    print(f"  Language:  {args.lang} ({_default_tgt_lang})")
    print(f"  TTS:       {get_backend_name(_default_tgt_lang)}")
    print(f"  Status:    {msg}")
    print(f"  Listening: http://{args.host}:{args.port}")
    if lan_ip:
        print(f"  LAN URL:   http://{lan_ip}:{args.port}")
    print()
    print("  Pi should POST audio to: http://<laptop-ip>:{}/api/process-audio".format(args.port))
    print("=" * 60)

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
