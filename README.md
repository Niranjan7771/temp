# Wearable Voice Translator v2

Low-latency voice translation pipeline designed for **Raspberry Pi earbuds**.  
The Pi captures audio and plays output; a laptop handles all heavy processing (STT, translation, TTS) over the local network.

## Architecture

```
[Raspberry Pi Zero 2W]                      [Laptop — Edge Server]
USB Mic → sounddevice capture → VAD          ┌──────────────────────────┐
  → HTTP POST (WAV) ─────────────────────►   │ faster-whisper (small)   │
                                              │ NLLB-200 translation     │
  ◄───────── JSON (text + base64 WAV) ────   │ IndicF5 / Piper TTS      │
sounddevice playback → Bluetooth Speaker     └──────────────────────────┘
```

**Expected latency: ~1.0-2.0s** end-to-end over WiFi

## Model Stack

| Stage       | Model                            | Languages              | Notes              |
|-------------|----------------------------------|------------------------|--------------------|
| STT         | faster-whisper `small` or Deepgram Nova | 99 languages | Local or cloud |
| Translation | NLLB-200-distilled-600M          | 200 languages          | CT2, int8          |
| TTS (Indic) | IndicF5 (ai4bharat, 400M)        | 11 Indian languages    | Near-human quality |
| TTS (en/hi) | Piper ONNX                       | English, Hindi, Malayalam | Offline, fast   |
| TTS (fallback) | espeak-ng                     | All languages          | Robotic, offline   |

### Supported Indian Languages

English, Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi

## Quick Start — Laptop (Edge Server)

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: install IndicF5 TTS for high-quality Indic speech
pip install git+https://github.com/ai4bharat/IndicF5.git

# Download models (Whisper small + NLLB-200, one-time ~1.5GB)
python download_local_models.py

# Start the edge server (accessible on your LAN)
python edge_server.py --lang hindi
# Server starts on http://0.0.0.0:5555

# Optional: Deepgram Nova STT + local NLLB translation
export DEEPGRAM_API_KEY="your_deepgram_key"
python edge_server.py --lang hindi --inference-backend deepgram
```

## Quick Start — Raspberry Pi (Audio Client)

```bash
# System dependencies
sudo apt update
sudo apt install -y portaudio19-dev

# Python dependencies (lightweight — only requests, sounddevice, numpy)
pip install requests sounddevice numpy

# Find the laptop's IP from the edge server output, then:
python pi_client.py --server http://192.168.x.x:5555 --lang hindi

# List audio devices first if needed
python pi_client.py --list-devices --server http://localhost:5555
```

## Standalone Mode (No Pi)

Run everything on the laptop with a local mic:

```bash
# Download models
python download_local_models.py

# Run headless with local backend
python main.py --inference-backend local --lang hindi

# Or run web prototype
python serve_demo.py --port 8080 --inference-backend local
```

## TTS Priority Chain

| Priority | Engine     | Speed       | Quality  | Offline? | Languages              |
|----------|-----------|-------------|----------|----------|------------------------|
| 1        | IndicF5   | ~1-2s       | Natural  | ✓        | 11 Indian languages    |
| 2        | Piper TTS | ~200-400ms  | Natural  | ✓        | English, Hindi, Malayalam |
| 3        | espeak-ng | ~50ms       | Robotic  | ✓        | All languages          |
| 4        | edge-tts  | ~300-500ms  | Natural  | ✗        | All languages          |

## Command-Line Options

### edge_server.py (Laptop)

```
python edge_server.py [options]

  --host HOST           Bind address (default: 0.0.0.0)
  --port PORT           Port (default: 5555)
  --lang LANG           Default target language (default: hindi)
  --inference-backend   sarvam, local, or deepgram (default: local)
```

### pi_client.py (Raspberry Pi)

```
python pi_client.py [options]

  --server URL          Edge server URL (required, e.g. http://192.168.1.100:5555)
  --lang LANG           Target language (default: hindi)
  --input-device N      Mic device index
  --output-device N     Speaker device index
  --gain N              Playback volume multiplier (default: 1.0)
  --list-devices        Show available audio devices and exit
```

### main.py (Standalone)

```
python main.py [options]

  --lang LANG           Output language (default: hindi)
  --inference-backend   sarvam, local, or deepgram (default: local)
  --list-devices        Show available audio devices
  --input-device N      Mic device index
  --threshold N         RMS speech detection threshold (default: 300)
```

## File Structure

```
wearable-v2/
├── main.py                   # Standalone mic capture + VAD + processing
├── edge_server.py            # Laptop HTTP server (Pi sends audio here)
├── pi_client.py              # Pi audio client (mic → server → speaker)
├── config.py                 # All settings (languages, models, thresholds)
├── local_inference_client.py # STT (Whisper) + Translation (NLLB-200)
├── inference_client.py       # Backend router (sarvam / local)
├── tts_engine.py             # TTS: IndicF5 → Piper → espeak → edge-tts
├── sarvam_client.py          # Sarvam cloud API (optional)
├── download_models.py        # Piper voice model downloader
├── download_local_models.py  # Whisper + NLLB-200 model downloader
├── serve_demo.py             # Web prototype UI
├── requirements.txt          # pip dependencies
└── piper_models/             # (created by download_models.py)
```

## Notes

- The Pi only needs `requests`, `sounddevice`, and `numpy` — no ML libraries.
- Internet is required for cloud backends (Sarvam, Deepgram) and one-time model downloads.
- On Pi, connect Bluetooth earbuds via `bluetoothctl` — they appear as a regular audio device.
- NLLB-200 covers **all** Indian languages including Tamil and Telugu (which Argos lacked).
