# Wearable Voice Translator v2

A low-latency, real-time voice translation pipeline designed for **Raspberry Pi-based wearable earbuds**. The system captures spoken audio, transcribes it, translates it to a target Indian language, and speaks the translation back — all within 1–2 seconds.

The Raspberry Pi handles audio capture and playback while a laptop (edge server) performs the compute-intensive processing (STT, translation, TTS) over the local WiFi network.

---

## Table of Contents

- [Architecture](#architecture)
- [Model Stack](#model-stack)
- [Supported Languages](#supported-languages)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Mode 1: Edge Server + Pi Client](#mode-1-edge-server--pi-client-recommended)
  - [Mode 2: Streaming (WebSocket)](#mode-2-streaming-websocket)
  - [Mode 3: Standalone (No Pi)](#mode-3-standalone-no-pi)
  - [Mode 4: Web Prototype](#mode-4-web-prototype)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Inference Backends](#inference-backends)
  - [TTS Priority Chain](#tts-priority-chain)
  - [Voice Activity Detection (VAD)](#voice-activity-detection-vad)
- [Command-Line Reference](#command-line-reference)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## Architecture

The system follows a split-architecture design where the resource-constrained Pi offloads heavy computation to a laptop over the LAN.

### HTTP Mode (Batch)

```
┌─────────────────────────────┐          ┌──────────────────────────────────┐
│    Raspberry Pi Zero 2W     │          │       Laptop (Edge Server)       │
│                             │          │                                  │
│  USB Mic                    │          │  ┌─────────────────────────────┐ │
│    ↓                        │          │  │ STT                         │ │
│  sounddevice capture        │  HTTP    │  │  faster-whisper (small)     │ │
│    ↓                        │  POST    │  │  OR Sarvam Cloud            │ │
│  Voice Activity Detection   │ ──────►  │  │  OR Deepgram Nova           │ │
│    ↓                        │  (WAV)   │  ├─────────────────────────────┤ │
│  Build WAV                  │          │  │ Translation                  │ │
│                             │  JSON +  │  │  NLLB-200 (local, 200 langs)│ │
│  Play translated audio ◄── │ ◄─────── │  │  OR Sarvam Mayura (cloud)   │ │
│    ↓                        │ base64   │  ├─────────────────────────────┤ │
│  Bluetooth Speaker          │  WAV     │  │ TTS                         │ │
│                             │          │  │  IndicF5 / Piper / MMS      │ │
│                             │          │  │  / espeak-ng / edge-tts     │ │
└─────────────────────────────┘          └──┴─────────────────────────────┘ │
                                          └──────────────────────────────────┘
```

### WebSocket Mode (Streaming)

```
┌─────────────────────────────┐          ┌──────────────────────────────────┐
│    Raspberry Pi Zero 2W     │    WS    │   Laptop (Streaming Edge)        │
│                             │ ──────►  │                                  │
│  Live PCM audio chunks      │  (PCM)   │  Sarvam Streaming STT            │
│                             │          │    ↓                              │
│  Play TTS audio ◄───────── │ ◄─────── │  NLLB-200 local translation      │
│                             │  (WAV)   │    ↓                              │
│                             │          │  Local TTS synthesis              │
└─────────────────────────────┘          └──────────────────────────────────┘
```

**Expected end-to-end latency:** ~1.0–2.0s over WiFi (HTTP), lower with streaming.

---

## Model Stack

| Stage       | Model                                | Type   | Languages           | Notes                         |
|-------------|--------------------------------------|--------|---------------------|-------------------------------|
| **STT**     | faster-whisper `small`               | Local  | 99 languages        | CTranslate2, int8 quantized   |
| **STT**     | Deepgram Nova-2                      | Cloud  | Multi-language       | Low-latency cloud alternative |
| **STT**     | Sarvam Saaras v3                     | Cloud  | Indian languages     | Optimized for Indian accents  |
| **Translate** | NLLB-200-distilled-600M           | Local  | 200 languages        | CTranslate2, int8             |
| **Translate** | Sarvam Mayura v1                  | Cloud  | Indian languages     | Formal/informal modes         |
| **TTS**     | IndicF5 (ai4bharat)                  | Local  | 11 Indian languages  | Near-human quality, ~1-2s     |
| **TTS**     | Piper ONNX                           | Local  | English, Hindi, Telugu | Fast (~200-400ms), natural  |
| **TTS**     | Meta MMS-TTS                         | Local  | Tamil (expandable)   | Offline, ~350ms               |
| **TTS**     | espeak-ng                            | Local  | All languages        | Ultra-fast (~50ms), robotic   |
| **TTS**     | Microsoft edge-tts                   | Cloud  | All languages        | Natural, requires internet    |

---

## Supported Languages

| Language   | Code    | STT | Translation | TTS (Primary)  |
|------------|---------|-----|-------------|----------------|
| English    | `en-IN` | ✅  | ✅          | Piper          |
| Hindi      | `hi-IN` | ✅  | ✅          | IndicF5 / Piper|
| Tamil      | `ta-IN` | ✅  | ✅          | IndicF5 / MMS  |
| Telugu     | `te-IN` | ✅  | ✅          | IndicF5 / Piper|
| Kannada    | `kn-IN` | ✅  | ✅          | IndicF5        |
| Malayalam  | `ml-IN` | ✅  | ✅          | IndicF5        |
| Bengali    | `bn-IN` | ✅  | ✅          | IndicF5        |
| Marathi    | `mr-IN` | ✅  | ✅          | IndicF5        |
| Gujarati   | `gu-IN` | ✅  | ✅          | IndicF5        |
| Punjabi    | `pa-IN` | ✅  | ✅          | IndicF5        |

---

## Prerequisites

### Laptop (Edge Server)

- Python 3.10+
- ~2 GB disk space for models (Whisper small + NLLB-200)
- Recommended: NVIDIA GPU for faster inference (CPU works, just slower)

### Raspberry Pi

- Raspberry Pi Zero 2W (or any Pi with WiFi)
- USB microphone
- Bluetooth speaker or 3.5mm audio output
- `portaudio19-dev` system package

---

## Installation

### Laptop Setup

```bash
# Clone / navigate to the project
cd DESIGN_PROJECT

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

# Install Python dependencies
pip install -r requirements.txt

# (Optional) Install IndicF5 TTS for high-quality Indian language speech
pip install git+https://github.com/ai4bharat/IndicF5.git

# Download STT and translation models (one-time, ~1.5 GB)
python download_local_models.py

# Download Piper TTS voice models (one-time, ~100 MB)
python download_models.py
```

### Raspberry Pi Setup

```bash
# System dependencies
sudo apt update
sudo apt install -y portaudio19-dev espeak-ng

# Python dependencies (lightweight)
pip install requests sounddevice numpy
```

### Environment Variables

Create a `.env` file in the project root (or parent directory):

```env
# Required for Sarvam cloud backend
SARVAM_API_KEY=your_sarvam_api_key

# Required for Deepgram cloud STT
DEEPGRAM_API_KEY=your_deepgram_api_key

# Optional overrides
INFERENCE_BACKEND=local          # sarvam | local | deepgram
LOCAL_STT_MODEL=small            # tiny | base | small
LOCAL_STT_DEVICE=cpu             # cpu | cuda
USE_INDICF5=true                 # Enable IndicF5 TTS
EDGE_SERVER_PORT=5555
```

---

## Quick Start

### Mode 1: Edge Server + Pi Client (Recommended)

This is the primary deployment mode for the wearable device.

**On the laptop:**

```bash
# Start the edge server with local inference (no API keys needed)
python edge_server.py --lang hindi

# The server prints its LAN IP, e.g.:
#   Edge server running on http://0.0.0.0:5555
#   LAN URL: http://192.168.1.42:5555
```

**On the Raspberry Pi:**

```bash
# Connect to the edge server
python pi_client.py --server http://192.168.1.42:5555 --lang hindi

# With specific audio devices
python pi_client.py --server http://192.168.1.42:5555 --lang tamil \
    --input-device 1 --output-device 3 --gain 2.0
```

### Mode 2: Streaming (WebSocket)

Lower-latency mode using WebSockets for continuous audio streaming. Requires Sarvam API for streaming STT.

**On the laptop:**

```bash
export SARVAM_API_KEY="your_key"
python edge_stream_server.py
# Starts on ws://0.0.0.0:5556
```

**On the Raspberry Pi:**

```bash
python pi_stream_client.py --server ws://192.168.1.42:5556 --lang hindi
```

### Mode 3: Standalone (No Pi)

Run the entire pipeline on a single machine using its microphone:

```bash
# Using local models (offline)
python main.py --lang hindi --inference-backend local

# Using Sarvam cloud
python main.py --lang telugu --inference-backend sarvam

# Using Deepgram STT + local translation
python main.py --lang tamil --inference-backend deepgram

# Tune VAD sensitivity
python main.py --lang hindi --threshold 200 --silence-timeout 0.3
```

### Mode 4: Web Prototype

A browser-based demo with text and voice input:

```bash
python serve_demo.py --port 8080 --inference-backend local
# Open http://localhost:8080 in your browser
# Or from your phone: http://<laptop-ip>:8080
```

The web UI provides:
- **Text translation** — type text and get instant translation + TTS audio
- **Voice translation** — record from browser mic, get STT + translation + audio
- **Edge mode** (`/edge`) — phone-as-mic mode, sends text to Pi for TTS playback
- **Latency metrics** — real-time timing breakdown for each pipeline stage

---

## Configuration

All settings are centralized in `config.py` and can be overridden via environment variables or `.env`.

### Inference Backends

| Backend    | STT Source         | Translation Source   | API Key Required | Offline |
|------------|--------------------|----------------------|------------------|---------|
| `local`    | faster-whisper     | NLLB-200 (CT2)      | No               | Yes     |
| `sarvam`   | Sarvam Saaras v3   | NLLB-200 (local)    | Yes              | No      |
| `deepgram` | Deepgram Nova-2    | NLLB-200 (local)    | Yes              | No      |

The `sarvam` backend uses cloud STT but routes translation through the local NLLB model for speed. Set `DIRECT_TRANSLATE=true` in config to use Sarvam's single-call STT+translate endpoint instead.

### TTS Priority Chain

When `backend="auto"` (default), the TTS engine tries each backend in order and uses the first one that succeeds:

| Priority | Engine     | Speed       | Quality  | Offline | Languages                   |
|----------|-----------|-------------|----------|---------|------------------------------|
| 1        | IndicF5   | ~1–2s       | Natural  | ✅      | Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi |
| 2        | Piper TTS | ~200–400ms  | Natural  | ✅      | English, Hindi, Telugu        |
| 3        | MMS-TTS   | ~350ms      | Good     | ✅      | Tamil (expandable)            |
| 4        | espeak-ng | ~50ms       | Robotic  | ✅      | All languages                 |
| 5        | edge-tts  | ~300–500ms  | Natural  | ❌      | All languages                 |

You can force a specific TTS backend via the `--tts` flag or API parameter.

### Voice Activity Detection (VAD)

The pipeline uses energy-based VAD to detect speech boundaries:

| Parameter          | Default | Description                                     |
|--------------------|---------|-------------------------------------------------|
| `RMS_THRESHOLD`    | 300     | Minimum RMS energy to consider as speech         |
| `SILENCE_TIMEOUT`  | 0.4s    | Silence duration before triggering processing    |
| `MAX_RECORD_SECS`  | 5s      | Maximum recording length before forced processing|
| `MIN_SPEECH_RMS`   | 200     | Overall RMS must exceed this to avoid noise       |
| `MAX_WAV_KB`       | 200     | Maximum WAV size sent to API (skip if larger)    |

---

## Command-Line Reference

### `edge_server.py` — Laptop HTTP Edge Server

```
python edge_server.py [options]

Options:
  --host HOST               Bind address (default: 0.0.0.0)
  --port PORT               Port number (default: 5555)
  --lang LANGUAGE            Default target language (default: hindi)
  --inference-backend NAME   STT/translate backend: sarvam, local, deepgram (default: local)
```

### `edge_stream_server.py` — Laptop WebSocket Streaming Server

```
python edge_stream_server.py

Environment:
  SARVAM_API_KEY             Required for Sarvam streaming STT
  SARVAM_STREAM_MODEL        Streaming model (default: saarika:v2.5)
```

Listens on `ws://0.0.0.0:5556`.

### `pi_client.py` — Raspberry Pi HTTP Audio Client

```
python pi_client.py [options]

Options:
  --server URL               Edge server URL (required, e.g. http://192.168.1.100:5555)
  --lang LANGUAGE             Target language (default: hindi)
  --input-device N           Microphone device index
  --output-device N          Speaker device index
  --gain FLOAT               Playback volume multiplier (default: 1.0)
  --wait-timeout SECS        Seconds to wait for server readiness (default: 60, 0=forever)
  --list-devices             List available audio devices and exit
```

### `pi_stream_client.py` — Raspberry Pi WebSocket Streaming Client

```
python pi_stream_client.py [options]

Options:
  --server URL               WebSocket server URL (required, e.g. ws://192.168.1.100:5556)
  --lang LANGUAGE             Target language (default: hindi)
  --input-lang LANGUAGE       Source language hint (default: english)
  --input-device N           Microphone device index
  --output-device N          Speaker device index
  --gain FLOAT               Playback volume (default: 1.5)
  --list-devices             List available audio devices and exit
```

### `main.py` — Standalone Pipeline (No Pi)

```
python main.py [options]

Options:
  --lang LANGUAGE             Output language (default: hindi)
  --inference-backend NAME   Backend: sarvam, local, deepgram
  --list-devices             Show available audio devices
  --input-device N           Microphone device index
  --output-device N          Speaker device index
  --threshold N              RMS speech detection threshold (default: 300)
  --gain FLOAT               Playback volume multiplier (default: 1.0)
  --tts BACKEND              Force TTS engine: auto, indicf5, piper, espeak, edge, mms
```

### `serve_demo.py` — Web Prototype Server

```
python serve_demo.py [options]

Options:
  --port PORT               HTTP port (default: 8080)
  --host HOST               Bind address (default: 0.0.0.0)
  --inference-backend NAME  Backend: sarvam, local, deepgram
  --lang LANGUAGE            Default target language (default: hindi)
```

### Utility Scripts

```bash
# Download STT (Whisper) + Translation (NLLB-200) models
python download_local_models.py

# Download Piper TTS voice models
python download_models.py

# List audio devices (quick check)
python investigate_device.py
```

---

## API Reference

### Edge Server HTTP Endpoints (`edge_server.py` / `serve_demo.py`)

#### `GET /api/health`

Health check and server status.

**Response:**

```json
{
  "status": "ok",
  "inference_backend": "local",
  "message": "Local backend ready (Whisper small + NLLB-200)",
  "default_target_lang": "hi-IN",
  "supported_languages": ["english", "hindi", "tamil", "telugu", ...]
}
```

#### `POST /api/process-audio`

Full pipeline: audio → STT → translation → TTS.

**Request:** `multipart/form-data`

| Field         | Type   | Required | Description                          |
|---------------|--------|----------|--------------------------------------|
| `audio`       | File   | Yes      | WAV audio file (16kHz, mono, 16-bit) |
| `target_lang` | String | No       | BCP-47 code (e.g. `hi-IN`), defaults to server setting |

**Response:**

```json
{
  "english_text": "hello how are you",
  "translated_text": "नमस्ते आप कैसे हैं",
  "tts_audio": "<base64-encoded WAV>",
  "tts_backend": "indicf5",
  "timings": {
    "stt_ms": 450,
    "translate_ms": 120,
    "tts_ms": 800,
    "total_ms": 1370
  }
}
```

#### `POST /api/translate-text`

Text-only translation (no audio input/output).

**Request:** `application/json`

```json
{
  "text": "Where is the train station?",
  "source_lang": "en-IN",
  "target_lang": "hi-IN"
}
```

**Response:**

```json
{
  "translated_text": "ट्रेन स्टेशन कहाँ है?",
  "translate_ms": 95
}
```

#### `POST /api/text-translate` (Web Prototype)

Text translation with optional TTS audio. Available in `serve_demo.py`.

**Request:** `application/json`

```json
{
  "text": "good morning",
  "sourceLang": "english",
  "targetLang": "hindi",
  "ttsBackend": "auto",
  "includeAudio": true
}
```

#### `POST /api/voice-translate` (Web Prototype)

Browser-recorded voice translation. Available in `serve_demo.py`.

**Request:** `multipart/form-data` with `audio` file and optional `targetLang`, `ttsBackend` fields.

---

## Project Structure

```
DESIGN_PROJECT/
│
├── config.py                    # Central configuration (languages, models, thresholds, API keys)
│
├── ── Core Pipeline ──────────────────────────────────────────────────
├── inference_client.py          # Backend router — dispatches to sarvam/local/deepgram
├── sarvam_client.py             # Sarvam cloud API client (STT + translation)
├── local_inference_client.py    # Local STT (faster-whisper) + translation (NLLB-200)
├── deepgram_client.py           # Deepgram Nova STT + local NLLB translation
├── tts_engine.py                # TTS engine with priority chain (IndicF5→Piper→MMS→espeak→edge-tts)
│
├── ── Deployment Modes ───────────────────────────────────────────────
├── edge_server.py               # HTTP edge server (laptop) — Pi sends WAV, gets back translated audio
├── edge_stream_server.py        # WebSocket streaming server — real-time audio forwarding via Sarvam
├── pi_client.py                 # HTTP audio client (Raspberry Pi) — mic capture → server → speaker
├── pi_stream_client.py          # WebSocket streaming client (Pi) — continuous audio stream
├── main.py                      # Standalone headless pipeline — runs everything on one machine
├── serve_demo.py                # Web prototype server — browser-based demo UI
│
├── ── Setup & Utilities ──────────────────────────────────────────────
├── download_local_models.py     # Downloads faster-whisper + NLLB-200 models from HuggingFace
├── download_models.py           # Downloads Piper TTS ONNX voice models
├── generate_speech.py           # Standalone TTS testing script
├── investigate_device.py        # Utility to list available audio devices
├── patch.py                     # One-time code fix utility
│
├── ── Web Frontend ───────────────────────────────────────────────────
├── local_demo/
│   ├── index.html               # Main web prototype UI (text + voice translation)
│   ├── edge.html                # Phone-as-mic edge mode UI
│   ├── app.js                   # Frontend logic for index.html
│   ├── edge.js                  # Frontend logic for edge.html
│   └── styles.css               # Shared styles
│
├── ── TTS Models ─────────────────────────────────────────────────────
├── piper_models/
│   ├── en_US-lessac-medium.onnx       # English Piper voice
│   ├── hi_IN-pratham-medium.onnx      # Hindi Piper voice
│   └── te_IN-maya-medium.onnx         # Telugu Piper voice
│
├── ── Configuration & Docs ───────────────────────────────────────────
├── requirements.txt             # Python dependencies
├── .env                         # API keys and environment overrides (not committed)
│
├── ── Test Scripts ───────────────────────────────────────────────────
├── test_sarvam.py               # Sarvam API integration tests
├── test_sarvam_wav.py           # Sarvam WAV format tests
├── test_sarvam_wav_proper.py    # Sarvam WAV proper format tests
├── test_speech.py               # Speech synthesis tests
├── test_speech_wav.py           # Speech WAV output tests
├── test_speech_flush.py         # TTS flush/streaming tests
└── test_speech_silence.py       # Silence detection tests
```

---

## Troubleshooting

### Audio Issues

**No audio input detected:**

```bash
# List available devices to find the correct index
python investigate_device.py
# or
python pi_client.py --list-devices --server http://localhost:5555

# Specify the correct device
python pi_client.py --server http://... --input-device 2
```

**Audio playback too quiet:**

```bash
# Increase playback gain (default: 1.0)
python pi_client.py --server http://... --gain 3.0
```

**Bluetooth audio not working on Pi:**

- Ensure the speaker is paired and connected via `bluetoothctl`
- The streaming client auto-selects `pulse` as the output device
- If `sounddevice` fails, `pi_client.py` automatically falls back to `aplay`

### Server Connectivity

**Pi cannot reach the edge server:**

- Verify both devices are on the same WiFi network
- Use the exact LAN IP printed by `edge_server.py` at startup
- Check firewall rules: the port (default 5555) must be open
- The Pi client retries automatically for up to 60 seconds (configurable via `--wait-timeout`)

### Model Issues

**"Local backend import failed":**

```bash
# Install all dependencies
pip install -r requirements.txt

# Re-download models
python download_local_models.py
```

**IndicF5 not loading:**

```bash
# IndicF5 requires a separate install
pip install git+https://github.com/ai4bharat/IndicF5.git

# If IndicF5 fails, the system falls back to Piper → MMS → espeak → edge-tts
# Set USE_INDICF5=false to disable and use fallbacks directly
```

**Sarvam streaming model error:**

The streaming server requires model name `saarika:v2.5` (not `saaras:v2.5`). Set the correct model:

```bash
export SARVAM_STREAM_MODEL=saarika:v2.5
```

### Performance Tuning

- **Reduce latency:** Lower `SILENCE_TIMEOUT` (e.g., 0.25s) for faster speech-end detection
- **Avoid false triggers:** Increase `RMS_THRESHOLD` in noisy environments
- **Faster STT:** Use `LOCAL_STT_MODEL=tiny` for speed over accuracy
- **GPU acceleration:** Set `LOCAL_STT_DEVICE=cuda` if you have an NVIDIA GPU

## Notes

- The Pi only needs `requests`, `sounddevice`, and `numpy` — no ML libraries.
- Internet is required for cloud backends (Sarvam, Deepgram) and one-time model downloads.
- On Pi, connect Bluetooth earbuds via `bluetoothctl` — they appear as a regular audio device.
- NLLB-200 covers **all** Indian languages including Tamil and Telugu (which Argos lacked).
