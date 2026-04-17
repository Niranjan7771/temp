# Wearable Voice Translator v2

## Technical Design Document

---

### Document Information

| Field              | Detail                                              |
|--------------------|------------------------------------------------------|
| **Project Title**  | Wearable Voice Translator v2                         |
| **Version**        | 2.0                                                  |
| **Date**           | April 2026                                           |
| **Platform**       | Raspberry Pi Zero 2W + Laptop (Edge Computing)       |
| **Domain**         | Embedded Systems, Natural Language Processing, Speech |

---

## 1. Abstract

This document presents the design, architecture, and implementation of a real-time wearable voice translation system. The system captures spoken audio through a Raspberry Pi-based wearable device, processes it through a multi-stage NLP pipeline consisting of Speech-to-Text (STT), Machine Translation, and Text-to-Speech (TTS), and plays back the translated audio — all within an end-to-end latency of 1–2 seconds.

The design addresses key challenges in wearable computing: limited on-device processing power, real-time latency requirements, multilingual support for 10 Indian languages, and reliable operation over wireless networks. A split-architecture approach offloads compute-intensive inference to a laptop acting as a local edge server, while the Raspberry Pi handles only audio capture and playback.

---

## 2. Introduction

### 2.1 Problem Statement

India's linguistic diversity — with 22 official languages and hundreds of dialects — creates significant communication barriers in daily life, tourism, healthcare, and commerce. Existing translation solutions either require continuous internet connectivity, introduce unacceptable latency, or lack support for Indian languages. A wearable, low-latency translation device that works over a local network and supports major Indian languages addresses a tangible need.

### 2.2 Objectives

1. **Real-time translation** with end-to-end latency under 2 seconds.
2. **Support for 10 Indian languages**: English, Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati, and Punjabi.
3. **Wearable form factor** using Raspberry Pi Zero 2W with USB microphone and Bluetooth speaker.
4. **Offline capability** for STT, translation, and TTS — no cloud dependency required.
5. **Modular architecture** allowing easy swapping of STT, translation, and TTS backends.

### 2.3 Scope

The system covers the complete speech translation pipeline: voice activity detection, speech-to-text transcription, language translation, and text-to-speech synthesis. It supports four deployment modes: edge server with Pi client (HTTP), streaming via WebSocket, standalone single-machine operation, and a web browser prototype.

---

## 3. System Architecture

### 3.1 Architecture Overview

The system follows a **split-architecture** pattern where computation is distributed between two devices:

- **Raspberry Pi Zero 2W (Client):** Handles audio I/O — microphone capture, voice activity detection, WAV encoding, and speaker playback.
- **Laptop (Edge Server):** Performs all compute-intensive tasks — speech-to-text, translation, and text-to-speech synthesis.

Communication between the devices occurs over the local WiFi network using either HTTP REST or WebSocket protocols.

### 3.2 High-Level Data Flow

```
┌──────────────────────────┐                    ┌──────────────────────────────┐
│     Raspberry Pi         │                    │     Laptop (Edge Server)     │
│                          │                    │                              │
│  ┌────────────────────┐  │    HTTP POST       │  ┌────────────────────────┐  │
│  │ USB Microphone     │  │    (WAV audio)     │  │ Speech-to-Text (STT)   │  │
│  │       ↓            │  │ ───────────────►   │  │  faster-whisper small  │  │
│  │ Audio Capture      │  │                    │  │  OR Deepgram Nova-2    │  │
│  │  (sounddevice)     │  │                    │  │  OR Sarvam Saaras v3   │  │
│  │       ↓            │  │                    │  └──────────┬─────────────┘  │
│  │ Voice Activity     │  │                    │             ↓                │
│  │  Detection (VAD)   │  │                    │  ┌────────────────────────┐  │
│  │       ↓            │  │                    │  │ Translation            │  │
│  │ WAV Encoding       │  │                    │  │  NLLB-200 (600M, int8) │  │
│  └────────────────────┘  │                    │  │  OR Sarvam Mayura v1   │  │
│                          │                    │  └──────────┬─────────────┘  │
│  ┌────────────────────┐  │    JSON Response   │             ↓                │
│  │ Audio Playback     │  │    (base64 WAV     │  ┌────────────────────────┐  │
│  │  (sounddevice)     │  │    + text + timing) │  │ Text-to-Speech (TTS)   │  │
│  │       ↓            │  │ ◄───────────────── │  │  IndicF5 / Piper /     │  │
│  │ Bluetooth Speaker  │  │                    │  │  MMS / espeak / edge   │  │
│  └────────────────────┘  │                    │  └────────────────────────┘  │
└──────────────────────────┘                    └──────────────────────────────┘
```

### 3.3 Communication Protocols

The system supports two communication modes:

**HTTP Mode (Batch Processing):**
- The Pi records a complete utterance, packages it as a WAV file, and sends it via HTTP POST to the edge server.
- The edge server processes the entire pipeline and returns a JSON response containing transcribed text, translated text, base64-encoded TTS audio, and timing metrics.
- Latency: ~1.0–2.0 seconds per utterance.

**WebSocket Mode (Streaming):**
- The Pi streams raw PCM audio chunks continuously over a WebSocket connection.
- The edge server forwards audio to Sarvam's streaming STT API, performs local translation on completed utterances, synthesizes TTS audio, and streams results back.
- Latency: lower than HTTP mode due to overlapped processing.

### 3.4 Deployment Modes

| Mode          | Description                                          | Use Case                  |
|---------------|------------------------------------------------------|---------------------------|
| **Edge + Pi** | Pi captures audio → HTTP → Laptop processes → Pi plays | Primary wearable deployment |
| **Streaming** | Pi streams PCM → WebSocket → Laptop → Pi plays        | Low-latency wearable       |
| **Standalone**| Single machine: mic → pipeline → speaker               | Development and testing     |
| **Web Demo**  | Browser UI → HTTP API → server → audio in browser      | Demonstration and testing   |

---

## 4. Component Design

### 4.1 Voice Activity Detection (VAD)

The system uses an energy-based Voice Activity Detection algorithm to identify speech boundaries in the continuous audio stream. This is critical for determining when a user has started and stopped speaking.

**Algorithm:**
1. Continuously compute the Root Mean Square (RMS) energy of incoming audio blocks (30ms frames at 16 kHz).
2. If RMS exceeds `RMS_THRESHOLD` (default: 300), mark the frame as speech.
3. Once speech is detected, continue recording until silence persists for `SILENCE_TIMEOUT` (default: 0.4 seconds).
4. Enforce a `MAX_RECORD_SECS` (default: 5 seconds) cap to prevent excessively long recordings.
5. Apply a minimum speech RMS filter to reject noise triggers.
6. Trim leading and trailing silence from the recorded audio to reduce WAV file size.

**VAD Parameters:**

| Parameter          | Default | Description                                        |
|--------------------|---------|----------------------------------------------------|
| `RMS_THRESHOLD`    | 300     | Minimum RMS energy to classify a frame as speech    |
| `SILENCE_TIMEOUT`  | 0.4 s   | Duration of silence that ends an utterance          |
| `MAX_RECORD_SECS`  | 5 s     | Maximum utterance length before forced processing   |
| `MIN_SPEECH_RMS`   | 200     | Overall RMS must exceed this to avoid noise triggers |
| `MAX_WAV_KB`       | 200     | Maximum WAV file size sent to the server            |

A noise calibration utility is included that records 3 seconds of ambient audio and recommends an appropriate `RMS_THRESHOLD` value for the environment.

### 4.2 Speech-to-Text (STT)

The system supports three STT backends, selectable at runtime:

#### 4.2.1 faster-whisper (Local)

- **Model:** OpenAI Whisper `small` (244M parameters), converted to CTranslate2 format with int8 quantization.
- **Operation:** The audio is decoded from WAV to float32 samples at 16 kHz. Whisper runs in `translate` mode, which simultaneously detects the source language and translates to English.
- **Optimizations:**
  - VAD pre-filtering to skip silence regions.
  - Beam search with beam size 5 and temperature 0.0 for deterministic output.
  - Hallucination detection filters (repetition, known noise phrases, prompt leak detection).
  - CPU thread count set to 8 for optimized throughput on multi-core processors.
- **Latency:** ~0.5–1.5 seconds depending on utterance length and hardware.

#### 4.2.2 Deepgram Nova-2 (Cloud)

- **Model:** Deepgram Nova-2 with smart formatting and punctuation.
- **Operation:** Audio bytes are sent directly to Deepgram's REST API. Supports automatic language detection or fixed source language.
- **Latency:** ~0.3–0.8 seconds (network-dependent).

#### 4.2.3 Sarvam Saaras v3 (Cloud)

- **Model:** Sarvam AI's Saaras v3, optimized for Indian accents and languages.
- **Operation:** Supports both batch (REST) and streaming (WebSocket) modes. Batch mode sends WAV audio and receives English transcription. Streaming mode (via `saarika:v2.5`) accepts continuous PCM chunks.
- **Latency:** ~0.4–1.0 seconds (batch), lower for streaming.

#### 4.2.4 Transcript Quality Assurance

Post-STT processing includes several quality checks:

- **Filler phrase filtering:** Common noise transcriptions ("thank you", "bye", single characters) are discarded.
- **Suspicious short transcript detection:** Heuristic to identify when a long audio clip produces an implausibly short transcript, triggering a retry with an alternate STT strategy.
- **Cross-strategy retry:** If the initial transcription is low-confidence, the system retries using the alternate mode (direct vs. two-step) and selects the higher-quality result.

### 4.3 Machine Translation

#### 4.3.1 NLLB-200 (Local)

- **Model:** Meta's NLLB-200-distilled-600M, converted to CTranslate2 format with int8 quantization.
- **Tokenization:** SentencePiece BPE tokenizer with FLORES-200 language codes.
- **Operation:** Source text is tokenized, prefixed with the source language code, and translated with the target language code as a decoding prefix.
- **Optimizations:**
  - Beam size 4 with repetition penalty (1.2) and no-repeat n-gram blocking (size 3).
  - Output length capped at 3× input length (max 200 tokens) to prevent degeneration.
  - Thread count set to 8 for CPU parallelism.
  - Common English→Hindi shortcuts for frequent phrases bypass the model entirely.
- **Supported Language Pairs:** All combinations among the 10 supported languages, routed through FLORES-200 codes.

**Language Code Mapping (BCP-47 → FLORES-200):**

| Language   | BCP-47  | FLORES-200  |
|------------|---------|-------------|
| English    | en-IN   | eng_Latn    |
| Hindi      | hi-IN   | hin_Deva    |
| Tamil      | ta-IN   | tam_Taml    |
| Telugu     | te-IN   | tel_Telu    |
| Kannada    | kn-IN   | kan_Knda    |
| Malayalam  | ml-IN   | mal_Mlym    |
| Bengali    | bn-IN   | ben_Beng    |
| Marathi    | mr-IN   | mar_Deva    |
| Gujarati   | gu-IN   | guj_Gujr    |
| Punjabi    | pa-IN   | pan_Guru    |

#### 4.3.2 Sarvam Mayura v1 (Cloud)

- Used as an optional cloud translation backend via the Sarvam API.
- Supports formal and informal translation modes.

#### 4.3.3 Translation Retry Logic

If the output text does not contain characters from the expected target script (e.g., Devanagari for Hindi, Tamil script for Tamil), the system retries translation explicitly to ensure correct-language output.

### 4.4 Text-to-Speech (TTS)

The TTS engine implements a **priority chain** — it tries each backend in order and uses the first one that produces valid audio output.

#### 4.4.1 IndicF5 (Priority 1 — Indian Languages)

- **Model:** ai4bharat IndicF5 (400M parameters), supporting 11 Indian languages.
- **Quality:** Near-human naturalness with reference-audio-based voice cloning.
- **Operation:** Uses bundled reference prompt audio files per language for voice consistency. Falls back gracefully if no reference is available.
- **Latency:** ~1–2 seconds.
- **Languages:** Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi.

#### 4.4.2 Piper TTS (Priority 2 — English and Select Languages)

- **Model:** ONNX-based neural TTS models from the Piper project.
- **Quality:** Natural-sounding with low resource requirements.
- **Operation:** Voice models are loaded once and cached. Supports both full synthesis and streaming output.
- **Latency:** ~200–400ms.
- **Languages:** English (`en_US-lessac-medium`), Hindi (`hi_IN-pratham-medium`), Telugu (`te_IN-maya-medium`).

#### 4.4.3 Meta MMS-TTS (Priority 3 — Tamil)

- **Model:** Meta's Massively Multilingual Speech TTS (`facebook/mms-tts-tam`), ~145MB.
- **Quality:** Good quality, fully offline.
- **Operation:** Uses HuggingFace's `VitsModel` and `AutoTokenizer` for synthesis. Output is converted to 16 kHz WAV.
- **Latency:** ~350ms.
- **Languages:** Tamil (expandable to other MMS-supported languages).

#### 4.4.4 espeak-ng (Priority 4 — Universal Fallback)

- **Quality:** Robotic but functional.
- **Operation:** Invoked via subprocess. Available on all Linux systems.
- **Latency:** ~50ms.
- **Languages:** All supported languages.

#### 4.4.5 Microsoft edge-tts (Priority 5 — Cloud Fallback)

- **Quality:** Natural (neural voices).
- **Operation:** Asynchronous cloud call to Microsoft's edge-tts service. Returns MP3, decoded via `miniaudio`.
- **Latency:** ~300–500ms (network-dependent).
- **Languages:** All supported languages with dedicated Indian voice models.

**TTS Backend Summary:**

| Priority | Backend   | Latency     | Quality | Offline | Primary Languages                                            |
|----------|-----------|-------------|---------|---------|---------------------------------------------------------------|
| 1        | IndicF5   | ~1–2s       | Natural | Yes     | Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi |
| 2        | Piper     | ~200–400ms  | Natural | Yes     | English, Hindi, Telugu                                        |
| 3        | MMS-TTS   | ~350ms      | Good    | Yes     | Tamil                                                         |
| 4        | espeak-ng | ~50ms       | Robotic | Yes     | All languages                                                 |
| 5        | edge-tts  | ~300–500ms  | Natural | No      | All languages                                                 |

### 4.5 Audio Playback

- Audio is decoded from WAV format and played through `sounddevice` using the system's default or specified output device.
- Microphone capture is paused during playback to prevent echo feedback.
- A configurable post-playback deaf period (default: 0.35 seconds) suppresses residual echo from Bluetooth speaker latency.
- A playback gain multiplier is available to boost volume for quiet Bluetooth paths.
- On Raspberry Pi, an `aplay` subprocess fallback handles cases where `sounddevice` PortAudio encounters Bluetooth issues.

---

## 5. Software Design

### 5.1 Module Architecture

The codebase follows a modular design with clear separation of concerns:

```
                    ┌─────────────────┐
                    │    config.py     │  ← Central configuration
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ↓              ↓              ↓
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │inference_    │  │  tts_engine  │  │  Deployment   │
   │  client.py   │  │     .py      │  │   Modules     │
   │  (Router)    │  │  (Priority   │  │              │
   └──┬───┬───┬───┘  │   Chain)     │  │  edge_server │
      │   │   │      └──────────────┘  │  pi_client   │
      ↓   ↓   ↓                        │  main.py     │
   ┌────┐┌────┐┌──────┐                │  serve_demo  │
   │Sar-││Lo- ││Deep- │                │  stream_*    │
   │vam ││cal ││gram  │                └──────────────┘
   └────┘└────┘└──────┘
```

### 5.2 Module Descriptions

| Module                       | Purpose                                                              |
|------------------------------|----------------------------------------------------------------------|
| `config.py`                  | Central configuration: API keys, model paths, language maps, thresholds, TTS voice mappings |
| `inference_client.py`        | Backend router — dispatches STT/translation calls to the selected backend |
| `sarvam_client.py`           | Sarvam cloud API client with connection pooling, retry logic, and warmup |
| `local_inference_client.py`  | Local STT (faster-whisper) and translation (NLLB-200) with lazy model loading |
| `deepgram_client.py`         | Deepgram Nova STT with local NLLB translation fallback              |
| `tts_engine.py`              | TTS priority chain with lazy model loading and caching               |
| `edge_server.py`             | Flask HTTP server exposing `/api/process-audio`, `/api/translate-text`, `/api/health` |
| `edge_stream_server.py`      | WebSocket server for streaming audio via Sarvam's real-time STT API  |
| `pi_client.py`               | Raspberry Pi HTTP client with VAD, server health checking, and retry logic |
| `pi_stream_client.py`        | Raspberry Pi WebSocket streaming client with automatic device selection |
| `main.py`                    | Standalone headless pipeline — complete mic-to-speaker on one machine |
| `serve_demo.py`              | Flask web server serving the browser-based demo frontend              |

### 5.3 Key Design Patterns

**Lazy Loading and Caching:**
All ML models (Whisper, NLLB, IndicF5, Piper, MMS) are loaded on first use and cached in memory. This avoids long startup times and allows the server to start accepting health checks immediately while models load in the background.

**Backend Router Pattern:**
`inference_client.py` abstracts the STT/translation backend behind a uniform API. Callers invoke `transcribe_and_translate()` and `translate_text()` without knowing which backend is active. Backends are swappable at runtime via `set_backend()`.

**Priority Chain (TTS):**
The TTS engine tries backends in priority order, automatically falling through to the next option on failure. This ensures audio output is always available even if the preferred engine is unavailable.

**Thread Safety:**
Model loading uses threading locks (`_stt_lock`, `_nllb_lock`) to prevent race conditions during concurrent access. TTS and playback run on a background thread pool to allow the main VAD loop to continue listening.

**Connection Pooling:**
HTTP clients (`sarvam_client.py`, `deepgram_client.py`) use persistent `requests.Session` objects with configured connection pools and retry strategies to minimize TCP/TLS handshake overhead.

---

## 6. API Specification

### 6.1 Edge Server REST API

#### `GET /api/health`

Returns the server status, active backend, and supported languages.

**Response (200 OK):**

```json
{
  "status": "ok",
  "inference_backend": "local",
  "message": "Local backend ready (Whisper + NLLB-200)",
  "default_target_lang": "hi-IN",
  "supported_languages": [
    "english", "hindi", "tamil", "telugu", "kannada",
    "malayalam", "bengali", "marathi", "gujarati", "punjabi"
  ]
}
```

#### `POST /api/process-audio`

Full translation pipeline: receives audio, returns transcription, translation, and synthesized speech.

**Request:** `multipart/form-data`

| Field         | Type   | Required | Description                                      |
|---------------|--------|----------|--------------------------------------------------|
| `audio`       | File   | Yes      | WAV audio file (16 kHz, mono, 16-bit PCM)        |
| `target_lang` | String | No       | Target language BCP-47 code (default: server setting) |

**Response (200 OK):**

```json
{
  "english_text": "where is the train station",
  "translated_text": "ट्रेन स्टेशन कहाँ है",
  "tts_audio": "<base64-encoded WAV bytes>",
  "tts_backend": "indicf5",
  "timings": {
    "stt_ms": 520,
    "translate_ms": 85,
    "tts_ms": 1100,
    "total_ms": 1705
  }
}
```

**Error Responses:**

| Status | Condition                   |
|--------|-----------------------------|
| 400    | Missing or invalid audio     |
| 503    | Backend not ready            |
| 500    | STT/Translation/TTS failure  |

#### `POST /api/translate-text`

Text-only translation without audio processing.

**Request:** `application/json`

```json
{
  "text": "good morning",
  "source_lang": "en-IN",
  "target_lang": "hi-IN"
}
```

**Response (200 OK):**

```json
{
  "translated_text": "सुप्रभात",
  "translate_ms": 72
}
```

### 6.2 WebSocket Streaming Protocol

**Connection:** `ws://<server-ip>:5556`

**Initial handshake:** Client sends a JSON configuration message:

```json
{
  "target_lang": "hi-IN",
  "input_lang": "en-IN"
}
```

**Audio streaming:** Client sends binary frames containing raw PCM audio (16-bit, 16 kHz, mono, 200ms chunks).

**Server responses:** JSON messages with transcription/translation results, followed by binary WAV audio frames:

```json
{
  "transcript": "hello how are you",
  "translation": "नमस्ते आप कैसे हैं",
  "is_final": true,
  "timings": {
    "stt_ms": 380,
    "trans_ms": 65,
    "tts_ms": 950,
    "total_ms": 1395
  }
}
```

---

## 7. Hardware Specification

### 7.1 Target Hardware

| Component            | Specification                              |
|----------------------|--------------------------------------------|
| **Compute (Client)** | Raspberry Pi Zero 2W (1 GHz quad-core ARM, 512 MB RAM) |
| **Microphone**       | USB microphone (16 kHz capable)            |
| **Speaker**          | Bluetooth A2DP speaker or 3.5mm audio out  |
| **Network**          | WiFi (802.11n) on shared LAN with server   |
| **Compute (Server)** | Laptop with 8+ GB RAM (GPU optional)       |

### 7.2 Audio Specifications

| Parameter       | Value                    |
|-----------------|--------------------------|
| Sample Rate     | 16,000 Hz                |
| Channels        | 1 (Mono)                 |
| Sample Width    | 16-bit (signed integer)  |
| Block Duration  | 30 ms per capture block  |
| Block Size      | 480 samples              |
| Audio Format    | WAV (PCM) for transport  |

---

## 8. Performance Analysis

### 8.1 Latency Breakdown (HTTP Mode, Local Backend)

| Pipeline Stage        | Typical Latency | Notes                                |
|-----------------------|-----------------|--------------------------------------|
| VAD end-of-speech     | 400 ms          | Configurable via `SILENCE_TIMEOUT`   |
| WAV encoding + send   | 20–50 ms        | Depends on audio length and WiFi     |
| STT (faster-whisper)  | 500–1500 ms     | Depends on utterance length, model   |
| Translation (NLLB)    | 50–150 ms       | Fast on CPU with int8 quantization   |
| TTS (varies)          | 50–2000 ms      | espeak: ~50ms, Piper: ~300ms, IndicF5: ~1.5s |
| Network return        | 10–30 ms        | LAN WiFi                             |
| Audio playback start  | 20–50 ms        | Bluetooth adds ~100ms buffer         |
| **Total**             | **~1.0–2.5 s**  | Typical for short sentences          |

### 8.2 Optimization Strategies Employed

1. **Reduced VAD silence timeout** (0.4s → configurable to 0.25s) for faster speech-end detection.
2. **Connection pre-warming** for cloud APIs — TCP+TLS handshakes are completed at startup.
3. **Persistent HTTP sessions** with connection pooling to reuse established connections.
4. **Int8 quantization** for both Whisper and NLLB models, reducing memory and compute requirements.
5. **Silence trimming** on audio edges to reduce WAV file size and API processing time.
6. **Background TTS+playback** via thread pool so the VAD loop resumes listening immediately.
7. **Model caching** — all models loaded once and kept in memory for subsequent requests.
8. **Piper streaming** — audio is synthesized and played in chunks rather than waiting for complete generation.

### 8.3 Resource Requirements

| Resource                | Edge Server (Laptop)          | Pi Client               |
|-------------------------|-------------------------------|--------------------------|
| **RAM (idle)**          | ~200 MB                       | ~30 MB                   |
| **RAM (models loaded)** | ~2.5 GB                       | ~30 MB                   |
| **Disk (models)**       | ~1.5 GB (Whisper + NLLB)     | ~0 (no models on Pi)     |
| **Disk (TTS models)**   | ~100 MB (Piper) + optional   | ~0                       |
| **CPU (inference)**     | Multi-core recommended        | Minimal (audio I/O only) |
| **Network**             | WiFi LAN                      | WiFi LAN                 |

---

## 9. Supported Languages

### 9.1 Language Coverage Matrix

| Language   | BCP-47 Code | STT | Translation | TTS Primary    | TTS Fallback        |
|------------|-------------|-----|-------------|----------------|----------------------|
| English    | `en-IN`     | ✅  | ✅          | Piper          | espeak-ng, edge-tts  |
| Hindi      | `hi-IN`     | ✅  | ✅          | IndicF5        | Piper, espeak, edge  |
| Tamil      | `ta-IN`     | ✅  | ✅          | IndicF5        | MMS, edge-tts        |
| Telugu     | `te-IN`     | ✅  | ✅          | IndicF5        | Piper, edge-tts      |
| Kannada    | `kn-IN`     | ✅  | ✅          | IndicF5        | espeak-ng, edge-tts  |
| Malayalam  | `ml-IN`     | ✅  | ✅          | IndicF5        | espeak-ng, edge-tts  |
| Bengali    | `bn-IN`     | ✅  | ✅          | IndicF5        | espeak-ng, edge-tts  |
| Marathi    | `mr-IN`     | ✅  | ✅          | IndicF5        | espeak-ng, edge-tts  |
| Gujarati   | `gu-IN`     | ✅  | ✅          | IndicF5        | espeak-ng, edge-tts  |
| Punjabi    | `pa-IN`     | ✅  | ✅          | IndicF5        | espeak-ng, edge-tts  |

### 9.2 Translation Direction

All translation flows through English as a pivot language:
- **Input:** Any supported language → Whisper transcribes/translates to English.
- **Translation:** English → Target language via NLLB-200.
- **Exception:** When source and target are the same, translation is skipped.

---

## 10. Installation and Setup

### 10.1 Laptop (Edge Server) Setup

**System Requirements:**
- Python 3.10 or later
- 4+ GB RAM (8+ GB recommended)
- ~2 GB free disk space for models

**Steps:**

```bash
# 1. Navigate to the project directory
cd DESIGN_PROJECT

# 2. Create and activate a Python virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. (Optional) Install IndicF5 TTS for high-quality Indian language synthesis
pip install git+https://github.com/ai4bharat/IndicF5.git

# 5. Download STT and translation models (~1.5 GB, one-time)
python download_local_models.py

# 6. Download Piper TTS voice models (~100 MB, one-time)
python download_models.py

# 7. (Optional) Configure API keys in .env file
echo "SARVAM_API_KEY=your_key_here" > .env
echo "DEEPGRAM_API_KEY=your_key_here" >> .env
```

### 10.2 Raspberry Pi Setup

**System Requirements:**
- Raspberry Pi Zero 2W (or any Pi with WiFi)
- Raspberry Pi OS (Bookworm or later)
- USB microphone and Bluetooth speaker

**Steps:**

```bash
# 1. Install system dependencies
sudo apt update
sudo apt install -y portaudio19-dev espeak-ng python3-pip

# 2. Install Python dependencies
pip install requests sounddevice numpy

# 3. Pair Bluetooth speaker (if applicable)
bluetoothctl
# > scan on
# > pair <MAC_ADDRESS>
# > connect <MAC_ADDRESS>
# > trust <MAC_ADDRESS>
```

### 10.3 Dependencies

**Core Dependencies (requirements.txt):**

| Package          | Version  | Purpose                                |
|------------------|----------|----------------------------------------|
| requests         | ≥2.31    | HTTP client for API calls              |
| python-dotenv    | ≥1.0     | Environment variable management        |
| numpy            | ≥1.24    | Numerical computing for audio processing |
| sounddevice      | ≥0.4.6   | Audio capture and playback             |
| Flask            | ≥3.0     | HTTP server framework                  |
| piper-tts        | ≥1.2.0   | Local neural TTS                       |
| edge-tts         | ≥6.1     | Cloud TTS fallback                     |
| miniaudio        | ≥1.50    | MP3 decoding for edge-tts output       |
| faster-whisper   | ≥1.1     | Local STT (CTranslate2 Whisper)        |
| ctranslate2      | ≥4.0     | Efficient transformer inference        |
| sentencepiece    | ≥0.2     | Tokenization for NLLB translation      |

---

## 11. Usage Guide

### 11.1 Starting the Edge Server

```bash
# Default: local backend, Hindi output, port 5555
python edge_server.py --lang hindi

# With Deepgram STT
python edge_server.py --lang tamil --inference-backend deepgram

# Custom port and host
python edge_server.py --host 0.0.0.0 --port 6000 --lang telugu
```

The server prints its LAN IP address for the Pi client to connect to.

### 11.2 Starting the Pi Client

```bash
# Basic usage
python pi_client.py --server http://192.168.1.42:5555 --lang hindi

# With audio device selection and gain boost
python pi_client.py --server http://192.168.1.42:5555 \
    --lang tamil --input-device 1 --output-device 3 --gain 2.0

# List available audio devices
python pi_client.py --list-devices --server http://localhost:5555
```

### 11.3 Starting the Streaming Mode

```bash
# On the laptop
python edge_stream_server.py

# On the Raspberry Pi
python pi_stream_client.py --server ws://192.168.1.42:5556 --lang hindi
```

### 11.4 Standalone Mode

```bash
# Full pipeline on one machine
python main.py --lang hindi --inference-backend local

# With VAD tuning
python main.py --lang telugu --threshold 200 --silence-timeout 0.3

# Calibrate noise threshold for your environment
python main.py --calibrate
```

---

## 12. Web Prototype

### 12.1 Overview

A browser-based demonstration interface is provided for testing and showcasing the translation pipeline without requiring a Raspberry Pi.

**Starting the web server:**

```bash
python serve_demo.py --port 8080
# Access at http://localhost:8080
```

### 12.2 Features

**Main Page (`/`):**
- Text input with quick-phrase chips for common expressions.
- Source and target language selectors.
- Voice recording via browser microphone (WebRTC).
- Real-time latency metrics display (VAD, STT, Translation, TTS).
- Audio playback of synthesized translation.
- TTS backend selector (auto, piper, espeak, edge-tts).

**Edge Mode (`/edge`):**
- Designed for phone-as-microphone use case.
- Browser speech recognition for dictation input.
- Sends text to the server for translation and TTS.
- Audio playback through the server's connected speaker.

### 12.3 Frontend Architecture

| File          | Purpose                                              |
|---------------|------------------------------------------------------|
| `index.html`  | Main demo page layout                                |
| `edge.html`   | Phone edge mode layout                               |
| `app.js`      | Frontend logic: recording, API calls, audio playback  |
| `edge.js`     | Edge mode: speech recognition, text submission        |
| `styles.css`  | Shared responsive styles                             |

---

## 13. Project File Structure

```
DESIGN_PROJECT/
│
├── config.py                    # Central configuration
├── inference_client.py          # Backend router
├── sarvam_client.py             # Sarvam cloud client
├── local_inference_client.py    # Local STT + Translation
├── deepgram_client.py           # Deepgram STT + Local Translation
├── tts_engine.py                # TTS priority chain engine
│
├── edge_server.py               # HTTP edge server (laptop)
├── edge_stream_server.py        # WebSocket streaming server
├── pi_client.py                 # HTTP audio client (Pi)
├── pi_stream_client.py          # WebSocket streaming client (Pi)
├── main.py                      # Standalone headless pipeline
├── serve_demo.py                # Web demo server
│
├── download_local_models.py     # STT + Translation model downloader
├── download_models.py           # Piper TTS model downloader
├── generate_speech.py           # TTS testing utility
├── investigate_device.py        # Audio device listing utility
│
├── local_demo/                  # Web frontend files
│   ├── index.html
│   ├── edge.html
│   ├── app.js
│   ├── edge.js
│   └── styles.css
│
├── piper_models/                # Downloaded Piper TTS ONNX models
│   ├── en_US-lessac-medium.onnx
│   ├── hi_IN-pratham-medium.onnx
│   └── te_IN-maya-medium.onnx
│
├── requirements.txt             # Python dependencies
├── .env                         # API keys (not version controlled)
├── README.md                    # Project README
└── PROJECT_DOCUMENTATION.md     # This document
```

---

## 14. Testing

### 14.1 Test Scripts

| Script                      | Purpose                                         |
|-----------------------------|-------------------------------------------------|
| `test_sarvam.py`            | Validates Sarvam API connectivity and responses  |
| `test_sarvam_wav.py`        | Tests WAV format compatibility with Sarvam STT   |
| `test_sarvam_wav_proper.py` | Tests proper WAV encoding for Sarvam upload      |
| `test_speech.py`            | Tests TTS synthesis across backends              |
| `test_speech_wav.py`        | Validates TTS WAV output format                  |
| `test_speech_flush.py`      | Tests TTS streaming/flushing behavior            |
| `test_speech_silence.py`    | Tests silence detection and handling             |

### 14.2 Manual Testing Procedure

1. **Health Check:** Verify server responds to `GET /api/health` with status `ok`.
2. **Text Translation:** Send a known phrase via `POST /api/translate-text` and verify output.
3. **Audio Pipeline:** Record a short spoken phrase, send to `POST /api/process-audio`, and verify non-empty `english_text`, `translated_text`, and `tts_audio`.
4. **End-to-End:** Speak into the Pi microphone and verify that translated audio plays back within 2 seconds.

---

## 15. Known Limitations and Future Work

### 15.1 Current Limitations

1. **Pivot Language Constraint:** All translation passes through English. Direct language-to-language translation (e.g., Hindi → Tamil) would reduce latency and potential translation loss.
2. **Energy-Based VAD:** The RMS-based voice activity detection is less robust than neural VAD in noisy environments.
3. **Single-Speaker Assumption:** The system processes one speaker at a time and does not support overlapping speech.
4. **Network Dependency:** The edge architecture requires WiFi connectivity between the Pi and laptop. Network interruptions halt translation.
5. **IndicF5 Latency:** The highest-quality TTS backend (IndicF5) has ~1–2 second latency, which is the largest single contributor to end-to-end delay.

### 15.2 Future Enhancements

1. **On-device inference:** As Raspberry Pi hardware improves, running lighter STT and translation models directly on the Pi.
2. **Neural VAD:** Replacing energy-based VAD with Silero VAD or WebRTC VAD for more robust speech detection.
3. **Speaker diarization:** Supporting multi-speaker conversations.
4. **Direct translation models:** Bypassing the English pivot for common language pairs.
5. **Offline-first streaming:** Implementing local streaming STT to eliminate cloud dependency for the WebSocket mode.
6. **Battery optimization:** Power management for extended wearable use.

---

## 16. References

1. OpenAI Whisper — Radford, A. et al. "Robust Speech Recognition via Large-Scale Weak Supervision." 2022.
2. NLLB Team. "No Language Left Behind: Scaling Human-Centered Machine Translation." Meta AI, 2022.
3. Piper TTS — https://github.com/rhasspy/piper
4. IndicF5 — AI4Bharat, https://github.com/ai4bharat/IndicF5
5. Meta MMS — Pratap, V. et al. "Scaling Speech Technology to 1,000+ Languages." Meta AI, 2023.
6. CTranslate2 — https://github.com/OpenNMT/CTranslate2
7. faster-whisper — https://github.com/SYSTRAN/faster-whisper
8. Sarvam AI — https://www.sarvam.ai
9. Deepgram — https://deepgram.com
10. Flask — https://flask.palletsprojects.com
11. sounddevice — https://python-sounddevice.readthedocs.io

---

*End of Document*
