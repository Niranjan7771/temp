# Wearable Voice Translator v2

Low-latency voice translation pipeline designed for **Raspberry Pi earbuds**.  
Core mode is headless audio-in -> audio-out, with an optional web prototype UI.

## Architecture

```
Earbud Mic → sounddevice capture → VAD (RMS-based)
    → Sarvam STT + Translate  [cloud, ~1-1.5s]
    → Piper TTS               [LOCAL, ~0.2-0.4s]
    → sounddevice playback → Earbud Speaker
```

**Expected latency: ~1.3-2.0s** (down from 3-5s in v1)

The key improvement: TTS runs **locally** via Piper ONNX instead of a cloud API call (which was 2.4-3.2s in v1).

## TTS Priority Chain

| Priority | Engine     | Speed       | Quality  | Offline? | Languages          |
|----------|-----------|-------------|----------|----------|--------------------|
| 1        | Piper TTS | ~200-400ms  | Natural  | ✓        | English, Hindi     |
| 2        | espeak-ng | ~50ms       | Robotic  | ✓        | All 4 languages    |
| 3        | edge-tts  | ~300-500ms  | Natural  | ✗        | All 4 languages    |

Tamil and Telugu don't have Piper models yet, so they fall back to espeak-ng (offline) or edge-tts (cloud).

## Quick Start (Windows — Development)

```bash
cd wearable-v2

# Install dependencies
pip install -r requirements.txt

# Download Piper voice models (~100MB each for English + Hindi)
python download_models.py

# Run (defaults to Hindi output)
python main.py

# Or specify a language
python main.py --lang telugu
python main.py --lang tamil
python main.py --lang english
```

## Web Prototype (Browser UI)

Run a full end-to-end prototype from browser input to translated audio output:

```bash
# Requires SARVAM_API_KEY in environment or .env
pip install -r requirements.txt
python serve_demo.py --port 8080
```

Then open:

```text
http://localhost:8080
```

The web prototype supports:

- Live backend health checks
- Text translation via Sarvam Translate API
- Voice upload (browser recording) -> Sarvam STT + Translate
- TTS audio synthesis and browser playback
- Per-call latency metrics (STT / translate / TTS / total)

## Quick Start (Raspberry Pi — Production)

```bash
cd wearable-v2

# System dependencies
sudo apt update
sudo apt install -y espeak-ng portaudio19-dev

# Python dependencies
pip install -r requirements.txt

# Download Piper models
python download_models.py

# List audio devices (find your mic + speaker)
python main.py --list-devices

# Run with specific mic
python main.py --lang hindi --input-device 1

# Run on boot (add to /etc/rc.local or systemd service)
```

## Command-Line Options

```
python main.py [options]

Options:
  --lang {english,hindi,tamil,telugu}   Output language (default: hindi)
  --list-devices                        Show available audio devices
  --input-device N                      Mic device index (from --list-devices)
  --threshold N                         RMS speech detection threshold (default: 300)
```

## File Structure

```
wearable-v2/
├── main.py              # Entry point — mic capture, VAD, orchestration
├── config.py            # All settings (languages, thresholds, model paths)
├── sarvam_client.py     # Sarvam API: STT + Translation (cloud, no TTS)
├── tts_engine.py        # Local TTS: Piper → espeak-ng → edge-tts fallback
├── download_models.py   # One-time Piper model downloader
├── requirements.txt     # pip dependencies
├── piper_models/        # (created by download_models.py)
│   ├── en_US-lessac-medium.onnx
│   ├── en_US-lessac-medium.onnx.json
│   ├── hi_IN-swara-medium.onnx
│   └── hi_IN-swara-medium.onnx.json
└── README.md            # This file
```

## Latency Comparison

| Component              | v1 (Sarvam TTS) | v2 (Piper TTS) |
|------------------------|------------------|-----------------|
| STT (saaras:v2.5)     | 0.3-1.0s         | 0.3-1.0s        |
| Translate (mayura:v1)  | 0.3-0.7s         | 0.3-0.7s        |
| **TTS**                | **2.4-3.2s** ☁️  | **0.2-0.4s** 💻 |
| **End-to-end**         | **3-5s**         | **~1.3-2s**     |

## Notes

- The `.env` file with `SARVAM_API_KEY` is shared with the v1 folder (auto-detected).
- Internet is still required for STT + Translation (Sarvam cloud).
- On Pi, connect Bluetooth earbuds via `bluetoothctl` — they appear as a regular audio device.
- Piper models are ~60-100MB each. Download once, use forever.
