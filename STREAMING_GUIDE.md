# Real-Time WebSocket Streaming Pipeline

This document explains the experimental ultra-low latency streaming architecture built for the wearable voice translator.

Unlike the default HTTP chunked approach, this pipeline uses WebSockets to stream audio constantly, resulting in faster Speech-to-Text and near-instant translation feedback.

## Architecture

```text
[Raspberry Pi Zero 2W]                          [Laptop — Edge Stream Server]
USB Mic → raw PCM chunks (50ms) ──────────────►  ws://0.0.0.0:5556
                                                    │
                                                    ▼
                                                 Sarvam AI WebSocket (Cloud STT)
                                                    │
                                                    ▼ (English Text)
                                                 NLLB-200 (Local Translation)
                                                    │
                                                    ▼ (Hindi Text)
                                                 Piper/IndicF5 (Local TTS)
                                                    │
◄──────── JSON (timings + text) & Audio WAV ────────
sounddevice playback → Bluetooth Speaker
```

## Files Explained

1. **`edge_stream_server.py`**: Runs on your MacBook.
   - Listens on `ws://0.0.0.0:5556`.
   - Forwards incoming audio chunks directly to `api.sarvam.ai`.
   - When speech ends, it instantly translates the text locally and generates TTS offline.
   - Sends the exact processing times (latency) and the generated audio bytes back to the Pi.

2. **`pi_stream_client.py`**: Runs on your Raspberry Pi.
   - Connects to the MacBook via WebSockets.
   - Streams 50ms chunks of microphone audio over the local network.
   - Listens for returning text transcripts, translation results, and latency metrics.
   - Catches raw audio bytes and plays them over your attached Bluetooth speaker.

## How to Run

### 1. Start the Server (MacBook / Edge Device)
Make sure you are in your virtual environment and have the `websockets` and `sarvamai` packages installed.
```bash
# Start the WebSocket server
python edge_stream_server.py
```
*Note your MacBook's local IP address (e.g., `10.75.38.186`).*

### 2. Start the Client (Raspberry Pi)
Make sure to copy `pi_stream_client.py` to the Pi first. Use the IP address from step 1.

**Default (English to Hindi):**
```bash
python pi_stream_client.py --server ws://10.75.38.186:5556
```

**With specific languages (e.g., Tamil to Telugu):**
```bash
python pi_stream_client.py --server ws://10.75.38.62:5556 --in-lang ta-IN --out-lang te-IN
```

**With auto-detect input:**
*(Note: Auto-detect streaming might be slower on the Sarvam side)*
```bash
python pi_stream_client.py --server ws://10.75.38.62:5556 --in-lang unknown --out-lang hi-IN
```

## Latency Metrics
When using this pipeline, you will see real-time latency numbers print to your terminal:
`⏱️ [LATENCY] STT: 150ms | Trans: 80ms | TTS: 200ms | Total Stream: 430ms`
- **STT**: The time Sarvam took to transcribe the audio.
- **Trans**: The time your MacBook's CPU took to run NLLB.
- **TTS**: The time your MacBook's CPU took to synthesize the voice.
