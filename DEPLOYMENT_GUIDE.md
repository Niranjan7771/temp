# Raspberry Pi Deployment Guide — Wearable Voice Translator v2

Complete step-by-step commands to deploy the wearable voice translator on a Raspberry Pi Zero 2W.

---

## 1. Prerequisites

- Raspberry Pi Zero 2W (or any Pi with WiFi)
- MicroSD card with Raspberry Pi OS (64-bit Lite recommended)
- USB sound card or I2S DAC + speaker/earbuds for audio output
- USB microphone or I2S MEMS mic
- WiFi configured and Pi connected to your network
- Sarvam AI API key (stored in `.env` file)

---

## 2. Verify Pi is Reachable (from Windows)

```powershell
ping raspberry-pi-2w
```

Expected output:
```
Reply from 10.80.236.149: bytes=32 time=119ms TTL=64
```

---

## 3. SSH into the Pi (from Windows)

```bash
ssh raspberrypi2w@raspberry-pi-2w.local
```

Enter your Pi password when prompted.

---

## 4. Transfer Project Files (from a NEW Windows terminal, not the SSH session)

### Copy the wearable-v2 project folder:
```bash
scp -r D:/voice-translator/wearable-v2 raspberrypi2w@raspberry-pi-2w.local:~/wearable-v2
```

### Copy the .env file (contains Sarvam API key):
```bash
scp D:/voice-translator/.env raspberrypi2w@raspberry-pi-2w.local:~/wearable-v2/.env
```

---

## 5. Install System Dependencies (on Pi, via SSH)

```bash
sudo apt update && sudo apt install -y espeak-ng portaudio19-dev python3-pip
```

- `espeak-ng` — Offline TTS engine (fallback, supports all 4 languages)
- `portaudio19-dev` — Audio I/O library needed by `sounddevice`
- `python3-pip` — Python package manager

---

## 6. Install Python Dependencies (on Pi)

```bash
cd ~/wearable-v2
pip install -r requirements.txt --break-system-packages
```

The `--break-system-packages` flag is needed on newer Raspberry Pi OS (Bookworm+) which uses externally managed Python.

---

## 7. Download TTS Models (on Pi)

```bash
cd ~/wearable-v2
python3 download_models.py
```

This downloads Piper ONNX voice models for English and Hindi (~60-100MB each).

---

## 8. Check Audio Devices (on Pi)

```bash
python3 main.py --list-devices
```

Look for your USB microphone (input) and USB speaker/DAC (output). Note the device index numbers.

---

## 9. Run the Translator (on Pi)

### Default (Hindi output):
```bash
python3 main.py
```

### Specify output language:
```bash
python3 main.py --lang hindi
python3 main.py --lang telugu
python3 main.py --lang tamil
python3 main.py --lang english
```

### Specify microphone device:
```bash
python3 main.py --lang hindi --input-device 1
```

### Adjust speech detection sensitivity:
```bash
python3 main.py --lang hindi --threshold 800
```

Lower threshold = more sensitive (picks up quieter speech).  
Higher threshold = less sensitive (needs louder speech).

### Full sentence capture profile (recommended with Bluetooth buds):
```bash
python3 main.py --lang hindi --input-device 1 --output-device 1 --threshold 1000 --silence-timeout 0.35 --min-record-secs 1.2 --max-record-secs 4.0 --trim-threshold 220 --post-playback-deaf-secs 0.45 --tts piper --playback-gain 1.6
```

Use this when speech gets split word-by-word. The flags work as follows:
- `--silence-timeout`: wait longer before ending an utterance
- `--min-record-secs`: prevent very short clips from triggering too early
- `--trim-threshold`: keep softer word edges during silence trimming
- `--post-playback-deaf-secs`: ignore mic briefly after TTS playback to avoid echo capture
- suspicious one-word replies on long audio are filtered by default (use `--allow-short-replies` to disable)

Latency vs accuracy:
- Add `--direct-translate` for lowest latency.
- Keep `--direct-translate` off for best accuracy (default behavior).
- Direct mode now auto-retries with 2-step STT+Translate when long speech returns suspiciously short output.

---

## 10. Run on Boot (Optional — Systemd Service)

Create a systemd service so the translator starts automatically when the Pi boots:

```bash
sudo nano /etc/systemd/system/voice-translator.service
```

Paste this:
```ini
[Unit]
Description=Wearable Voice Translator
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=raspberrypi2w
WorkingDirectory=/home/raspberrypi2w/wearable-v2
ExecStart=/usr/bin/python3 /home/raspberrypi2w/wearable-v2/main.py --lang hindi
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable voice-translator
sudo systemctl start voice-translator
```

Check status:
```bash
sudo systemctl status voice-translator
```

View logs:
```bash
journalctl -u voice-translator -f
```

---

## 11. TTS Backend Status

| Language | Piper (local) | espeak-ng (local) | edge-tts (cloud) |
|----------|--------------|-------------------|-------------------|
| English  | Available    | Available          | Available         |
| Hindi    | Available    | Available          | Available         |
| Tamil    | Not available| Available          | Available         |
| Telugu   | Not available| Available          | Available         |

On the Pi with `espeak-ng` installed, Tamil and Telugu will use espeak-ng (offline, instant, robotic voice) instead of edge-tts (cloud, natural but adds latency).

---

## 12. Troubleshooting

### No audio output
```bash
# Check ALSA devices
aplay -l
arecord -l

# Test speaker
speaker-test -t wav -c 1

# Test recording
arecord -d 3 -f S16_LE -r 16000 test.wav
aplay test.wav
```

### Permission denied on audio
```bash
sudo usermod -a -G audio raspberrypi2w
# Then logout and login again
```

### Python module not found
```bash
pip install -r requirements.txt --break-system-packages
```

### High latency
- Check WiFi signal: `iwconfig wlan0`
- Move closer to router
- Use 5GHz WiFi if available (Pi Zero 2W only supports 2.4GHz)
- Use espeak-ng instead of edge-tts (saves cloud round-trip for TTS)

---

## 13. Latency Expectations

| Component              | Expected Time     |
|------------------------|-------------------|
| Silence detection      | 0.5s              |
| Sarvam STT (cloud)     | 0.3–1.0s          |
| Sarvam Translate       | 0.3–0.7s          |
| Piper TTS (local)      | 0.2–0.5s          |
| espeak-ng TTS (local)  | ~0.05s            |
| **Total (to audio)**   | **~1.3–2.5s**     |

WiFi latency from Pi (~73-314ms per ping) adds to the cloud API round-trip times.


Bluetooth device D6:93:57:C7:35:F9 is connected.

