# Raspberry Pi Run Commands

## 1) SSH Into Raspberry Pi

```bash
ssh raspberrypi2w@raspberry-pi-2w.local
```

## 2) Go To Project Folder

```bash
cd ~/app_design_test_20260405
```

## 3) Pull Latest Branch (low-latency work)

```bash
git fetch origin
git checkout feature/low-latency-rpi-zero2w
git pull --ff-only origin feature/low-latency-rpi-zero2w
```

## 4) One-Time System Setup

```bash
sudo apt update
sudo apt install -y espeak-ng portaudio19-dev python3-venv
```

## 5) Python Environment Setup

```bash
cd ~/app_design_test_20260405
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 6) Add API Key

```bash
nano .env
```

Put this line in `.env`:

```text
SARVAM_API_KEY=YOUR_REAL_KEY
```

## 6A) Check If API Key Is Already Loaded

Check shell environment only:

```bash
if [ -n "$SARVAM_API_KEY" ]; then
	echo "SARVAM_API_KEY is set (${#SARVAM_API_KEY} chars)"
else
	echo "SARVAM_API_KEY is NOT set in current shell"
fi
```

Check what this project sees (includes `.env` loading from `config.py`):

```bash
python3 -c "from config import SARVAM_API_KEY as k; print(f'set ({len(k)} chars)' if k else 'not set')"
```

Optional masked preview (never print full key):

```bash
python3 -c "from config import SARVAM_API_KEY as k; print((k[:4] + '...' + k[-4:]) if k else 'not set')"
```

## 7) One-Time Model Download

```bash
python3 download_models.py
```

## 8) Find Mic Device

```bash
python3 main.py --list-devices
```

## 9) Calibrate Threshold (recommended)

Replace `1` with your mic index.

```bash
python3 main.py --calibrate --input-device 1
```

Based on your latest calibration logs:

- `--input-device 1` (Pulse/Bluetooth path) suggested a high threshold due to spikes.
- `--input-device 0` suggested `1120`.

Good starting point for your current setup: `--threshold 1200`.

## 10) Run Headless Translator (Low-Latency Profile)

Replace `1` with your mic index.

Replace `3` with your Bluetooth output index.

```bash
python3 main.py --lang hindi --input-device 1 --output-device 3 --threshold 1200 --direct-translate --silence-timeout 0.15 --max-record-secs 2.8 --tts piper --playback-gain 1.5
```

Quick output test before full run:

```bash
python3 main.py --list-devices
python3 main.py --test-audio --output-device 3 --playback-gain 1.5
```

## 11) Run Web Prototype

```bash
python3 serve_demo.py --host 0.0.0.0 --port 8080
```

In another terminal, get Pi IP:

```bash
hostname -I
```

Open on phone browser:

```text
http://PI_IP:8080
```

## 12) Quick Troubleshooting

### If audio modules fail

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### If mic not detected

```bash
python3 main.py --list-devices
```

### If latency is high

Use this command profile:

```bash
python3 main.py --lang hindi --input-device 1 --output-device 3 --direct-translate --silence-timeout 0.12 --max-record-secs 2.5 --tts piper --playback-gain 1.5
```

### If Bluetooth buds have no sound

Check current default sink:

```bash
pactl info | grep "Default Sink"
```

List all sinks:

```bash
pactl list short sinks
```

Set Bluetooth sink as default (replace with your sink name):

```bash
pactl set-default-sink bluez_output.XX_XX_XX_XX_XX_XX.a2dp-sink
```

Unmute and raise volume:

```bash
pactl set-sink-mute @DEFAULT_SINK@ 0
pactl set-sink-volume @DEFAULT_SINK@ 100%
```

Quick Linux audio test (outside app):

```bash
speaker-test -D pulse -t sine -f 1000 -l 1
```

## 13) Shutdown Raspberry Pi

Shutdown immediately:

```bash
sudo shutdown -h now
```

Alternative (power off after 1 minute):

```bash
sudo shutdown -h +1
```
