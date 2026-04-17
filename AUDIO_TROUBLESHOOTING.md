# Audio & Noise Troubleshooting Guide

This guide provides commands to test your Bluetooth earbuds on the Raspberry Pi and explains how to improve speech recognition in noisy environments.

---

## Part 1: Testing Bluetooth and Audio on Raspberry Pi

If you are unsure whether your Bluetooth earbuds are correctly connected or if the Pi can hear/play audio through them, run these commands on the Pi terminal.

### 1. Check Bluetooth Connection
Verify that your earbuds are actively connected:
```bash
bluetoothctl info
```
*(Look for "Connected: yes" and "Audio Sink / Audio Source" in the UUIDs).*

### 2. List Audio Devices
Find out how Linux sees your microphone and speaker. You can use ALSA or our built-in Python script:
```bash
aplay -l     # Lists all playback (speaker) devices
arecord -l   # Lists all capture (microphone) devices

# Or use the python client to see the exact IDs sounddevice sees:
python pi_client.py --list-devices
```

### 3. Test the Speaker (Output)
Play a static white noise or a test voice directly to the speaker to ensure audio is working:
```bash
# Plays a woman's voice saying "Front, Center"
speaker-test -t wav -c 2
```
Hit `Ctrl+C` to stop. If you hear nothing, your Bluetooth earbuds are not set as the default output device. You may need to configure `~/.asoundrc` or use `alsamixer`.

### 4. Test the Microphone (Input)
Record 5 seconds of audio from your microphone, then play it back to yourself:
```bash
# Record a 5-second WAV file
arecord -d 5 -f S16_LE -r 16000 test_mic.wav

# Play it back
aplay test_mic.wav
```
If the recording is pure static or silence, the Pi is using the wrong microphone.

---

## Part 2: Handling High Background Noise

When surrounding noise is high, the microphone captures everything, confusing the Sarvam STT engine. Here are three ways to fix it:

### 1. Lower the Microphone Gain (Hardware level)
If the mic is too sensitive, it captures the whole room. You can lower the mic volume so it only catches audio very close to your mouth:
```bash
alsamixer
```
- Press `F4` to go to **Capture** (Microphone).
- Use the **Down Arrow** to lower the microphone volume to ~40-50%.
- Press `Esc` to exit.
- Save the settings so they persist after reboot: `sudo alsactl store`

### 2. Tweak Sarvam VAD Sensitivity (Software level)
By default, we set `high_vad_sensitivity="true"` in the `edge_stream_server.py`. In a noisy room, Sarvam might think background noise is speech. 
On your **MacBook**, open `edge_stream_server.py` and change `high_vad_sensitivity` to `"false"`:

```python
    async with sarvam_client.speech_to_text_streaming.connect(
        model="saaras:v3",
        mode="transcribe",
        language_code=input_lang,
        input_audio_codec="pcm_s16le",
        sample_rate="16000",
        high_vad_sensitivity="false",  # <--- Change this to false
        vad_signals="true"
    ) as sarvam_ws:
```
Restart the MacBook server after making this change.

### 3. Physical Solutions
- **Positioning**: Move the microphone receiver closer to your mouth. Earbud mics are naturally far from the mouth, which makes them struggle in noisy rooms.
- **Directional Mics**: If Bluetooth earbuds continue to fail in public, using a wired USB lavalier (lapel) mic clipped to your collar will provide vastly superior noise isolation for the Pi.
