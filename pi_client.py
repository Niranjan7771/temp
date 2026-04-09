"""
Pi Audio Client — runs on the Raspberry Pi wearable device.

Records audio from USB mic, sends to the laptop edge server for processing,
receives translated audio back, and plays through Bluetooth speaker.

Usage:
    python pi_client.py --server http://192.168.x.x:5555
    python pi_client.py --server http://192.168.x.x:5555 --lang tamil
    python pi_client.py --list-devices
"""

import argparse
import base64
import io
import queue
import sys
import threading
import time
import wave

import numpy as np
import requests
import sounddevice as sd

# ── Audio settings ─────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_DURATION_MS = 30
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION_MS / 1000)

# VAD settings
RMS_THRESHOLD = 300
SILENCE_TIMEOUT = 0.8
MAX_RECORD_SECS = 10
MIN_SPEECH_RMS = 300

LANG_MAP = {
    "english":  "en-IN",
    "hindi":    "hi-IN",
    "tamil":    "ta-IN",
    "telugu":   "te-IN",
    "kannada":  "kn-IN",
    "malayalam":"ml-IN",
    "bengali":  "bn-IN",
    "marathi":  "mr-IN",
    "gujarati": "gu-IN",
    "punjabi":  "pa-IN",
}

# ── Audio queue ────────────────────────────────────────────────────────────
_audio_q: queue.Queue = queue.Queue(maxsize=5000)
_paused = threading.Event()
_mic_rate: int = SAMPLE_RATE
_output_device = None
_playback_gain = 1.0
_capture_block_until = 0.0


def mic_callback(indata, frames, time_info, status):
    if _paused.is_set() or time.time() < _capture_block_until:
        return
    pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
    if _mic_rate != SAMPLE_RATE:
        pcm = _resample_int16(pcm, _mic_rate, SAMPLE_RATE)
    try:
        _audio_q.put_nowait(pcm)
    except queue.Full:
        pass


def _resample_int16(pcm_bytes, from_rate, to_rate):
    if from_rate == to_rate:
        return pcm_bytes
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    ratio = to_rate / from_rate
    n_out = int(len(samples) * ratio)
    indices = np.arange(n_out) / ratio
    indices = np.clip(indices, 0, len(samples) - 1).astype(np.int32)
    return samples[indices].astype(np.int16).tobytes()


def calculate_rms(raw_bytes):
    if len(raw_bytes) < 2:
        return 0.0
    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples ** 2)))


def build_wav(raw_pcm):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw_pcm)
    return buf.getvalue()


def play_wav(wav_bytes):
    global _capture_block_until
    if not wav_bytes or len(wav_bytes) < 44:
        return

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

        if _playback_gain != 1.0:
            audio = np.clip(audio * _playback_gain, -1.0, 1.0)

        _paused.set()

        # Use OutputStream for more reliable playback (esp. Bluetooth)
        audio_int16 = (audio * 32767).astype(np.int16)
        try:
            with sd.RawOutputStream(
                samplerate=sr,
                channels=1,
                dtype="int16",
                device=_output_device,
            ) as stream:
                # Write in chunks to avoid buffer issues
                chunk_size = sr  # 1 second chunks
                for i in range(0, len(audio_int16), chunk_size):
                    chunk = audio_int16[i : i + chunk_size]
                    stream.write(chunk.tobytes())
                # Small drain delay for Bluetooth
                time.sleep(0.1)
        except sd.PortAudioError as e:
            # Fallback: write temp file and play with aplay
            print(f"  [sd fallback] {e}")
            _play_wav_aplay(wav_bytes)
    except Exception as e:
        print(f"  [Play error] {e}")
    finally:
        _capture_block_until = max(_capture_block_until, time.time() + 0.35)
        _paused.clear()
        while not _audio_q.empty():
            try:
                _audio_q.get_nowait()
            except queue.Empty:
                break


def _play_wav_aplay(wav_bytes):
    """Fallback: play WAV via aplay (works reliably with Bluetooth on Pi)."""
    import subprocess
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            f.write(wav_bytes)
            f.flush()
            subprocess.run(
                ["aplay", f.name],
                timeout=30,
                capture_output=True,
            )
    except FileNotFoundError:
        pass  # aplay not available
    except Exception as e:
        print(f"  [aplay error] {e}")


def send_to_server(server_url, wav_bytes, target_lang):
    """Send audio to the edge server and get back translated text + TTS audio."""
    url = f"{server_url.rstrip('/')}/api/process-audio"
    try:
        files = {"audio": ("audio.wav", wav_bytes, "audio/wav")}
        data = {"target_lang": target_lang}
        resp = requests.post(url, files=files, data=data, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        print("  [ERROR] Cannot connect to edge server. Is it running?")
        return None
    except Exception as e:
        print(f"  [Server error] {e}")
        return None


def check_server(server_url):
    """Check if the edge server is reachable."""
    try:
        resp = requests.get(f"{server_url.rstrip('/')}/api/health", timeout=5)
        data = resp.json()
        return data.get("status") == "ok", data.get("message", "")
    except Exception as e:
        return False, str(e)


def main():
    global _mic_rate, _output_device, _playback_gain

    parser = argparse.ArgumentParser(description="Pi Wearable Audio Client")
    parser.add_argument("--server", required=True, help="Edge server URL (e.g., http://192.168.1.100:5555)")
    parser.add_argument("--lang", default="hindi", choices=list(LANG_MAP.keys()), help="Target language")
    parser.add_argument("--input-device", type=int, default=None, help="Mic device index")
    parser.add_argument("--output-device", type=int, default=None, help="Speaker device index")
    parser.add_argument("--gain", type=float, default=1.0, help="Playback gain multiplier")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    tgt_lang = LANG_MAP.get(args.lang, "hi-IN")
    _output_device = args.output_device
    _playback_gain = args.gain

    # Check server connectivity
    print(f"Connecting to edge server: {args.server}")
    ok, msg = check_server(args.server)
    if not ok:
        print(f"WARNING: Server not ready — {msg}")
        print("Will keep trying...")
    else:
        print(f"Server ready: {msg}")

    # Determine mic sample rate
    if args.input_device is not None:
        dev_info = sd.query_devices(args.input_device, "input")
        _mic_rate = int(dev_info["default_samplerate"])
    else:
        _mic_rate = SAMPLE_RATE

    mic_block = int(_mic_rate * BLOCK_DURATION_MS / 1000)

    print()
    print("=" * 50)
    print("  PI AUDIO CLIENT")
    print("=" * 50)
    print(f"  Server:   {args.server}")
    print(f"  Language:  {args.lang} ({tgt_lang})")
    print(f"  Mic rate:  {_mic_rate} Hz")
    print()
    print("  Speak into the mic — audio is sent to laptop for processing.")
    print("  Press Ctrl+C to stop.")
    print("=" * 50)

    # Start mic stream
    stream = sd.InputStream(
        device=args.input_device,
        channels=CHANNELS,
        samplerate=_mic_rate,
        blocksize=mic_block,
        dtype="float32",
        callback=mic_callback,
    )
    stream.start()

    recording = False
    audio_chunks = []
    silence_start = None
    speech_detected = False
    record_start = 0.0

    try:
        while True:
            try:
                chunk = _audio_q.get(timeout=0.05)
            except queue.Empty:
                continue

            rms = calculate_rms(chunk)

            if not recording:
                if rms > RMS_THRESHOLD:
                    recording = True
                    audio_chunks = [chunk]
                    silence_start = None
                    speech_detected = True
                    record_start = time.time()
                    print("\n🎤 Recording...", end="", flush=True)
            else:
                audio_chunks.append(chunk)

                if rms > RMS_THRESHOLD:
                    silence_start = None
                    speech_detected = True
                else:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= SILENCE_TIMEOUT:
                        # End of speech
                        recording = False

                elapsed = time.time() - record_start
                if elapsed >= MAX_RECORD_SECS:
                    recording = False

                if not recording and speech_detected:
                    print(" Done.", flush=True)
                    raw_pcm = b"".join(audio_chunks)
                    overall_rms = calculate_rms(raw_pcm)

                    if overall_rms < MIN_SPEECH_RMS:
                        print("  (noise, skipping)")
                        audio_chunks = []
                        speech_detected = False
                        continue

                    wav_bytes = build_wav(raw_pcm)
                    audio_secs = len(raw_pcm) / (SAMPLE_RATE * 2)
                    print(f"  Audio: {audio_secs:.1f}s, RMS={overall_rms:.0f}")

                    # Send to edge server
                    print("  Sending to server...", end="", flush=True)
                    t0 = time.time()
                    result = send_to_server(args.server, wav_bytes, tgt_lang)
                    t_round = time.time() - t0

                    if result and (result.get("english_text") or result.get("translated_text")):
                        en = result.get("english_text", "")
                        tr = result.get("translated_text", "")
                        timings = result.get("timings", {})

                        print(f" {int(t_round*1000)}ms")
                        print(f"  EN: {en}")
                        print(f"  TR: {tr}")
                        print(f"  Timings: STT={timings.get('stt_ms',0)}ms "
                              f"Translate={timings.get('translate_ms',0)}ms "
                              f"TTS={timings.get('tts_ms',0)}ms "
                              f"Total={timings.get('total_ms',0)}ms")

                        # Play TTS audio
                        tts_b64 = result.get("tts_audio", "")
                        if tts_b64:
                            tts_wav = base64.b64decode(tts_b64)
                            print("  🔊 Playing...", end="", flush=True)
                            play_wav(tts_wav)
                            print(" Done.")
                    else:
                        print(" No result.")

                    audio_chunks = []
                    speech_detected = False

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    main()
