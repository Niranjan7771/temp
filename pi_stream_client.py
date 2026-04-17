"""
Streaming Audio Client for Raspberry Pi.
Runs independently of pi_client.py.

Usage:
  python pi_stream_client.py --server ws://192.168.x.x:5556
"""

import argparse
import asyncio
import json
import base64
import sounddevice as sd
import websockets
import threading
import queue

import io
import wave
import numpy as np

# Prevent librosa/ALSA warnings hiding output on Pi
import logging
logging.getLogger('sox').setLevel(logging.ERROR)

TARGET_SAMPLE_RATE = 16000
BLOCK_MS = 200  # Send 200ms chunks to prevent Wi-Fi dropouts

# Playback queue ensures audio files play one after another completely
playback_queue = queue.Queue()

# Global variables to store selected device IDs
TARGET_IN_DEVICE = None
TARGET_OUT_DEVICE = None
PREFERRED_CAPTURE_RATE = None


def auto_select_input_device():
    """Prefer a physical USB microphone over pulse/default virtual devices."""
    try:
        devices = sd.query_devices()
    except Exception as e:
        print(f"⚠️  Unable to query audio devices for auto input selection: {e}")
        return None

    ranked = []
    for idx, dev in enumerate(devices):
        max_in = int(dev.get("max_input_channels", 0))
        if max_in <= 0:
            continue

        name = str(dev.get("name", ""))
        lname = name.lower().strip()

        # Skip virtual endpoints unless there is no better explicit device.
        if lname in {"pulse", "default"}:
            continue

        score = 0
        if "usb" in lname:
            score += 6
        if "uac" in lname:
            score += 4
        if "mic" in lname or "microphone" in lname:
            score += 2
        if int(dev.get("max_output_channels", 0)) == 0:
            score += 2

        ranked.append((score, idx, name))

    if not ranked:
        return None

    ranked.sort(reverse=True)
    best_score, best_idx, best_name = ranked[0]

    if best_score <= 0:
        return None

    print(f"🎤 Auto-selected USB/physical input device: {best_idx} ({best_name})")
    return best_idx


def auto_select_output_device():
    """Prefer pulse/default virtual output so Bluetooth sink routing works."""
    try:
        devices = sd.query_devices()
    except Exception as e:
        print(f"⚠️  Unable to query audio devices for auto output selection: {e}")
        return None

    pulse_idx = None
    default_idx = None
    fallback_idx = None

    for idx, dev in enumerate(devices):
        max_out = int(dev.get("max_output_channels", 0))
        if max_out <= 0:
            continue

        name = str(dev.get("name", ""))
        lname = name.lower().strip()

        if fallback_idx is None:
            fallback_idx = idx

        if lname == "pulse":
            pulse_idx = idx
        elif lname == "default":
            default_idx = idx

    chosen = pulse_idx if pulse_idx is not None else default_idx
    if chosen is None:
        chosen = fallback_idx

    if chosen is not None:
        print(f"🔊 Auto-selected output device: {chosen} ({devices[chosen]['name']})")
    return chosen


def choose_capture_sample_rate(device, preferred_rate=TARGET_SAMPLE_RATE):
    """Pick a supported microphone sample rate, preferring the target rate."""
    candidates = []
    if preferred_rate:
        candidates.append(int(preferred_rate))

    try:
        dev_info = sd.query_devices(device, "input")
        default_sr = int(round(float(dev_info.get("default_samplerate", 0) or 0)))
        if default_sr > 0 and default_sr not in candidates:
            candidates.append(default_sr)
    except Exception:
        pass

    for sr in (48000, 44100, 32000, 24000, 22050, 16000, 8000):
        if sr not in candidates:
            candidates.append(sr)

    for sr in candidates:
        try:
            sd.check_input_settings(device=device, channels=1, dtype="int16", samplerate=sr)
            return sr
        except Exception:
            continue

    raise RuntimeError("No compatible microphone sample rate found for selected input device.")


def resample_int16_mono(audio_mono_int16, src_rate, dst_rate):
    """Resample mono int16 PCM to the target sample rate using linear interpolation."""
    if src_rate == dst_rate:
        return audio_mono_int16

    if audio_mono_int16.size == 0:
        return audio_mono_int16

    src_len = int(audio_mono_int16.size)
    dst_len = max(1, int(round(src_len * float(dst_rate) / float(src_rate))))

    x_old = np.arange(src_len, dtype=np.float64)
    x_new = np.linspace(0, src_len - 1, num=dst_len, dtype=np.float64)
    y_new = np.interp(x_new, x_old, audio_mono_int16.astype(np.float32))
    return np.clip(np.round(y_new), -32768, 32767).astype(np.int16)

def audio_player_thread():
    while True:
        item = playback_queue.get()
        if item is None: break
        wav_bytes, gain = item
        
        try:
            buf = io.BytesIO(wav_bytes)
            with wave.open(buf, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                # Convert to float32
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
                
                # Apply volume gain (e.g. 2.0 = double the volume)
                if gain != 1.0:
                    audio = np.clip(audio * gain, -1.0, 1.0)
                    
                print("🔊 Playing Translation Audio...")
                # Play specifically on the chosen output device
                sd.play(audio, samplerate=wf.getframerate(), blocking=True, device=TARGET_OUT_DEVICE) 
        except Exception as e:
            print(f"⚠️ Error playing audio: {e}")

# Start background audio player
t = threading.Thread(target=audio_player_thread, daemon=True)
t.start()

async def capture_and_send(websocket):
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue()
    chunks_sent = 0
    all_zero_chunks = 0
    zero_warned = False

    preferred = PREFERRED_CAPTURE_RATE if PREFERRED_CAPTURE_RATE else TARGET_SAMPLE_RATE
    capture_rate = choose_capture_sample_rate(TARGET_IN_DEVICE, preferred_rate=preferred)
    if capture_rate != TARGET_SAMPLE_RATE:
        print(f"⚙️  [Mic] Using capture sample rate {capture_rate} Hz and resampling to {TARGET_SAMPLE_RATE} Hz.")

    def callback(indata, frames, time, status):
        if status:
            print(f"⚠️  [Mic Stream Status] {status}")
        loop.call_soon_threadsafe(queue.put_nowait, indata.copy())

    stream = sd.InputStream(
        samplerate=capture_rate,
        channels=1,
        dtype='int16',
        blocksize=int(capture_rate * BLOCK_MS / 1000),
        device=TARGET_IN_DEVICE,
        callback=callback
    )

    print("🎤 Microphone ON - Streaming to server...")
    with stream:
        while True:
            indata = await queue.get()
            mono = indata.reshape(-1).astype(np.int16, copy=False)
            chunks_sent += 1
            peak = int(np.max(np.abs(mono.astype(np.int32)))) if mono.size else 0
            if peak == 0:
                all_zero_chunks += 1

            # This warning catches muted / wrong input device quickly.
            if not zero_warned and chunks_sent >= 40 and all_zero_chunks == chunks_sent:
                print("⚠️  [Mic Warning] Input is completely silent (all-zero PCM).")
                print("⚠️  [Mic Warning] Try --input-device with a valid microphone index.")
                zero_warned = True

            if chunks_sent % 20 == 0:
                device_label = TARGET_IN_DEVICE if TARGET_IN_DEVICE is not None else "default"
                print(
                    f"🎙️  [Mic Level] chunk={chunks_sent} peak={peak} "
                    f"input_device={device_label} capture_rate={capture_rate} send_rate={TARGET_SAMPLE_RATE}"
                )

            # Send raw PCM audio as binary
            out_pcm = resample_int16_mono(mono, capture_rate, TARGET_SAMPLE_RATE)
            await websocket.send(out_pcm.tobytes())

async def receive_translations(websocket, gain=1.0):
    _expecting_english_audio = False
    try:
        async for message in websocket:
            if type(message) is bytes:
                if _expecting_english_audio:
                    print(f"📥 [English Audio] Got {len(message)} bytes from server.")
                    _expecting_english_audio = False
                else:
                    print(f"📥 [Translation Audio] Got {len(message)} bytes from server.")
                playback_queue.put((message, gain))
                # Also save the last received audio to disk for debugging
                try:
                    with open("debug_last_translation.wav", "wb") as f:
                        f.write(message)
                except Exception as ex:
                    print(f"Failed to save debug wav: {ex}")
                continue
                
            try:
                data = json.loads(message)
                if 'transcript' in data:
                    print(f"  [Heard] {data['transcript']}")
                if 'translation' in data:
                    print(f"  [Translated] {data['translation']}")
                
                # Check if English audio will follow
                _expecting_english_audio = data.get('has_english_audio', False)
                
                if 'timings' in data:
                    t = data['timings']
                    en_tts = f" | EN-TTS: {t.get('en_tts_ms', 0)}ms" if t.get('en_tts_ms') else ""
                    print(f"  ⏱️  [LATENCY] STT: {t.get('stt_ms')}ms | Trans: {t.get('trans_ms')}ms{en_tts} | TTS: {t.get('tts_ms')}ms | Total Stream: {t.get('total_ms')}ms")
            except Exception as e:
                # Binary audio (TTS) could be received here pass
                pass
    except websockets.exceptions.ConnectionClosed:
        print("Server disconnected.")
        raise  # Bubble up to trigger reconnect

async def main():
    global TARGET_IN_DEVICE
    global TARGET_OUT_DEVICE
    global PREFERRED_CAPTURE_RATE
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default=None, help="e.g. ws://192.168.x.x:5556")
    parser.add_argument("--in-lang", default="en-IN", help="Input language (e.g. 'en-IN' or 'unknown')")
    parser.add_argument("--out-lang", default="hi-IN", help="Target translation language (e.g. 'ta-IN', 'hi-IN')")
    parser.add_argument("--gain", type=float, default=1.0, help="Playback volume multiplier (e.g. 2.0 to double the volume)")
    parser.add_argument("--input-device", type=int, default=None, help="Force microphone to a specific sounddevice index")
    parser.add_argument("--input-samplerate", type=int, default=None, help="Preferred microphone capture sample rate (e.g. 48000)")
    parser.add_argument("--output-device", type=int, default=None, help="Force audio to a specific sounddevice index")
    parser.add_argument("--list-input-devices", action="store_true", help="List only microphone-capable devices and exit")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    if args.list_input_devices:
        devices = sd.query_devices()
        print("Input-capable devices:")
        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                print(f"{idx}: {dev['name']} (in={dev['max_input_channels']}, out={dev['max_output_channels']})")
        return

    if not args.server:
        parser.error("the following arguments are required: --server (unless --list-devices or --list-input-devices is used)")

    if args.input_device is not None:
        TARGET_IN_DEVICE = args.input_device
    else:
        TARGET_IN_DEVICE = auto_select_input_device()
        if TARGET_IN_DEVICE is None:
            print("ℹ️  No dedicated USB mic found. Falling back to default input device.")

    PREFERRED_CAPTURE_RATE = args.input_samplerate

    if args.output_device is not None:
        TARGET_OUT_DEVICE = args.output_device
    else:
        TARGET_OUT_DEVICE = auto_select_output_device()
        if TARGET_OUT_DEVICE is None:
            print("ℹ️  No output device auto-selected. Falling back to system default output.")

    while True:
        try:
            print(f"Connecting to Edge Stream Server at {args.server} ...")
            print(f"Input Language: {args.in_lang} | Target Language: {args.out_lang}")
            print(f"Input Device: {TARGET_IN_DEVICE if TARGET_IN_DEVICE is not None else 'default'} | Output Device: {TARGET_OUT_DEVICE if TARGET_OUT_DEVICE is not None else 'default'}")
            
            # Increased ping interval and timeout to 120 seconds to prevent Raspberry Pi disconnection
            async with websockets.connect(
                args.server,
                ping_interval=120,
                ping_timeout=120
            ) as websocket:
                print("Connected!")
                # 1. Send configuration JSON
                config_msg = json.dumps({
                    "input_lang": args.in_lang,
                    "target_lang": args.out_lang
                })
                await websocket.send(config_msg)
                
                # 2. Run sender and receiver concurrently
                await asyncio.gather(
                    capture_and_send(websocket),
                    receive_translations(websocket, gain=args.gain)
                )
        except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError, TimeoutError, OSError) as e:
            print(f"⚠️ Connection dropped or server offline ({e}). Retrying in 5 seconds...")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"❌ Unexpected error ({e}). Retrying in 5 seconds...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")