"""
Wearable Voice Translator v2 — Headless Audio Pipeline

No browser, no Streamlit. Pure terminal app designed for Raspberry Pi.
Captures mic audio, translates via Sarvam cloud, speaks via local Piper TTS.

Usage:
    python main.py                   # defaults to Hindi output
    python main.py --lang telugu     # output in Telugu
    python main.py --lang tamil      # output in Tamil
    python main.py --lang english    # output in English
    python main.py --list-devices    # show available audio devices
    python main.py --input-device 1  # use specific mic device index
"""

import argparse
import io
import subprocess
import sys
import time
import wave
import queue
import threading
import concurrent.futures

import numpy as np

# ── Suppress ALSA/PortAudio warnings on Linux (must be before sounddevice import) ──
if sys.platform == "linux":
    try:
        import ctypes
        _ALSA_ERR_CB = ctypes.CFUNCTYPE(
            None, ctypes.c_char_p, ctypes.c_int,
            ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
        _alsa_err_cb = _ALSA_ERR_CB(lambda *_: None)
        ctypes.cdll.LoadLibrary("libasound.so.2").snd_lib_error_set_handler(_alsa_err_cb)
    except Exception:
        pass

import sounddevice as sd

from config import (
    LANG_MAP, DEFAULT_TARGET_LANG,
    SAMPLE_RATE, CHANNELS, BLOCK_SIZE,
    RMS_THRESHOLD, SILENCE_TIMEOUT, MAX_RECORD_SECS, MIN_SPEECH_RMS,
    FILLER_PHRASES, MAX_WAV_KB,
)
from sarvam_client import transcribe_and_translate
from tts_engine import synthesize, synthesize_stream, print_status, get_backend_name, preload_piper


# ── Audio queue (mic callback → main thread) ──────────────────────────────
_audio_q: queue.Queue = queue.Queue(maxsize=5000)

# Flag to pause mic capture during playback (avoid echo)
_paused = threading.Event()

# Thread pool for TTS + playback (runs off the main loop)
_tts_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")

# Actual mic sample rate (may differ from SAMPLE_RATE if mic doesn't support 16kHz)
_mic_rate: int = SAMPLE_RATE

# Optional playback device and gain controls
_output_device: int | None = None
_playback_gain: float = 1.0


def _resample_int16(pcm_bytes: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample int16 PCM bytes from from_rate to to_rate."""
    if from_rate == to_rate:
        return pcm_bytes
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    ratio = to_rate / from_rate
    n_out = int(len(samples) * ratio)
    indices = np.arange(n_out) / ratio
    indices = np.clip(indices, 0, len(samples) - 1).astype(np.int32)
    resampled = samples[indices].astype(np.int16)
    return resampled.tobytes()


def mic_callback(indata, frames, time_info, status):
    """Called by sounddevice for each mic block. Puts raw PCM into queue."""
    if _paused.is_set():
        return
    # indata is float32 [-1, 1] → convert to int16
    pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
    # Resample to 16kHz if mic runs at a different rate
    if _mic_rate != SAMPLE_RATE:
        pcm = _resample_int16(pcm, _mic_rate, SAMPLE_RATE)
    try:
        _audio_q.put_nowait(pcm)
    except queue.Full:
        pass  # drop frames if queue is full


def calculate_rms(raw_bytes: bytes) -> float:
    """RMS energy of int16 PCM bytes."""
    if len(raw_bytes) < 2:
        return 0.0
    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples ** 2)))


def trim_silence(raw_pcm: bytes, threshold: int = 150, frame_size: int = 480) -> bytes:
    """Remove leading/trailing silence from int16 PCM to reduce WAV upload size (vectorized)."""
    if len(raw_pcm) < frame_size * 4:
        return raw_pcm
    samples = np.frombuffer(raw_pcm, dtype=np.int16)
    n_frames = len(samples) // frame_size
    if n_frames < 3:
        return raw_pcm
    
    # Truncate to exact frame boundary before reshape
    samples = samples[:n_frames * frame_size]
    
    # Compute RMS for all frames in single vectorized pass (much faster than loop)
    frames = samples.reshape(n_frames, frame_size).astype(np.float32)
    rms_values = np.sqrt(np.mean(frames ** 2, axis=1))
    
    # Find start and end indices (vectorized, no loop)
    loud_frames = np.where(rms_values > threshold)[0]
    if len(loud_frames) == 0:
        return raw_pcm
    
    start = max(0, loud_frames[0] - 3)
    end = min(n_frames, loud_frames[-1] + 4)
    return samples[start*frame_size:end*frame_size].tobytes()


def build_wav(raw_pcm: bytes) -> bytes:
    """Wrap raw int16 mono PCM in a WAV header."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw_pcm)
    return buf.getvalue()


def play_wav(wav_bytes: bytes):
    """Play WAV bytes through the default output device (speaker/earbud)."""
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
            print(f"  [Play] Unsupported sample width: {sw}")
            return

        if ch > 1:
            audio = audio.reshape(-1, ch)[:, 0]  # take first channel

        # Optional gain boost for quiet Bluetooth paths; clip to valid float32 range.
        if _playback_gain != 1.0:
            audio = np.clip(audio * _playback_gain, -1.0, 1.0)

        _paused.set()  # pause mic during playback
        sd.play(audio, samplerate=sr, blocking=True, device=_output_device)
        sd.wait()
    except Exception as e:
        print(f"  [Play error] {e}")
    finally:
        _paused.clear()
        # Drain any frames captured during playback
        while not _audio_q.empty():
            try:
                _audio_q.get_nowait()
            except queue.Empty:
                break


def play_stream(audio_generator):
    """Play raw PCM frames yielded by a stream generator."""
    if not audio_generator:
        return
    _paused.set()
    try:
        # Collect all chunks and play as one block (most reliable for Bluetooth)
        all_audio = []
        play_sr = None
        for chunk, sr, sw in audio_generator:
            play_sr = sr
            if sw == 2:
                audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767.0
            elif sw == 1:
                audio = np.frombuffer(chunk, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
            else:
                continue
            all_audio.append(audio)

        if all_audio and play_sr:
            combined = np.concatenate(all_audio)
            if _playback_gain != 1.0:
                combined = np.clip(combined * _playback_gain, -1.0, 1.0)
            sd.play(combined, samplerate=play_sr, blocking=True, device=_output_device)
            sd.wait()
    except Exception as e:
        print(f"  [Play stream error] {e}")
    finally:
        _paused.clear()
        while not _audio_q.empty():
            try:
                _audio_q.get_nowait()
            except queue.Empty:
                break


# Global TTS backend preference (set by --tts flag)
_tts_backend: str = "auto"


def _speak_background(text: str, tgt_code: str, t0: float, t_stt: float, t_translate: float, tts_backend: str = "auto"):
    """Run TTS + playback on a background thread so main loop resumes listening."""
    t1 = time.time()
    audio_generator = synthesize_stream(text, tgt_code, backend=tts_backend)

    t2 = time.time()
    played = False
    if audio_generator:
        play_stream(audio_generator)
        played = True
    t_play = time.time() - t2

    t_total = time.time() - t0
    tr_str = f" + Trans {t_translate:.2f}s" if t_translate > 0 else ""
    play_str = f" + Play {t_play:.2f}s" if played else " + Play FAILED"
    print(f"  \u2500\u2500\u2500 STT {t_stt:.2f}s{tr_str}{play_str} = Total {t_total:.2f}s \u2500\u2500\u2500")


def process_utterance(wav_bytes: bytes, tgt_code: str, tgt_name: str, direct_translate: bool):
    """Run the full pipeline: STT → Translate → TTS → Play."""
    t0 = time.time()

    # Audio info
    wav_kb = len(wav_bytes) / 1024
    audio_secs = (len(wav_bytes) - 44) / (SAMPLE_RATE * 2)
    print(f"  Audio : {audio_secs:.1f}s ({wav_kb:.0f} KB)")

    # ── Minimum duration filter (skip very short clips = noise) ────────
    if audio_secs < 0.3:
        print(f"  (too short, skipping — likely noise)")
        return

    # ── Max size filter (skip huge clips that will timeout the API) ────
    if wav_kb > MAX_WAV_KB:
        print(f"  (too large: {wav_kb:.0f} KB > {MAX_WAV_KB} KB limit — skipping)")
        return

    # ── STT + Translation (cloud) ──────────────────────────────────────
    result = transcribe_and_translate(wav_bytes, tgt_lang=tgt_code, direct_translate=direct_translate)
    
    # Handle both 3-tuple and 4-tuple returns for compatibility
    if len(result) == 4:
        text, english_text, t_stt, t_translate = result
    else:
        text, t_stt, t_translate = result
        english_text = None

    if not text or text.strip().lower() in FILLER_PHRASES or len(text.strip()) <= 1:
        print(f"  (filtered noise \u2014 stt: {t_stt:.2f}s)")
        return

    print(f"  STT   : {t_stt:.2f}s")
    if t_translate > 0:
        print(f"  Trans : {t_translate:.2f}s")
    
    # ── Show source + translated text clearly ──────────────────────────
    if english_text and tgt_code != "en-IN":
        print(f"  [English]  {english_text}")
        print(f"  [{tgt_name}]    {text}")
    else:
        print(f"  [{tgt_name}] {text}")
    print(f"  🔊 Speaking…", flush=True)

    # ── TTS + Playback (background thread) ─────────────────────────────
    _tts_pool.submit(_speak_background, text, tgt_code, t0, t_stt, t_translate, _tts_backend)


def _calibrate_noise(input_device=None):
    """Record 3 seconds of ambient noise and recommend a threshold."""
    print("=" * 56)
    print("  🎤 Noise Calibration")
    print("=" * 56)
    print("  Stay QUIET for 3 seconds…")
    print()

    # Detect mic rate
    mic_sr = SAMPLE_RATE
    try:
        dev_info = sd.query_devices(input_device, 'input')
        native = int(dev_info['default_samplerate'])
        if native != SAMPLE_RATE:
            mic_sr = native
    except Exception:
        pass

    # Record 3 seconds
    duration = 3.0
    blocksize = int(mic_sr * 30 / 1000)
    recording = sd.rec(int(duration * mic_sr), samplerate=mic_sr, channels=1, dtype='float32',
                       device=input_device, blocksize=blocksize)
    sd.wait()

    # Convert to int16 and compute RMS
    pcm = (recording[:, 0] * 32767).astype(np.int16)
    rms_values = []
    chunk_size = int(mic_sr * 0.1)  # 100ms chunks
    for i in range(0, len(pcm) - chunk_size, chunk_size):
        chunk = pcm[i:i+chunk_size].astype(np.float32)
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        rms_values.append(rms)

    if not rms_values:
        print("  ERROR: No audio captured.")
        return

    avg_rms = np.mean(rms_values)
    max_rms = np.max(rms_values)
    recommended = int(max_rms * 2.0)  # 2x the peak ambient noise

    print(f"  Ambient noise (avg) : {avg_rms:.0f}")
    print(f"  Ambient noise (max) : {max_rms:.0f}")
    print(f"  Current threshold   : {RMS_THRESHOLD}")
    print(f"  Recommended         : {recommended}")
    print()
    print(f"  Usage: python3 main.py --threshold {recommended}")
    print("=" * 56)


def _play_test_tone(output_device=None, gain: float = 1.0):
    """Play a short 1 kHz tone to verify speaker routing/volume."""
    sr = 24000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    tone = 0.2 * np.sin(2 * np.pi * 1000 * t)
    if gain != 1.0:
        tone = np.clip(tone * gain, -1.0, 1.0)
    print("  Playing 1 kHz test tone...")
    sd.play(tone, samplerate=sr, blocking=True, device=output_device)
    sd.wait()
    print("  Done.")


def _device_name_lower(device_index: int | None) -> str:
    """Best-effort device name lookup (lowercase)."""
    if device_index is None:
        return ""
    try:
        info = sd.query_devices(device_index)
        return str(info.get("name", "")).lower()
    except Exception:
        return ""


def _default_sink_name() -> str | None:
    """Return Pulse/PipeWire default sink name on Linux, else None."""
    if sys.platform != "linux":
        return None
    try:
        proc = subprocess.run(
            ["pactl", "info"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if proc.returncode != 0:
            return None
        for line in proc.stdout.splitlines():
            if line.startswith("Default Sink:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        return None
    return None


def _has_real_sink() -> bool:
    """True if at least one non-null Pulse sink exists."""
    if sys.platform != "linux":
        return True
    try:
        proc = subprocess.run(
            ["pactl", "list", "short", "sinks"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if proc.returncode != 0:
            return True  # Unknown state: do not block app.
        sinks = []
        for line in proc.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                sinks.append(parts[1])
        real_sinks = [s for s in sinks if s != "auto_null" and not s.startswith("null.")]
        return len(real_sinks) > 0
    except Exception:
        return True


def _guard_against_null_sink(output_device: int | None):
    """Fail fast when selected output is Pulse/default but only auto_null exists."""
    dev_name = _device_name_lower(output_device)

    # If caller selected explicit non-Pulse hardware device, skip this guard.
    if output_device is not None and "pulse" not in dev_name and "default" not in dev_name:
        return

    sink = _default_sink_name()
    if sink == "auto_null" or (sink is not None and not _has_real_sink()):
        print("ERROR: No real audio sink is available (Default Sink: auto_null).")
        print("Your Bluetooth buds are not connected as an active A2DP sink yet.")
        print("Run: pactl list short sinks")
        print("Then reconnect buds via bluetoothctl and set default sink to bluez_output...a2dp-sink")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Wearable Voice Translator v2")
    parser.add_argument(
        "--lang", choices=list(LANG_MAP.keys()),
        default=DEFAULT_TARGET_LANG,
        help=f"Output language (default: {DEFAULT_TARGET_LANG})",
    )
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--input-device", type=int, default=None, help="Mic device index")
    parser.add_argument("--output-device", type=int, default=None,
                        help="Speaker output device index (use for Bluetooth buds)")
    parser.add_argument("--test-audio", action="store_true",
                        help="Play a short test tone on output device and exit")
    parser.add_argument("--playback-gain", type=float, default=1.0,
                        help="Playback gain multiplier (e.g. 1.5 for quieter buds)")
    parser.add_argument("--threshold", type=int, default=RMS_THRESHOLD,
                        help=f"RMS speech threshold (default: {RMS_THRESHOLD}). Higher = less sensitive. Use --calibrate to find optimal value.")
    parser.add_argument("--silence-timeout", type=float, default=SILENCE_TIMEOUT,
                        help=f"Seconds of silence before processing (default: {SILENCE_TIMEOUT}). Lower = faster end-of-speech.")
    parser.add_argument("--max-record-secs", type=float, default=MAX_RECORD_SECS,
                        help=f"Max utterance length before forced processing (default: {MAX_RECORD_SECS}).")
    parser.add_argument("--tts", choices=["auto", "piper", "espeak", "edge"], default="auto",
                        help="TTS backend: auto (best available), piper, espeak (fastest), edge (cloud)")
    parser.add_argument("--direct-translate", action="store_true",
                        help="Use a single STT+Translate call (lower latency, skips English preview).")
    parser.add_argument("--calibrate", action="store_true",
                        help="Measure ambient noise for 3 seconds and recommend a threshold")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    if args.calibrate:
        _calibrate_noise(args.input_device)
        return

    tgt_name = args.lang.capitalize()
    tgt_code = LANG_MAP[args.lang]
    threshold = args.threshold
    silence_timeout = args.silence_timeout
    max_record_secs = args.max_record_secs
    direct_translate = args.direct_translate

    # Store playback controls globally for background playback thread.
    global _output_device, _playback_gain
    _output_device = args.output_device
    _playback_gain = max(0.1, args.playback_gain)

    # Set process-level defaults without passing None values.
    if args.input_device is not None and args.output_device is not None:
        sd.default.device = (args.input_device, args.output_device)
    elif args.input_device is not None:
        sd.default.device = args.input_device
    elif args.output_device is not None:
        sd.default.device = (sd.default.device[0], args.output_device)

    if args.output_device is not None:
        try:
            out_info = sd.query_devices(args.output_device, 'output')
            print(f"  Output device   : {args.output_device} ({out_info['name']})")
        except Exception as e:
            print(f"ERROR: Could not use output device {args.output_device}: {e}")
            print("Run with --list-devices and pick a device with output channels.")
            sys.exit(1)

    # Detect the active output path and block known silent setup (auto_null sink).
    selected_output = args.output_device
    if selected_output is None:
        try:
            selected_output = sd.default.device[1]
        except Exception:
            selected_output = None
    _guard_against_null_sink(selected_output)

    if args.test_audio:
        try:
            _play_test_tone(output_device=args.output_device, gain=_playback_gain)
        except Exception as e:
            print(f"ERROR: Test tone failed: {e}")
            print("Run with --list-devices and verify your Bluetooth sink is connected.")
            sys.exit(1)
        return

    print("=" * 56)
    print("  Wearable Voice Translator v2")
    print("=" * 56)
    print(f"  Output language : {tgt_name} ({tgt_code})")
    # Store TTS backend preference globally
    global _tts_backend
    _tts_backend = args.tts
    tts_display = args.tts if args.tts != "auto" else get_backend_name(tgt_code)
    print(f"  TTS backend     : {tts_display}")
    print(f"  Mic threshold   : {threshold}")
    print(f"  Silence timeout : {silence_timeout}")
    print(f"  Max record secs : {max_record_secs}")
    print(f"  Direct translate: {'on' if direct_translate else 'off'}")
    print(f"  Sample rate     : {SAMPLE_RATE} Hz")
    print(f"  Playback gain   : {_playback_gain:.2f}x")
    print()
    print("  TTS backends available:")
    print_status()
    print("=" * 56)
    print()

    # Pre-load Piper model now so first utterance is fast
    preload_piper(tgt_code)

    # ── Detect mic sample rate ─────────────────────────────────────────
    global _mic_rate
    mic_sr = None
    dev = args.input_device
    # Query device's default sample rate first to avoid PortAudio warnings
    try_rates = [SAMPLE_RATE]
    try:
        dev_info = sd.query_devices(dev, 'input')
        native = int(dev_info['default_samplerate'])
        # Put native rate first if it's not already SAMPLE_RATE
        if native != SAMPLE_RATE:
            try_rates = [native, SAMPLE_RATE]
    except Exception:
        pass
    try_rates += [r for r in [48000, 44100, 32000, 22050, 8000] if r not in try_rates]
    for try_rate in try_rates:
        try:
            s = sd.InputStream(
                samplerate=try_rate, channels=CHANNELS, dtype="float32",
                blocksize=int(try_rate * 30 / 1000), device=dev,
            )
            s.close()
            mic_sr = try_rate
            break
        except Exception:
            continue
    if mic_sr is None:
        print("ERROR: Could not find a supported sample rate for the microphone.")
        sys.exit(1)
    if mic_sr != SAMPLE_RATE:
        print(f"  Mic native rate : {mic_sr} Hz (resampling to {SAMPLE_RATE} Hz)")
    _mic_rate = mic_sr
    mic_block = int(mic_sr * 30 / 1000)  # 30ms block at mic's native rate

    # ── Open mic stream ────────────────────────────────────────────────
    try:
        stream = sd.InputStream(
            samplerate=mic_sr,
            channels=CHANNELS,
            dtype="float32",
            blocksize=mic_block,
            device=args.input_device,
            callback=mic_callback,
        )
    except Exception as e:
        print(f"ERROR: Could not open microphone: {e}")
        print("Run with --list-devices to see available devices.")
        sys.exit(1)

    buf = bytearray()
    speech_active = False
    silence_start = None
    last_reset = time.time()
    utterance_count = 0
    last_rms = 0.0  # Cache RMS to avoid redundant calculations

    print("🎧 Listening… (Ctrl+C to quit)\n")

    try:
        with stream:
            while True:
                time.sleep(0.01)  # Reduced from 0.02s for faster response

                # Drain queue into buffer (batch operation, avoid exceptions)
                try:
                    while True:
                        chunk = _audio_q.get_nowait()
                        buf.extend(chunk)
                except queue.Empty:
                    pass

                if len(buf) < 8000:
                    continue

                # ── VAD on last 0.25s (avoid unnecessary copying) ──────────
                # Reuse array slice without creating new bytes object
                start_idx = max(0, len(buf) - 8000)
                rms_samples = np.frombuffer(buf[start_idx:], dtype=np.int16).astype(np.float32)
                last_rms = float(np.sqrt(np.mean(rms_samples ** 2))) if len(rms_samples) > 0 else 0.0
                now = time.time()

                if last_rms > threshold:
                    if not speech_active:
                        speech_active = True
                        print("🎙️  Voice detected…", flush=True)
                    silence_start = None
                else:
                    if speech_active and silence_start is None:
                        silence_start = now
                    # If we haven't detected speech yet, don't let buffer grow forever
                    if not speech_active and len(buf) > 32000:
                        buf = bytearray(buf[-8000:])  # keep only last 0.25s

                # ── Trigger processing ─────────────────────────────────
                should_trigger = (
                    (speech_active and silence_start and now - silence_start > silence_timeout)
                    or (now - last_reset > max_record_secs and len(buf) > 32000)
                )

                if should_trigger:
                    overall_rms = calculate_rms(bytes(buf))
                    if overall_rms < MIN_SPEECH_RMS:
                        # Just noise, discard
                        buf = bytearray()
                        speech_active = False
                        silence_start = None
                        last_reset = now
                        continue

                    # Trim silence and process
                    raw_pcm = bytes(buf)
                    trimmed = trim_silence(raw_pcm, threshold=threshold)
                    wav_bytes = build_wav(trimmed)
                    buf = bytearray()
                    speech_active = False
                    silence_start = None
                    last_reset = now
                    utterance_count += 1

                    print(f"\n── Utterance #{utterance_count} ──")
                    process_utterance(wav_bytes, tgt_code, tgt_name, direct_translate)
                    print("🎧 Listening…\n", flush=True)

    except KeyboardInterrupt:
        print(f"\n\nStopped. Processed {utterance_count} utterances.")


if __name__ == "__main__":
    main()
