"""
Streaming Edge Server.

Starts an independent WebSocket server.
Receives live PCM audio -> Pipes to Sarvam API live -> On finish, Translates -> Sends to Pi.
"""
import asyncio
import json
import base64
import os
import time
import socket
import websockets
from sarvamai import AsyncSarvamAI

# Connect to local components
import local_inference_client

from tts_engine import synthesize, preload_piper

# Environment configuration
from dotenv import load_dotenv
load_dotenv()
SARVAM_KEY = os.getenv("SARVAM_API_KEY", "")


def _normalize_stream_model(model_name: str) -> str:
    # Sarvam streaming accepts saarika:v2.5 (not saaras:v2.5) and saaras:v3.
    if model_name == "saaras:v2.5":
        print("⚠️  [Config] 'saaras:v2.5' is invalid for streaming. Using 'saarika:v2.5'.")
        return "saarika:v2.5"
    return model_name


SARVAM_STREAM_MODEL = _normalize_stream_model(
    os.getenv("SARVAM_STREAM_MODEL", "saarika:v2.5")
)

async def handle_pi_connection(websocket):
    print("📱 Pi connected to streaming edge server!")
    
    # 1. Wait for initial configuration message from Pi
    try:
        first_msg = await websocket.recv()
        config = json.loads(first_msg)
        target_lang = config.get("target_lang", "hi-IN")
        input_lang = config.get("input_lang", "unknown")
        print(f"⚙️  [Config] Input Lang: {input_lang} | Target Lang: {target_lang}")
    except Exception as e:
        print(f"⚠️  [Config Error] Expected JSON config first, got: {e}")
        target_lang = "hi-IN"
        input_lang = "unknown"
    
    sarvam_client = AsyncSarvamAI(api_subscription_key=SARVAM_KEY)

    print(f"🌩  Connecting to Sarvam Cloud Streaming STT (model={SARVAM_STREAM_MODEL})...")
    try:
        async with sarvam_client.speech_to_text_streaming.connect(
            model=SARVAM_STREAM_MODEL,
            mode="transcribe",         
            language_code=input_lang,  # Use our dynamic input language
            input_audio_codec="pcm_s16le", # Raw PCM from Pi
            sample_rate="16000",
            # Removed high_vad_sensitivity to prevent background noise from keeping the mic open indefinitely
            vad_signals="true"
        ) as sarvam_ws:
            print("🌩  Sarvam STT Ready.")
            
            async def forward_audio_to_sarvam():
                chunk_count = 0
                async for pcm_chunk in websocket:
                    if type(pcm_chunk) == bytes:
                        chunk_count += 1
                        
                        if chunk_count % 20 == 0:
                            # Check amplitude to detect total silence from mic
                            import numpy as np
                            audio_np = np.frombuffer(pcm_chunk, dtype=np.int16)
                            max_amp = np.max(np.abs(audio_np)) if len(audio_np) > 0 else 0
                            print(f"📡 [Forwarding] Sent {chunk_count} chunks to Sarvam (Max Amp={max_amp})...")
                            
                        b64_audio = base64.b64encode(pcm_chunk).decode("utf-8")
                        await sarvam_ws.transcribe(
                            audio=b64_audio,
                            sample_rate=16000,
                            encoding="audio/wav"
                        )
                        
            async def receive_from_sarvam_and_translate():
                async for sarvam_msg in sarvam_ws:
                    msg_type = getattr(sarvam_msg, "type", "")
                    
                    if msg_type == "events":
                        # Output VAD signals to understand when Sarvam thinks we started/stopped speaking
                        event_data = getattr(sarvam_msg, "data", None)
                        signal = getattr(event_data, "signal_type", "") if event_data else ""
                        if signal == "START_SPEECH":
                            print("\n🟢 [VAD] Speech Detected...")
                        elif signal == "END_SPEECH":
                            print("🔴 [VAD] Target phrase captured. Analyzing...")
                            
                    # Check for transcription data
                    elif msg_type == "data":
                        msg_data = getattr(sarvam_msg, "data", None)
                        if msg_data:
                            text_str = getattr(msg_data, "transcript", "").strip()
                            
                            if text_str:
                                # 1. Fetch STT Latency straight from Sarvam
                                stt_ms = 0
                                metrics = getattr(msg_data, "metrics", None)
                                if metrics:
                                    stt_sec = getattr(metrics, "processing_latency", 0)
                                    if stt_sec:
                                        stt_ms = int(stt_sec * 1000)
                                
                                print(f"\n========================================")
                                print(f"🌩  [Sarvam STT] => {text_str} ({stt_ms}ms)")
                                
                                # 2. Measure Local Translation Latency
                                if target_lang == "en-IN":
                                    # No translation needed — English in, English out
                                    hi_text = text_str
                                    trans_ms = 0
                                    print(f"💻  [No Translation] => Target is English, skipping")
                                else:
                                    print(f"💻  [Local Translate] => Engine triggering... (to {target_lang})")
                                    t0_trans = time.time()
                                    hi_text, _ = local_inference_client.translate_text(
                                        text_str, source_lang="auto", target_lang=target_lang
                                    )
                                    trans_ms = int((time.time() - t0_trans) * 1000)
                                    print(f"💻  [Local Translate] => {hi_text} ({trans_ms}ms)")
                                
                                # 3. Generate Target Language TTS
                                print(f"🎵  [Local TTS] => Generating Audio for {target_lang}...")
                                t0_tts = time.time()
                                tts_wav = b""
                                try:
                                    tts_wav = synthesize(hi_text, lang_code=target_lang, backend="auto")
                                except Exception as e:
                                    print(f"⚠️  [TTS Error] => {e}")
                                tts_ms = int((time.time() - t0_tts) * 1000)
                                
                                if tts_wav:
                                    print(f"🎵  [Local TTS] => Generated {len(tts_wav)} bytes ({tts_ms}ms)")
                                
                                total_server_ms = stt_ms + trans_ms + tts_ms
                                print(f"⏱️  [LATENCY PIPELINE] STT: {stt_ms}ms | Trans: {trans_ms}ms | TTS: {tts_ms}ms | Total: {total_server_ms}ms")
                                print(f"========================================\n")
                                
                                # Send JSON data & timings back to Pi
                                await websocket.send(json.dumps({
                                    "transcript": text_str,
                                    "translation": hi_text,
                                    "is_final": True,
                                    "timings": {
                                        "stt_ms": stt_ms,
                                        "trans_ms": trans_ms,
                                        "tts_ms": tts_ms,
                                        "total_ms": total_server_ms
                                    }
                                }))
                                
                                # Send target language audio to Pi
                                if tts_wav:
                                    await websocket.send(tts_wav)
                                    
            # Run both the audio forwarding and result receiving at the exact same time
            try:
                await asyncio.gather(
                    forward_audio_to_sarvam(),
                    receive_from_sarvam_and_translate()
                )
            except websockets.exceptions.ConnectionClosed as e:
                print(f"⚠️  [Stream Closed] {e}")
                if "Invalid model" in str(e):
                    print("💡 Set SARVAM_STREAM_MODEL to 'saarika:v2.5' or 'saaras:v3'.")
            except Exception as e:
                print(f"⚠️  [Stream Error] {e}")
    except Exception as e:
        print(f"🌩  [Sarvam / Network Error] Stream disconnected: {e}")

def get_lan_ip():
    try:
        # Create a dummy socket to find the local IP routed to external networks
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

async def main():
    lan_ip = get_lan_ip()
    
    # Pre-load English Piper model so first TTS call is fast (~50ms vs ~520ms cold)
    print("⏳ Pre-loading English TTS model...")
    preload_piper("en-IN")
    print("✅ English TTS ready.")
    
    print("====================================")
    print("  EDGE STREAM SERVER (WebSockets)")
    print("  Listening on ws://0.0.0.0:5556")
    print(f"  LAN URL:   ws://{lan_ip}:5556")
    print("====================================")
    
    # Increased ping interval and timeout to 120 seconds to prevent Raspberry Pi disconnection
    server = await websockets.serve(
        handle_pi_connection, 
        "0.0.0.0", 
        5556,
        ping_interval=120,
        ping_timeout=120
    )
    await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown.")