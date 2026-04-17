import os
import asyncio

import base64
from sarvamai import AsyncSarvamAI

import wave

async def test():
    sarvam_client = AsyncSarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY", ""))
    async with sarvam_client.speech_to_text_streaming.connect(
        model="saaras:v3",
        mode="transcribe",
        language_code="hi-IN",
        input_audio_codec="pcm_s16le",
        sample_rate="16000",
        vad_signals="true"
    ) as sarvam_ws:
        print("Connected!")
        # Send 5 seconds of pseudo-random noise to force VAD to trigger
        
        for _ in range(25):
            noise = os.urandom(6400) # 200ms of 16kHz 16bit PCM
            b64_audio = base64.b64encode(noise).decode("utf-8")
            await sarvam_ws.transcribe(audio=b64_audio, sample_rate=16000, encoding="audio/wav")
            await asyncio.sleep(0.2)
            
        print("Sent noise, waiting for response...")
        while True:
            # Wait for Sarvam to respond
            msg = await asyncio.wait_for(sarvam_ws.recv(), timeout=5.0)
            print("Response:", msg)
            
asyncio.run(test())
