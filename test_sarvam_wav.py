import asyncio
import os
import base64
import wave
import io
from sarvamai import AsyncSarvamAI

async def test():
    sarvam_client = AsyncSarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY", ""))
    async with sarvam_client.speech_to_text_streaming.connect(
        model="saaras:v3",
        mode="transcribe",
        language_code="hi-IN",
        input_audio_codec="pcm_s16le", # <--- I'll try changing this to wav too! Wait.
        sample_rate="16000"
    ) as sarvam_ws:
        print("Connected!")
        for _ in range(25):
            noise = os.urandom(6400) # 200ms of 16kHz 16bit PCM
            b64_audio = base64.b64encode(noise).decode("utf-8")
            await sarvam_ws.transcribe(audio=b64_audio, sample_rate=16000, encoding="audio/wav")
            await asyncio.sleep(0.2)
            
        print("Sent noise, waiting for response...")
        try:
            msg = await asyncio.wait_for(sarvam_ws.recv(), timeout=5.0)
            print("Response:", msg)
        except Exception as e:
            print("Timeout!")

asyncio.run(test())
