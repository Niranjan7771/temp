import wave
import io
import base64
import asyncio
import os
from sarvamai import AsyncSarvamAI
from config import SARVAM_API_KEY

async def test():
    sarvam_client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
    
    with wave.open("audio.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
    
        # Let's just download a tiny real speech 16kHz sample of an English sentence
        # I'll just use curl to pull something from a known github repo or I'll just use TTS engine locally!
