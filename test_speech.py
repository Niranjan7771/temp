import wave, io, base64, asyncio, os
from sarvamai import AsyncSarvamAI
from config import SARVAM_API_KEY
from tts_engine import synthesize

# generate speech
wav_bytes = synthesize("Hello my name is Niranjan and I am testing this microphone stream, checking if it works properly right now.", backend="auto", lang_code="en-IN")

async def test():
    sarvam_client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
    # The bytes have a WAV header, let's extract just the PCM payload by opening it via wave
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        pcm_bytes = wf.readframes(wf.getnframes())
        
    print(f"Got {len(pcm_bytes)} bytes of speech.")
    
    async with sarvam_client.speech_to_text_streaming.connect(
        model="saaras:v3",
        mode="transcribe",
        language_code="en-IN", # match
        input_audio_codec="pcm_s16le",
        sample_rate="16000"
    ) as sarvam_ws:
        print("Connected!")
        chunk_size = 6400  # 200 ms
        for i in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[i:i+chunk_size]
            if len(chunk) < chunk_size:
                chunk += b'\x00' * (chunk_size - len(chunk))
            
            b64_audio = base64.b64encode(chunk).decode("utf-8")
            await sarvam_ws.transcribe(audio=b64_audio, sample_rate=16000, encoding="audio/wav")
            await asyncio.sleep(0.2)
            
            try:
                # Poll immediately to see if any partials arrive
                msg = await asyncio.wait_for(sarvam_ws.recv(), timeout=0.1)
                print("Partial Response:", getattr(getattr(msg, "data", object()), "transcript", ""))
            except TimeoutError:
                pass

        print("Finished sending. Waiting up to 5 sec for final...")
        while True:
            try:
                msg = await asyncio.wait_for(sarvam_ws.recv(), timeout=5.0)
                print("Response:", msg)
            except Exception as e:
                break
                
asyncio.run(test())
