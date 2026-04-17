import wave, io, base64, asyncio, os
from sarvamai import AsyncSarvamAI
from tts_engine import synthesize
from config import SARVAM_API_KEY

wav_bytes = synthesize("Hello my name is Niranjan and I am testing this microphone stream.", backend="auto", lang_code="en-IN")

async def test():
    sarvam_client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        pcm_bytes = wf.readframes(wf.getnframes())
    
    # ADD 3 seconds of SILENCE at the end to trigger END_SPEECH!
    pcm_bytes += b'\x00' * (16000 * 2 * 3)
    
    async with sarvam_client.speech_to_text_streaming.connect(
        model="saaras:v3",
        mode="transcribe",
        language_code="en-IN",
        input_audio_codec="pcm_s16le",
        sample_rate="16000",
        vad_signals="true"
    ) as sarvam_ws:
        print("Connected!")
        chunk_size = 6400  # 200 ms
        for i in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[i:i+chunk_size]
            b64_audio = base64.b64encode(chunk).decode("utf-8")
            await sarvam_ws.transcribe(audio=b64_audio, sample_rate=16000, encoding="audio/wav")
            await asyncio.sleep(0.2)
            try:
                msg = await asyncio.wait_for(sarvam_ws.recv(), timeout=0.1)
                print("Partial / Final response during stream:", msg)
            except TimeoutError:
                pass
                
asyncio.run(test())
