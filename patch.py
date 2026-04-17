import re
with open("edge_stream_server.py", "r") as f:
    code = f.read()

fixed = code.replace("""                        await sarvam_ws.transcribe(
                            audio=b64_audio,
                            sample_rate=16000,
                        encoding="audio/wav"
                async for sarvam_msg in sarvam_ws:""",
"""                        await sarvam_ws.transcribe(
                            audio=b64_audio,
                            sample_rate=16000,
                            encoding="audio/wav"
                        )
                        
            async def receive_from_sarvam_and_translate():
                async for sarvam_msg in sarvam_ws:""")

with open("edge_stream_server.py", "w") as f:
    f.write(fixed)
