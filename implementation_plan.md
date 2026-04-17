# Add Meta MMS Offline TTS for Tamil

## Goal Description
The user wants to reduce the ~800ms generation latency for Tamil (`ta-IN`) voice output. Since there is no official Piper model for Tamil and `IndicF5` is not launching properly, the system falls back to the cloud-based Microsoft Edge-TTS engine, which causes major internet-bound delays.

To achieve <100ms latency without relying on the internet, we will integrate Meta's **Massively Multilingual Speech (MMS)** TTS model specifically for Tamil (`facebook/mms-tts-tam`). It runs completely offline using the existing `transformers` and PyTorch libraries you already have installed and generates high-quality audio almost instantly.

## User Review Required
> [!IMPORTANT]
> The MMS-TTS model is roughly ~145MB and will download automatically the first time `edge_stream_server.py` runs and tries to synthesize Tamil. We are currently downloading it in the background as a test. Please confirm you are okay dropping Edge-TTS for this local framework.

## Proposed Changes

### `tts_engine.py`
- Add a new backend option specifically for MMS models (`mms`).
- Introduce a lazy-loading setup `_get_mms_model(lang_code)` which uses `VitsModel` and `AutoTokenizer` from HuggingFace to load `facebook/mms-tts-tam`.
- Create the `_mms_synthesize(text, lang_code)` function to generate raw WAV bytes at 16kHz directly from the PyTorch tensor outputs.
- Update `get_backend_name()` to return `mms` for `ta-IN` as a higher priority over `edge-tts`.

### `config.py`
- Add a configuration block `MMS_TTS_VOICES` mapped to their HuggingFace paths (e.g., `"ta-IN": "facebook/mms-tts-tam"`).

## Open Questions
> [!NOTE]  
> Are there any other languages besides Tamil that you want to test with this Meta MMS framework, or is just Tamil fine for now?

## Verification Plan

### Automated Tests
- The background command `mms-tts-tam` latency benchmark script will print `MMS Tamil took XX.Xms`. We expect this number to be `< 150ms`.

### Manual Verification
- We will replace `tts_engine.py` and have you restart `python edge_stream_server.py`.
- You will test speaking English and getting a Tamil response; the turnaround time should be near instant without internet lag.
