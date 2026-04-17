# Task: Integrate MMS TTS for Tamil

- [x] Add `MMS_TTS_VOICES` configuration to `config.py` for Tamil
- [x] Implement MMS logic in `tts_engine.py`
  - [x] Add imports (`VitsModel`, `AutoTokenizer`, `torch`)
  - [x] Add lazy loading function `_get_mms_model(lang_code)`
  - [x] Implement `_mms_synthesize(text, lang_code)` 
  - [x] Update `get_backend_name` to prioritize `mms` for `ta-IN`
  - [x] Update `synthesize` priority chain to invoke MMS
- [x] Verify synthesis and update walkthrough
