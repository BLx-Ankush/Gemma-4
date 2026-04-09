# AEGIS

AEGIS is an offline-first medical first-response assistant built with:

- `Gemma 4 E2B` via `llama-cpp-python`
- `MIRROR` safety-audit pass for response validation
- Drug cache enrichment for medicine-aware guidance
- Voice input/output pipeline (STT + TTS)
- Flask API + browser frontend

## Project Structure

- `core/`
  - `compute_router.py` connectivity/RAM mode detection
  - `gemma_core.py` local model loading and inference
  - `mirror.py` safety audit parsing and verdicting
- `modules/`
  - `drug_lookup.py` local medicine cache lookups
  - `voice_utils.py` transcription and speech synthesis helpers
  - `veda.py` end-to-end orchestration pipeline
  - `voice_handler.py` multi-turn voice session manager
- `app.py` Flask backend and API routes
- `templates/index.html` frontend page
- `static/style.css` and `static/app.js` frontend assets
- `tests/` unit tests for core, modules, and API

## Setup

From the workspace root:

```powershell
& "d:/Gemma 4/.venv/Scripts/python.exe" -m pip install -r aegis/requirements.txt
```

If you need to regenerate drug cache:

```powershell
& "d:/Gemma 4/.venv/Scripts/python.exe" aegis/setup_drug_cache.py
```

## Run

```powershell
& "d:/Gemma 4/.venv/Scripts/python.exe" -m aegis.app
```

Then open: `http://127.0.0.1:5000`

## Test

```powershell
& "d:/Gemma 4/.venv/Scripts/python.exe" -m pytest aegis/tests -q
```

## API Endpoints

- `GET /api/health`
- `POST /api/query` (`multipart/form-data`: `text`, optional `language`, optional `image`)
- `POST /api/voice/session`
- `DELETE /api/voice/session/<session_id>`
- `GET /api/voice/history/<session_id>`
- `POST /api/voice/query` (`multipart/form-data`: `audio`, optional `session_id`, optional `language`, optional `image`)

## Known Runtime Note

On some Windows setups, currently available `llama-cpp-python` builds may not support `gemma4` architecture in all versions. If model load fails with `unknown model architecture: 'gemma4'`, use a backend build/version that explicitly supports Gemma 4 GGUF for inference execution.
