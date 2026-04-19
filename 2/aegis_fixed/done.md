# AEGIS Completed Features (Current State)

## Verification snapshot
- Date: 2026-04-07
- Test status: 37 tests passing
- Verified with: `python -m pytest aegis/tests -q`

## Core modules implemented
- `core/compute_router.py`
  - Detects connectivity with HEAD request.
  - Uses strict 2-second network timeout and defaults safely to offline on exceptions.
  - Detects device RAM with `psutil`.
  - Exposes binary mode: `online` or `offline`.
  - Provides `refresh()` and `get_status()`.
- `core/gemma_core.py`
  - Uses HTTP-only routing to local llama-server and cloud API.
  - No Python-side model loading runtime dependency for local inference.
  - Supports text and image payload flow through OpenAI-compatible chat/completion endpoints.
  - Online cloud inference branch implemented (when cloud API config is present).
  - Automatic fallback to local llama-server if cloud call fails.
  - Backend/mode visibility exposed via `get_model_info()`.
  - Exposes `load_model()`, `infer()`, and `get_model_info()`.
- `core/mirror.py`
  - `MirrorReport` dataclass implemented.
  - `audit()` implemented with structured parser.
  - `apply_verdict()` implemented (`pass`/`warn`/`block`).
  - In-memory audit log implemented with `get_audit_log()`.

## Application modules implemented
- `modules/drug_lookup.py`
  - Loads local `data/drug_cache.json`.
  - Generic and brand-name lookup.
  - Drug extraction from text.
  - Prompt-ready drug context rendering.
  - Live OpenFDA lookup path when online.
  - Offline cache-first behavior preserved with live fallback.
- `modules/voice_utils.py`
  - Local STT via `faster-whisper`.
  - Local TTS via `pyttsx3`.
  - Test audio generation helpers.
- `modules/veda.py`
  - End-to-end orchestration for text/audio queries.
  - Language mapping and prompt composition.
  - Drug context injection.
  - MIRROR audit and verdict application.
  - Audio response generation with fail-safe fallback when pipeline errors occur.
- `modules/voice_handler.py`
  - Session lifecycle (`create/get/clear`).
  - Turn history and multi-turn voice flow.

## Backend implemented
- `app.py` Flask app and CORS enabled.
- Implemented routes:
  - `GET /`
  - `GET /api/health`
  - `GET /api/status`
  - `GET /status`
  - `POST /api/query`
  - `POST /veda`
  - `GET /api/mirror/log`
  - `GET /mirror/log`
  - `POST /api/voice/session`
  - `DELETE /api/voice/session/<session_id>`
  - `GET /api/voice/history/<session_id>`
  - `POST /api/voice/query`
  - `POST /voice`
- Upload storage, base64 image decode support, and response audio data-url conversion present.
- `/veda` now accepts either text or audio input.

## Runtime scripts implemented
- `setup.py`
  - Verifies required model/projector/binary assets.
  - Supports optional llama-server binary download.
- `start.sh`
  - Starts llama-server in background.
  - Waits for health endpoint readiness.
  - Starts Flask app on configured host/port.

## Frontend implemented
- `templates/index.html`
  - Query form and voice session controls.
  - Response and history sections.
- `static/style.css`
  - Responsive layout and visual styling.
- `static/app.js`
  - Health check integration.
  - Query submission.
  - Voice session start/end.
  - Browser recording with `MediaRecorder`.
  - Voice history rendering.

## Data and model assets present
- `data/drug_cache.json` present.
- `models/gemma4-e2b.gguf` present.
- `models/mmproj-gemma4-e2b.gguf` present.

## Working tests currently present
- `tests/test_compute_router.py`
- `tests/test_mirror.py`
- `tests/test_drug_lookup.py`
- `tests/test_voice_utils.py`
- `tests/test_veda.py`
- `tests/test_voice_handler.py`
- `tests/test_app.py`
- `tests/test_architecture_online_offline.py`
- `tests/test_gemma_core.py`
- `tests/test_offline.py`
