"""
AEGIS Flask Application
Serves the web UI and API endpoints for all AEGIS modules.
Runs on 0.0.0.0 so devices connected to the phone hotspot can reach it.
"""

import base64
import os
import uuid
import time
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

from aegis.core import compute_router, gemma_core, mirror
from aegis.modules import veda, voice_handler


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
IMAGE_UPLOADS_DIR = UPLOADS_DIR / "images"
AUDIO_UPLOADS_DIR = UPLOADS_DIR / "audio"
REQUEST_TIMEOUT_SEC = max(10, int(os.environ.get("AEGIS_REQUEST_TIMEOUT_SEC", "30")))
_REQUEST_EXECUTOR = ThreadPoolExecutor(max_workers=4)
APP_STARTED_AT = int(time.time())
APP_BUILD_ID = os.environ.get("AEGIS_BUILD_ID", "2026-04-11-r2")


def _run_with_timeout(func, timeout_sec: int):
    future = _REQUEST_EXECUTOR.submit(func)
    try:
        return future.result(timeout=max(1, int(timeout_sec)))
    except FuturesTimeout as exc:
        future.cancel()
        raise TimeoutError(f"Request timed out after {timeout_sec}s") from exc


def _ensure_dirs() -> None:
    IMAGE_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def _save_upload(file_storage, directory: Path) -> str:
    suffix = Path(file_storage.filename or "").suffix.lower()
    if not suffix:
        suffix = ".bin"
    filename = f"{uuid.uuid4().hex}{suffix}"
    output_path = directory / filename
    file_storage.save(output_path)
    return str(output_path)


def _audio_to_data_url(audio_path: str) -> str:
    if not audio_path or not os.path.exists(audio_path):
        return ""
    with open(audio_path, "rb") as file_handle:
        encoded = base64.b64encode(file_handle.read()).decode("utf-8")
    return f"data:audio/wav;base64,{encoded}"


def _save_image_b64(image_b64: str, directory: Path) -> str:
    if not image_b64:
        return ""
    payload = image_b64.strip()
    suffix = ".jpg"
    if payload.startswith("data:image") and "," in payload:
        header, payload = payload.split(",", 1)
        if "image/png" in header:
            suffix = ".png"
        elif "image/webp" in header:
            suffix = ".webp"
    filename = f"{uuid.uuid4().hex}{suffix}"
    output_path = directory / filename
    output_path.write_bytes(base64.b64decode(payload))
    return str(output_path)


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    CORS(app)
    _ensure_dirs()

    @app.after_request
    def add_cache_headers(response):
        path = request.path or ""
        if (
            path == "/"
            or path == "/sw.js"
            or path == "/manifest.json"
            or path.startswith("/static/")
        ):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    # ── PWA manifest ──────────────────────────────────────────────────────────
    @app.route("/manifest.json")
    def manifest():
        return send_from_directory(BASE_DIR / "static", "manifest.json")

    @app.route("/sw.js")
    def service_worker():
        return send_from_directory(BASE_DIR / "static", "sw.js",
                                   mimetype="application/javascript")

    # ── Main UI ───────────────────────────────────────────────────────────────
    @app.route("/")
    def index():
        return render_template("index.html")

    # ── Status / health ───────────────────────────────────────────────────────
    @app.route("/api/health", methods=["GET"])
    @app.route("/api/status", methods=["GET"])
    @app.route("/status", methods=["GET"])
    def health():
        # Only /status triggers a real refresh; inference routes use get_status()
        compute = compute_router.force_refresh()
        model_info = gemma_core.get_model_info()
        return jsonify(
            {
                "ok": True,
                "compute": compute,
                "model": model_info,
                "runtime": {
                    "build_id": APP_BUILD_ID,
                    "pid": os.getpid(),
                    "started_at": APP_STARTED_AT,
                },
            }
        )

    # ── VEDA: text + image medical query ─────────────────────────────────────
    @app.route("/api/query", methods=["POST"])
    @app.route("/veda", methods=["POST"])
    def query():
        # Use cached status — no extra network ping per request
        data = request.get_json(silent=True) if request.is_json else {}

        text_query = request.form.get("text", "").strip()
        if not text_query and isinstance(data, dict):
            text_query = str(data.get("text", "")).strip()

        language = request.form.get("language", "English").strip() or "English"
        if language == "English" and isinstance(data, dict):
            language = str(data.get("language", "English")).strip() or "English"

        image_path = None
        image_file = request.files.get("image")
        if image_file and image_file.filename:
            image_path = _save_upload(image_file, IMAGE_UPLOADS_DIR)
        elif isinstance(data, dict) and data.get("image_b64"):
            try:
                image_path = _save_image_b64(str(data.get("image_b64")), IMAGE_UPLOADS_DIR)
            except Exception:
                image_path = None

        audio_path = None
        audio_file = request.files.get("audio")
        if audio_file and audio_file.filename:
            audio_path = _save_upload(audio_file, AUDIO_UPLOADS_DIR)

        if not text_query and not audio_path:
            return jsonify({"ok": False, "error": "Provide either 'text' or 'audio'."}), 400

        timeout_sec = max(
            10,
            int(os.environ.get("AEGIS_VEDA_REQUEST_TIMEOUT_SEC", str(REQUEST_TIMEOUT_SEC))),
        )

        try:
            result = _run_with_timeout(
                lambda: veda.process_medical_query(
                    image_path=image_path,
                    audio_path=audio_path,
                    text_query=text_query if text_query else None,
                    language=language,
                ),
                timeout_sec,
            )
            return jsonify({
                "ok": True,
                "data": {
                    "query_text": result.query_text,
                    "response_text": result.response_text,
                    "detected_language": result.detected_language,
                    "drug_context": result.drug_context,
                    "processing_time_ms": result.processing_time_ms,
                    "mirror": {
                        "confidence_score": result.mirror_report.confidence_score,
                        "flags": result.mirror_report.flags,
                        "verdict": result.mirror_report.verdict,
                        "reasoning_trace": result.mirror_report.reasoning_trace,
                        "audit_time_ms": result.mirror_report.audit_time_ms,
                    },
                    "audio_response_data_url": _audio_to_data_url(result.audio_response_path),
                    "mode": compute_router.get_status()["mode"],
                },
            })
        except TimeoutError as exc:
            return jsonify({"ok": False, "error": str(exc), "timeout": True}), 504
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    # ── MIRROR audit log ──────────────────────────────────────────────────────
    @app.route("/api/mirror/log", methods=["GET"])
    @app.route("/mirror/log", methods=["GET"])
    def mirror_log():
        try:
            limit = int(request.args.get("limit", "20"))
        except ValueError:
            limit = 20
        return jsonify({"ok": True, "data": mirror.get_audit_log(limit=limit)})

    # ── Voice session management ──────────────────────────────────────────────
    @app.route("/api/voice/session", methods=["POST"])
    def create_voice_session():
        session_id = voice_handler.create_session()
        return jsonify({"ok": True, "session_id": session_id})

    @app.route("/api/voice/session/<session_id>", methods=["DELETE"])
    def clear_voice_session(session_id: str):
        cleared = voice_handler.clear_session(session_id)
        return jsonify({"ok": True, "cleared": cleared, "session_id": session_id})

    @app.route("/api/voice/history/<session_id>", methods=["GET"])
    def voice_history(session_id: str):
        history = voice_handler.get_session_history(session_id)
        return jsonify({"ok": True, "session_id": session_id, "history": history})

    # ── Voice query ───────────────────────────────────────────────────────────
    @app.route("/api/voice/query", methods=["POST"])
    @app.route("/voice", methods=["POST"])
    def voice_query():
        language = request.form.get("language", "English").strip() or "English"
        session_id = request.form.get("session_id", "").strip() or None

        audio_file = request.files.get("audio")
        if not audio_file or not audio_file.filename:
            return jsonify({"ok": False, "error": "Field 'audio' is required."}), 400

        audio_path = _save_upload(audio_file, AUDIO_UPLOADS_DIR)

        image_path = None
        image_file = request.files.get("image")
        if image_file and image_file.filename:
            image_path = _save_upload(image_file, IMAGE_UPLOADS_DIR)

        timeout_sec = max(
            10,
            int(os.environ.get("AEGIS_VOICE_REQUEST_TIMEOUT_SEC", str(REQUEST_TIMEOUT_SEC))),
        )

        try:
            result = _run_with_timeout(
                lambda: voice_handler.process_voice_turn(
                    audio_path=audio_path,
                    session_id=session_id,
                    image_path=image_path,
                    language=language,
                ),
                timeout_sec,
            )
            result["audio_response_data_url"] = _audio_to_data_url(result.get("audio_response_path", ""))
            result["mode"] = compute_router.get_status()["mode"]
            return jsonify({"ok": True, "data": result})
        except TimeoutError as exc:
            return jsonify({"ok": False, "error": str(exc), "timeout": True}), 504
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    return app


app = create_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AEGIS Flask server")
    parser.add_argument("--host", default=os.environ.get("AEGIS_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AEGIS_PORT", "5000")))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=args.debug)
