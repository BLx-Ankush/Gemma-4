#!/usr/bin/env python3
"""
AEGIS Secure Launcher

Prompts for a startup password and launches the local llama-server + Flask app.
Prints local and LAN access links for self-use and nearby users on the same network.
"""

from __future__ import annotations

import getpass
import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import requests


PASSWORD = "0007"
MAX_ATTEMPTS = 3

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent


def _default_models_dir() -> Path:
    local_models = ROOT_DIR / "models"
    if local_models.exists():
        return local_models

    workspace_models = PROJECT_ROOT / "models"
    if workspace_models.exists():
        return workspace_models

    return local_models


DEFAULT_MODELS_DIR = _default_models_dir()
DEFAULT_LLAMA_BIN = DEFAULT_MODELS_DIR / ("llama-server.exe" if os.name == "nt" else "llama-server")
DEFAULT_MODEL_PATH = DEFAULT_MODELS_DIR / "gemma4-e2b.gguf"
DEFAULT_MMPROJ_PATH = DEFAULT_MODELS_DIR / "mmproj-gemma4-e2b.gguf"

LLAMA_LOG = ROOT_DIR / "aegis-llama.log"
FLASK_LOG = ROOT_DIR / "aegis-flask.log"


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _safe_int_env(name: str, default: int, minimum: int = 0) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(minimum, value)


def _safe_float_env(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError:
        value = default
    return max(minimum, value)


def _get_launch_password(cli_password: str | None) -> str:
    env_password = os.environ.get("AEGIS_LAUNCH_PASSWORD", "").strip()
    if cli_password:
        return cli_password.strip()
    if env_password:
        return env_password
    return PASSWORD


def _resolve_existing_path(raw_value: str, fallback: Path, *, allow_windows_exe: bool = False) -> Path:
    raw_path = Path(raw_value).expanduser()
    candidates: list[Path] = []

    def add_candidate(path_value: Path) -> None:
        candidates.append(path_value)
        if allow_windows_exe and os.name == "nt" and path_value.suffix == "":
            candidates.append(path_value.with_suffix(".exe"))

    if raw_path.is_absolute():
        add_candidate(raw_path)
    else:
        add_candidate((ROOT_DIR / raw_path).resolve())
        add_candidate((PROJECT_ROOT / raw_path).resolve())
        add_candidate((DEFAULT_MODELS_DIR / raw_path.name).resolve())

    add_candidate(fallback.resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def _check_password(expected_password: str) -> bool:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        entered = getpass.getpass("Enter AEGIS launch password: ")
        if entered == expected_password:
            return True
        remaining = MAX_ATTEMPTS - attempt
        if remaining > 0:
            print(f"Incorrect password. {remaining} attempt(s) remaining.")
    return False


def _is_http_ok(url: str, timeout_sec: float = 2.0) -> bool:
    try:
        response = requests.get(url, timeout=timeout_sec)
        return response.status_code < 500
    except Exception:
        return False


def _wait_for_url(url: str, timeout_sec: int) -> bool:
    for _ in range(timeout_sec):
        if _is_http_ok(url):
            return True
        time.sleep(1)
    return False


def _wait_for_llama_ready(base_url: str, timeout_sec: int, warmup_timeout_sec: int) -> bool:
    completion_url = f"{base_url}/completion"
    for _ in range(timeout_sec):
        health_ok = _is_http_ok(f"{base_url}/health")
        slots_ok = _is_http_ok(f"{base_url}/slots")
        if health_ok and slots_ok:
            try:
                response = requests.post(
                    completion_url,
                    json={"prompt": "warmup", "n_predict": 1, "temperature": 0, "stream": False},
                    timeout=max(2, warmup_timeout_sec),
                )
                if response.status_code < 500:
                    return True
            except Exception:
                pass
        time.sleep(1)
    return False


def _guess_lan_ip() -> str:
    # Preferred path: discover outbound interface IP
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
    except Exception:
        pass

    # Fallback path: hostname resolution
    try:
        host_ips = socket.gethostbyname_ex(socket.gethostname())[2]
        for ip in host_ips:
            if ip and not ip.startswith("127."):
                return ip
    except Exception:
        pass

    return ""


def _build_llama_args(llama_bin: Path, model_path: Path, mmproj_path: Path) -> list[str]:
    host = os.environ.get("AEGIS_LOCAL_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = _safe_int_env("AEGIS_LOCAL_PORT", 8080, minimum=1)
    ctx_size = _safe_int_env("AEGIS_CTX_SIZE", 4096, minimum=512)
    threads = _safe_int_env("AEGIS_THREADS", os.cpu_count() or 4, minimum=1)
    parallel = _safe_int_env("AEGIS_LLAMA_PARALLEL", 1, minimum=1)

    args = [
        str(llama_bin),
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        str(model_path),
        "--ctx-size",
        str(ctx_size),
        "--threads",
        str(threads),
        "--n-gpu-layers",
        "0",
        "--parallel",
        str(parallel),
        "--cont-batching",
    ]

    if mmproj_path.exists():
        args.extend(["--mmproj", str(mmproj_path)])

    if os.environ.get("AEGIS_ENABLE_MLOCK", "1").strip().lower() in {"1", "true", "yes", "on"}:
        args.append("--mlock")

    if os.environ.get("AEGIS_DISABLE_MMAP", "0").strip().lower() in {"1", "true", "yes", "on"}:
        args.append("--no-mmap")

    return args


def main() -> int:
    _load_env_file(ROOT_DIR / ".env")

    parser = argparse.ArgumentParser(description="Launch AEGIS securely with a startup password.")
    parser.add_argument("--password", help="Optional password override. Defaults to AEGIS_LAUNCH_PASSWORD or 0007.")
    args = parser.parse_args()

    launch_password = _get_launch_password(args.password)

    if not _check_password(launch_password):
        print("Access denied.")
        return 1

    host = os.environ.get("AEGIS_HOST", "0.0.0.0").strip() or "0.0.0.0"
    app_port = _safe_int_env("AEGIS_APP_PORT", 5000, minimum=1)
    llama_host = os.environ.get("AEGIS_LOCAL_HOST", "127.0.0.1").strip() or "127.0.0.1"
    llama_port = _safe_int_env("AEGIS_LOCAL_PORT", 8080, minimum=1)
    os.environ["AEGIS_LOCAL_LLM_URL"] = f"http://{llama_host}:{llama_port}"

    llama_timeout = _safe_int_env("AEGIS_LLAMA_STARTUP_TIMEOUT_SEC", 120, minimum=10)
    warmup_timeout = _safe_int_env("AEGIS_WARMUP_REQUEST_TIMEOUT_SEC", 8, minimum=2)
    flask_timeout = _safe_int_env("AEGIS_FLASK_STARTUP_TIMEOUT_SEC", 60, minimum=10)

    llama_bin = _resolve_existing_path(
        os.environ.get("AEGIS_LLAMA_SERVER_BIN", str(DEFAULT_LLAMA_BIN)),
        DEFAULT_LLAMA_BIN,
        allow_windows_exe=True,
    )
    model_path = _resolve_existing_path(
        os.environ.get("AEGIS_MODEL_PATH", str(DEFAULT_MODEL_PATH)),
        DEFAULT_MODEL_PATH,
    )
    mmproj_path = _resolve_existing_path(
        os.environ.get("AEGIS_MMPROJ_PATH", str(DEFAULT_MMPROJ_PATH)),
        DEFAULT_MMPROJ_PATH,
    )

    llama_proc: Optional[subprocess.Popen] = None
    flask_proc: Optional[subprocess.Popen] = None
    owned_processes: list[subprocess.Popen] = []

    llama_url = f"http://{llama_host}:{llama_port}"
    app_health_url = f"http://127.0.0.1:{app_port}/api/health"

    try:
        if _is_http_ok(f"{llama_url}/health"):
            print("llama-server is already running. Reusing existing process.")
        else:
            if not llama_bin.exists():
                print(f"ERROR: llama-server binary not found: {llama_bin}")
                return 1
            if not model_path.exists():
                print(f"ERROR: Model file not found: {model_path}")
                return 1

            llama_args = _build_llama_args(llama_bin, model_path, mmproj_path)
            llama_log_fh = open(LLAMA_LOG, "a", encoding="utf-8")
            llama_proc = subprocess.Popen(
                llama_args,
                cwd=str(ROOT_DIR),
                stdout=llama_log_fh,
                stderr=subprocess.STDOUT,
            )
            owned_processes.append(llama_proc)
            print("Starting llama-server...")
            if not _wait_for_llama_ready(llama_url, llama_timeout, warmup_timeout):
                print(f"ERROR: llama-server did not become ready in {llama_timeout}s.")
                print(f"Check logs: {LLAMA_LOG}")
                return 1
            print("llama-server is ready.")

        if _is_http_ok(app_health_url):
            print("AEGIS Flask API is already running. Reusing existing process.")
        else:
            flask_log_fh = open(FLASK_LOG, "a", encoding="utf-8")
            flask_proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "aegis.app",
                    "--host",
                    host,
                    "--port",
                    str(app_port),
                ],
                cwd=str(PROJECT_ROOT),
                stdout=flask_log_fh,
                stderr=subprocess.STDOUT,
            )
            owned_processes.append(flask_proc)
            print("Starting AEGIS Flask API...")
            if not _wait_for_url(app_health_url, flask_timeout):
                print(f"ERROR: Flask API did not become ready in {flask_timeout}s.")
                print(f"Check logs: {FLASK_LOG}")
                return 1
            print("AEGIS Flask API is ready.")

        lan_ip = _guess_lan_ip()
        print("\nAEGIS is running.")
        print(f"Your local access: http://127.0.0.1:{app_port}")
        if lan_ip:
            print(f"Local network access for others: http://{lan_ip}:{app_port}")
        else:
            print("Local network access for others: unable to auto-detect LAN IP.")
        print(f"llama log: {LLAMA_LOG}")
        print(f"flask log: {FLASK_LOG}")
        print("If phone access fails: ensure both devices are on the same Wi-Fi/hotspot, disable VPN on phone and PC, and allow inbound TCP port 5000 in Windows Firewall (Admin PowerShell).")

        if not owned_processes:
            return 0

        print("\nPress Ctrl+C to stop services started by this launcher.")
        while True:
            time.sleep(1)
            for proc in list(owned_processes):
                if proc.poll() is not None:
                    print("A launched service exited unexpectedly. Stopping launcher.")
                    return 1

    except KeyboardInterrupt:
        print("\nStopping AEGIS services...")
        return 0
    finally:
        for proc in reversed(owned_processes):
            if proc.poll() is None:
                try:
                    proc.send_signal(signal.SIGINT)
                    proc.wait(timeout=10)
                except Exception:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
        for proc in reversed(owned_processes):
            if proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
