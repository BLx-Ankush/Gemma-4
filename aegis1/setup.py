import argparse
import os
from pathlib import Path

import requests
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"

MODEL_PATH = MODELS_DIR / "gemma4-e2b.gguf"
MMPROJ_PATH = MODELS_DIR / "mmproj-gemma4-e2b.gguf"
LLAMA_SERVER_PATH = MODELS_DIR / "llama-server"
LLAMA_SERVER_EXE_PATH = MODELS_DIR / "llama-server.exe"


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path) -> None:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with destination.open("wb") as fh, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=f"Downloading {destination.name}",
    ) as progress:
        for chunk in response.iter_content(chunk_size=8192):
            if not chunk:
                continue
            fh.write(chunk)
            progress.update(len(chunk))


def verify_assets() -> bool:
    ok = True
    if LLAMA_SERVER_PATH.exists():
        llama_path = LLAMA_SERVER_PATH
    elif LLAMA_SERVER_EXE_PATH.exists():
        llama_path = LLAMA_SERVER_EXE_PATH
    else:
        llama_path = LLAMA_SERVER_PATH

    checks = [
        (MODEL_PATH, "Main Gemma model"),
        (MMPROJ_PATH, "Vision projector"),
        (llama_path, "llama-server binary"),
    ]

    print("=== AEGIS Asset Verification ===")
    for path, label in checks:
        if path.exists():
            size_gb = path.stat().st_size / (1024 ** 3)
            print(f"{label}: OK - {path} ({size_gb:.2f} GB)")
        else:
            ok = False
            print(f"{label}: MISSING - {path}")

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="AEGIS setup helper")
    parser.add_argument("--verify", action="store_true", help="Verify required model and binary files")
    parser.add_argument("--llama-server-url", type=str, default="", help="Optional URL to download prebuilt llama-server")
    parser.add_argument("--download-llama-server", action="store_true", help="Download llama-server from --llama-server-url")
    args = parser.parse_args()

    ensure_dirs()

    if args.download_llama_server:
        if not args.llama_server_url:
            raise SystemExit("--download-llama-server requires --llama-server-url")
        download_file(args.llama_server_url, LLAMA_SERVER_PATH)
        os.chmod(LLAMA_SERVER_PATH, 0o755)
        print(f"Downloaded binary: {LLAMA_SERVER_PATH}")

    if args.verify:
        success = verify_assets()
        raise SystemExit(0 if success else 1)

    print("=== AEGIS Setup Notes ===")
    print("1. Place gemma4-e2b.gguf in aegis/models/")
    print("2. Place mmproj-gemma4-e2b.gguf in aegis/models/")
    print("3. Place precompiled llama-server binary in aegis/models/llama-server (or llama-server.exe on Windows)")
    print("4. Run: python setup.py --verify")


if __name__ == "__main__":
    main()
