import os
import subprocess
import sys

import requests

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found. Installing tqdm...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm


MODELS_DIR = "aegis/models/"
GEMMA_MODEL_PATH = os.path.join(MODELS_DIR, "gemma4-e2b.gguf")

os.makedirs(MODELS_DIR, exist_ok=True)


def download_file(url, destination):
    if os.path.exists(destination):
        print(f"File already exists, skipping: {destination}")
        return

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    filename = os.path.basename(destination)

    with open(destination, "wb") as file_handle, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=filename,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file_handle.write(chunk)
                progress_bar.update(len(chunk))

    print(f"Downloaded: {filename}")


def verify_gemma_model():
    if os.path.exists(GEMMA_MODEL_PATH):
        file_size_gb = os.path.getsize(GEMMA_MODEL_PATH) / (1024 ** 3)
        print(f"Model size: {file_size_gb:.2f} GB")
    else:
        print("Model not found at expected path.")


if __name__ == "__main__":
    print(
        """=== MODEL DOWNLOAD INSTRUCTIONS ===

1. Go to https://huggingface.co/google/gemma-4-e2b
2. Accept the license agreement
3. Download the quantized GGUF file (Q4_K_M variant)
4. Save it to: aegis/models/gemma4-e2b.gguf

For faster-whisper, the model will auto-download on first use.

5. After downloading the Gemma model, run this script again 
   to verify: python aegis/setup_models.py --verify"""
    )

    if "--verify" in sys.argv:
        verify_gemma_model()