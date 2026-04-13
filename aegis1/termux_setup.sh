#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# AEGIS Termux Bootstrap Script
# Run this ONCE on a fresh Termux install to set up everything.
# Requires internet for first-time package + model downloads only.
#
# Usage (inside Termux):
#   chmod +x termux_setup.sh
#   ./termux_setup.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

AEGIS_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$AEGIS_DIR/models"
mkdir -p "$MODELS_DIR"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║          AEGIS Termux Setup                      ║"
echo "║   Offline-First Medical First-Response AI        ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Termux system packages ───────────────────────────────────────────
echo "▶ Step 1/6 — Installing Termux packages..."
pkg update -y -q
pkg install -y python curl tmux libexpat openssl-tool clang cmake make

# Allow Termux to access phone storage (for model files if downloaded externally)
if command -v termux-setup-storage &>/dev/null; then
    echo "  Requesting storage permission (tap Allow if prompted)..."
    termux-setup-storage 2>/dev/null || true
fi
echo "  ✓ Termux packages ready"

# ── Step 2: Python dependencies ──────────────────────────────────────────────
echo ""
echo "▶ Step 2/6 — Installing Python packages..."
pip install --upgrade pip --quiet
pip install -r "$AEGIS_DIR/requirements.txt" --quiet
echo "  ✓ Python packages installed"

# ── Step 3: llama-server binary ──────────────────────────────────────────────
echo ""
echo "▶ Step 3/6 — Downloading llama-server binary (AArch64)..."

LLAMA_BIN="$MODELS_DIR/llama-server"

if [[ -x "$LLAMA_BIN" ]]; then
    echo "  llama-server already present — skipping download"
else
    # Get latest release tag from GitHub API
    LATEST_TAG="$(curl -s https://api.github.com/repos/ggml-org/llama.cpp/releases/latest \
        | python -c "import sys,json; print(json.load(sys.stdin)['tag_name'])" 2>/dev/null || echo "b5310")"

    TAR_NAME="llama-${LATEST_TAG}-bin-android-aarch64.tar.gz"
    DOWNLOAD_URL="https://github.com/ggml-org/llama.cpp/releases/download/${LATEST_TAG}/${TAR_NAME}"

    echo "  Downloading: $DOWNLOAD_URL"
    curl -L "$DOWNLOAD_URL" -o /tmp/llama-android.tar.gz --progress-bar

    # Extract — find the llama-server binary wherever it lands in the archive
    tar -xzf /tmp/llama-android.tar.gz -C /tmp/
    EXTRACTED_BIN="$(find /tmp -name 'llama-server*' -type f 2>/dev/null | head -1)"

    if [[ -z "$EXTRACTED_BIN" ]]; then
        echo ""
        echo "WARNING: Could not auto-extract llama-server from archive."
        echo "Please manually extract and place at: $LLAMA_BIN"
        echo "Then run: chmod +x $LLAMA_BIN"
    else
        cp "$EXTRACTED_BIN" "$LLAMA_BIN"
        chmod +x "$LLAMA_BIN"
        echo "  ✓ llama-server installed at $LLAMA_BIN"
    fi

    rm -f /tmp/llama-android.tar.gz
fi

# ── Step 4: Gemma model instructions ─────────────────────────────────────────
echo ""
echo "▶ Step 4/6 — Gemma 4 E2B model"

if [[ -f "$MODELS_DIR/gemma4-e2b.gguf" ]]; then
    SIZE_MB=$(( $(stat -c%s "$MODELS_DIR/gemma4-e2b.gguf" 2>/dev/null || echo 0) / 1048576 ))
    echo "  ✓ Model found ($SIZE_MB MB)"
else
    echo ""
    echo "  Model file NOT found. You must download it manually:"
    echo ""
    echo "  Option A — Hugging Face (browser on phone):"
    echo "    1. Open: https://huggingface.co/google/gemma-4-e2b-it-GGUF"
    echo "    2. Accept license, download gemma-4-e2b-it-Q4_K_M.gguf"
    echo "    3. Move to: $MODELS_DIR/gemma4-e2b.gguf"
    echo ""
    echo "  Option B — hf_transfer in Termux (needs login token):"
    echo "    pip install huggingface_hub hf_transfer"
    echo "    HF_HUB_ENABLE_HF_TRANSFER=1 python -c \\"
    echo "      \"from huggingface_hub import hf_hub_download; \\"
    echo "       hf_hub_download('google/gemma-4-e2b-it-GGUF', \\"
    echo "       'gemma-4-e2b-it-Q4_K_M.gguf', \\"
    echo "       local_dir='$MODELS_DIR', local_dir_use_symlinks=False)\""
    echo "    mv $MODELS_DIR/gemma-4-e2b-it-Q4_K_M.gguf $MODELS_DIR/gemma4-e2b.gguf"
    echo ""
fi

# ── Step 5: .env file ─────────────────────────────────────────────────────────
echo "▶ Step 5/6 — Creating .env config..."

if [[ -f "$AEGIS_DIR/.env" ]]; then
    echo "  .env already exists — skipping"
else
    cat > "$AEGIS_DIR/.env" <<'ENVEOF'
# AEGIS environment configuration
# Edit this file before running start.sh

# ── Local llama-server ────────────────────────────────────────────────────────
# AEGIS_LLAMA_SERVER_BIN=./models/llama-server
# AEGIS_MODEL_PATH=./models/gemma4-e2b.gguf
# AEGIS_MMPROJ_PATH=./models/mmproj-gemma4-e2b.gguf
# AEGIS_LOCAL_HOST=127.0.0.1
# AEGIS_LOCAL_PORT=8080
# AEGIS_THREADS=4
# AEGIS_CTX_SIZE=4096

# ── Flask ─────────────────────────────────────────────────────────────────────
# AEGIS_HOST=0.0.0.0
# AEGIS_APP_PORT=5000

# ── Cloud inference (optional — AEGIS works fully offline without these) ──────
# AEGIS_CLOUD_API_URL=https://generativelanguage.googleapis.com/v1beta/models/gemma-3-27b-it:generateContent
# AEGIS_CLOUD_API_KEY=your_google_api_key_here
# AEGIS_CLOUD_MODEL_LABEL=gemma-27b-cloud

# ── Connectivity check ────────────────────────────────────────────────────────
# AEGIS_CONNECTIVITY_URL=http://clients3.google.com/generate_204
# AEGIS_REFRESH_INTERVAL=30
ENVEOF
    echo "  ✓ .env created — edit it to add your cloud API key (optional)"
fi

# ── Step 6: Battery optimization reminder ────────────────────────────────────
echo ""
echo "▶ Step 6/6 — Android battery optimization"
echo ""
echo "  IMPORTANT: Disable battery optimization for Termux so processes"
echo "  stay alive when the screen is off during a demo."
echo ""
echo "  Steps:"
echo "    Settings → Apps → Termux → Battery → select 'Unrestricted'"
echo ""

# ── Done ──────────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════╗"
echo "║  Setup complete!                                 ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Place the Gemma model at: $MODELS_DIR/gemma4-e2b.gguf"
echo "  2. Disable battery optimization for Termux (see above)"
echo "  3. Run: ./start.sh"
echo "  4. Open http://localhost:5000 in your phone browser"
echo "  5. For hotspot demo: turn on WiFi Hotspot, other devices open"
echo "     http://<your-phone-ip>:5000"
echo ""
