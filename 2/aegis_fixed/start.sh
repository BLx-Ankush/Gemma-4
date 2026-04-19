#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# AEGIS start.sh — Termux / Android production startup script
#
# Usage:
#   ./start.sh              # start with tmux (Termux default)
#   ./start.sh --no-tmux    # run in foreground without tmux (CI / desktop)
#   ./start.sh --stop       # kill any running AEGIS tmux sessions
#
# Environment overrides (set in .env or export before running):
#   AEGIS_LLAMA_SERVER_BIN  path to llama-server binary
#   AEGIS_MODEL_PATH        path to .gguf model file
#   AEGIS_MMPROJ_PATH       path to multimodal projector .gguf (optional)
#   AEGIS_LOCAL_HOST        llama-server bind host  (default 127.0.0.1)
#   AEGIS_LOCAL_PORT        llama-server bind port  (default 8080)
#   AEGIS_APP_PORT          Flask bind port         (default 5000)
#   AEGIS_THREADS           CPU threads for llama   (default: nproc)
#   AEGIS_CTX_SIZE          context window tokens   (default 4096)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# ── Load .env if present ──────────────────────────────────────────────────────
if [[ -f "$ROOT_DIR/.env" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ROOT_DIR/.env"
    set +a
    echo "[aegis] Loaded .env"
fi

# ── Defaults ──────────────────────────────────────────────────────────────────
LLAMA_BIN="${AEGIS_LLAMA_SERVER_BIN:-$ROOT_DIR/models/llama-server}"
MODEL_PATH="${AEGIS_MODEL_PATH:-$ROOT_DIR/models/gemma4-e2b.gguf}"
MMPROJ_PATH="${AEGIS_MMPROJ_PATH:-$ROOT_DIR/models/mmproj-gemma4-e2b.gguf}"
LLAMA_HOST="${AEGIS_LOCAL_HOST:-127.0.0.1}"
LLAMA_PORT="${AEGIS_LOCAL_PORT:-8080}"
APP_PORT="${AEGIS_APP_PORT:-5000}"
CTX_SIZE="${AEGIS_CTX_SIZE:-4096}"
USE_TMUX=true

# Auto-detect thread count — nproc works on Android/Termux
if [[ -z "${AEGIS_THREADS:-}" ]]; then
    THREADS="$(nproc 2>/dev/null || echo 4)"
else
    THREADS="$AEGIS_THREADS"
fi

# ── Argument parsing ──────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --no-tmux) USE_TMUX=false ;;
        --stop)
            echo "[aegis] Killing AEGIS tmux sessions..."
            tmux kill-session -t aegis-llama 2>/dev/null || true
            tmux kill-session -t aegis-flask 2>/dev/null || true
            echo "[aegis] Done."
            exit 0
            ;;
    esac
done

# ── Validate required assets ──────────────────────────────────────────────────
if [[ ! -x "$LLAMA_BIN" ]]; then
    echo ""
    echo "ERROR: llama-server binary not found or not executable."
    echo "  Expected: $LLAMA_BIN"
    echo ""
    echo "On Termux/Android, download the AArch64 binary:"
    echo "  pkg install curl"
    echo "  curl -L https://github.com/ggml-org/llama.cpp/releases/latest/download/llama-b5310-bin-android-aarch64.tar.gz \\"
    echo "       -o /tmp/llama.tar.gz"
    echo "  tar -xzf /tmp/llama.tar.gz -C $ROOT_DIR/models/ --wildcards '*llama-server*'"
    echo "  mv $ROOT_DIR/models/llama-server-android-aarch64 $ROOT_DIR/models/llama-server 2>/dev/null || true"
    echo "  chmod +x $ROOT_DIR/models/llama-server"
    echo ""
    exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
    echo ""
    echo "ERROR: Model file not found: $MODEL_PATH"
    echo "Run: python setup_models.py --verify"
    echo ""
    exit 1
fi

export AEGIS_LOCAL_LLM_URL="http://${LLAMA_HOST}:${LLAMA_PORT}"

# ── Build llama-server argument list ─────────────────────────────────────────
LLAMA_ARGS=(
    -m "$MODEL_PATH"
    --host "$LLAMA_HOST"
    --port "$LLAMA_PORT"
    --ctx-size "$CTX_SIZE"
    --threads "$THREADS"
    --n-gpu-layers 0
    --parallel 2
    --cont-batching
)

if [[ -f "$MMPROJ_PATH" ]]; then
    LLAMA_ARGS+=(--mmproj "$MMPROJ_PATH")
    echo "[aegis] Vision projector found — multimodal enabled"
else
    echo "[aegis] No mmproj file — text-only mode"
fi

# ── Wait for llama-server health ──────────────────────────────────────────────
wait_for_llama() {
    local health_url="${AEGIS_LOCAL_LLM_URL}/health"
    echo "[aegis] Waiting for llama-server at $health_url ..."
    for i in $(seq 1 40); do
        if curl -fsS "$health_url" >/dev/null 2>&1; then
            echo "[aegis] llama-server healthy after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "ERROR: llama-server did not become ready after 40s. Check /tmp/aegis-llama.log"
    return 1
}

# ── Print access URLs for hotspot sharing ─────────────────────────────────────
print_access_info() {
    echo ""
    echo "═══════════════════════════════════════════════════"
    echo "  AEGIS is running"
    echo "═══════════════════════════════════════════════════"
    echo "  Local:    http://127.0.0.1:${APP_PORT}"

    # Try wlan0 first (hotspot network on Android), then fallback
    LOCAL_IP="$(ip addr show wlan0 2>/dev/null | grep 'inet ' | awk '{print $2}' | cut -d/ -f1 | head -1)"
    if [[ -z "$LOCAL_IP" ]]; then
        LOCAL_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
    fi
    if [[ -n "$LOCAL_IP" ]]; then
        echo "  Network:  http://${LOCAL_IP}:${APP_PORT}"
        echo "  Share this URL with devices connected to your hotspot"
    fi
    echo "═══════════════════════════════════════════════════"
    echo ""
}

# ── Check tmux availability ───────────────────────────────────────────────────
if $USE_TMUX && ! command -v tmux &>/dev/null; then
    echo "[aegis] tmux not found — run: pkg install tmux"
    echo "[aegis] Falling back to foreground mode"
    USE_TMUX=false
fi

# ── TMUX mode (Termux / production) ──────────────────────────────────────────
if $USE_TMUX; then
    # Kill any stale sessions from a previous run
    tmux kill-session -t aegis-llama 2>/dev/null || true
    tmux kill-session -t aegis-flask 2>/dev/null || true

    # Window 1: llama-server
    tmux new-session -d -s aegis-llama -x 220 -y 50
    tmux send-keys -t aegis-llama \
        "\"$LLAMA_BIN\" ${LLAMA_ARGS[*]} 2>&1 | tee /tmp/aegis-llama.log" Enter

    wait_for_llama

    # Window 2: Flask
    tmux new-session -d -s aegis-flask -x 220 -y 50
    tmux send-keys -t aegis-flask \
        "cd \"$ROOT_DIR\" && python app.py --host 0.0.0.0 --port $APP_PORT 2>&1 | tee /tmp/aegis-flask.log" Enter

    sleep 2
    print_access_info
    echo "  Logs — llama-server : tmux attach -t aegis-llama"
    echo "  Logs — Flask        : tmux attach -t aegis-flask"
    echo "  Stop everything     : ./start.sh --stop"
    echo ""

# ── Foreground mode (no tmux, CI/desktop) ────────────────────────────────────
else
    echo "[aegis] Starting llama-server in background..."
    "$LLAMA_BIN" "${LLAMA_ARGS[@]}" >/tmp/aegis-llama.log 2>&1 &
    LLAMA_PID=$!
    trap 'echo "[aegis] Shutting down..."; kill $LLAMA_PID 2>/dev/null || true' EXIT INT TERM

    wait_for_llama
    print_access_info

    echo "[aegis] Starting Flask (foreground — Ctrl+C to stop)..."
    python app.py --host 0.0.0.0 --port "$APP_PORT"
fi
