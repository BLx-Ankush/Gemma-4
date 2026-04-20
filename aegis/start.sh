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
#   AEGIS_LLAMA_STARTUP_TIMEOUT_SEC seconds to wait for llama warmup (default 120)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

WATCHDOG_PID_FILE="/tmp/aegis-watchdog.pid"
WATCHDOG_SCRIPT="/tmp/aegis-watchdog.sh"

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
LLAMA_STARTUP_TIMEOUT_SEC="${AEGIS_LLAMA_STARTUP_TIMEOUT_SEC:-120}"
WARMUP_REQUEST_TIMEOUT_SEC="${AEGIS_WARMUP_REQUEST_TIMEOUT_SEC:-8}"
WATCHDOG_INTERVAL_SEC="${AEGIS_WATCHDOG_INTERVAL_SEC:-8}"
USE_TMUX=true

# Auto-detect thread count — nproc works on Android/Termux
if [[ -z "${AEGIS_THREADS:-}" ]]; then
    THREADS="$(nproc 2>/dev/null || echo 4)"
else
    THREADS="$AEGIS_THREADS"
fi

# ── Argument parsing ──────────────────────────────────────────────────────────
stop_watchdog() {
    if [[ -f "$WATCHDOG_PID_FILE" ]]; then
        local watchdog_pid
        watchdog_pid="$(cat "$WATCHDOG_PID_FILE" 2>/dev/null || true)"
        if [[ -n "$watchdog_pid" ]] && kill -0 "$watchdog_pid" 2>/dev/null; then
            kill "$watchdog_pid" 2>/dev/null || true
        fi
        rm -f "$WATCHDOG_PID_FILE"
    fi
    rm -f "$WATCHDOG_SCRIPT"
}

for arg in "$@"; do
    case "$arg" in
        --no-tmux) USE_TMUX=false ;;
        --stop)
            echo "[aegis] Killing AEGIS tmux sessions..."
            stop_watchdog
            if command -v tmux >/dev/null 2>&1; then
                tmux kill-session -t aegis-llama 2>/dev/null || true
                tmux kill-session -t aegis-flask 2>/dev/null || true
            fi
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

detect_ram_gb() {
    if [[ -r /proc/meminfo ]]; then
        awk '/MemTotal:/ { printf "%.1f", $2 / (1024 * 1024); exit }' /proc/meminfo
        return 0
    fi
    if command -v free >/dev/null 2>&1; then
        free -g | awk '/Mem:/ { print $2; exit }'
        return 0
    fi
    echo "2.0"
}

RAM_GB="$(detect_ram_gb)"
LLAMA_PARALLEL="${AEGIS_LLAMA_PARALLEL:-2}"
if awk "BEGIN { exit !(${RAM_GB:-0} < 4.0) }"; then
    LLAMA_PARALLEL="${AEGIS_LOW_RAM_PARALLEL:-1}"
    echo "[aegis] Low RAM detected (${RAM_GB}GB) — using safer llama profile (parallel=${LLAMA_PARALLEL})"
fi

# ── Build llama-server argument list ─────────────────────────────────────────
LLAMA_ARGS=(
    -m "$MODEL_PATH"
    --host "$LLAMA_HOST"
    --port "$LLAMA_PORT"
    --ctx-size "$CTX_SIZE"
    --threads "$THREADS"
    --n-gpu-layers 0
    --parallel "$LLAMA_PARALLEL"
    --cont-batching
)

if [[ "${AEGIS_ENABLE_MLOCK:-1}" == "1" ]] && awk "BEGIN { exit !(${RAM_GB:-0} < 6.0) }"; then
    LLAMA_ARGS+=(--mlock)
fi

if [[ "${AEGIS_DISABLE_MMAP:-0}" == "1" ]]; then
    LLAMA_ARGS+=(--no-mmap)
fi

if [[ -f "$MMPROJ_PATH" ]]; then
    LLAMA_ARGS+=(--mmproj "$MMPROJ_PATH")
    echo "[aegis] Vision projector found — multimodal enabled"
else
    echo "[aegis] No mmproj file — text-only mode"
fi

# ── Wait for llama-server health ──────────────────────────────────────────────
wait_for_llama() {
    local health_url="${AEGIS_LOCAL_LLM_URL}/health"
    local slots_url="${AEGIS_LOCAL_LLM_URL}/slots"
    local completion_url="${AEGIS_LOCAL_LLM_URL}/completion"
    local warmup_payload='{"prompt":"warmup","n_predict":1,"temperature":0,"stop":["\\n"],"stream":false}'

    echo "[aegis] Waiting for llama-server and warm model (timeout=${LLAMA_STARTUP_TIMEOUT_SEC}s)..."
    for i in $(seq 1 "$LLAMA_STARTUP_TIMEOUT_SEC"); do
        if curl -fsS --max-time 2 "$health_url" >/dev/null 2>&1 \
           && curl -fsS --max-time 2 "$slots_url" >/dev/null 2>&1 \
           && curl -fsS --max-time "$WARMUP_REQUEST_TIMEOUT_SEC" \
               -H "Content-Type: application/json" \
               -d "$warmup_payload" \
               "$completion_url" >/dev/null 2>&1; then
            printf "\n"
            echo "[aegis] llama-server ready after ${i}s"
            return 0
        fi

        if (( i % 10 == 0 )); then
            printf "."
        fi

        sleep 1
    done
    printf "\n"
    echo "ERROR: llama-server did not become ready after ${LLAMA_STARTUP_TIMEOUT_SEC}s. Check /tmp/aegis-llama.log"
    return 1
}

wait_for_flask() {
    local flask_health_url="http://127.0.0.1:${APP_PORT}/api/health"
    echo "[aegis] Waiting for Flask API at ${flask_health_url} ..."
    for i in $(seq 1 60); do
        if curl -fsS --max-time 2 "$flask_health_url" >/dev/null 2>&1; then
            echo "[aegis] Flask API ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "ERROR: Flask API did not become ready after 60s. Check /tmp/aegis-flask.log"
    return 1
}

build_llama_command() {
    local cmd="\"$LLAMA_BIN\""
    local arg
    for arg in "${LLAMA_ARGS[@]}"; do
        cmd+=" $(printf '%q' "$arg")"
    done
    cmd+=" 2>&1 | tee /tmp/aegis-llama.log"
    printf '%s' "$cmd"
}

build_flask_command() {
    local root_q
    local port_q
    printf -v root_q '%q' "$ROOT_DIR"
    printf -v port_q '%q' "$APP_PORT"
    printf 'cd %s && python app.py --host 0.0.0.0 --port %s 2>&1 | tee /tmp/aegis-flask.log' "$root_q" "$port_q"
}

LLAMA_CMD="$(build_llama_command)"
FLASK_CMD="$(build_flask_command)"

launch_llama_tmux() {
    tmux kill-session -t aegis-llama 2>/dev/null || true
    tmux new-session -d -s aegis-llama -x 220 -y 50
    tmux send-keys -t aegis-llama "$LLAMA_CMD" Enter
}

launch_flask_tmux() {
    tmux kill-session -t aegis-flask 2>/dev/null || true
    tmux new-session -d -s aegis-flask -x 220 -y 50
    tmux send-keys -t aegis-flask "$FLASK_CMD" Enter
}

start_tmux_watchdog() {
    stop_watchdog

    local local_url_q
    local app_port_q
    local interval_q
    local llama_cmd_q
    local flask_cmd_q

    printf -v local_url_q '%q' "$AEGIS_LOCAL_LLM_URL"
    printf -v app_port_q '%q' "$APP_PORT"
    printf -v interval_q '%q' "$WATCHDOG_INTERVAL_SEC"
    printf -v llama_cmd_q '%q' "$LLAMA_CMD"
    printf -v flask_cmd_q '%q' "$FLASK_CMD"

    cat > "$WATCHDOG_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail

AEGIS_LOCAL_LLM_URL=$local_url_q
APP_PORT=$app_port_q
WATCHDOG_INTERVAL_SEC=$interval_q
LLAMA_CMD=$llama_cmd_q
FLASK_CMD=$flask_cmd_q

restart_llama() {
    tmux kill-session -t aegis-llama 2>/dev/null || true
    tmux new-session -d -s aegis-llama -x 220 -y 50
    tmux send-keys -t aegis-llama "\$LLAMA_CMD" Enter
}

restart_flask() {
    tmux kill-session -t aegis-flask 2>/dev/null || true
    tmux new-session -d -s aegis-flask -x 220 -y 50
    tmux send-keys -t aegis-flask "\$FLASK_CMD" Enter
}

while true; do
    if ! tmux has-session -t aegis-llama 2>/dev/null; then
        echo "[watchdog] llama tmux session missing — restarting"
        restart_llama
    elif ! curl -fsS --max-time 3 "\$AEGIS_LOCAL_LLM_URL/health" >/dev/null 2>&1; then
        echo "[watchdog] llama health failed — restarting"
        restart_llama
    fi

    if ! tmux has-session -t aegis-flask 2>/dev/null; then
        echo "[watchdog] flask tmux session missing — restarting"
        restart_flask
    elif ! curl -fsS --max-time 3 "http://127.0.0.1:\$APP_PORT/api/health" >/dev/null 2>&1; then
        echo "[watchdog] flask health failed — restarting"
        restart_flask
    fi

    sleep "\$WATCHDOG_INTERVAL_SEC"
done
EOF

    chmod +x "$WATCHDOG_SCRIPT"
    nohup bash "$WATCHDOG_SCRIPT" >/tmp/aegis-watchdog.log 2>&1 &
    echo "$!" > "$WATCHDOG_PID_FILE"
    echo "[aegis] Watchdog active (pid $(cat "$WATCHDOG_PID_FILE"), interval=${WATCHDOG_INTERVAL_SEC}s)"
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
    stop_watchdog

    # Window 1: llama-server
    launch_llama_tmux
    wait_for_llama

    # Window 2: Flask
    launch_flask_tmux
    wait_for_flask

    start_tmux_watchdog

    print_access_info
    echo "  Logs — llama-server : tmux attach -t aegis-llama"
    echo "  Logs — Flask        : tmux attach -t aegis-flask"
    echo "  Logs — watchdog     : tail -f /tmp/aegis-watchdog.log"
    echo "  Stop everything     : ./start.sh --stop"
    echo ""

# ── Foreground mode (no tmux, CI/desktop) ────────────────────────────────────
else
    echo "[aegis] Starting llama-server in background..."
    "$LLAMA_BIN" "${LLAMA_ARGS[@]}" >/tmp/aegis-llama.log 2>&1 &
    LLAMA_PID=$!
    trap 'echo "[aegis] Shutting down..."; kill "$LLAMA_PID" 2>/dev/null || true' EXIT INT TERM

    wait_for_llama
    print_access_info

    echo "[aegis] Starting Flask (foreground — Ctrl+C to stop)..."
    python app.py --host 0.0.0.0 --port "$APP_PORT"
fi
