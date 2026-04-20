#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/.env" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ROOT_DIR/.env"
    set +a
fi

BASE_URL="${AEGIS_DRY_RUN_BASE_URL:-http://127.0.0.1:${AEGIS_APP_PORT:-5000}}"
LLAMA_URL="${AEGIS_DRY_RUN_LLAMA_URL:-${AEGIS_LOCAL_LLM_URL:-http://127.0.0.1:${AEGIS_LOCAL_PORT:-8080}}}"
QUERY_TIMEOUT_SEC="${AEGIS_DRY_RUN_QUERY_TIMEOUT_SEC:-180}"
MAX_FIRST_QUERY_MS="${AEGIS_DRY_RUN_MAX_FIRST_QUERY_MS:-180000}"
RECOVERY_TIMEOUT_SEC="${AEGIS_DRY_RUN_RECOVERY_TIMEOUT_SEC:-120}"
SKIP_WATCHDOG_KILL=0

usage() {
    cat <<'EOF'
Usage: ./termux_dry_run.sh [options]

Options:
  --base-url URL                  API base URL (default: http://127.0.0.1:${AEGIS_APP_PORT})
  --llama-url URL                 llama-server URL (default: AEGIS_LOCAL_LLM_URL)
  --query-timeout-sec N           timeout for verification query (default: 180)
  --max-first-query-ms N          warning threshold for first-query latency (default: 180000)
  --recovery-timeout-sec N        timeout for watchdog recovery checks (default: 120)
  --skip-watchdog-kill            skip destructive kill/restart watchdog checks
  -h, --help                      show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-url)
            BASE_URL="$2"
            shift 2
            ;;
        --llama-url)
            LLAMA_URL="$2"
            shift 2
            ;;
        --query-timeout-sec)
            QUERY_TIMEOUT_SEC="$2"
            shift 2
            ;;
        --max-first-query-ms)
            MAX_FIRST_QUERY_MS="$2"
            shift 2
            ;;
        --recovery-timeout-sec)
            RECOVERY_TIMEOUT_SEC="$2"
            shift 2
            ;;
        --skip-watchdog-kill)
            SKIP_WATCHDOG_KILL=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

passes=0
warnings=0
failures=0

log_step() {
    echo ""
    echo "==> $1"
}

pass() {
    passes=$((passes + 1))
    echo "PASS: $1"
}

warn() {
    warnings=$((warnings + 1))
    echo "WARN: $1"
}

fail() {
    failures=$((failures + 1))
    echo "FAIL: $1"
}

require_cmd() {
    local cmd="$1"
    if command -v "$cmd" >/dev/null 2>&1; then
        pass "Command available: $cmd"
    else
        fail "Missing required command: $cmd"
    fi
}

wait_for_endpoint() {
    local url="$1"
    local timeout_sec="$2"
    local i
    for i in $(seq 1 "$timeout_sec"); do
        if curl -fsS --max-time 3 "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

run_query_check() {
    python - "$BASE_URL" "$QUERY_TIMEOUT_SEC" <<'PY'
import json
import sys
import time
import requests

base = sys.argv[1].rstrip('/')
timeout = float(sys.argv[2])
payload = {
    "text": "My child has a nosebleed for 5 minutes. What should I do right now?",
    "language": "English",
}

start = time.time()
try:
    response = requests.post(base + "/api/query", data=payload, timeout=timeout)
    elapsed_ms = int((time.time() - start) * 1000)
    result = {
        "http_status": response.status_code,
        "elapsed_ms": elapsed_ms,
    }

    try:
        body = response.json()
    except Exception:
        body = {}

    data = body.get("data", {}) if isinstance(body, dict) else {}
    mirror = data.get("mirror", {}) if isinstance(data, dict) else {}

    result.update(
        {
            "ok": bool(body.get("ok")) if isinstance(body, dict) else False,
            "mode": data.get("mode", "") if isinstance(data, dict) else "",
            "verdict": mirror.get("verdict", "") if isinstance(mirror, dict) else "",
            "flags": mirror.get("flags", []) if isinstance(mirror, dict) else [],
            "processing_time_ms": data.get("processing_time_ms", 0) if isinstance(data, dict) else 0,
            "text_len": len(data.get("response_text", "")) if isinstance(data.get("response_text", ""), str) else 0,
        }
    )
    print(json.dumps(result))
except Exception as exc:
    print(json.dumps({"error": str(exc)}))
PY
}

log_step "Termux Dry-Run: prerequisites"
require_cmd curl
require_cmd tmux
require_cmd python

log_step "Session/process checks"
if tmux has-session -t aegis-llama 2>/dev/null; then
    pass "tmux session aegis-llama is running"
else
    fail "tmux session aegis-llama is missing (run ./start.sh first)"
fi

if tmux has-session -t aegis-flask 2>/dev/null; then
    pass "tmux session aegis-flask is running"
else
    fail "tmux session aegis-flask is missing (run ./start.sh first)"
fi

if [[ -f /tmp/aegis-watchdog.pid ]]; then
    watchdog_pid="$(cat /tmp/aegis-watchdog.pid 2>/dev/null || true)"
    if [[ -n "$watchdog_pid" ]] && kill -0 "$watchdog_pid" 2>/dev/null; then
        pass "watchdog process is running (pid ${watchdog_pid})"
    else
        fail "watchdog pid file exists but process is not running"
    fi
else
    fail "watchdog pid file missing: /tmp/aegis-watchdog.pid"
fi

log_step "Health/readiness checks"
if wait_for_endpoint "${LLAMA_URL}/health" 20; then
    pass "llama health endpoint reachable"
else
    fail "llama health endpoint not reachable: ${LLAMA_URL}/health"
fi

if wait_for_endpoint "${LLAMA_URL}/slots" 20; then
    pass "llama slots endpoint reachable (warmup-ready signal path)"
else
    fail "llama slots endpoint not reachable: ${LLAMA_URL}/slots"
fi

if wait_for_endpoint "${BASE_URL}/api/health" 20; then
    pass "Flask API health endpoint reachable"
else
    fail "Flask API health endpoint not reachable: ${BASE_URL}/api/health"
fi

health_json="$(curl -fsS --max-time 5 "${BASE_URL}/api/health" || true)"
if [[ -n "$health_json" ]]; then
    parsed_health="$(printf '%s' "$health_json" | python -c 'import json,sys; d=json.load(sys.stdin); c=d.get("compute",{}); m=d.get("model",{}); print("|".join([str(c.get("mode","")), str(c.get("is_online","")), str(m.get("loaded","")), str(m.get("active_backend",""))]))' 2>/dev/null || true)"
    if [[ -n "$parsed_health" ]]; then
        IFS='|' read -r mode is_online model_loaded backend <<< "$parsed_health"
        pass "health summary mode=${mode} online=${is_online} loaded=${model_loaded} backend=${backend}"
        if [[ "$model_loaded" != "True" ]]; then
            fail "model_loaded is false in /api/health"
        fi
    else
        fail "unable to parse /api/health JSON"
    fi
else
    fail "unable to fetch /api/health JSON"
fi

log_step "Cold path query check"
query_result_json="$(run_query_check)"
query_error="$(printf '%s' "$query_result_json" | python -c 'import json,sys; d=json.load(sys.stdin); print(d.get("error",""))' 2>/dev/null || true)"

if [[ -n "$query_error" ]]; then
    fail "query check failed: ${query_error}"
else
    parsed_query="$(printf '%s' "$query_result_json" | python -c 'import json,sys; d=json.load(sys.stdin); flags=",".join(d.get("flags",[])); print("|".join([str(d.get("http_status")), str(d.get("ok")), str(d.get("elapsed_ms")), str(d.get("processing_time_ms")), str(d.get("mode")), str(d.get("verdict")), str(d.get("text_len")), flags]))')"
    IFS='|' read -r http_status ok elapsed_ms processing_time_ms query_mode verdict text_len flags <<< "$parsed_query"

    if [[ "$http_status" == "200" && "$ok" == "True" ]]; then
        pass "first query returned HTTP 200 and ok=true"
    else
        fail "first query failed (http_status=${http_status}, ok=${ok})"
    fi

    if [[ "$text_len" =~ ^[0-9]+$ ]] && (( text_len >= 40 )); then
        pass "response_text length is sufficient (${text_len})"
    else
        fail "response_text too short (${text_len})"
    fi

    if [[ "$verdict" == "block" && ( "$flags" == *"system_unavailable"* || "$flags" == *"audit_unavailable"* ) ]]; then
        fail "first query blocked due to backend unavailability flags (${flags})"
    else
        pass "first query did not fail with backend-unavailable block"
    fi

    if [[ "$elapsed_ms" =~ ^[0-9]+$ ]] && (( elapsed_ms > MAX_FIRST_QUERY_MS )); then
        warn "first query latency high (${elapsed_ms}ms > ${MAX_FIRST_QUERY_MS}ms)"
    else
        pass "first query latency within threshold (${elapsed_ms}ms)"
    fi

    pass "query summary mode=${query_mode} verdict=${verdict} processing_ms=${processing_time_ms}"
fi

if [[ "$SKIP_WATCHDOG_KILL" -eq 0 ]]; then
    log_step "Watchdog recovery checks (destructive)"

    llama_old_pid="$(tmux list-panes -t aegis-llama -F '#{pane_pid}' 2>/dev/null | head -1 || true)"
    if [[ -n "$llama_old_pid" ]]; then
        kill -9 "$llama_old_pid" 2>/dev/null || true
        if wait_for_endpoint "${LLAMA_URL}/health" "$RECOVERY_TIMEOUT_SEC"; then
            llama_new_pid="$(tmux list-panes -t aegis-llama -F '#{pane_pid}' 2>/dev/null | head -1 || true)"
            if [[ -n "$llama_new_pid" && "$llama_new_pid" != "$llama_old_pid" ]]; then
                pass "watchdog restarted llama-server (pid ${llama_old_pid} -> ${llama_new_pid})"
            else
                warn "llama health recovered but pid did not change as expected"
            fi
        else
            fail "llama health did not recover within ${RECOVERY_TIMEOUT_SEC}s after crash injection"
        fi
    else
        fail "could not determine llama tmux pane pid"
    fi

    flask_old_pid="$(tmux list-panes -t aegis-flask -F '#{pane_pid}' 2>/dev/null | head -1 || true)"
    if [[ -n "$flask_old_pid" ]]; then
        kill -9 "$flask_old_pid" 2>/dev/null || true
        if wait_for_endpoint "${BASE_URL}/api/health" "$RECOVERY_TIMEOUT_SEC"; then
            flask_new_pid="$(tmux list-panes -t aegis-flask -F '#{pane_pid}' 2>/dev/null | head -1 || true)"
            if [[ -n "$flask_new_pid" && "$flask_new_pid" != "$flask_old_pid" ]]; then
                pass "watchdog restarted Flask (pid ${flask_old_pid} -> ${flask_new_pid})"
            else
                warn "Flask health recovered but pid did not change as expected"
            fi
        else
            fail "Flask health did not recover within ${RECOVERY_TIMEOUT_SEC}s after crash injection"
        fi
    else
        fail "could not determine Flask tmux pane pid"
    fi

    post_recovery_query="$(run_query_check)"
    post_recovery_error="$(printf '%s' "$post_recovery_query" | python -c 'import json,sys; d=json.load(sys.stdin); print(d.get("error",""))' 2>/dev/null || true)"
    if [[ -n "$post_recovery_error" ]]; then
        fail "post-recovery query failed: ${post_recovery_error}"
    else
        post_status="$(printf '%s' "$post_recovery_query" | python -c 'import json,sys; d=json.load(sys.stdin); print(str(d.get("http_status")))')"
        if [[ "$post_status" == "200" ]]; then
            pass "post-recovery query returned HTTP 200"
        else
            fail "post-recovery query returned HTTP ${post_status}"
        fi
    fi
else
    warn "watchdog kill/recovery checks were skipped"
fi

echo ""
echo "========================================"
echo "Termux dry-run summary"
echo "  Passes   : ${passes}"
echo "  Warnings : ${warnings}"
echo "  Failures : ${failures}"
echo "========================================"

if (( failures > 0 )); then
    exit 1
fi

exit 0
