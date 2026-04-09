#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

LLAMA_BIN="${AEGIS_LLAMA_SERVER_BIN:-$ROOT_DIR/models/llama-server}"
MODEL_PATH="${AEGIS_MODEL_PATH:-$ROOT_DIR/models/gemma4-e2b.gguf}"
MMPROJ_PATH="${AEGIS_MMPROJ_PATH:-$ROOT_DIR/models/mmproj-gemma4-e2b.gguf}"
LLAMA_HOST="${AEGIS_LOCAL_HOST:-127.0.0.1}"
LLAMA_PORT="${AEGIS_LOCAL_PORT:-8080}"
APP_PORT="${AEGIS_APP_PORT:-5000}"
THREADS="${AEGIS_THREADS:-4}"
CTX_SIZE="${AEGIS_CTX_SIZE:-4096}"

if [[ ! -x "$LLAMA_BIN" ]]; then
  echo "ERROR: llama-server binary not found or not executable: $LLAMA_BIN"
  echo "Set AEGIS_LLAMA_SERVER_BIN or place llama-server at aegis/models/llama-server"
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "ERROR: model file not found: $MODEL_PATH"
  exit 1
fi

export AEGIS_LOCAL_LLM_URL="http://${LLAMA_HOST}:${LLAMA_PORT}"

echo "Starting local llama-server on ${AEGIS_LOCAL_LLM_URL}..."
LLAMA_ARGS=(
  -m "$MODEL_PATH"
  --host "$LLAMA_HOST"
  --port "$LLAMA_PORT"
  --ctx-size "$CTX_SIZE"
  --threads "$THREADS"
  --n-gpu-layers 0
)

if [[ -n "$MMPROJ_PATH" && -f "$MMPROJ_PATH" ]]; then
  LLAMA_ARGS+=(--mmproj "$MMPROJ_PATH")
fi

"$LLAMA_BIN" "${LLAMA_ARGS[@]}" >/tmp/aegis-llama.log 2>&1 &

LLAMA_PID=$!
trap 'kill $LLAMA_PID >/dev/null 2>&1 || true' EXIT

for i in {1..20}; do
  if curl -fsS "${AEGIS_LOCAL_LLM_URL}/health" >/dev/null 2>&1; then
    echo "llama-server is healthy"
    break
  fi
  sleep 1
done

if ! curl -fsS "${AEGIS_LOCAL_LLM_URL}/health" >/dev/null 2>&1; then
  echo "ERROR: llama-server did not become ready. Check /tmp/aegis-llama.log"
  exit 1
fi

echo "Starting Flask app on 0.0.0.0:${APP_PORT}"
python app.py --host 0.0.0.0 --port "$APP_PORT"
