# AEGIS Termux Production Dry-Run Checklist

Use this checklist on the actual Android/Termux target device to validate startup, warmup, and watchdog resilience before demos or unattended use.

## Scope

This verifies:

- startup waits for real model readiness
- first request after boot is usable
- watchdog restarts crashed llama-server and Flask tmux sessions
- service recovers without manual intervention

## Preconditions

1. Run initial setup:

```bash
chmod +x termux_setup.sh
./termux_setup.sh
```

2. Ensure model files exist:

- `models/gemma4-e2b.gguf`
- optional: `models/mmproj-gemma4-e2b.gguf`

3. Ensure startup scripts are executable:

```bash
chmod +x start.sh termux_dry_run.sh
```

4. Start services in tmux mode:

```bash
./start.sh
```

## Automated Dry-Run (Recommended)

Run the verifier:

```bash
./termux_dry_run.sh
```

Expected result:

- summary ends with `Failures : 0`
- script exits code `0`

Optional flags:

```bash
./termux_dry_run.sh --skip-watchdog-kill
./termux_dry_run.sh --max-first-query-ms 240000
./termux_dry_run.sh --recovery-timeout-sec 180
```

## Manual Spot Checks (Fast)

1. Check watchdog is alive:

```bash
cat /tmp/aegis-watchdog.pid
ps -p "$(cat /tmp/aegis-watchdog.pid)"
```

2. Check health endpoints:

```bash
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/slots
curl -fsS http://127.0.0.1:5000/api/health
```

3. First-query smoke test:

```bash
curl -s -X POST \
  -F "text=My child has a nosebleed for 5 minutes. What should I do right now?" \
  -F "language=English" \
  http://127.0.0.1:5000/api/query
```

Check response JSON:

- `ok: true`
- no `system_unavailable` or `audit_unavailable` in mirror flags
- `response_text` is non-empty and actionable

## Recovery Injection Tests (Manual)

1. Kill llama-server tmux pane process:

```bash
LLAMA_PID="$(tmux list-panes -t aegis-llama -F '#{pane_pid}' | head -1)"
kill -9 "$LLAMA_PID"
```

2. Wait and verify restart:

```bash
curl -fsS http://127.0.0.1:8080/health
NEW_LLAMA_PID="$(tmux list-panes -t aegis-llama -F '#{pane_pid}' | head -1)"
echo "old=$LLAMA_PID new=$NEW_LLAMA_PID"
```

3. Kill Flask tmux pane process:

```bash
FLASK_PID="$(tmux list-panes -t aegis-flask -F '#{pane_pid}' | head -1)"
kill -9 "$FLASK_PID"
```

4. Wait and verify restart:

```bash
curl -fsS http://127.0.0.1:5000/api/health
NEW_FLASK_PID="$(tmux list-panes -t aegis-flask -F '#{pane_pid}' | head -1)"
echo "old=$FLASK_PID new=$NEW_FLASK_PID"
```

## Evidence to Capture

For release confidence and incident debugging, collect:

- `/tmp/aegis-llama.log`
- `/tmp/aegis-flask.log`
- `/tmp/aegis-watchdog.log`
- output of `./termux_dry_run.sh`

## Pass Criteria

Production confidence is considered closed when:

1. `./termux_dry_run.sh` returns exit code `0`
2. dry-run summary shows `Failures : 0`
3. first query after cold start succeeds without unavailable flags
4. both crash-injection restart checks recover within timeout
