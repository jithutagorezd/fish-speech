#!/usr/bin/env bash
# Lightweight start/stop alternative to systemd (scripts/cadence-tts.service).
# Useful for quick checks or environments without systemd access.
#
# Usage: scripts/start.sh [run_server.py args...]
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_DIR="run"
PID_FILE="$RUN_DIR/cadence-tts.pid"
LOG_FILE="$RUN_DIR/cadence-tts.log"
mkdir -p "$RUN_DIR"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Already running (PID $(cat "$PID_FILE")). Use scripts/status.sh or scripts/stop.sh."
  exit 0
fi

PYTHON_BIN="${CADENCE_PYTHON:-python3}"

nohup "$PYTHON_BIN" run_server.py "$@" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
disown

echo "Started Cadence TTS Control Panel (PID $(cat "$PID_FILE"))."
echo "Logs: $LOG_FILE"
