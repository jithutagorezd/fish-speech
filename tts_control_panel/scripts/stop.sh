#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PID_FILE="run/cadence-tts.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "Not running (no $PID_FILE)."
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "Stopped Cadence TTS Control Panel (PID $PID)."
else
  echo "PID $PID from $PID_FILE is not running (stale pid file)."
fi
rm -f "$PID_FILE"
