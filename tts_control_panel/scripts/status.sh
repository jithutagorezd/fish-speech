#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PID_FILE="run/cadence-tts.pid"
PORT="${TCP_PORT:-8731}"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Process: running (PID $(cat "$PID_FILE"))"
else
  echo "Process: not running"
  exit 1
fi

echo -n "Model:   "
curl -fsS "http://127.0.0.1:${PORT}/api/status" || echo "(no response on port $PORT)"
echo
