#!/usr/bin/env bash
# Generates /etc/systemd/system/cadence-tts.service from the current
# checkout (auto-detecting paths/user/python) and starts it — the "create
# and run" step that scripts/cadence-tts.service otherwise needs done by hand.
#
# Usage (as root, or with sudo):
#   sudo scripts/install-service.sh
#   sudo scripts/install-service.sh --device cuda --port 8731
#   sudo scripts/install-service.sh --no-start   # install + enable, don't start yet
#
# Re-run any time to update the unit file (e.g. after moving checkpoints).
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"

SERVICE_NAME="cadence-tts"
SERVICE_USER="${SUDO_USER:-$(whoami)}"
# tts_control_panel/ lives directly inside the fish-speech repo.
FISH_SPEECH_DIR="$(cd "$APP_DIR/.." && pwd)"
PYTHON_BIN=""
DEVICE="cuda"
HOST="0.0.0.0"
PORT="8731"
LLAMA_CKPT=""
DECODER_CKPT=""
WHISPER_DIR=""
DO_START=1

while [ $# -gt 0 ]; do
  case "$1" in
    --user) SERVICE_USER="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --fish-speech-dir) FISH_SPEECH_DIR="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --llama-checkpoint-path) LLAMA_CKPT="$2"; shift 2 ;;
    --decoder-checkpoint-path) DECODER_CKPT="$2"; shift 2 ;;
    --whisper-model-dir) WHISPER_DIR="$2"; shift 2 ;;
    --no-start) DO_START=0; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ "$(id -u)" -ne 0 ]; then
  echo "Must run as root (try: sudo $0 ...)" >&2
  exit 1
fi

if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$APP_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$APP_DIR/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

LLAMA_CKPT="${LLAMA_CKPT:-$FISH_SPEECH_DIR/checkpoints/s2-pro}"
DECODER_CKPT="${DECODER_CKPT:-$FISH_SPEECH_DIR/checkpoints/s2-pro/codec.pth}"
WHISPER_DIR="${WHISPER_DIR:-$FISH_SPEECH_DIR/checkpoints/whisper-small-pt}"

UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

echo "Writing $UNIT_PATH"
echo "  user:              $SERVICE_USER"
echo "  working dir:       $APP_DIR"
echo "  python:            $PYTHON_BIN"
echo "  device:            $DEVICE"
echo "  llama checkpoint:  $LLAMA_CKPT"
echo "  decoder checkpoint: $DECODER_CKPT"
echo "  whisper model dir: $WHISPER_DIR"

cat > "$UNIT_PATH" <<EOF
[Unit]
Description=Cadence TTS Control Panel (Fish Speech S2)
After=network.target

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_USER}
WorkingDirectory=${APP_DIR}
ExecStart=${PYTHON_BIN} run_server.py \\
    --host ${HOST} \\
    --port ${PORT} \\
    --device ${DEVICE} \\
    --llama-checkpoint-path ${LLAMA_CKPT} \\
    --decoder-checkpoint-path ${DECODER_CKPT} \\
    --whisper-model-dir ${WHISPER_DIR}
Restart=on-failure
RestartSec=5
TimeoutStartSec=600

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

if [ "$DO_START" -eq 1 ]; then
  systemctl restart "$SERVICE_NAME"
  echo "Started. Check progress with: systemctl status $SERVICE_NAME  /  journalctl -u $SERVICE_NAME -f"
else
  echo "Installed and enabled, not started (--no-start). Start with: systemctl start $SERVICE_NAME"
fi
