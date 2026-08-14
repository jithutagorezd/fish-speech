"""
Launch the Cadence TTS Control Panel (FastAPI + Jinja2, real Fish Speech S2 backend).

Model loading mirrors fish-speech/tools/run_webui_v2.py: same llama checkpoint,
decoder checkpoint, and whisper model dir flags, same warmup inference on start.

This script lives at fish-speech/tts_control_panel/run_server.py — its parent
directory is the fish-speech repo root.

Example (run from tts_control_panel/):

    python run_server.py \\
        --llama-checkpoint-path ../checkpoints/s2-pro \\
        --decoder-checkpoint-path ../checkpoints/s2-pro/codec.pth \\
        --whisper-model-dir ../checkpoints/whisper-small-pt

With no arguments, checkpoint paths default to that same location under
../checkpoints relative to this file.
"""

import argparse
import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
DEFAULT_FISH_SPEECH_DIR = APP_DIR.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cadence TTS Control Panel server (FastAPI)."
    )
    parser.add_argument(
        "--fish-speech-dir",
        type=Path,
        default=DEFAULT_FISH_SPEECH_DIR,
        help="Path to the fish-speech repo (for model/checkpoint imports).",
    )
    parser.add_argument("--llama-checkpoint-path", type=Path, default=None)
    parser.add_argument("--decoder-checkpoint-path", type=Path, default=None)
    parser.add_argument("--decoder-config-name", type=str, default="modded_dac_vq")
    parser.add_argument(
        "--whisper-model-dir",
        type=Path,
        default=None,
        help="Path to local Whisper model directory (for Auto-transcribe).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--quantization",
        type=str,
        default="none",
        choices=["none", "int8", "int4"],
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8731)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn autoreload (development only).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["TCP_FISH_SPEECH_DIR"] = str(args.fish_speech_dir)
    if args.llama_checkpoint_path:
        os.environ["TCP_LLAMA_CHECKPOINT_PATH"] = str(args.llama_checkpoint_path)
    if args.decoder_checkpoint_path:
        os.environ["TCP_DECODER_CHECKPOINT_PATH"] = str(args.decoder_checkpoint_path)
    os.environ["TCP_DECODER_CONFIG_NAME"] = args.decoder_config_name
    if args.whisper_model_dir:
        os.environ["TCP_WHISPER_MODEL_DIR"] = str(args.whisper_model_dir)
    os.environ["TCP_DEVICE"] = args.device
    os.environ["TCP_HALF"] = "1" if args.half else "0"
    os.environ["TCP_COMPILE"] = "1" if args.compile else "0"
    os.environ["TCP_QUANTIZATION"] = args.quantization
    os.environ["TCP_HOST"] = args.host
    os.environ["TCP_PORT"] = str(args.port)

    import uvicorn

    uvicorn.run("main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
