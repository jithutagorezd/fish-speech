import os
from dataclasses import dataclass
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
# tts_control_panel/ lives directly inside the fish-speech repo, so its
# parent directory *is* the fish-speech root.
DEFAULT_FISH_SPEECH_DIR = APP_DIR.parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


@dataclass(frozen=True)
class Settings:
    fish_speech_dir: Path
    llama_checkpoint_path: Path
    decoder_checkpoint_path: Path
    decoder_config_name: str
    whisper_model_dir: Path
    device: str
    half: bool
    compile: bool
    quantization: str
    host: str
    port: int


def get_settings() -> Settings:
    fish_speech_dir = _env_path("TCP_FISH_SPEECH_DIR", DEFAULT_FISH_SPEECH_DIR)
    return Settings(
        fish_speech_dir=fish_speech_dir,
        llama_checkpoint_path=_env_path(
            "TCP_LLAMA_CHECKPOINT_PATH", fish_speech_dir / "checkpoints" / "s2-pro"
        ),
        decoder_checkpoint_path=_env_path(
            "TCP_DECODER_CHECKPOINT_PATH",
            fish_speech_dir / "checkpoints" / "s2-pro" / "codec.pth",
        ),
        decoder_config_name=os.environ.get("TCP_DECODER_CONFIG_NAME", "modded_dac_vq"),
        whisper_model_dir=_env_path(
            "TCP_WHISPER_MODEL_DIR", fish_speech_dir / "checkpoints" / "whisper-small-pt"
        ),
        device=os.environ.get("TCP_DEVICE", "cuda"),
        half=os.environ.get("TCP_HALF") == "1",
        compile=os.environ.get("TCP_COMPILE") == "1",
        quantization=os.environ.get("TCP_QUANTIZATION", "none"),
        host=os.environ.get("TCP_HOST", "0.0.0.0"),
        port=int(os.environ.get("TCP_PORT", "8731")),
    )
