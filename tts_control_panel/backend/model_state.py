"""
Loads and holds the Fish Speech S2 inference engine.

Mirrors the model-loading sequence in fish-speech/tools/run_webui_v2.py
(llama queue -> decoder model -> TTSInferenceEngine -> warmup inference),
but runs it on a background thread so the web server can start immediately
and report progress via /api/status instead of blocking process startup.
"""

import sys
import threading
from typing import Optional

from loguru import logger

from .config import Settings


class ModelState:
    def __init__(self) -> None:
        self.engine = None
        self.whisper_model_dir: Optional[str] = None
        self.device: Optional[str] = None
        self.ready = False
        self.loading = False
        self.error: Optional[str] = None
        self._lock = threading.Lock()

    def load_async(self, settings: Settings) -> None:
        with self._lock:
            if self.loading or self.ready:
                return
            self.loading = True
        threading.Thread(target=self._load, args=(settings,), daemon=True).start()

    def _load(self, settings: Settings) -> None:
        try:
            if str(settings.fish_speech_dir) not in sys.path:
                sys.path.insert(0, str(settings.fish_speech_dir))

            import torch

            from fish_speech.inference_engine import TTSInferenceEngine
            from fish_speech.models.dac.inference import load_model as load_decoder_model
            from fish_speech.models.text2semantic.inference import (
                launch_thread_safe_queue,
            )
            from fish_speech.utils.schema import ServeTTSRequest

            device = settings.device
            if torch.backends.mps.is_available():
                device = "mps"
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                device = "xpu"
            elif not torch.cuda.is_available():
                device = "cpu"
            precision = torch.half if settings.half else torch.bfloat16

            logger.info("Loading Llama model from {}...", settings.llama_checkpoint_path)
            llama_queue = launch_thread_safe_queue(
                checkpoint_path=settings.llama_checkpoint_path,
                device=device,
                precision=precision,
                compile=settings.compile,
                quantization=settings.quantization,
            )

            logger.info(
                "Loading VQ-GAN decoder from {}...", settings.decoder_checkpoint_path
            )
            decoder_model = load_decoder_model(
                config_name=settings.decoder_config_name,
                checkpoint_path=settings.decoder_checkpoint_path,
                device=device,
            )

            engine = TTSInferenceEngine(
                llama_queue=llama_queue,
                decoder_model=decoder_model,
                compile=settings.compile,
                precision=precision,
            )

            logger.info("Warming up inference engine...")
            list(
                engine.inference(
                    ServeTTSRequest(
                        text="Hello world.",
                        references=[],
                        reference_id=None,
                        max_new_tokens=1024,
                        chunk_length=200,
                        top_p=0.7,
                        repetition_penalty=1.5,
                        temperature=0.7,
                        format="wav",
                    )
                )
            )

            self.engine = engine
            self.device = device
            self.whisper_model_dir = str(settings.whisper_model_dir)
            self.ready = True
            logger.info("Model ready to serve on device '{}'.", device)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI via /api/status
            logger.exception("Model load failed")
            self.error = str(exc)
        finally:
            self.loading = False


model_state = ModelState()
