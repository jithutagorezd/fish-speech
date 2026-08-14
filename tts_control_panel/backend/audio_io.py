import io
import os
import tempfile
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import soundfile as sf
from fastapi import UploadFile


def encode_wav(sample_rate: int, audio: np.ndarray) -> bytes:
    # Match Gradio's gr.Audio(type="numpy") postprocessing, which peak-normalizes
    # float audio before writing 16-bit PCM. Without this, the raw decoder output
    # (often well below full scale) gets written at its native — much quieter — level.
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak * 0.98

    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


@contextmanager
def save_upload_to_tempfile(upload: UploadFile) -> Iterator[str]:
    suffix = os.path.splitext(upload.filename or "")[1] or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(upload.file.read())
        yield path
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
