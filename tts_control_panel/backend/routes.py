import re
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from starlette import status

from .audio_io import encode_wav, save_upload_to_tempfile
from .model_state import model_state

router = APIRouter(prefix="/api")

_TAG_RE = re.compile(r"<[^>]+>")

# Advanced-config values: the panel in the UI is a disabled preview (no live
# controls yet), so requests always synthesize with the engine's defaults.
_MAX_NEW_TOKENS = 1024
_CHUNK_LENGTH = 200
_TOP_P = 0.8
_REPETITION_PENALTY = 1.1
_TEMPERATURE = 0.8


def _plain_error(message: str) -> str:
    return _TAG_RE.sub("", message).strip() or "Generation failed."


def _not_ready_error() -> HTTPException:
    if model_state.loading:
        return HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Model is still loading. Try again shortly.",
        )
    return HTTPException(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        model_state.error or "Model is not loaded.",
    )


@router.get("/status")
def get_status():
    return {
        "ready": model_state.ready,
        "loading": model_state.loading,
        "error": model_state.error,
        "device": model_state.device,
    }


@router.post("/generate")
def generate(
    text: str = Form(...),
    mode: str = Form("single"),
    reference_id: str = Form(""),
    reference_text: str = Form(""),
    memory_cache: str = Form("on"),
    chunk_words: int = Form(250),
    reference_audio: Optional[UploadFile] = File(None),
):
    if not model_state.ready:
        raise _not_ready_error()
    if not text.strip():
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Input text is empty. Add a script before generating.",
        )

    from webui_v2.inference import inference_long_form, inference_single

    def run(ref_audio_path: Optional[str]):
        if mode == "long":
            return inference_long_form(
                text=text,
                reference_id=reference_id or None,
                reference_audio=ref_audio_path,
                reference_text=reference_text,
                max_new_tokens=_MAX_NEW_TOKENS,
                chunk_length=_CHUNK_LENGTH,
                top_p=_TOP_P,
                repetition_penalty=_REPETITION_PENALTY,
                temperature=_TEMPERATURE,
                seed=None,
                use_memory_cache=memory_cache,
                max_words_per_chunk=chunk_words,
                engine=model_state.engine,
            )
        return inference_single(
            text=text,
            reference_id=reference_id or None,
            reference_audio=ref_audio_path,
            reference_text=reference_text,
            max_new_tokens=_MAX_NEW_TOKENS,
            chunk_length=_CHUNK_LENGTH,
            top_p=_TOP_P,
            repetition_penalty=_REPETITION_PENALTY,
            temperature=_TEMPERATURE,
            seed=None,
            use_memory_cache=memory_cache,
            engine=model_state.engine,
        )

    if reference_audio is not None and reference_audio.filename:
        with save_upload_to_tempfile(reference_audio) as ref_path:
            audio, err = run(ref_path)
    else:
        audio, err = run(None)

    if err is not None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, _plain_error(err))
    if audio is None:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "No audio generated."
        )

    sample_rate, samples = audio
    wav_bytes = encode_wav(sample_rate, samples)
    return Response(content=wav_bytes, media_type="audio/wav")


@router.post("/transcribe")
def transcribe(reference_audio: UploadFile = File(...)):
    if not model_state.ready:
        raise _not_ready_error()

    from tools.webui.whisper_utils import transcribe_reference_audio

    with save_upload_to_tempfile(reference_audio) as ref_path:
        try:
            text = transcribe_reference_audio(
                audio_path=ref_path,
                model_dir=model_state.whisper_model_dir,
                language=None,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                f"Whisper transcription error: {exc}",
            ) from exc

    return {"text": text}
