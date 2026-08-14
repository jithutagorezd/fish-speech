import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend.config import get_settings
from backend.model_state import model_state
from backend.routes import router as api_router

BASE_DIR = Path(__file__).resolve().parent

settings = get_settings()
if str(settings.fish_speech_dir) not in sys.path:
    sys.path.insert(0, str(settings.fish_speech_dir))


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_state.load_async(settings)
    yield


app = FastAPI(title="Cadence — TTS Control Panel", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")
app.include_router(api_router)

EMOTION_TAGS = ["laugh", "whisper", "excited", "sigh", "pause", "gasp", "angry", "curious"]

DEFAULT_TEXT = (
    "Welcome back. [excited] This is going to be a good one. [pause] "
    "Let's get started — [whisper] just between us, this part's my favorite."
)


@app.get("/")
def control_panel(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "brand_name": "Cadence",
            "brand_initial": "C",
            "brand_tagline": "voiceover generation workspace",
            "docs_url": "#",
            "initial_text": DEFAULT_TEXT,
            "emotion_tags": EMOTION_TAGS,
            "default_chunk_words": 250,
            "generate_label": "Generate speech",
        },
    )
