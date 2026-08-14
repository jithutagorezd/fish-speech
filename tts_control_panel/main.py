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

# All 34 tags below are the exact "common tags" list documented for this
# checkpoint — see checkpoints/s2-pro/README.md, "Fine-Grained Inline
# Control". S2 Pro accepts free-form bracketed text (not a fixed enum), but
# these are the tags it documents as reliably supported; anything else is
# unverified. Grouped here (patch-bay categories) purely for the UI's
# expanded view — the groups are a presentation grouping, not part of the
# checkpoint's documentation.
EMOTION_TAG_GROUPS = [
    ("Pacing", ["pause", "short pause", "emphasis", "interrupting"]),
    ("Breath & texture", ["inhale", "exhale", "sigh", "panting", "moaning", "clearing throat", "tsk"]),
    ("Laughter", ["laughing", "chuckle", "chuckling", "laughing tone", "audience laughter"]),
    ("Emotion", ["excited", "excited tone", "angry", "sad", "delight", "surprised", "shocked"]),
    ("Dynamics", ["volume up", "volume down", "low volume", "loud", "echo", "low voice"]),
    ("Performance", ["singing", "screaming", "shouting", "whisper", "with strong accent"]),
]
EMOTION_COMMON_TAGS = [
    "pause", "whisper", "excited", "laughing", "sigh",
    "sad", "angry", "singing", "emphasis", "volume up",
]
MAX_SCRIPT_WORDS = 5000

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
            "brand_name": "Fish Speech Console",
            "brand_initial": "F",
            "brand_tagline": "long-form synthesis · voice cloning",
            "initial_text": DEFAULT_TEXT,
            "emotion_common_tags": EMOTION_COMMON_TAGS,
            "emotion_tag_groups": EMOTION_TAG_GROUPS,
            "emotion_tag_more_count": sum(len(tags) for _, tags in EMOTION_TAG_GROUPS) - len(EMOTION_COMMON_TAGS),
            "default_chunk_words": 250,
            "max_script_words": MAX_SCRIPT_WORDS,
            "generate_label": "Generate speech",
        },
    )
