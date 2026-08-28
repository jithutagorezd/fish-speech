import os
import logging
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

logger = logging.getLogger(__name__)

if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not found in environment.")

# Groq official client
client = Groq(
    api_key=GROQ_API_KEY
)

EMOTIONS = [
    "angry",
    "excited",
    "fearful",
    "happy",
    "romantic",
    "sad",
    "serious",
    "suspenseful",
    "neutral",
    "warm"
]


def call_groq(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 150
) -> str:

    if not GROQ_API_KEY:
        return ""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )

        content = response.choices[0].message.content

        if content:
            return content.strip()

        return ""

    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return ""


def auto_tag_text(text: str) -> str:
    """
    Uses Llama to insert [pause] tags into the text.
    The original words must not be changed.
    """

    system_prompt = (
        "You are a TTS (Text-to-Speech) script editor.\n\n"
        "Your task is ONLY to insert [pause] tags into the "
        "provided text where natural speech pauses should occur.\n\n"
        "Rules:\n"
        "1. Do NOT change any original words.\n"
        "2. Do NOT remove any original words.\n"
        "3. Do NOT rewrite or paraphrase the text.\n"
        "4. ONLY insert [pause] tags.\n"
        "5. Use [pause] after sentences and important clauses "
        "when a natural speech pause is appropriate.\n"
        "6. Do not add explanations.\n"
        "7. Return ONLY the final tagged text."
    )

    prompt = f"Text to tag:\n{text}"

    tagged = call_groq(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=1024
    )

    return tagged if tagged else text


def detect_emotion(text: str) -> str:
    """
    Detects the best matching emotion from the allowed list.
    """

    system_prompt = (
        "You are an emotion classification system.\n\n"
        f"You MUST choose exactly ONE emotion from this list:\n"
        f"{', '.join(EMOTIONS)}\n\n"
        "Rules:\n"
        "1. Return exactly one emotion.\n"
        "2. Return only the emotion word.\n"
        "3. Do not provide an explanation.\n"
        "4. Do not use punctuation.\n"
        "5. Choose neutral if the emotion is unclear."
    )

    prompt = f"Text:\n{text}"

    emotion = call_groq(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=10
    ).lower().strip()

    # Remove punctuation/spaces
    emotion = "".join(
        character for character in emotion
        if character.isalpha()
    )

    if emotion in EMOTIONS:
        return emotion

    return "neutral"
