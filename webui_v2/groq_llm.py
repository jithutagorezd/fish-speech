import os
import time
import re
import logging
import requests
import json
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPEN_ROUTER")

logger = logging.getLogger(__name__)

if not OPENROUTER_API_KEY:
    logger.warning("OPENROUTER API key not found in environment.")

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

    if not OPENROUTER_API_KEY:
        return ""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "nvidia/nemotron-3.5-lightning:free",
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "reasoning": {"enabled": True},
                    "max_tokens": max_tokens,
                    "temperature": 0.3
                },
                timeout=30
            )

            if response.status_code == 429:
                sleep_time = 2.0
                retry_after = response.headers.get("retry-after")
                if retry_after:
                    try:
                        sleep_time = float(retry_after)
                    except ValueError:
                        pass
                sleep_time += 0.5
                logger.warning(f"OpenRouter Rate limit hit. Waiting {sleep_time:.2f} seconds before retry (Attempt {attempt + 1}/{max_retries})")
                time.sleep(sleep_time)
                continue
                
            response.raise_for_status()
            
            data = response.json()
            message = data['choices'][0]['message']
            content = message.get('content')

            if content:
                return content.strip()

            return ""

        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"OpenRouter API error (Attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(2.0)
            else:
                logger.error(f"OpenRouter API error after {max_retries} attempts: {e}")
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
