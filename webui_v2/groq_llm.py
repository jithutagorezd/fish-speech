import os
import time
import re
import logging
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIStatusError, APITimeoutError

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPEN_ROUTER")

logger = logging.getLogger(__name__)

if not OPENROUTER_API_KEY:
    logger.warning("OPENROUTER API key not found in environment.")

# OpenRouter client using OpenAI SDK
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
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

    if not OPENROUTER_API_KEY:
        return ""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="poolside/laguna-xs-2.1:free",
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
                temperature=0.3,
                extra_body={"reasoning": {"enabled": True}}
            )

            content = response.choices[0].message.content

            if content:
                return content.strip()

            return ""

        except RateLimitError as e:
            if attempt < max_retries - 1:
                # Exponential backoff: 2s, 4s, 8s...
                sleep_time = 2.0 * (2 ** attempt)
                
                headers = getattr(e.response, "headers", {})
                retry_after = headers.get("retry-after")
                
                if retry_after:
                    try:
                        # Use the larger of our backoff or their requested wait
                        sleep_time = max(sleep_time, float(retry_after))
                    except ValueError:
                        pass
                
                # Add a small buffer to the sleep time
                sleep_time += 0.5
                
                logger.warning(f"OpenRouter Rate limit hit (429). Waiting {sleep_time:.2f} seconds before retry (Attempt {attempt + 1}/{max_retries})")
                time.sleep(sleep_time)
            else:
                logger.error(f"OpenRouter API rate limit error after {max_retries} attempts: {e}")
                return ""
        except APIStatusError as e:
            # Handle upstream provider errors (404, 502, 529 etc on OpenRouter)
            if attempt < max_retries - 1:
                sleep_time = 2.0 * (2 ** attempt)
                logger.warning(f"OpenRouter Provider error ({e.status_code}). Waiting {sleep_time:.2f} seconds before retry (Attempt {attempt + 1}/{max_retries})")
                time.sleep(sleep_time)
            else:
                logger.error(f"OpenRouter Provider error ({e.status_code}) after {max_retries} attempts: {e}")
                return ""
        except APITimeoutError as e:
            if attempt < max_retries - 1:
                sleep_time = 2.0 * (2 ** attempt)
                logger.warning(f"OpenRouter Timeout. Waiting {sleep_time:.2f} seconds before retry (Attempt {attempt + 1}/{max_retries})")
                time.sleep(sleep_time)
            else:
                logger.error(f"OpenRouter Timeout after {max_retries} attempts: {e}")
                return ""
        except Exception as e:
            logger.error(f"OpenRouter API unknown error: {e}")
            return ""


def auto_tag_text(text: str) -> str:
    """
    Uses an LLM to insert creative emotion and pacing tags into the text.
    The original words must not be changed.
    """

    system_prompt = (
        "You are a TTS (Text-to-Speech) script editor.\n\n"
        "Your task is to creatively insert appropriate emotion, pacing, and breath tags "
        "into the provided text to make the speech sound natural and expressive.\n\n"
        "Feel free to use any descriptive tag inside brackets (e.g., [sigh], [laughing], [whispering], [clearing throat], [pause], [emphasis], etc.). "
        "Be creative and use tags that best fit the mood and context of the text.\n\n"
        "Rules:\n"
        "1. Do NOT change any original words.\n"
        "2. Do NOT remove any original words.\n"
        "3. Do NOT rewrite or paraphrase the text.\n"
        "4. Enclose ONLY the tags in square brackets [like this]. Do NOT wrap the actual spoken text in brackets.\n"
        "5. Insert tags *between* words or sentences.\n"
        "6. Do not overuse tags; add them only where they naturally fit the context and emotion of the text.\n"
        "7. Return ONLY the final tagged text without any explanations."
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
