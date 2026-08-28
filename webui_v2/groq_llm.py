import os
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY not found in environment.")

EMOTIONS = ["angry", "excited", "fearful", "happy", "romantic", "sad", "serious", "suspenseful", "neutral", "warm"]

def call_groq(prompt: str, system_prompt: str = "You are a helpful assistant.", max_tokens: int = 150) -> str:
    if not GROQ_API_KEY:
        return ""
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }
    
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Groq API error: {e}")
        return ""

def auto_tag_text(text: str) -> str:
    """Uses LLM to inject TTS tags like [pause] into the text."""
    system_prompt = (
        "You are a TTS (Text-to-Speech) script editor. Your job is to add [pause] tags where appropriate in the provided text "
        "to make the speech sound more natural. Add [pause] after sentences or major clauses (like commas). "
        "Do not change the words, only insert [pause]. Return ONLY the tagged text."
    )
    prompt = f"Text to tag:\n{text}"
    tagged = call_groq(prompt, system_prompt, max_tokens=1024)
    return tagged if tagged else text

def detect_emotion(text: str) -> str:
    """Uses LLM to detect the best matching emotion from the allowed list."""
    system_prompt = (
        f"You are an emotion detection AI. Analyze the text and return EXACTLY ONE of the following emotions that best matches the tone: {', '.join(EMOTIONS)}. "
        "Do not return any other text, punctuation, or explanation. Just the single word."
    )
    prompt = f"Text:\n{text}"
    emotion = call_groq(prompt, system_prompt, max_tokens=10).lower()
    
    # Strip punctuation and whitespace
    emotion = "".join(c for c in emotion if c.isalpha())
    
    if emotion in EMOTIONS:
        return emotion
    return "neutral"  # Fallback
