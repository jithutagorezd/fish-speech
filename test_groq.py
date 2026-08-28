import sys
import os

# Add the parent directory to the sys path so we can import webui_v2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webui_v2.groq_llm import auto_tag_text, detect_emotion

print("--- Testing Auto Tagging ---")
text = "Hello everyone! Today we are going to learn about Python. It is an amazing programming language."
print("Original:", text)
tagged = auto_tag_text(text)
print("Tagged:", tagged)

print("\n--- Testing Emotion Detection ---")
test_texts = [
    "I can't believe you did this to me! I hate you!",
    "I won the lottery! This is the best day of my life!",
    "I'm feeling a bit down today. Nothing seems to be going right.",
    "The data indicates a 20% increase in revenue for Q3."
]

for t in test_texts:
    emotion = detect_emotion(t)
    print(f"Text: '{t}' -> Detected Emotion: [{emotion}]")
