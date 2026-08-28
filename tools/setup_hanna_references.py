import os
import shutil
from webui_v2.inference import get_whisper_transcribe_wrapper

hanna_dir = "reference_audio/hanna"
references_dir = "references"
whisper_dir = "checkpoints/whisper-small-pt"

transcribe_fn = get_whisper_transcribe_wrapper(whisper_dir)

emotions = ["Angry", "Excited", "Fearful", "Happy", "Romantic", "Sad", "Serious", "Suspenseful", "neutral", "warm"]

for emotion in emotions:
    src_mp3 = os.path.join(hanna_dir, f"{emotion}.mp3")
    if not os.path.exists(src_mp3):
        print(f"Skipping {emotion} - file not found: {src_mp3}")
        continue
    
    preset_id = f"Hanna - {emotion.capitalize()}"
    dest_dir = os.path.join(references_dir, preset_id)
    os.makedirs(dest_dir, exist_ok=True)
    
    dest_mp3 = os.path.join(dest_dir, "sample.mp3")
    shutil.copy2(src_mp3, dest_mp3)
    
    try:
        transcript = transcribe_fn(dest_mp3)
        lab_path = os.path.join(dest_dir, "sample.lab")
        with open(lab_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        print(f"Successfully processed {preset_id}: {transcript}")
    except Exception as e:
        print(f"Failed to transcribe {preset_id}: {e}")
