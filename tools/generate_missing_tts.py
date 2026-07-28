"""
Generate missing ai_generated.wav files under test_audio/ using a running
fish-speech Gradio instance (the https://xxxx.gradio.live link).

Folder layout expected:
    test_audio/
        male/<Author Name>/<original_recording>.wav
        male/<Author Name>/ai_generated.wav      (may be missing)
        female/<Author Name>/...

For every author folder missing ai_generated.wav, the script:
    1. Finds the original recording (any .wav that isn't ai_generated.wav)
    2. Sends it as the reference/clone audio to the Gradio app
    3. Sends TARGET_TEXT as the text to synthesize
    4. Saves the returned audio as ai_generated.wav in that same folder

USAGE
-----
Step 1 - inspect the app's real API once (names/order of params differ
between fish-speech versions), and copy the api_name + param order into
API_NAME / PARAM_ORDER below:

    python generate_missing_tts.py --url https://xxxx.gradio.live --inspect

Step 2 - run for real:

    python generate_missing_tts.py --url https://xxxx.gradio.live
"""

import argparse
import shutil
from pathlib import Path

from gradio_client import Client, handle_file

# ---------------------------------------------------------------------------
# EDIT THESE two values after running with --inspect once.
# api_name: the endpoint shown in view_api() output, e.g. "/inference" or "/tts"
# PARAM_ORDER: the positional args in the SAME order view_api() lists them.
# Use None for any param you want left at the app's default.
# ---------------------------------------------------------------------------
API_NAME = "/inference"

TARGET_TEXT = (
    "[laugh] That's actually pretty funny!\n"
    "[whisper] I don't think anyone noticed.\n"
    "[excited] We finally made it!\n"
    "[sad] I really miss those days.\n"
    "[angry] This is completely unacceptable!\n"
    "[surprised] Wait... what just happened?\n"
    "[happy] Today has been amazing!\n"
    "[calm] Take your time. There's no rush.\n"
    "[nervous] I hope everything goes well.\n"
    "[confident] I know we can do this."
)

REFERENCE_TEXT = ""  # left blank -> webui's own Whisper ASR fills this in


def build_kwargs(reference_audio_path: Path):
    """
    Keyword args matching the fish-speech webui `/inference` endpoint.
    Adjust names to match whatever --inspect prints for your build.
    """
    return {
        "text": TARGET_TEXT,
        "reference_audio": handle_file(str(reference_audio_path)),
        "reference_text": REFERENCE_TEXT,
        "max_new_tokens": 1024,
        "chunk_length": 200,
        "top_p": 0.7,
        "repetition_penalty": 1.5,
        "temperature": 0.7,
        "seed": 0,
    }


def find_original_recording(author_dir: Path) -> Path | None:
    for f in sorted(author_dir.iterdir()):
        if not f.is_file():
            continue
        if f.suffix.lower() not in (".wav", ".mp3", ".flac", ".m4a", ".ogg"):
            continue
        if f.name.lower() == "ai_generated.wav":
            continue
        return f
    return None


def iter_author_dirs(root: Path):
    for gender_dir in root.iterdir():
        if not gender_dir.is_dir():
            continue
        for author_dir in gender_dir.iterdir():
            if author_dir.is_dir():
                yield author_dir


def main():
    parser = argparse.ArgumentParser(description="Fill in missing ai_generated.wav files.")
    parser.add_argument("--url", required=True, help="Gradio share link, e.g. https://xxxx.gradio.live")
    parser.add_argument("--test-audio-root", type=Path, default=Path("test_audio"))
    parser.add_argument("--inspect", action="store_true", help="Print the app's API and exit.")
    parser.add_argument("--dry-run", action="store_true", help="List what would be generated, don't call API.")
    args = parser.parse_args()

    client = Client(args.url)

    if args.inspect:
        client.view_api(all_endpoints=True)
        return

    root = args.test_audio_root
    missing = []
    for author_dir in iter_author_dirs(root):
        out_path = author_dir / "ai_generated.wav"
        if out_path.exists():
            print(f"[skip] {author_dir} -> ai_generated.wav already exists")
            continue

        original = find_original_recording(author_dir)
        if original is None:
            print(f"[warn] {author_dir} -> no original recording found, skipping")
            continue

        missing.append((author_dir, original, out_path))

    print(f"\nFound {len(missing)} author folder(s) needing ai_generated.wav\n")

    if args.dry_run:
        for author_dir, original, out_path in missing:
            print(f"  {author_dir.name}: reference={original.name} -> {out_path}")
        return

    for author_dir, original, out_path in missing:
        print(f"[gen] {author_dir} using reference {original.name}")
        try:
            result = client.predict(api_name=API_NAME, **build_kwargs(original))
        except Exception as e:
            print(f"[error] {author_dir}: {e}")
            continue

        # result is usually a filepath (str) or dict with 'path' key returned by gradio_client
        result_path: str | None
        if isinstance(result, dict) and "path" in result:
            result_path = str(result["path"])
        elif isinstance(result, (str, Path)):
            result_path = str(result)
        else:
            result_path = None

        if not result_path or not Path(result_path).exists():
            print(f"[error] {author_dir}: no output audio returned ({result})")
            continue

        shutil.copy(str(result_path), str(out_path))
        print(f"[done] {author_dir} -> {out_path}")

    print("\nAll done.")


if __name__ == "__main__":
    main()