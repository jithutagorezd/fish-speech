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
import logging
import shutil
import time
from pathlib import Path

from gradio_client import Client, handle_file

LOG_PATH = Path("generate_missing_tts.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),          # prints to terminal, same as before
        logging.FileHandler(LOG_PATH),    # also saved to disk, survives crashes
    ],
)
log = logging.getLogger("tts")

# ---------------------------------------------------------------------------
# EDIT THESE two values after running with --inspect once.
# api_name: the endpoint shown in view_api() output, e.g. "/inference" or "/tts"
# PARAM_ORDER: the positional args in the SAME order view_api() lists them.
# Use None for any param you want left at the app's default.
# ---------------------------------------------------------------------------
API_NAME = "/dispatch"
ASR_API_NAME = "/_wrapper"  # auto-transcribes reference_audio -> reference_text

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


def build_kwargs(reference_audio_path: Path, reference_id: str, reference_text: str):
    """
    Keyword args matching the fish-speech webui `/dispatch` endpoint exactly,
    in the order shown by --inspect.
    """
    return {
        "text": TARGET_TEXT,
        "reference_id": reference_id,
        "reference_audio": handle_file(str(reference_audio_path)),
        "reference_text": reference_text,
        "max_new_tokens": 1024,
        "chunk_length": 200,
        "top_p": 0.8,
        "repetition_penalty": 1.1,
        "temperature": 0.8,
        "seed": 0,
        "use_memory_cache": "on",
        "max_words_per_chunk": 200,
        "mode_radio": "Long-form (chunked)",
    }


def run_with_progress(client: Client, api_name: str, log_prefix: str, poll_interval: float = 3.0, *args, **kwargs):
    """
    Like client.predict(), but non-blocking + prints live progress while the
    job runs, instead of going silent until it finishes.
    """
    job = client.submit(*args, api_name=api_name, **kwargs)

    last_msg = None
    while not job.done():
        status = job.status()
        pct = None
        desc = None
        if status.progress_data:
            for p in status.progress_data:
                if p.get("length"):
                    pct = 100 * (p.get("index") or 0) / p["length"]
                desc = p.get("desc") or desc
        eta = f", eta {status.eta:.0f}s" if status.eta else ""
        if pct is not None:
            msg = f"{log_prefix} progress: {pct:.0f}%{f' - {desc}' if desc else ''}{eta}"
        elif status.code:
            msg = f"{log_prefix} status: {status.code.name.lower()}{eta}"
        else:
            msg = f"{log_prefix} still running...{eta}"

        if msg != last_msg:
            log.info(msg)
            last_msg = msg
        time.sleep(poll_interval)

    return job.result()


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

    log.info(f"Connecting to {args.url}")
    client = Client(args.url)

    if args.inspect:
        client.view_api(all_endpoints=True)
        return

    root = args.test_audio_root
    missing = []
    for author_dir in iter_author_dirs(root):
        out_path = author_dir / "ai_generated.wav"
        if out_path.exists():
            log.info(f"[skip] {author_dir} -> ai_generated.wav already exists")
            continue

        original = find_original_recording(author_dir)
        if original is None:
            log.warning(f"[warn] {author_dir} -> no original recording found, skipping")
            continue

        missing.append((author_dir, original, out_path))

    log.info(f"Found {len(missing)} author folder(s) needing ai_generated.wav")

    if args.dry_run:
        for author_dir, original, out_path in missing:
            log.info(f"  {author_dir.name}: reference={original.name} -> {out_path}")
        return

    succeeded, failed = [], []
    run_start = time.time()

    for i, (author_dir, original, out_path) in enumerate(missing, start=1):
        log.info(f"[{i}/{len(missing)}] [gen] {author_dir} using reference {original.name}")
        item_start = time.time()

        try:
            reference_text = run_with_progress(
                client, ASR_API_NAME, f"[{i}/{len(missing)}] [asr]",
                handle_file(str(original)),
            )
            reference_text = str(reference_text) if reference_text else ""
            log.info(f"[{i}/{len(missing)}] [asr] transcribed reference_text: {reference_text!r}")
        except Exception as e:
            log.error(f"[{i}/{len(missing)}] [error] {author_dir}: ASR transcribe failed: {e}")
            failed.append((author_dir, f"ASR failed: {e}"))
            continue

        try:
            result = run_with_progress(
                client, API_NAME, f"[{i}/{len(missing)}] [gen]",
                **build_kwargs(original, reference_id=author_dir.name, reference_text=reference_text),
            )
        except Exception as e:
            log.error(f"[{i}/{len(missing)}] [error] {author_dir}: {e}")
            failed.append((author_dir, str(e)))
            continue

        # /dispatch returns a tuple: (generated_audio, error_message)
        generated_audio, error_message = result if isinstance(result, (tuple, list)) else (result, "")
        if error_message:
            log.error(f"[{i}/{len(missing)}] [error] {author_dir}: webui returned error: {error_message}")
            failed.append((author_dir, str(error_message)))
            continue

        # result is usually a filepath (str) or dict with 'path' key returned by gradio_client
        result_path: str | None
        if isinstance(generated_audio, dict) and "path" in generated_audio:
            result_path = str(generated_audio["path"])
        elif isinstance(generated_audio, (str, Path)):
            result_path = str(generated_audio)
        else:
            result_path = None

        if not result_path or not Path(result_path).exists():
            log.error(f"[{i}/{len(missing)}] [error] {author_dir}: no output audio returned ({result})")
            failed.append((author_dir, "no output audio returned"))
            continue

        try:
            shutil.copy(str(result_path), str(out_path))
        except Exception as e:
            log.error(f"[{i}/{len(missing)}] [error] {author_dir}: failed saving file: {e}")
            failed.append((author_dir, f"save failed: {e}"))
            continue

        elapsed = time.time() - item_start
        log.info(f"[{i}/{len(missing)}] [done] {author_dir} -> {out_path} ({elapsed:.1f}s)")
        succeeded.append(author_dir)

    total_elapsed = time.time() - run_start
    log.info("")
    log.info("===== SUMMARY =====")
    log.info(f"Total attempted : {len(missing)}")
    log.info(f"Succeeded       : {len(succeeded)}")
    log.info(f"Failed          : {len(failed)}")
    log.info(f"Total time      : {total_elapsed:.1f}s")
    if failed:
        log.info("Failed items:")
        for author_dir, reason in failed:
            log.info(f"  - {author_dir}: {reason}")
        log.info("Re-run the script again - already-succeeded files are skipped automatically,")
        log.info("only failed/missing ones will be retried.")
    log.info(f"Full log saved to: {LOG_PATH.resolve()}")


if __name__ == "__main__":
    main()