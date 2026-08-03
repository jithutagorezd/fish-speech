"""
Blend two reference audios into one hybrid voice, clone it via a running
fish-speech Gradio instance, then measure how close the resulting clone
sounds to each of the two original speakers (cosine similarity).

WORKFLOW
--------
1. Load audio1 and audio2, resample to a common rate, trim to equal length.
2. Blend them: mixed = ratio * audio1 + (1 - ratio) * audio2, peak-normalized.
3. Save the blended audio as a new reference file.
4. Auto-transcribe the blended reference (webui's ASR endpoint).
5. Clone TARGET_TEXT using the blended reference through /dispatch.
6. Compute speaker_similarity(clone, audio1) and speaker_similarity(clone, audio2)
   so you can see which parent voice the hybrid leans toward.

USAGE
-----
    python tools/mix_and_clone.py \\
        --url https://xxxx.gradio.live \\
        --audio1 "test_audio/male/Andrew White/Books_68_..._Retail-Sample.mp3" \\
        --audio2 "test_audio/female/Sophia Williams/Books_423_..._RetailSample.mp3" \\
        --ratio 0.5 \\
        --out-dir mixed_voice_test

    pip install librosa resemblyzer soundfile --break-system-packages   (if not already installed)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("mix_and_clone")

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

ASR_API_NAME = "/_wrapper"
DISPATCH_API_NAME = "/dispatch"
SR = 24000  # common working rate for blending; server resamples internally anyway


def blend_audios(path1: Path, path2: Path, ratio: float, out_path: Path) -> Path:
    import librosa
    import soundfile as sf

    log.info(f"Loading {path1.name} and {path2.name} at {SR} Hz")
    y1, _ = librosa.load(str(path1), sr=SR, mono=True)
    y2, _ = librosa.load(str(path2), sr=SR, mono=True)

    n = min(len(y1), len(y2))
    if n == 0:
        raise ValueError("One of the audio files is empty after loading.")
    y1, y2 = y1[:n], y2[:n]

    # normalize each to unit peak before blending, so one loud file doesn't dominate
    y1 = y1 / (np.max(np.abs(y1)) + 1e-9)
    y2 = y2 / (np.max(np.abs(y2)) + 1e-9)

    mixed = ratio * y1 + (1 - ratio) * y2
    peak = np.max(np.abs(mixed)) + 1e-9
    if peak > 1.0:
        mixed = mixed / peak * 0.98  # leave a little headroom, avoid clipping

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), mixed, SR)
    log.info(f"Blended (overlapped) audio saved to {out_path} (ratio={ratio:.2f}, duration={n / SR:.1f}s)")
    return out_path


def concat_audios(path1: Path, path2: Path, out_path: Path, seconds_each: float = 15.0) -> Path:
    """
    Sequential concatenation: first `seconds_each` of audio1, followed by
    first `seconds_each` of audio2 - one voice, then the other, back to back.
    This is what "first half is author A, second half is author B" means.
    """
    import librosa
    import soundfile as sf

    log.info(f"Loading {path1.name} and {path2.name} at {SR} Hz for concatenation")
    y1, _ = librosa.load(str(path1), sr=SR, mono=True, duration=seconds_each)
    y2, _ = librosa.load(str(path2), sr=SR, mono=True, duration=seconds_each)

    concatenated = np.concatenate([y1, y2])
    peak = np.max(np.abs(concatenated)) + 1e-9
    if peak > 1.0:
        concatenated = concatenated / peak * 0.98

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), concatenated, SR)
    total_dur = len(concatenated) / SR
    log.info(f"Concatenated audio saved to {out_path} (audio1: {len(y1)/SR:.1f}s + audio2: {len(y2)/SR:.1f}s = {total_dur:.1f}s)")
    return out_path


def run_with_progress(client, api_name: str, log_prefix: str, *args, poll_interval: float = 3.0, **kwargs):
    job = client.submit(*args, api_name=api_name, **kwargs)
    last_msg = None
    while not job.done():
        status = job.status()
        eta = f", eta {status.eta:.0f}s" if status.eta else ""
        msg = f"{log_prefix} status: {status.code.name.lower() if status.code else 'running'}{eta}"
        if msg != last_msg:
            log.info(msg)
            last_msg = msg
        time.sleep(poll_interval)
    return job.result()


def clone_voice(url: str, reference_audio: Path, target_text: str, out_path: Path) -> Path:
    from gradio_client import Client, handle_file

    log.info(f"Connecting to {url}")
    client = Client(url)

    log.info("Transcribing blended reference audio (ASR)...")
    reference_text = run_with_progress(client, ASR_API_NAME, "[asr]", handle_file(str(reference_audio)))
    reference_text = str(reference_text) if reference_text else ""
    log.info(f"Transcribed reference_text: {reference_text!r}")

    log.info("Generating cloned audio from blended reference...")
    kwargs = {
        "text": target_text,
        "reference_id": "",  # must stay empty - non-empty loads from preset library instead of our audio
        "reference_audio": handle_file(str(reference_audio)),
        "reference_text": reference_text,
        "max_new_tokens": 400,
        "chunk_length": 200,
        "top_p": 0.8,
        "repetition_penalty": 1.1,
        "temperature": 0.8,
        "seed": 0,
        "use_memory_cache": "off",
        "max_words_per_chunk": 200,
        "mode_radio": "Single shot",
    }
    result = run_with_progress(client, DISPATCH_API_NAME, "[gen]", **kwargs)
    generated_audio, error_message = result if isinstance(result, (tuple, list)) else (result, "")
    if error_message:
        raise RuntimeError(f"webui returned error: {error_message}")

    result_path = generated_audio["path"] if isinstance(generated_audio, dict) and "path" in generated_audio else generated_audio
    if not result_path or not Path(result_path).exists():
        raise RuntimeError(f"no output audio returned ({result})")

    import shutil
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(result_path), str(out_path))
    log.info(f"Cloned hybrid voice saved to {out_path}")
    return out_path


def speaker_similarity(path_a: Path, path_b: Path, encoder) -> float:
    from resemblyzer import preprocess_wav
    wav_a = preprocess_wav(str(path_a))
    wav_b = preprocess_wav(str(path_b))
    emb_a = encoder.embed_utterance(wav_a)
    emb_b = encoder.embed_utterance(wav_b)
    return float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))


def main():
    parser = argparse.ArgumentParser(description="Blend 2 reference audios, clone the hybrid, measure similarity to each.")
    parser.add_argument("--url", required=True, help="Gradio share link")
    parser.add_argument("--audio1", required=True, type=Path)
    parser.add_argument("--audio2", required=True, type=Path)
    parser.add_argument("--mode", choices=["blend", "concat"], default="concat",
                         help="'concat' = first half author A, second half author B (sequential). "
                              "'blend' = both voices overlapped/averaged simultaneously.")
    parser.add_argument("--seconds-each", type=float, default=15.0,
                         help="For --mode concat: how many seconds to take from each audio.")
    parser.add_argument("--ratio", type=float, default=0.5, help="For --mode blend: weight of audio1 in the mix (0-1).")
    parser.add_argument("--target-text", type=str, default=TARGET_TEXT)
    parser.add_argument("--out-dir", type=Path, default=Path("mixed_voice_test"))
    args = parser.parse_args()

    if not args.audio1.exists():
        sys.exit(f"audio1 not found: {args.audio1}")
    if not args.audio2.exists():
        sys.exit(f"audio2 not found: {args.audio2}")

    blended_path = args.out_dir / "combined_reference.wav"
    cloned_path = args.out_dir / "hybrid_generated.wav"

    if args.mode == "concat":
        concat_audios(args.audio1, args.audio2, blended_path, seconds_each=args.seconds_each)
    else:
        if not (0.0 <= args.ratio <= 1.0):
            sys.exit("--ratio must be between 0 and 1")
        blend_audios(args.audio1, args.audio2, args.ratio, blended_path)
    clone_voice(args.url, blended_path, args.target_text, cloned_path)

    log.info("Loading speaker encoder (CPU) for similarity scoring...")
    from resemblyzer import VoiceEncoder
    encoder = VoiceEncoder(device="cpu")

    sim_to_1 = speaker_similarity(cloned_path, args.audio1, encoder)
    sim_to_2 = speaker_similarity(cloned_path, args.audio2, encoder)
    # also measure how the raw blend itself compares, before cloning - useful context
    blend_sim_to_1 = speaker_similarity(blended_path, args.audio1, encoder)
    blend_sim_to_2 = speaker_similarity(blended_path, args.audio2, encoder)

    print("\n===== RESULTS =====")
    print(f"Mode                : {args.mode}")
    if args.mode == "concat":
        print(f"Seconds each        : {args.seconds_each}s from audio1, {args.seconds_each}s from audio2")
    else:
        print(f"Blend ratio         : {args.ratio:.2f} (audio1) / {1 - args.ratio:.2f} (audio2)")
    print(f"Combined ref vs audio1 similarity : {blend_sim_to_1:.4f}")
    print(f"Combined ref vs audio2 similarity : {blend_sim_to_2:.4f}")
    print(f"Cloned hybrid vs audio1 similarity : {sim_to_1:.4f}")
    print(f"Cloned hybrid vs audio2 similarity : {sim_to_2:.4f}")
    if sim_to_1 > sim_to_2:
        print(f"-> Hybrid clone leans toward audio1 ({args.audio1.name})")
    elif sim_to_2 > sim_to_1:
        print(f"-> Hybrid clone leans toward audio2 ({args.audio2.name})")
    else:
        print("-> Hybrid clone is equally balanced between both speakers")
    print(f"\nFiles saved in: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()