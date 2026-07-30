"""
Automated evaluation of cloned/generated TTS audio vs reference audio.

For every author folder under test_audio/{male,female}/<author>/ that has
both an original reference recording AND ai_generated.wav, computes 12
objective metrics and writes them to a CSV - no manual listening required.

Metrics computed:
    1.  speaker_similarity   - voice cloning fidelity (cosine sim of speaker embeddings)
    2.  wer                  - Word Error Rate (generated speech vs target text)
    3.  cer                  - Character Error Rate
    4.  utmos                - predicted naturalness/MOS score (no reference needed)
    5.  snr_db               - estimated signal-to-noise ratio of generated audio
    6.  clipping_pct         - % of samples clipped/distorted
    7.  silence_ratio        - fraction of generated audio that is silence
    8.  duration_sec         - duration sanity check
    9.  pitch_mean_hz        - mean F0 (pitch) of generated audio
    9b. pitch_std_hz         - pitch variation (monotone vs expressive)
    10. speaking_rate_wpm    - words per minute
    11. mfcc_distance        - timbre/spectral distance to reference (DTW)
    12. rms_loudness_db      - average loudness level

USAGE
-----
    pip install librosa jiwer resemblyzer soundfile speechmos --break-system-packages

    python tools/evaluate_generated_audio.py
    python tools/evaluate_generated_audio.py --test-audio-root test_audio --out evaluation_results.csv
"""

import argparse
import csv
import re
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Pass/fail thresholds - each metric has a "good" range; anything outside it
# gets flagged with a plain-English reason in the output CSV.
# Format: (metric_key, comparator, threshold, human_description)
#   comparator "min" -> value must be >= threshold to pass
#   comparator "max" -> value must be <= threshold to pass
# ---------------------------------------------------------------------------
THRESHOLDS = [
    ("speaker_similarity", "min", 0.75, "voice doesn't match reference speaker closely enough"),
    ("wer", "max", 0.30, "too many wrong/missing words"),
    ("cer", "max", 0.20, "too many character-level transcription errors"),
    ("utmos", "min", 2.5, "sounds unnatural/robotic (low predicted MOS)"),
    ("snr_db", "min", 10.0, "too much background noise"),
    ("clipping_pct", "max", 0.1, "audio is clipped/distorted"),
    ("silence_ratio", "max", 0.40, "too much dead air/silence, likely cut off"),
    ("speaking_rate_wpm", "min", 100.0, "speaking too slowly/dragging"),
    ("speaking_rate_wpm_max", "max", 200.0, "speaking too fast/rushed"),
    ("pitch_std_hz", "min", 5.0, "monotone delivery, low pitch variation"),
]


def evaluate_thresholds(row: Dict[str, Any]) -> tuple:
    """Returns (overall_flag, reasons_str, score_summary_str)."""
    reasons = []
    summary_parts = []

    for key, comparator, threshold, description in THRESHOLDS:
        actual_key = "speaking_rate_wpm" if key == "speaking_rate_wpm_max" else key
        value = row.get(actual_key)
        if value is None:
            continue  # metric failed to compute, skip rather than penalize
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value != value:  # NaN check
            continue

        passed = (value >= threshold) if comparator == "min" else (value <= threshold)
        op_symbol = ">=" if comparator == "min" else "<="
        summary_parts.append(f"{actual_key}={value:.3f} (need {op_symbol}{threshold})")
        if not passed:
            reasons.append(f"{actual_key}={value:.3f} fails {op_symbol}{threshold}: {description}")

    overall_flag = "FAIL" if reasons else "PASS"
    reasons_str = "; ".join(reasons) if reasons else "all thresholds met"
    return overall_flag, reasons_str

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
# Strip emotion tags for WER/CER comparison - Whisper won't transcribe "[laugh]" literally
TARGET_TEXT_CLEAN = re.sub(r"\[[a-z]+\]\s*", "", TARGET_TEXT, flags=re.IGNORECASE).replace("\n", " ").strip()
TARGET_WORD_COUNT = len(TARGET_TEXT_CLEAN.split())


def find_files(author_dir: Path):
    """Return (reference_audio_path, generated_audio_path) or (None, None) if incomplete."""
    generated = author_dir / "ai_generated.wav"
    if not generated.exists():
        return None, None
    reference = None
    for f in sorted(author_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in (".wav", ".mp3", ".flac", ".m4a", ".ogg") and f.name.lower() != "ai_generated.wav":
            reference = f
            break
    if reference is None:
        return None, None
    return reference, generated


def iter_author_dirs(root: Path):
    for gender_dir in sorted(root.iterdir()):
        if not gender_dir.is_dir():
            continue
        for author_dir in sorted(gender_dir.iterdir()):
            if author_dir.is_dir():
                yield gender_dir.name, author_dir


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------

def load_audio(path: Path, sr: int = 16000):
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y, sr


def metric_speaker_similarity(ref_path: Path, gen_path: Path, encoder) -> float:
    from resemblyzer import preprocess_wav
    ref_wav = preprocess_wav(str(ref_path))
    gen_wav = preprocess_wav(str(gen_path))
    ref_emb = encoder.embed_utterance(ref_wav)
    gen_emb = encoder.embed_utterance(gen_wav)
    return float(np.dot(ref_emb, gen_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(gen_emb)))


def metric_wer_cer(gen_path: Path, whisper_model) -> tuple:
    import jiwer
    result = whisper_model.transcribe(str(gen_path), language="en")
    hyp = result["text"].strip()
    wer = jiwer.wer(TARGET_TEXT_CLEAN, hyp)
    cer = jiwer.cer(TARGET_TEXT_CLEAN, hyp)
    return wer, cer, hyp


def metric_utmos(gen_path: Path) -> float:
    try:
        import speechmos
        y, sr = load_audio(gen_path, sr=16000)
        score = speechmos.dnsmos.run(y, sr=sr)
        return float(score.get("ovrl_mos", score.get("mos", np.nan)))
    except Exception:
        return float("nan")


def metric_snr_db(gen_path: Path) -> float:
    y, sr = load_audio(gen_path)
    # crude VAD: top 20% energy frames = signal, bottom 20% = noise floor
    frame_len = int(0.025 * sr)
    hop = frame_len // 2
    energies = np.array([
        np.sum(y[i:i + frame_len] ** 2)
        for i in range(0, len(y) - frame_len, hop)
    ])
    if len(energies) < 5:
        return float("nan")
    sorted_e = np.sort(energies)
    noise_floor = np.mean(sorted_e[:max(1, len(sorted_e) // 5)]) + 1e-12
    signal_level = np.mean(sorted_e[-max(1, len(sorted_e) // 5):]) + 1e-12
    return float(10 * np.log10(signal_level / noise_floor))


def metric_clipping_pct(gen_path: Path) -> float:
    y, sr = load_audio(gen_path)
    clipped = np.sum(np.abs(y) >= 0.99)
    return float(100 * clipped / len(y))


def metric_silence_ratio(gen_path: Path) -> float:
    import librosa
    y, sr = load_audio(gen_path)
    intervals = librosa.effects.split(y, top_db=30)
    voiced = sum(end - start for start, end in intervals)
    total = len(y)
    return float(1 - (voiced / total)) if total else float("nan")


def metric_duration(gen_path: Path) -> float:
    y, sr = load_audio(gen_path)
    return float(len(y) / sr)


def metric_pitch_stats(gen_path: Path) -> tuple:
    import librosa
    y, sr = load_audio(gen_path)
    f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
    f0_voiced = f0[voiced_flag] if f0 is not None else np.array([])
    if len(f0_voiced) == 0:
        return float("nan"), float("nan")
    return float(np.nanmean(f0_voiced)), float(np.nanstd(f0_voiced))


def metric_speaking_rate(duration_sec: float) -> float:
    if duration_sec <= 0:
        return float("nan")
    return float(TARGET_WORD_COUNT / (duration_sec / 60))


def metric_mfcc_distance(ref_path: Path, gen_path: Path) -> float:
    import librosa
    y_ref, sr = load_audio(ref_path)
    y_gen, _ = load_audio(gen_path)
    mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr, n_mfcc=13)
    mfcc_gen = librosa.feature.mfcc(y=y_gen, sr=sr, n_mfcc=13)
    D, wp = librosa.sequence.dtw(mfcc_ref, mfcc_gen)
    return float(D[-1, -1] / len(wp))


def metric_rms_loudness_db(gen_path: Path) -> float:
    import librosa
    y, sr = load_audio(gen_path)
    rms = librosa.feature.rms(y=y)[0]
    mean_rms = np.mean(rms) + 1e-12
    return float(20 * np.log10(mean_rms))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate generated TTS audio against reference audio.")
    parser.add_argument("--test-audio-root", type=Path, default=Path("test_audio"))
    parser.add_argument("--out", type=Path, default=Path("evaluation_results.csv"))
    parser.add_argument("--whisper-model", type=str, default="small", help="Whisper model size for WER/CER")
    args = parser.parse_args()

    print("Loading models (whisper + speaker encoder)... this takes a moment")
    import whisper
    from resemblyzer import VoiceEncoder

    whisper_model = whisper.load_model(args.whisper_model)
    encoder = VoiceEncoder()

    fieldnames = [
        "gender", "author", "reference_file", "generated_file",
        "overall_flag", "fail_reasons",
        "speaker_similarity", "wer", "cer", "utmos", "snr_db",
        "clipping_pct", "silence_ratio", "duration_sec",
        "pitch_mean_hz", "pitch_std_hz", "speaking_rate_wpm",
        "mfcc_distance", "rms_loudness_db", "whisper_transcript",
    ]

    rows = []
    for gender, author_dir in iter_author_dirs(args.test_audio_root):
        ref_path, gen_path = find_files(author_dir)
        if ref_path is None or gen_path is None:
            print(f"[skip] {author_dir} - missing reference or generated audio")
            continue

        print(f"[eval] {gender}/{author_dir.name} ...")
        row: Dict[str, Any] = {
            "gender": gender,
            "author": author_dir.name,
            "reference_file": ref_path.name,
            "generated_file": gen_path.name,
        }

        try:
            row["speaker_similarity"] = round(metric_speaker_similarity(ref_path, gen_path, encoder), 4)
        except Exception as e:
            print(f"  speaker_similarity failed: {e}")
            row["speaker_similarity"] = None

        try:
            wer, cer, transcript = metric_wer_cer(gen_path, whisper_model)
            row["wer"] = round(wer, 4)
            row["cer"] = round(cer, 4)
            row["whisper_transcript"] = transcript
        except Exception as e:
            print(f"  wer/cer failed: {e}")
            row["wer"] = row["cer"] = None
            row["whisper_transcript"] = ""

        try:
            row["utmos"] = round(metric_utmos(gen_path), 4)
        except Exception as e:
            print(f"  utmos failed: {e}")
            row["utmos"] = None

        try:
            row["snr_db"] = round(metric_snr_db(gen_path), 2)
        except Exception as e:
            print(f"  snr failed: {e}")
            row["snr_db"] = None

        try:
            row["clipping_pct"] = round(metric_clipping_pct(gen_path), 4)
        except Exception as e:
            print(f"  clipping failed: {e}")
            row["clipping_pct"] = None

        try:
            row["silence_ratio"] = round(metric_silence_ratio(gen_path), 4)
        except Exception as e:
            print(f"  silence failed: {e}")
            row["silence_ratio"] = None

        try:
            duration = metric_duration(gen_path)
            row["duration_sec"] = round(duration, 2)
        except Exception as e:
            print(f"  duration failed: {e}")
            duration = 0
            row["duration_sec"] = None

        try:
            pitch_mean, pitch_std = metric_pitch_stats(gen_path)
            row["pitch_mean_hz"] = round(pitch_mean, 2) if pitch_mean == pitch_mean else None
            row["pitch_std_hz"] = round(pitch_std, 2) if pitch_std == pitch_std else None
        except Exception as e:
            print(f"  pitch failed: {e}")
            row["pitch_mean_hz"] = row["pitch_std_hz"] = None

        try:
            row["speaking_rate_wpm"] = round(metric_speaking_rate(duration), 2)
        except Exception as e:
            print(f"  speaking rate failed: {e}")
            row["speaking_rate_wpm"] = None

        try:
            row["mfcc_distance"] = round(metric_mfcc_distance(ref_path, gen_path), 2)
        except Exception as e:
            print(f"  mfcc failed: {e}")
            row["mfcc_distance"] = None

        try:
            row["rms_loudness_db"] = round(metric_rms_loudness_db(gen_path), 2)
        except Exception as e:
            print(f"  rms failed: {e}")
            row["rms_loudness_db"] = None

        row["overall_flag"], row["fail_reasons"] = evaluate_thresholds(row)
        print(f"  -> {row['overall_flag']}" + (f" ({row['fail_reasons']})" if row["overall_flag"] == "FAIL" else ""))

        rows.append(row)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Evaluated {len(rows)} authors.")
    passed = sum(1 for r in rows if r["overall_flag"] == "PASS")
    failed = len(rows) - passed
    print(f"PASS: {passed}  FAIL: {failed}")
    print(f"Results saved to: {args.out.resolve()}")
    print("\nThreshold reference (used for overall_flag):")
    for key, comparator, threshold, description in THRESHOLDS:
        op = ">=" if comparator == "min" else "<="
        actual_key = "speaking_rate_wpm" if key == "speaking_rate_wpm_max" else key
        print(f"  {actual_key} {op} {threshold}  ({description})")


if __name__ == "__main__":
    main()