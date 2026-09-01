"""
Fish Speech WebUI v2 — rich interface with long-form TTS support.
"""

import os
from typing import Optional

import gradio as gr

from fish_speech.i18n import i18n

from webui_v2.inference import (
    build_html_error_message,
    get_inference_long_form_wrapper,
    get_inference_single_wrapper,
    get_whisper_transcribe_wrapper,
)
import re

from webui_v2.utils import count_words
from webui_v2.groq_llm import auto_tag_text, detect_emotion

def _auto_tag_running():
    return gr.update(value="⏳ Tagging…", interactive=False)

def _auto_tag_done(text):
    if not text or not text.strip():
        return gr.update(), gr.update(value="✨ Auto Tag Text", interactive=True)
    try:
        return auto_tag_text(text), gr.update(value="✨ Auto Tag Text", interactive=True)
    except Exception:
        return gr.update(), gr.update(value="✨ Auto Tag Text", interactive=True)

def _tags_html(text):
    found = sorted(set(m.lower().strip() for m in re.findall(r"\[([a-zA-Z ]+)\]", text or "")))
    if not found:
        return ""
    chips = "".join(f'<span class="tag-chip">[{t}]</span>' for t in found)
    return f'<div class="tag-chip-row">{chips}</div>'

def _update_auto_tag_btn(text):
    return gr.update(interactive=bool(text and text.strip()))

# Preset "pick a ready-made voice" cards. Each id must match a folder under
# references/<id>/ containing at least one audio file + matching .lab
# transcript (fish_speech.inference_engine.reference_loader.ReferenceLoader
# reads that folder structure directly — see load_by_id/list_reference_ids).
# reference_id already flows straight through dispatch() -> run_single /
# run_long_form -> ServeTTSRequest, and the engine prefers reference_id over
# any uploaded reference_audio, so selecting a preset needs no changes on
# the inference side at all.
# Removed PRESET_VOICES as Author dropdown now dynamically reads from reference_audio

HEADER_HTML = """
<div style="margin-bottom: 28px;">
  <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
    <span style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted-c);">
      Voice Synthesis
    </span>
  </div>
  <h1 style="font-family: 'Space Grotesk', sans-serif; font-size: 26px; font-weight: 700; margin: 0; letter-spacing: -0.01em; color: var(--text-c);">
    Voice Cloning Studio
  </h1>
  <p style="color: var(--muted-c); font-size: 13.5px; margin-top: 6px; max-width: 480px;">
    Write your text, record or upload a voice to clone, and let the emotion detect itself — or set it yourself.
  </p>
</div>
"""

STEP_RAIL_HTML = """
<div style="padding: 16px 18px 4px; border: 1px solid var(--border-c); border-radius: 10px; background: var(--panel); margin-bottom: 28px;">
  <div style="display: flex; align-items: center; width: 100%;">
    <div style="display: flex; align-items: center; flex: 1;">
      <div style="display: flex; flex-direction: column; align-items: center; gap: 6px;">
        <div style="width: 9px; height: 9px; border-radius: 50%; background: var(--accent); border: 1.5px solid var(--accent);"></div>
        <span style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted-c);">Text</span>
      </div>
      <div style="height: 1px; flex: 1; background: var(--border-c); margin: 0 8px 18px 8px; position: relative;">
        <div style="position: absolute; inset: 0; background: var(--accent); transform: scaleX(1); transform-origin: left;"></div>
      </div>
    </div>
    <div style="display: flex; align-items: center; flex: 1;">
      <div style="display: flex; flex-direction: column; align-items: center; gap: 6px;">
        <div style="width: 9px; height: 9px; border-radius: 50%; background: var(--accent); border: 1.5px solid var(--accent);"></div>
        <span style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted-c);">Voice</span>
      </div>
      <div style="height: 1px; flex: 1; background: var(--border-c); margin: 0 8px 18px 8px; position: relative;">
        <div style="position: absolute; inset: 0; background: var(--accent); transform: scaleX(0); transform-origin: left;"></div>
      </div>
    </div>
    <div style="display: flex; align-items: center; flex: 0 0 auto;">
      <div style="display: flex; flex-direction: column; align-items: center; gap: 6px;">
        <div style="width: 9px; height: 9px; border-radius: 50%; background: transparent; border: 1.5px solid var(--border-c);"></div>
        <span style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted-c);">Audio</span>
      </div>
    </div>
  </div>
</div>
"""


# The exact "common tags" list documented for this checkpoint — see
# checkpoints/s2-pro/README.md, "Fine-Grained Inline Control". S2 Pro accepts
# free-form bracketed text (not a fixed enum), but these are the tags it
# documents as reliably supported; anything else is unverified. Grouped here
# purely for the UI's expanded view.
EMOTION_TAG_GROUPS = [
    ("Pacing", ["pause", "short pause", "emphasis", "interrupting"]),
    ("Breath & texture", ["inhale", "exhale", "sigh", "panting", "moaning", "clearing throat", "tsk"]),
    ("Laughter", ["laughing", "chuckle", "chuckling", "laughing tone", "audience laughter"]),
    ("Dynamics", ["volume up", "volume down", "low volume", "loud", "echo", "low voice"]),
]

# A small, always-visible subset of the verified tags above, presented as
# single-tap "mood" pills (mirroring a common voice-app pattern) — the full
# grouped list is still available right below for anything not covered here.
QUICK_EMOTION_TAGS = ["excited", "angry", "sad", "delight", "surprised", "shocked"]


def _build_emotion_tags_html() -> str:
    def chip(tag: str) -> str:
        return f'<button type="button" class="emotion-tag" data-tag="{tag}">[{tag}]</button>'

    groups_html = ""
    for group_name, group_tags in EMOTION_TAG_GROUPS:
        tags_html = "".join(chip(t) for t in group_tags)
        groups_html += (
            f'<div class="emotion-tag-group">'
            f'<div class="emotion-tag-group-label">{group_name}</div>'
            f'<div class="emotion-tags">{tags_html}</div>'
            f"</div>"
        )

    # Collapsed by default — the whole tag picker is one line until opened,
    # so the default view stays to text / voice / generate.
    return f"""
<details class="emotion-tag-details">
  <summary>🎭 More tags — click to insert</summary>
  <div class="emotion-tag-groups">{groups_html}</div>
</details>
"""


def _build_quick_emotion_html() -> str:
    chips = ['<button type="button" class="emotion-pill emotion-pill-active" data-tag="">None</button>']
    for tag in QUICK_EMOTION_TAGS:
        chips.append(
            f'<button type="button" class="emotion-pill" data-tag="{tag}">{tag.capitalize()}</button>'
        )
    return f'<div class="emotion-pills">{"".join(chips)}</div>'


EMOTION_TAGS_HTML = _build_emotion_tags_html()
QUICK_EMOTION_HTML = _build_quick_emotion_html()

# Delegated click handler + native insert-at-cursor, injected once via
# app.load(js=...) below. Targets the real <textarea> Gradio renders inside
# the #script-input wrapper and dispatches a real "input" event so Gradio's
# own reactivity (word count, etc.) picks up the change exactly as if the
# user had typed it. Handles both the full grouped list (.emotion-tag) and
# the quick pill row (.emotion-pill), which also tracks a visual
# active/selected pill purely as UI feedback.
EMOTION_TAG_INSERT_JS = """
() => {
  function insertTag(tag) {
    const ta = document.querySelector('#script-input textarea');
    if (!ta) return;
    const insert = '[' + tag + '] ';
    const start = ta.selectionStart ?? ta.value.length;
    const end = ta.selectionEnd ?? ta.value.length;
    ta.value = ta.value.slice(0, start) + insert + ta.value.slice(end);
    ta.dispatchEvent(new Event('input', { bubbles: true }));
    ta.focus();
    const pos = start + insert.length;
    ta.setSelectionRange(pos, pos);
  }
  document.addEventListener('click', (e) => {
    const chip = e.target.closest('.emotion-tag[data-tag]');
    if (chip) {
      e.preventDefault();
      insertTag(chip.dataset.tag);
      return;
    }
    const pill = e.target.closest('.emotion-pill[data-tag]');
    if (pill) {
      e.preventDefault();
      const group = pill.closest('.emotion-pills');
      if (group) {
        group.querySelectorAll('.emotion-pill').forEach((el) => el.classList.remove('emotion-pill-active'));
      }
      pill.classList.add('emotion-pill-active');
      if (pill.dataset.tag) insertTag(pill.dataset.tag);
    }
  });
}
"""

FOOTER_HTML = """
<div class="fish-footer"><b>Audifyz</b> — clone any voice</div>
"""

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700;800&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --bg: #0F1114;
    --panel: #15181D;
    --panel-2: #1A1D22;
    --border-c: #2A2E35;
    --text-c: #EEEFF1;
    --muted-c: #8B9099;
    --accent: #E8A33D;
    --accent-hover: #F3B860;
    --accent-dark: #3A3F49;
    --font-mono: 'IBM Plex Mono', 'SFMono-Regular', monospace;
}

html, body {
    margin: 0 !important;
    background: var(--bg) !important;
}

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding-left: 28px !important;
    padding-right: 28px !important;
    padding-bottom: 24px !important;
    display: flex !important;
    flex-direction: column !important;
    box-sizing: border-box !important;
    background: var(--bg) !important;
    color: var(--text-c) !important;
    --body-background-fill: var(--bg) !important;
    --background-fill-primary: var(--panel) !important;
    --background-fill-secondary: var(--panel-2) !important;
    --block-background-fill: var(--panel) !important;
    --border-color-primary: var(--border-c) !important;
    --border-color-accent: var(--accent) !important;
    --body-text-color: var(--text-c) !important;
    --body-text-color-subdued: var(--muted-c) !important;
    --input-background-fill: var(--panel-2) !important;
    --button-secondary-background-fill: var(--panel-2) !important;
    --button-secondary-text-color: var(--text-c) !important;
    --button-secondary-border-color: var(--border-c) !important;
}

.centered-container {
    max-width: 640px !important;
    margin: 0 auto !important;
    padding: 24px 0 !important;
    width: 100% !important;
}

.gradio-container .block,
.gradio-container .form {
    background: var(--panel) !important;
    border-color: var(--border-c) !important;
    color: var(--text-c) !important;
}

.gradio-container textarea,
.gradio-container input[type="text"],
.gradio-container input[type="number"],
.gradio-container select {
    background: var(--panel-2) !important;
    color: var(--text-c) !important;
    border-color: var(--border-c) !important;
}
.gradio-container label span,
.gradio-container .label-wrap span {
    color: var(--muted-c) !important;
}

/* Fix dropdown styling */
.gradio-dropdown {
    background: var(--panel-2) !important;
    border: 1px solid var(--border-c) !important;
    border-radius: 8px !important;
    padding: 0 !important;
}

.fish-card {
    background: var(--panel) !important;
    border: 1px solid var(--border-c) !important;
    border-radius: 14px !important;
    box-shadow: none !important;
}

.fish-card-nested {
    background: var(--panel-2) !important;
    border: 1px solid var(--border-c) !important;
    border-radius: 12px !important;
    box-shadow: none !important;
}

/* Audio inputs */
#voice-source-row {
    gap: 12px;
}
#mic-audio, #upload-audio {
    width: 100% !important;
    height: 180px;
    max-width: 100% !important;
    min-width: 0 !important;
    overflow: hidden !important;
    border: 1px dashed var(--border-c) !important;
    border-radius: 12px !important;
}
#mic-audio *, #upload-audio * {
    transition: none !important;
}

/* Author / Emotion — override Gradio's mobile row-stacking */
#voice-dropdown-row {
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    gap: 12px !important;
}
#voice-dropdown-row > * {
    min-width: 0 !important;
    flex: 1 1 0 !important;
}
/* only stack on genuinely tiny screens */
@media (max-width: 380px) {
    #voice-dropdown-row { flex-direction: column !important; }
}

/* Generate Button */
#generate-row {
    display: flex !important;
    justify-content: center !important;
    margin-top: 16px !important;
}
#generate-row > * {
    width: auto !important;
    flex: 0 0 auto !important;
}
#generate-btn {
    width: auto !important;
    height: auto !important;
    font-size: 14.5px !important;
    font-weight: 600 !important;
    padding: 13px 28px !important;
    border-radius: 999px !important;
    background: var(--accent) !important;
    border: none !important;
    color: #171106 !important;
    box-shadow: 0 8px 24px rgba(232, 163, 61, 0.22) !important;
    margin-top: 0 !important;
}
#generate-btn:hover {
    background: var(--accent-hover) !important;
    box-shadow: 0 10px 28px rgba(232, 163, 61, 0.32) !important;
}

/* Advanced settings */
#advanced-settings {
    background: var(--panel-2) !important;
    border: 1px solid var(--border-c) !important;
    border-radius: 12px !important;
    margin-top: 16px;
}

/* Emotion tags */
.emotion-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 6px;
}
.emotion-tag {
    font: inherit;
    font-size: 0.78rem;
    padding: 5px 11px;
    border-radius: 999px;
    background: rgba(232, 163, 61, 0.08);
    border: 1px solid rgba(232, 163, 61, 0.25);
    color: var(--text-c);
    cursor: pointer;
}
.emotion-tag:hover {
    background: rgba(232, 163, 61, 0.16);
}

.emotion-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 8px 0 4px 0;
}
.emotion-pill {
    font: inherit;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 999px;
    background: var(--panel-2);
    border: 1px solid var(--border-c);
    color: var(--text-c);
    cursor: pointer;
}
.emotion-pill:hover { border-color: var(--accent); }
.emotion-pill-active {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #000000 !important;
}

#audio-output {
    border-radius: 12px !important;
    height: 180px;
    background: var(--panel) !important;
    transition: none !important;
}

#output-placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    min-height: 90px;
    color: var(--muted-c);
    font-size: 0.85rem;
    background: var(--panel);
    border-radius: 10px;
    margin-bottom: 8px;
}
#output-placeholder:empty { display: none; }

.word-badge {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--muted-c);
}
.longform-nudge {
    font-size: 0.75rem;
    color: var(--accent);
    margin-left: 8px;
}
.word-count-wrapper {
    text-align: right;
    margin-top: -6px;
    margin-bottom: 12px;
    padding-right: 8px;
}
.auto-tag-row {
    margin-bottom: 8px;
}
.auto-tag-btn {
    background: rgba(232, 163, 61, 0.12) !important;
    border: 1px solid rgba(232, 163, 61, 0.35) !important;
    color: #F3B860 !important;
    border-radius: 999px !important;
    font-weight: 600 !important;
    font-size: 12.5px !important;
    padding: 8px 14px !important;
    width: auto !important;
    flex: 0 0 auto !important;
}
.auto-tag-btn:hover { background: rgba(232, 163, 61, 0.2) !important; }

.tag-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 8px;
}
.tag-chip {
    font-family: var(--font-mono);
    font-size: 11px;
    background: var(--panel-2);
    border: 1px solid var(--border-c);
    color: var(--accent);
    padding: 3px 6px;
    border-radius: 6px;
}

/* Emotion tag details panel */
.emotion-tag-details {
    background: var(--panel-2);
    border: 1px solid var(--border-c);
    border-radius: 10px;
    margin-top: 8px;
    padding: 0;
}
.emotion-tag-details summary {
    padding: 10px 14px;
    cursor: pointer;
    font-size: 13px;
    color: var(--muted-c);
    font-weight: 500;
    list-style: none;
    user-select: none;
}
.emotion-tag-details summary::-webkit-details-marker {
    display: none;
}
.emotion-tag-details summary:hover {
    color: var(--text-c);
}
.emotion-tag-groups {
    padding: 0 14px 14px 14px;
    border-top: 1px solid var(--border-c);
    margin-top: 4px;
}
"""

FISH_THEME = gr.themes.Base(
    primary_hue="orange",
    secondary_hue="stone",
    neutral_hue="zinc",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    block_radius="14px",
    block_shadow="none",
    button_primary_background_fill="#E8A33D",
    button_primary_background_fill_hover="#F3B860",
    button_primary_text_color="#171106",
    button_large_radius="12px",
    button_large_padding="14px 24px",
    button_small_padding="8px 16px",
)


# Past this many words, single-shot generation tends to get slow/unwieldy —
# nudge toward Long-form instead of making the user go find the toggle.
LONG_FORM_NUDGE_THRESHOLD = 500


def _word_count_display(text: Optional[str]) -> str:
    n = count_words(text) if text and text.strip() else 0
    html = f'<span class="word-badge">📝 {n} words</span>'
    if n > LONG_FORM_NUDGE_THRESHOLD:
        html += (
            '<span class="longform-nudge">Long document — enable '
            "Long-form in Advanced settings</span>"
        )
    return html


def _voice_status_display(
    source: str, preset_id: Optional[str], reference_audio_path: Optional[str]
) -> str:
    if source == "preset" and preset_id:
        return f'<span class="voice-status voice-status-active">🎭 Using preset voice: {preset_id}</span>'
    if reference_audio_path:
        return '<span class="voice-status voice-status-active">🎙️ Using: Your voice</span>'
    return '<span class="voice-status voice-status-none">⚠️ No voice selected — record, upload, or pick a preset</span>'


def _resolve_reference_audio(
    mic_path: Optional[str], upload_path: Optional[str], source: str
) -> Optional[str]:
    """Pick whichever of the two source widgets should actually be used.

    Prefers whichever widget the source state says was touched last, but
    falls back to whichever one actually has a value (e.g. the active one
    was just cleared but the other still holds an earlier recording).
    """
    if source == "upload" and upload_path:
        return upload_path
    if source == "mic" and mic_path:
        return mic_path
    return mic_path or upload_path


def build_app(
    engine,
    theme: str = "light",
    whisper_model_dir: str = "checkpoints/whisper-small-pt",
) -> gr.Blocks:
    inference_single_fn = get_inference_single_wrapper(engine)
    inference_long_fn = get_inference_long_form_wrapper(engine)
    whisper_transcribe_fn = get_whisper_transcribe_wrapper(whisper_model_dir)

    def run_single(
        text,
        reference_id,
        reference_audio,
        reference_text,
        max_new_tokens,
        chunk_length,
        top_p,
        repetition_penalty,
        temperature,
        seed,
        use_memory_cache,
    ):
        audio, err = inference_single_fn(
            text=text,
            reference_id=reference_id,
            reference_audio=reference_audio,
            reference_text=reference_text,
            max_new_tokens=max_new_tokens,
            chunk_length=chunk_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            seed=seed,
            use_memory_cache=use_memory_cache,
        )
        return audio, err or ""

    def run_long_form(
        text,
        reference_id,
        reference_audio,
        reference_text,
        max_new_tokens,
        chunk_length,
        top_p,
        repetition_penalty,
        temperature,
        seed,
        use_memory_cache,
        max_words_per_chunk,
        progress=gr.Progress(),
    ):
        progress(0, "Preparing chunks…")
        audio, err = inference_long_fn(
            text=text,
            reference_id=reference_id,
            reference_audio=reference_audio,
            reference_text=reference_text,
            max_new_tokens=max_new_tokens,
            chunk_length=chunk_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            seed=seed,
            use_memory_cache=use_memory_cache,
            max_words_per_chunk=max_words_per_chunk,
            progress=lambda frac, msg: progress(frac, msg),
        )
        return audio, err
    import os
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    REF_AUDIO_DIR = os.path.join(ROOT_DIR, "reference_audio")
    
    AUTHOR_EMOTIONS = {}
    
    if os.path.isdir(REF_AUDIO_DIR):
        subfolders = [f for f in os.listdir(REF_AUDIO_DIR) if os.path.isdir(os.path.join(REF_AUDIO_DIR, f))]
        for speaker in subfolders:
            speaker_dir = os.path.join(REF_AUDIO_DIR, speaker)
            emotions = []
            for file in os.listdir(speaker_dir):
                if file.endswith((".mp3", ".wav", ".flac", ".ogg", ".m4a")):
                    emotion = os.path.splitext(file)[0]
                    emotions.append((emotion.capitalize(), os.path.join(speaker_dir, file)))
            emotions.sort(key=lambda x: x[0])
            if emotions:
                AUTHOR_EMOTIONS[speaker] = emotions
                
    with gr.Blocks(
        title="Audifyz Voice Cloner",
        theme=FISH_THEME,
        css=CUSTOM_CSS,
    ) as app:
        gr.HTML(HEADER_HTML)

        app.load(
            None,
            None,
            js="() => {const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {params.set('__theme', 'dark');window.location.search = params.toString();}}"
        )
        app.load(None, None, js=EMOTION_TAG_INSERT_JS)

        with gr.Tabs():
            with gr.Tab("🎙️ Generate Speech"):
                with gr.Column(elem_classes="centered-container"):
                    gr.HTML(STEP_RAIL_HTML)

            # --- STEP 1 ---
            gr.HTML('<div style="margin-bottom: 10px"><div style="display: flex; align-items: center; gap: 9px;"><span style="width: 20px; height: 20px; border-radius: 50%; border: 1px solid var(--accent); color: var(--accent); display: flex; align-items: center; justify-content: center; font-family: var(--font-mono); font-size: 10.5px; font-weight: 600;">1</span><span style="font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted-c); font-weight: 500;">Text</span></div><h2 style="font-family: \'Space Grotesk\', sans-serif; font-size: 18px; font-weight: 600; color: var(--text-c); margin: 16px 0 0 0;">Text to speech</h2></div>')
            with gr.Group(elem_classes="fish-card"):
                text_input = gr.Textbox(
                    label="Input Text",
                    placeholder="Paste or type text here. For long documents (3000–5000 words), turn on Long-form in Advanced settings.",
                    lines=4,
                    max_lines=30,
                    show_label=False,
                    elem_id="script-input",
                )
                word_count = gr.HTML(_word_count_display(None), elem_classes="word-count-wrapper")
                
                with gr.Row(elem_classes="auto-tag-row"):
                    auto_tag_btn = gr.Button("✨ Auto Tag Text", size="sm", variant="secondary", elem_classes=["auto-tag-btn"], interactive=False)
                tags_readout = gr.HTML("")
                
                gr.HTML(QUICK_EMOTION_HTML)
                gr.HTML(EMOTION_TAGS_HTML)

                auto_tag_btn.click(
                    fn=_auto_tag_running, inputs=None, outputs=auto_tag_btn,
                ).then(
                    fn=_auto_tag_done, inputs=[text_input], outputs=[text_input, auto_tag_btn],
                )

                text_input.change(
                    fn=_word_count_display,
                    inputs=[text_input],
                    outputs=[word_count],
                )
                text_input.change(
                    fn=_tags_html,
                    inputs=[text_input],
                    outputs=[tags_readout],
                )
                text_input.change(
                    fn=_update_auto_tag_btn,
                    inputs=[text_input],
                    outputs=[auto_tag_btn],
                )

            # --- STEP 2 ---
            gr.HTML('<div style="margin-bottom: 10px; margin-top: 30px;"><div style="display: flex; align-items: center; gap: 9px;"><span style="width: 20px; height: 20px; border-radius: 50%; border: 1px solid var(--accent); color: var(--accent); display: flex; align-items: center; justify-content: center; font-family: var(--font-mono); font-size: 10.5px; font-weight: 600;">2</span><span style="font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted-c); font-weight: 500;">Voice</span></div><h2 style="font-family: \'Space Grotesk\', sans-serif; font-size: 18px; font-weight: 600; color: var(--text-c); margin: 16px 0 4px 0;">Reference voice</h2><p style="font-size: 13.5px; color: var(--muted-c); margin: 0;">Record or upload a reference sample to clone the selected author voice. Emotion is chosen separately for generation.</p></div>')
            
            with gr.Row(elem_id="voice-dropdown-row"):
                with gr.Column(scale=1):
                    gr.Markdown("<label style='font-size:12.5px;font-weight:500;color:var(--muted-c);margin-bottom:8px;display:block'>Author</label>")
                    author_choices = [("— None —", "")] + [(k.capitalize(), k) for k in sorted(AUTHOR_EMOTIONS.keys())]
                    preset_radio = gr.Dropdown(
                        show_label=False,
                        choices=author_choices,
                        value="",
                        container=False,
                    )
                with gr.Column(scale=1):
                    gr.Markdown("<label style='font-size:12.5px;font-weight:500;color:var(--muted-c);margin-bottom:8px;display:block'>Emotion</label>")
                    hanna_radio = gr.Dropdown(
                        show_label=False,
                        choices=[("— None —", ""), ("— Auto from text —", "auto")],
                        value="",
                        container=False,
                    )
            
            preset_preview = gr.Audio(
                show_label=False,
                type="filepath",
                interactive=False,
                visible=False,
                elem_id="preset-preview-audio",
            )
            
            gr.Markdown("<label style='font-size:12.5px;font-weight:500;color:var(--muted-c);margin:20px 0 8px 0;display:block'>Reference audio</label>")
            with gr.Row(elem_id="voice-source-row"):
                mic_audio = gr.Audio(
                    show_label=False,
                    type="filepath",
                    sources=["microphone"],
                    elem_id="mic-audio",
                )
                upload_audio = gr.Audio(
                    show_label=False,
                    type="filepath",
                    sources=["upload"],
                    elem_id="upload-audio",
                )
            
            voice_status = gr.HTML(
                _voice_status_display("mic", "", None),
                elem_id="voice-status",
            )
            active_source = gr.State("mic")

            with gr.Row(elem_id="generate-row"):
                generate_btn = gr.Button(
                    "🎙️ " + i18n("Generate speech"),
                    variant="primary",
                    size="lg",
                    elem_id="generate-btn",
                )

            # --- STEP 3 ---
            gr.HTML('<div style="margin-bottom: 10px; margin-top: 30px;"><div style="display: flex; align-items: center; gap: 9px;"><span style="width: 20px; height: 20px; border-radius: 50%; border: 1px solid var(--accent); color: var(--accent); display: flex; align-items: center; justify-content: center; font-family: var(--font-mono); font-size: 10.5px; font-weight: 600;">3</span><span style="font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted-c); font-weight: 500;">Audio</span></div><h2 style="font-family: \'Space Grotesk\', sans-serif; font-size: 18px; font-weight: 600; color: var(--text-c); margin: 16px 0 0 0;">Generated audio</h2></div>')
            
            with gr.Group(elem_classes=["fish-card-nested"]):
                error_out = gr.HTML(label=i18n("Error Message"), value="", elem_id="error-box")
                output_placeholder = gr.HTML(
                    '<div class="output-placeholder-text">🔈 Your generated audio will appear here</div>',
                    elem_id="output-placeholder",
                )
                audio_out = gr.Audio(
                    label=i18n("Generated Audio"),
                    type="numpy",
                    interactive=False,
                    autoplay=True,
                    elem_id="audio-output",
                )
                regenerate_btn = gr.Button(
                    "🔁 Regenerate",
                    variant="secondary",
                    size="sm",
                    elem_id="regenerate-btn",
                )
            
            with gr.Accordion("⚙️ " + i18n("Advanced settings"), open=False, elem_id="advanced-settings"):
                mode = gr.Radio(
                    label="Mode",
                    choices=["Single shot", "Long-form (chunked)"],
                    value="Single shot",
                )
                max_words_per_chunk = gr.Slider(
                    label="Max words per chunk (long-form only)",
                    minimum=100,
                    maximum=500,
                    value=200,
                    step=50,
                )
                reference_id = gr.Textbox(
                    label=i18n("Reference ID"),
                    placeholder="Auto-filled when you pick a preset voice, or leave empty to use the reference audio above",
                    value="",
                )
                transcribe_btn = gr.Button(
                    "🎤 " + i18n("Auto-transcribe reference audio with Whisper"),
                    variant="secondary",
                    size="sm",
                )
                reference_text = gr.Textbox(
                    label=i18n("Reference Text"),
                    placeholder="Transcription of the reference audio, or use Auto-transcribe. Improves voice-clone accuracy but isn't required.",
                    lines=2,
                    value="",
                )
                use_memory_cache = gr.Radio(
                    label=i18n("Use Memory Cache"),
                    choices=["on", "off"],
                    value="on",
                )
                chunk_length = gr.Slider(
                    label=i18n("Iterative Prompt Length, 0 means off"),
                    minimum=100,
                    maximum=300,
                    value=200,
                    step=8,
                )
                max_new_tokens = gr.Slider(
                    label=i18n("Maximum tokens per batch, 0 means no limit"),
                    minimum=0,
                    maximum=2048,
                    value=0,
                    step=8,
                )
                with gr.Row():
                    top_p = gr.Slider(label="Top-P", minimum=0.7, maximum=0.95, value=0.8, step=0.01)
                    repetition_penalty = gr.Slider(
                        label=i18n("Repetition Penalty"),
                        minimum=1,
                        maximum=1.2,
                        value=1.1,
                        step=0.01,
                    )
                with gr.Row():
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.7,
                        maximum=1.0,
                        value=0.8,
                        step=0.01,
                    )
                    seed = gr.Number(
                        label="Seed",
                        value=0,
                        precision=0,
                    )

            with gr.Tab("➕ Create Voice"):
                with gr.Column(elem_classes="centered-container"):
                    gr.HTML('<h2 style="font-family: \'Space Grotesk\', sans-serif; font-size: 22px; font-weight: 700; color: var(--text-c); margin: 0 0 8px 0;">Create Your Voice</h2><p style="color: var(--muted-c); font-size: 14px; margin-bottom: 24px;">Upload voice samples for different emotions to create your own voice profile. At least "Neutral" is required.</p>')
                    
                    voice_name_input = gr.Textbox(label="Voice Name (e.g., your_name)", placeholder="No spaces or special characters", lines=1)
                    
                    EMOTIONS_DATA = [
                        ("Neutral", "Every day, the city operates on a steady rhythm. People commute to work, complete their tasks, and return home by evening. Weather patterns shift with the seasons, businesses open and close, and traffic flows through familiar intersections. There is no particular urgency to any of it — simply the consistent movement of life proceeding as expected. Reports are filed, meetings are held, and decisions are made based on available information. The process is straightforward and methodical. Nothing about today stands apart from yesterday, and tomorrow is expected to follow in much the same way. It is, by all measures, an ordinary day — functional, predictable, and unremarkable in the best possible sense."),
                        ("Warm", "Welcome — truly, genuinely welcome. This is a place where you belong, where your presence matters, and where the door is always open. Whether you've been here before or this is your very first time, know that you are seen, valued, and appreciated exactly as you are. Pull up a chair, settle in, and let yourself breathe. There's no rush here, no pressure, no judgment. Just good company, honest conversation, and the simple comfort of being among people who care. Whatever brought you here today, we're glad it did. Life is so much richer when we share it with one another — and right now, in this moment, we are so very glad to share it with you."),
                        ("Happy", "Oh, what a magnificent day it is! The sun is doing its absolute best up there, the birds seem to have rehearsed something special just for this morning, and everything — everything — feels like it's smiling right along with you. Have you ever noticed how good news seems to make the coffee taste better, the commute feel shorter, and even the most mundane tasks feel oddly enjoyable? That's the magic of a truly happy day. It bubbles up from somewhere deep inside and spills out into every corner of your world. Laugh a little louder today. Smile at a stranger. Dance, if the mood strikes — and honestly, let it strike. Joy like this deserves to be celebrated fully and without apology."),
                        ("Sad", "There are days when the weight of missing someone becomes almost too much to carry quietly. You move through the familiar spaces — the kitchen, the hallway, the chair by the window — and everywhere you look, there is the echo of someone who is no longer there. It's not dramatic, not loud. It's soft, and slow, and persistent. A song comes on the radio and suddenly your chest is full of something that has no name. Grief doesn't follow a schedule. It arrives uninvited, lingers in the small things, and reminds you that love, even when it hurts, is the most human thing we carry. And so you sit with it, gently, because that's all you can do."),
                        ("Angry", "Enough. That word — enough — has been building for a long time, and it is finally here. You were patient. You gave chances that weren't deserved, extended grace that was exploited, and stayed silent when you should have spoken. But there is a limit to how long a person can watch injustice unfold before something inside them refuses to stay quiet any longer. This is that moment. The truth is coming out — not with cruelty, but with clarity and force. Every lie, every dismissal, every moment of being made to feel small — it all ends now. This isn't about anger for its own sake. This is about reclaiming what was taken and making absolutely certain it never happens again."),
                        ("Fearful", "Something is wrong. It's hard to explain exactly — there's no obvious reason, no clear threat — but the feeling is undeniable. A quiet that's too quiet. A stillness that feels less like calm and more like the moment just before something breaks. Your pulse quickens without permission. Your eyes move to every shadow, every corner, every flicker at the edge of your vision. You try to steady your breath, but it keeps catching. The rational part of your brain says everything is fine. The rest of you is not convinced. You stand very still, listening — straining to hear anything that might explain this cold, creeping unease that has settled into your bones and refuses, absolutely refuses, to let go."),
                        ("Excited", "It's here! It is finally, actually, undeniably here — the thing you've been waiting for, planning for, counting down to for longer than you care to admit! And oh, the wait was absolutely worth it because right now, in this very moment, everything feels electric. Your mind is running ten steps ahead, imagining all the possibilities, all the moments yet to come. The energy is almost impossible to contain — you want to tell everyone, celebrate everything, and soak up every single second of this feeling. This is what it means to be fully alive — completely present, completely thrilled, completely invested in what's unfolding right in front of you. Don't hold back. Dive in headfirst. This kind of excitement doesn't come along every day, and it deserves every ounce of your enthusiasm."),
                        ("Romantic", "There are moments that don't need grand gestures or carefully chosen words — moments that exist simply in the quiet between two people who truly know each other. The way you reach for my hand without thinking. The sound of your laugh from the next room. The particular way you look at me sometimes, like I am somehow exactly where I am supposed to be. I don't always have the right language for what I feel, but I know it lives in the small things — in shared mornings, and the way time softens when you're near. You have changed the shape of my ordinary days in the most extraordinary way, and I am grateful, deeply and quietly grateful, for every single moment that is ours."),
                        ("Suspenseful", "The door at the end of the hallway had been closed for three days. Nobody talked about it. Nobody went near it. But tonight, something shifted — a sound, faint and low, like a breath that didn't belong. She pressed her back against the wall, heart hammering, eyes locked on the thin strip of darkness beneath that door. The lights flickered once. Twice. Then died completely. In the absolute dark, she heard it again — closer now. Much closer. Her phone was in the other room. The exit was behind her. Whatever was on the other side of that door knew exactly where she was. And in the silence that followed, the most terrifying realization settled over her: it wasn't coming. It was already there."),
                        ("Serious", "What we choose in moments like these defines not just who we are, but who we will become. History does not remember the comfortable choices — the safe paths taken when courage was inconvenient, the truths left unspoken because silence was easier. History remembers the people who stood up when standing was costly, who spoke when speaking carried risk, who chose principle over comfort at a time when most were looking the other way. This is that kind of moment. The weight of it is real, and it should be felt. Because some decisions cannot be undone, some words cannot be taken back, and some opportunities to do what is right appear only once. Choose carefully. Choose honestly. And choose with full awareness of what is truly at stake.")
                    ]
                    
                    audio_inputs = []
                    
                    with gr.Group(elem_classes="fish-card"):
                        for emotion_name, script_text in EMOTIONS_DATA:
                            with gr.Accordion(f"🎙️ {emotion_name}" + (" (Required)" if emotion_name == "Neutral" else ""), open=(emotion_name == "Neutral")):
                                gr.Markdown(f"> {script_text}")
                                audio_in = gr.Audio(sources=["upload", "microphone"], type="filepath", label=f"{emotion_name} Audio", elem_id=f"audio-{emotion_name.lower()}")
                                audio_inputs.append(audio_in)
                    
                    save_btn = gr.Button("💾 Save Voice", variant="primary", size="lg")
                    save_status = gr.HTML("")
                    
                    def handle_save_voice(v_name, *audios):
                        if not v_name or not v_name.strip():
                            return "<span style='color: #ef4444;'>Error: Voice Name is required.</span>", gr.update()
                        
                        import re
                        import shutil
                        clean_name = re.sub(r'[^a-zA-Z0-9_]', '', v_name.strip().replace(' ', '_')).lower()
                        if not clean_name:
                            return "<span style='color: #ef4444;'>Error: Invalid Voice Name.</span>", gr.update()
                            
                        if not audios[0]: # Neutral is required
                            return "<span style='color: #ef4444;'>Error: Neutral audio is required.</span>", gr.update()
                            
                        voice_dir = os.path.join(REF_AUDIO_DIR, clean_name)
                        os.makedirs(voice_dir, exist_ok=True)
                        
                        saved_emotions = []
                        for i, (emotion_name, _) in enumerate(EMOTIONS_DATA):
                            if audios[i]:
                                src = audios[i]
                                ext = os.path.splitext(src)[1] or '.wav'
                                dst = os.path.join(voice_dir, f"{emotion_name.lower()}{ext}")
                                shutil.copy2(src, dst)
                                saved_emotions.append((emotion_name, dst))
                                
                                txt_dst = os.path.join(voice_dir, f"{emotion_name.lower()}.txt")
                                with open(txt_dst, "w", encoding="utf-8") as f:
                                    f.write(EMOTIONS_DATA[i][1])
                                    
                        saved_emotions.sort(key=lambda x: x[0])
                        AUTHOR_EMOTIONS[clean_name] = saved_emotions
                        
                        author_choices = [("— None —", "")] + [(k.capitalize(), k) for k in sorted(AUTHOR_EMOTIONS.keys())]
                        
                        return f"<span style='color: #10b981;'>Success: Voice '{clean_name}' saved! You can now use it in the Generate Speech tab.</span>", gr.update(choices=author_choices)

                    save_btn.click(
                        fn=handle_save_voice,
                        inputs=[voice_name_input] + audio_inputs,
                        outputs=[save_status, preset_radio]
                    )


        def dispatch(
            text,
            reference_id,
            mic_path,
            upload_path,
            source,
            reference_text,
            max_new_tokens,
            chunk_length,
            top_p,
            repetition_penalty,
            temperature,
            seed,
            use_memory_cache,
            max_words_per_chunk,
            mode_radio,
            hanna_emotion,
            progress=gr.Progress(),
        ):
            # Belt-and-suspenders: run_single/run_long_form already turn known
            # failure modes into a friendly (None, error_html) pair, but this
            # is the actual Gradio-facing function in the .click().then()
            # chain below — if *anything* unforeseen raises here instead of
            # returning normally, that chain aborts and its final .then()
            # step (which resets the Generate button back to idle) never
            # runs, leaving the button permanently stuck on "Generating…".
            # Catching broadly here is the one place that guarantees the
            # button always comes back, regardless of what specifically
            # went wrong deeper in the stack.
            try:
                reference_audio = _resolve_reference_audio(mic_path, upload_path, source)
                
                if hanna_emotion:
                    if hanna_emotion == "auto":
                        detected = detect_emotion(text)
                        
                        # Try exact match first for the selected author
                        # If no author selected, default to the first one available or 'hanna'
                        author_for_auto = reference_id
                        if not author_for_auto:
                            if "hanna" in AUTHOR_EMOTIONS:
                                author_for_auto = "hanna"
                            elif AUTHOR_EMOTIONS:
                                author_for_auto = list(AUTHOR_EMOTIONS.keys())[0]
                                
                        auto_path = ""
                        if author_for_auto:
                            auto_path = os.path.join(REF_AUDIO_DIR, author_for_auto, f"{detected}.mp3")
                            if not os.path.exists(auto_path):
                                # Try any extension for this author
                                found = False
                                for ext in [".mp3", ".wav", ".flac", ".ogg", ".m4a"]:
                                    test_path = os.path.join(REF_AUDIO_DIR, author_for_auto, f"{detected}{ext}")
                                    if os.path.exists(test_path):
                                        auto_path = test_path
                                        found = True
                                        break
                                
                                if not found:
                                    # Fallback to ANY emotion file for this author matching the emotion
                                    author_files = AUTHOR_EMOTIONS.get(author_for_auto, [])
                                    fallback = [path for name, path in author_files if name.lower() == detected.lower()]
                                    if fallback:
                                        auto_path = fallback[0]
                                        
                        reference_audio = auto_path if auto_path and os.path.exists(auto_path) else None
                    else:
                        reference_audio = hanna_emotion
                        
                # If an emotion/reference audio is selected but no text is provided, 
                # try to read its .lab or .txt file to completely bypass slow Whisper transcription
                if reference_audio and not reference_text:
                    base_path = os.path.splitext(reference_audio)[0]
                    for ext in [".lab", ".txt"]:
                        if os.path.exists(base_path + ext):
                            with open(base_path + ext, "r", encoding="utf-8") as f:
                                reference_text = f.read().strip()
                            break
                        
                # We're using reference_audio directly via the dropdowns, so we don't pass reference_id 
                # (which used to pull from references/ folder instead of reference_audio/ folder).
                active_ref_id = None
                        
                if mode_radio == "Long-form (chunked)":
                    audio, err = run_long_form(
                        text,
                        active_ref_id,
                        reference_audio,
                        reference_text,
                        max_new_tokens,
                        chunk_length,
                        top_p,
                        repetition_penalty,
                        temperature,
                        seed,
                        use_memory_cache,
                        max_words_per_chunk,
                        progress,
                    )
                else:
                    audio, err = run_single(
                        text,
                        active_ref_id,
                        reference_audio,
                        reference_text,
                        max_new_tokens,
                        chunk_length,
                        top_p,
                        repetition_penalty,
                        temperature,
                        seed,
                        use_memory_cache,
                    )
            except Exception as e:
                audio, err = None, build_html_error_message(e)
            # Third output clears the "will appear here" placeholder — once a
            # generation has run (success or error), it's no longer useful.
            return audio, err, ""

        dispatch_inputs = [
            text_input,
            reference_id,
            mic_audio,
            upload_audio,
            active_source,
            reference_text,
            max_new_tokens,
            chunk_length,
            top_p,
            repetition_penalty,
            temperature,
            seed,
            use_memory_cache,
            max_words_per_chunk,
            mode,
            hanna_radio,
        ]

        generate_btn.click(
            fn=lambda: gr.update(value="⏳ " + i18n("Generating…"), interactive=False),
            inputs=None,
            outputs=generate_btn,
        ).then(
            fn=dispatch,
            inputs=dispatch_inputs,
            outputs=[audio_out, error_out, output_placeholder],
            concurrency_limit=1,
        ).then(
            fn=lambda: gr.update(value="🎙️ " + i18n("Generate speech"), interactive=True),
            inputs=None,
            outputs=generate_btn,
        )

        regenerate_btn.click(
            fn=dispatch,
            inputs=dispatch_inputs,
            outputs=[audio_out, error_out, output_placeholder],
            concurrency_limit=1,
        )

        def _transcribe_active(mic_path, upload_path, source):
            resolved = _resolve_reference_audio(mic_path, upload_path, source)
            return whisper_transcribe_fn(resolved)

        transcribe_btn.click(
            fn=_transcribe_active,
            inputs=[mic_audio, upload_audio, active_source],
            outputs=[reference_text],
        )

        # Mic, upload, and preset are mutually exclusive — using one clears
        # the other two, so at most one voice source is ever active. Without
        # this, several widgets could hold content at once with no visible
        # signal of which one would actually be used at generate time. Each
        # handler only clears the *others* when it just received real
        # content (a truthy new value); if it was cleared instead, it leaves
        # the others alone and falls back to whichever one still has
        # something (mic > upload > preset, an arbitrary but fixed order).
        # Clearing preset_radio/reference_id here also re-triggers
        # preset_radio.change below — harmless, since by then mic_audio/
        # upload_audio already hold their new values and _on_preset_change
        # re-derives the same fallback source from them.
        def _on_mic_change(mic_path, upload_path, preset_id):
            if mic_path:
                return (
                    gr.update(value=None),
                    gr.update(value=""),
                    "mic",
                    "",
                    _voice_status_display("mic", "", mic_path),
                    gr.update(value=None, visible=False),
                    gr.update(value=""),
                )
            if upload_path:
                source = "upload"
            elif preset_id:
                source = "preset"
            else:
                source = "mic"
            return (
                gr.update(),
                gr.update(),
                source,
                preset_id if source == "preset" else "",
                _voice_status_display(source, preset_id, upload_path),
                gr.update(),
                gr.update(),
            )

        def _on_upload_change(mic_path, upload_path, preset_id):
            if upload_path:
                return (
                    gr.update(value=None),
                    gr.update(value=""),
                    "upload",
                    "",
                    _voice_status_display("upload", "", upload_path),
                    gr.update(value=None, visible=False),
                    gr.update(value=""),
                )
            if mic_path:
                source = "mic"
            elif preset_id:
                source = "preset"
            else:
                source = "upload"
            return (
                gr.update(),
                gr.update(),
                source,
                preset_id if source == "preset" else "",
                _voice_status_display(source, preset_id, mic_path),
                gr.update(),
                gr.update(),
            )

        def _on_preset_change(preset_id, mic_path, upload_path):
            base_emotions = [("— None —", ""), ("— Auto from text —", "auto")]
            if preset_id:
                emotions = AUTHOR_EMOTIONS.get(preset_id, [])
                preview_audio = emotions[0][1] if emotions else None
                return (
                    gr.update(value=None),
                    gr.update(value=None),
                    "preset",
                    preset_id,
                    _voice_status_display("preset", preset_id, None),
                    gr.update(value=preview_audio, visible=bool(preview_audio)),
                    gr.update(choices=base_emotions + emotions, value=""),
                )
            if mic_path:
                source = "mic"
            elif upload_path:
                source = "upload"
            else:
                source = "mic"
            return (
                gr.update(),
                gr.update(),
                source,
                preset_id if source == "preset" else "",
                _voice_status_display(source, preset_id, mic_path),
                gr.update(value=None, visible=False),
                gr.update(choices=base_emotions, value=""),
            )

        mic_audio.change(
            fn=_on_mic_change,
            inputs=[mic_audio, upload_audio, preset_radio],
            outputs=[upload_audio, preset_radio, active_source, reference_id, voice_status, preset_preview, hanna_radio],
        )
        upload_audio.change(
            fn=_on_upload_change,
            inputs=[mic_audio, upload_audio, preset_radio],
            outputs=[mic_audio, preset_radio, active_source, reference_id, voice_status, preset_preview, hanna_radio],
        )
        preset_radio.change(
            fn=_on_preset_change,
            inputs=[preset_radio, mic_audio, upload_audio],
            outputs=[mic_audio, upload_audio, active_source, reference_id, voice_status, preset_preview, hanna_radio],
        )

        def _on_emotion_change(emotion_path, preset_id):
            if emotion_path and emotion_path != "auto":
                return gr.update(value=emotion_path, visible=True)
            elif preset_id:
                emotions = AUTHOR_EMOTIONS.get(preset_id, [])
                preview_audio = emotions[0][1] if emotions else None
                return gr.update(value=preview_audio, visible=bool(preview_audio))
            return gr.update(value=None, visible=False)

        hanna_radio.change(
            fn=_on_emotion_change,
            inputs=[hanna_radio, preset_radio],
            outputs=[preset_preview],
        )

        gr.HTML(FOOTER_HTML)

    return app
