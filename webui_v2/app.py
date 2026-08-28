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
from webui_v2.utils import count_words

# Preset "pick a ready-made voice" cards. Each id must match a folder under
# references/<id>/ containing at least one audio file + matching .lab
# transcript (fish_speech.inference_engine.reference_loader.ReferenceLoader
# reads that folder structure directly — see load_by_id/list_reference_ids).
# reference_id already flows straight through dispatch() -> run_single /
# run_long_form -> ServeTTSRequest, and the engine prefers reference_id over
# any uploaded reference_audio, so selecting a preset needs no changes on
# the inference side at all.
PRESET_VOICES = [
    ("Amelia Taylor — Female", "Amelia Taylor"),
    ("John Taylor — Male", "John Taylor"),
    ("Ava Thompson — Female", "Ava Thompson"),
]
PRESET_PREVIEW_PATHS = {
    ref_id: os.path.join("references", ref_id, "sample.mp3") for _, ref_id in PRESET_VOICES
}

HEADER_HTML = """
<div class="fish-header">
  <div class="fish-header-row">
    <div class="fish-brand">
      <span class="fish-logo-bars">
        <span class="bar bar-1"></span><span class="bar bar-2"></span><span class="bar bar-3"></span>
      </span>
      <span class="fish-title">VOICE <span class="fish-title-accent">CLONING STUDIO</span></span>
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
    ("Emotion", ["excited", "excited tone", "angry", "sad", "delight", "surprised", "shocked"]),
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
    chips = ['<button type="button" class="emotion-pill emotion-pill-active" data-tag="">Default</button>']
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
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700;800&display=swap');

:root {
    /* The app fills the whole browser viewport (see .gradio-container
       below), so --bg and --panel are deliberately the same white —
       there's no separate gray page behind a smaller white card. Cards
       are distinguished purely by border lines, matching the reference
       design's white-on-white panel look without reintroducing the page
       gutters we removed earlier. */
    --bg: #0F1114;
    --panel: #15181D;
    --panel-2: #1A1D22;
    --border-c: #2A2E35;
    --text-c: #EEEFF1;
    --muted-c: #8B9099;
    --accent: #E8A33D;
    --accent-hover: #F3B860;
    --accent-dark: #3A3F49;
}

/* ---- Page shell ----
   The page scrolls naturally, like a normal document — no forced single-
   viewport height, no per-column internal scrolling. That tidy-looking
   "pin everything to 100dvh, let each column scroll internally" approach
   fell apart in practice: the Generate-speech column genuinely holds more
   content than the Voice-input column (textbox, tags, button, advanced
   settings, output player), so forcing both into the same fixed height
   just crushed the taller one — clipped text, a barely-visible Advanced
   Settings section, a collapsed-looking output card, and its own inner
   scrollbar stacked on top of the browser's outer one. Letting each
   column size to its own content and scrolling the whole page instead
   removes all of that in one move. */
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
    /* Best-effort override of Gradio's own theme tokens, so built-in
       components (labels, inputs, blocks) inherit this palette even
       where we haven't targeted them by id/class directly below. */
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

/* ---- Main layout ----
   Gradio wraps a Row's columns into a stacked, single-column layout
   whenever they can't fit side by side at their min_width, which reads
   as a tall "mobile" page even on a desktop-width window. Force the two
   main columns to stay side by side down to genuine phone widths — each
   one now just grows to its own natural content height (see the page
   shell comment above), so the shorter Voice-input column ends where its
   content ends instead of being stretched to match the taller one.

   flex-wrap:nowrap is also set on the columns themselves, not just the
   row: a flex-direction:column container that runs out of vertical room
   doesn't clip an overflowing child, it wraps it into a second column
   positioned off to the side instead (this genuinely happened once, when
   these columns still had a fixed height — a child taller than the
   column landed a full column-width to the right, needing a horizontal
   scrollbar to ever reach). Harmless to keep as insurance even without a
   height constraint forcing the issue. */
#main-row {
    flex-wrap: nowrap !important;
}
#voice-column, #studio-column {
    flex-wrap: nowrap !important;
}
/* Clear visual separation between the input (left) and output (right)
   zones. */
#studio-column {
    border-left: 1px solid var(--border-c);
    padding-left: 24px;
}
@media (max-width: 640px) {
    #main-row {
        flex-wrap: wrap !important;
    }
    #studio-column {
        border-left: none;
        padding-left: 0;
        border-top: 1px solid var(--border-c);
        padding-top: 16px;
        margin-top: 8px;
    }
}

/* ---- Header banner ---- */
.fish-header {
    border-bottom: 1px solid var(--border-c);
    padding: 10px 4px 18px 4px;
    margin-bottom: 16px;
}
.fish-header-row {
    display: flex;
    align-items: center;
}
.fish-brand {
    display: flex;
    align-items: center;
    gap: 8px;
}
.fish-logo-bars {
    display: flex;
    align-items: flex-end;
    gap: 2px;
    height: 20px;
}
.fish-logo-bars .bar {
    width: 4px;
    border-radius: 2px;
    background: var(--accent-dark);
}
.fish-logo-bars .bar-1 { height: 12px; }
.fish-logo-bars .bar-2 { height: 20px; }
.fish-logo-bars .bar-3 { height: 16px; }
.fish-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    letter-spacing: 0.02em;
    color: var(--text-c) !important;
}
.fish-title-accent {
    color: var(--accent) !important;
    font-weight: 800;
}

/* ---- Step kickers ---- */
.step-kicker {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted-c);
    margin: 0 0 4px 2px;
}

/* ---- Headings ---- */
.column-heading {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: var(--text-c) !important;
    margin: 0 0 10px 0 !important;
}
.heading-sub {
    font-weight: 500;
    color: var(--muted-c);
    font-size: 0.95rem;
}
.section-heading {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    margin-bottom: 2px !important;
    color: var(--text-c) !important;
}

/* ---- Cards ---- */
.fish-card {
    background: var(--panel) !important;
    border: 1px solid var(--border-c) !important;
    border-radius: 14px !important;
    box-shadow: none !important;
}
/* Nested card (e.g. the "Generated Speech" box inside the Generate panel) —
   a subtly different fill so it reads as a secondary surface within the
   card rather than a competing top-level card. */
.fish-card-nested {
    background: var(--panel-2) !important;
    border: 1px solid var(--border-c) !important;
    border-radius: 12px !important;
    box-shadow: none !important;
}

/* ---- Voice source row (record / upload side by side) ---- */
#voice-source-row {
    gap: 12px;
}
/* Flex items default to min-width:auto, which refuses to shrink below
   the content's natural width. An empty upload dropzone is narrow, but
   the loaded audio player (waveform canvas + play/download/share/trim
   controls) is intrinsically much wider — without min-width:0 that
   swap drags the whole #voice-source-row wider, which reflows #main-row
   and visibly shifts the Generate-speech column left/right. min-width:0
   on just the outer wrapper isn't enough on its own, since a deeply
   nested descendant (the controls row, the waveform canvas) can still
   carry its own content-based min-width floor that bubbles back up —
   so it's forced to 0 on every descendant too, and overflow:hidden is
   the hard backstop: nothing inside can ever push past the card's own
   width no matter what Gradio's internal player renders at. */
#voice-source-row > .fish-card {
    flex: 1 1 0% !important;
    min-width: 0 !important;
    overflow: hidden !important;
}
#mic-audio, #upload-audio {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    overflow: hidden !important;
}
#mic-audio *, #upload-audio * {
    min-width: 0 !important;
    max-width: 100% !important;
}

/* ---- Buttons: bigger and more tactile everywhere ---- */
.gradio-container button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: transform .12s ease, box-shadow .12s ease, filter .12s ease, background .12s ease;
}
.gradio-container button:active {
    transform: scale(0.97);
}
.gradio-container .lg.secondary,
.gradio-container button[class*="secondary"] {
    min-height: 44px !important;
    padding: 10px 20px !important;
    font-size: 0.95rem !important;
    background: var(--panel) !important;
    color: var(--text-c) !important;
    border: 1px solid var(--border-c) !important;
}
.gradio-container button[class*="secondary"]:hover {
    background: var(--panel-2) !important;
}
/* Icon-only buttons (record / upload / play toggles inside the audio
   widget) are tiny by default — give them real tap targets. */
.gradio-container .icon-button-wrapper button,
.gradio-container button.icon {
    min-width: 42px !important;
    min-height: 42px !important;
}
/* The mic/upload/output audio widgets are much narrower now (each voice
   card is roughly a quarter of the viewport). A row of several 42px
   icon buttons (play, trim, clear, ...) doesn't fit and can push the
   Clear/trash button out of the visible row entirely. Scale those three
   widgets' icon buttons down so the whole toolbar actually fits. */
#mic-audio .icon-button-wrapper button,
#mic-audio button.icon,
#upload-audio .icon-button-wrapper button,
#upload-audio button.icon,
#audio-output .icon-button-wrapper button,
#audio-output button.icon {
    min-width: 32px !important;
    min-height: 32px !important;
}
.gradio-container button svg {
    width: 18px;
    height: 18px;
}
/* Destructive actions (clear/remove a recording, etc.) get a distinct
   tint so they read differently from safe actions like download/play. */
.gradio-container button[aria-label*="Clear" i],
.gradio-container button[aria-label*="Remove" i],
.gradio-container button[aria-label*="Delete" i] {
    color: #dc2626 !important;
}
.gradio-container button[aria-label*="Clear" i]:hover,
.gradio-container button[aria-label*="Remove" i]:hover,
.gradio-container button[aria-label*="Delete" i]:hover {
    background: rgba(220, 38, 38, 0.08) !important;
}

/* ---- Emotion tags (full grouped list) ---- */
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
    background: rgba(37, 99, 235, 0.08);
    border: 1px solid rgba(37, 99, 235, 0.25);
    color: var(--text-c);
    cursor: pointer;
}
.emotion-tag:hover {
    background: rgba(37, 99, 235, 0.16);
}

.emotion-tag-details { margin-top: 8px; }
.emotion-tag-details summary {
    display: inline-block;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--muted-c);
    cursor: pointer;
    list-style: none;
}
.emotion-tag-details summary::-webkit-details-marker { display: none; }
.emotion-tag-details summary:hover { color: var(--text-c); }
.emotion-tag-groups {
    margin-top: 8px;
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.emotion-tag-group-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted-c);
    margin-bottom: 3px;
}

/* ---- Emotion pills (quick row) ---- */
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
    background: var(--accent-dark) !important;
    border-color: var(--accent-dark) !important;
    color: #ffffff !important;
}

/* ---- Word count badge ---- */
.word-badge {
    display: inline-block;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--muted-c);
    padding: 2px 4px;
}
.longform-nudge {
    display: inline-block;
    margin-left: 6px;
    font-size: 0.72rem;
    font-weight: 700;
    color: #b45309;
    background: rgba(217, 119, 6, 0.08);
    border: 1px solid rgba(217, 119, 6, 0.3);
    padding: 3px 9px;
    border-radius: 999px;
}

/* ---- Voice card hint ---- */
.voice-hint {
    font-size: 0.85rem !important;
    color: var(--muted-c) !important;
    margin: 0 0 10px 0 !important;
    line-height: 1.5 !important;
}

/* ---- Voice status (near Generate) ---- */
.voice-status {
    display: block;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--muted-c);
    margin: 6px 0 2px 2px;
}
.voice-status-active {
    color: var(--accent);
}
.voice-status-none {
    color: #b45309;
}

/* ---- Generate button ----
   Deliberately NOT sticky: Advanced settings and the "Generated Speech"
   card both come after it in the column, and a sticky element pins on
   top of whatever scrolls underneath it — with an opaque background and
   z-index, that meant it could visually cover the output card once you
   scrolled past it, which read as "I can't see the generated audio". */
#generate-btn {
    font-size: 1rem !important;
    font-weight: 700 !important;
    height: 52px !important;
    width: 100% !important;
    border-radius: 12px !important;
    margin-top: 8px;
    background: var(--accent) !important;
    border: none !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, .25);
}
#generate-btn:hover {
    background: var(--accent-hover) !important;
    box-shadow: 0 6px 18px rgba(37, 99, 235, .35);
}

/* ---- Advanced settings ----
   A proper bordered card instead of Gradio's default thin accordion row. */
#advanced-settings {
    background: var(--panel-2) !important;
    border: 1px solid var(--border-c) !important;
    border-radius: 12px !important;
    margin-top: 8px;
    margin-bottom: 8px;
}

/* ---- Output panel ---- */
#audio-output {
    border-radius: 12px !important;
    height: 180px;
    background: var(--panel) !important;
}
/* Gradio (Svelte) runs a FLIP-style resize animation on its own .block
   wrapper via the Web Animations API whenever a tracked element's box
   changes size between renders — visible as --start-top/--start-left/
   --start-width/--start-height custom properties it sets on itself.
   That's a JS/WAAPI animation, not a CSS transition, so `transition:
   none` alone doesn't touch it. FLIP only has something to animate when
   there's a size delta between the old and new box, so the real fix is
   giving these widgets a genuinely fixed height (not min-height, which
   still lets them grow when a recording/upload/result loads) — with no
   delta, there's nothing left for it to animate. transition: none stays
   too, as a no-cost second line of defense for anything that IS a plain
   CSS transition. */
#audio-output, #audio-output *,
#mic-audio, #mic-audio *,
#upload-audio, #upload-audio *,
#preset-preview-audio, #preset-preview-audio * {
    transition: none !important;
}
#mic-audio, #upload-audio {
    height: 180px;
}
#preset-preview-audio {
    height: 120px;
    margin-top: 8px;
}

/* Empty-state message shown until the first generation completes, so the
   output card doesn't read as broken/blank before anything's happened.
   Deliberately no border/dropzone styling — this isn't a drop target,
   just a lightweight placeholder. */
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

#regenerate-btn {
    margin-top: 8px;
}

/* ---- Error banner ---- */
.fish-error-banner {
    background: rgba(220, 38, 38, 0.08);
    border: 1px solid rgba(220, 38, 38, 0.3);
    color: #dc2626;
    font-weight: 600;
    font-size: 0.88rem;
    padding: 10px 14px;
    border-radius: 10px;
    margin-bottom: 8px;
}
#error-box:empty { display: none; }

/* ---- Footer ---- */
.fish-footer {
    text-align: center;
    color: var(--muted-c);
    font-size: 0.78rem;
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid var(--border-c);
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
        return audio, err or ""

    with gr.Blocks(
        title="Audifyz Voice Cloner",
        theme=FISH_THEME,
        css=CUSTOM_CSS,
    ) as app:
        gr.HTML(HEADER_HTML)

        app.load(
            None,
            None,
            js="() => {const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {params.set('__theme', '%s');window.location.search = params.toString();}}"
            % theme,
        )
        app.load(None, None, js=EMOTION_TAG_INSERT_JS)

        # Left: voice input only (record + upload, two separate cards).
        # Right: script, emotion, generate, advanced settings, and a nested
        # "Generated Speech" card for the result — mirrors a single
        # Source -> Generate two-step flow.
        with gr.Row(equal_height=False, elem_id="main-row"):
            with gr.Column(scale=1, min_width=320, elem_id="voice-column"):
                gr.HTML('<p class="step-kicker">Step 1 — Record or upload your voice</p>')
                gr.Markdown(
                    '## 🎤 Voice Input <span class="heading-sub">(Cloning)</span>',
                    elem_classes=["column-heading"],
                )
                gr.Markdown(
                    "Record or upload a short voice sample, or pick a ready-made "
                    "voice below — whatever you type on the right will be spoken "
                    "in that voice.",
                    elem_classes=["voice-hint"],
                )
                with gr.Row(elem_id="voice-source-row"):
                    with gr.Group(elem_classes=["fish-card"]):
                        gr.Markdown("### 🎙️ Mic Record", elem_classes=["section-heading"])
                        mic_audio = gr.Audio(
                            show_label=False,
                            type="filepath",
                            sources=["microphone"],
                            elem_id="mic-audio",
                        )
                    with gr.Group(elem_classes=["fish-card"]):
                        gr.Markdown("### 📁 Upload File", elem_classes=["section-heading"])
                        upload_audio = gr.Audio(
                            show_label=False,
                            type="filepath",
                            sources=["upload"],
                            elem_id="upload-audio",
                        )
                with gr.Group(elem_classes=["fish-card"], elem_id="preset-voice-card"):
                    gr.Markdown("### 🎭 Or Pick a Preset Voice", elem_classes=["section-heading"])
                    preset_radio = gr.Radio(
                        show_label=False,
                        choices=[("— None, use recording/upload above —", "")] + PRESET_VOICES,
                        value="",
                    )
                    preset_preview = gr.Audio(
                        show_label=False,
                        type="filepath",
                        interactive=False,
                        visible=False,
                        elem_id="preset-preview-audio",
                    )
                with gr.Group(elem_classes=["fish-card"], elem_id="hanna-voice-card"):
                    gr.Markdown("### 👩 Hanna Emotion Presets", elem_classes=["section-heading"])
                    hanna_radio = gr.Radio(
                        show_label=False,
                        choices=[("— None —", "")] + [(e.capitalize(), os.path.abspath(f"reference_audio/hanna/{e}.mp3")) for e in ["Angry", "Excited", "Fearful", "Happy", "Romantic", "Sad", "Serious", "Suspenseful", "neutral", "warm"]],
                        value="",
                    )
                active_source = gr.State("mic")

            with gr.Column(scale=1, min_width=320, elem_id="studio-column"):
                gr.HTML('<p class="step-kicker">Step 2 — Generate speech from your cloned voice</p>')
                gr.Markdown("## ✍️ Generate Speech", elem_classes=["column-heading"])

                with gr.Group(elem_classes=["fish-card"]):
                    text_input = gr.Textbox(
                        label=i18n("Input Text"),
                        placeholder="Paste or type text here. For long documents (3000–5000 words), turn on Long-form in Advanced settings.",
                        lines=4,
                        max_lines=30,
                        show_label=False,
                        elem_id="script-input",
                    )
                    word_count = gr.HTML(_word_count_display(None))
                    text_input.change(
                        fn=_word_count_display,
                        inputs=[text_input],
                        outputs=[word_count],
                    )

                    gr.HTML(QUICK_EMOTION_HTML)
                    gr.HTML(EMOTION_TAGS_HTML)

                    voice_status = gr.HTML(
                        _voice_status_display("mic", "", None),
                        elem_id="voice-status",
                    )

                    generate_btn = gr.Button(
                        "🎙️ " + i18n("Generate speech"),
                        variant="primary",
                        size="lg",
                        elem_id="generate-btn",
                    )

                    with gr.Accordion(
                        "⚙️ " + i18n("Advanced settings"), open=False, elem_id="advanced-settings"
                    ):
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

                    with gr.Group(elem_classes=["fish-card-nested"]):
                        gr.Markdown("### 🔊 Generated Speech", elem_classes=["section-heading"])
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
                if mode_radio == "Long-form (chunked)":
                    audio, err = run_long_form(
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
                        progress,
                    )
                else:
                    audio, err = run_single(
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
            if preset_id:
                return (
                    gr.update(value=None),
                    gr.update(value=None),
                    "preset",
                    preset_id,
                    _voice_status_display("preset", preset_id, None),
                    gr.update(value=PRESET_PREVIEW_PATHS.get(preset_id), visible=True),
                    gr.update(value=""),
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
                gr.update(),
                gr.update(),
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
            outputs=[mic_audio, upload_audio, active_source, reference_id, voice_status, preset_preview],
        )

        gr.HTML(FOOTER_HTML)

    return app
