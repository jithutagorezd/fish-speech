"""
Fish Speech WebUI v2 — rich interface with long-form TTS support.
"""

from typing import Optional

import gradio as gr

from fish_speech.i18n import i18n

from webui_v2.inference import (
    get_inference_long_form_wrapper,
    get_inference_single_wrapper,
    get_whisper_transcribe_wrapper,
)
from webui_v2.utils import count_words

HEADER_HTML = """
<div class="fish-header">
  <div class="fish-header-row">
    <div class="fish-logo">🐟</div>
    <div>
      <div class="fish-title">Fish Speech <span class="fish-title-accent">S2</span></div>
      <div class="fish-subtitle">Studio-grade voice cloning &amp; long-form narration</div>
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
    ("Performance", ["singing", "screaming", "shouting", "whisper", "with strong accent"]),
]


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
  <summary>🎭 Emotion tags — click to insert</summary>
  <div class="emotion-tag-groups">{groups_html}</div>
</details>
"""


EMOTION_TAGS_HTML = _build_emotion_tags_html()

# Delegated click handler + native insert-at-cursor, injected once via
# app.load(js=...) below. Targets the real <textarea> Gradio renders inside
# the #script-input wrapper and dispatches a real "input" event so Gradio's
# own reactivity (word count, etc.) picks up the change exactly as if the
# user had typed it.
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
    const btn = e.target.closest('.emotion-tag[data-tag]');
    if (!btn) return;
    e.preventDefault();
    insertTag(btn.dataset.tag);
  });
}
"""

FOOTER_HTML = """
<div class="fish-footer">🐟 Powered by <b>Fish Speech S2</b></div>
"""

CUSTOM_CSS = """
:root {
    --fish-rose: #fb7185;
    --fish-orange: #fb923c;
    --fish-amber: #fbbf24;
}

.gradio-container {
    max-width: 1440px !important;
    margin: 0 auto !important;
}

/* ---- Header banner ---- */
.fish-header {
    background: linear-gradient(135deg, #1e1145 0%, #be123c 55%, #fb923c 100%);
    border-radius: 14px;
    padding: 14px 22px;
    margin-bottom: 12px;
    color: #fff7ed;
    box-shadow: 0 8px 20px rgba(190, 18, 60, 0.25);
}
.fish-header-row {
    display: flex;
    align-items: center;
    gap: 12px;
}
.fish-logo {
    font-size: 1.7rem;
    line-height: 1;
    filter: drop-shadow(0 2px 6px rgba(0,0,0,0.25));
}
.fish-title {
    font-size: 1.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #fff7ed !important;
}
.fish-title-accent {
    background: linear-gradient(90deg, #fde68a, #fb923c);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent !important;
}
.fish-subtitle {
    font-size: 0.8rem;
    color: #fed7aa !important;
}

/* ---- Section headings ---- */
.section-heading {
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    margin-bottom: 2px !important;
}

/* ---- Cards ---- */
.fish-card {
    border-radius: 18px !important;
    box-shadow: 0 2px 14px rgba(190, 18, 60, 0.05);
    transition: box-shadow .2s ease;
}
.fish-card:hover {
    box-shadow: 0 8px 28px rgba(190, 18, 60, 0.1);
}

/* ---- Buttons: bigger and more tactile everywhere ---- */
.gradio-container button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: transform .12s ease, box-shadow .12s ease, filter .12s ease;
}
.gradio-container button:active {
    transform: scale(0.97);
}
.gradio-container .lg.secondary,
.gradio-container button[class*="secondary"] {
    min-height: 44px !important;
    padding: 10px 20px !important;
    font-size: 0.95rem !important;
}
/* Icon-only buttons (record / upload / play toggles inside the audio
   widget) are tiny by default — give them real tap targets. */
.gradio-container .icon-button-wrapper button,
.gradio-container button.icon {
    min-width: 42px !important;
    min-height: 42px !important;
}
.gradio-container button svg {
    width: 18px;
    height: 18px;
}

/* ---- Emotion tags ---- */
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
    background: rgba(251, 113, 133, 0.14);
    border: 1px solid rgba(251, 113, 133, 0.35);
    color: var(--body-text-color);
    cursor: pointer;
}
.emotion-tag:hover {
    background: rgba(251, 113, 133, 0.26);
}

.emotion-tag-details { margin-top: 8px; }
.emotion-tag-details summary {
    display: inline-block;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--body-text-color-subdued);
    cursor: pointer;
    list-style: none;
}
.emotion-tag-details summary::-webkit-details-marker { display: none; }
.emotion-tag-details summary:hover { color: var(--body-text-color); }
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
    color: var(--body-text-color-subdued);
    margin-bottom: 3px;
}

/* ---- Word count badge ---- */
.word-badge {
    display: inline-block;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--body-text-color-subdued);
    padding: 2px 4px;
}

/* ---- Generate button ---- */
#generate-btn {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    height: 56px !important;
    border-radius: 14px !important;
    letter-spacing: .2px;
    margin-top: 8px;
    background: linear-gradient(90deg, #fb7185 0%, #fb923c 100%) !important;
    border: none !important;
    color: #fff7ed !important;
    box-shadow: 0 8px 20px rgba(251, 113, 133, .35);
}
#generate-btn:hover {
    transform: translateY(-2px);
    filter: brightness(1.06);
    box-shadow: 0 12px 28px rgba(251, 113, 133, .45);
}

/* ---- Output panel ---- */
#audio-output {
    border-radius: 14px !important;
    min-height: 200px;
}
/* Gradio's audio player animates its own height/opacity while it switches
   between empty / loading / loaded states. Combined with autoplay kicking
   in immediately (and its timeupdate polling for the progress bar), that
   animation plus the waveform's ResizeObserver still settling reads as
   the whole card "shaking" in Chrome. Reserving a fixed height means
   there's no size jump to animate, and killing the transition makes any
   remaining state change instant instead of visibly animated. */
#audio-output, #audio-output * {
    transition: none !important;
}
#error-box:empty { display: none; }

/* ---- Footer ---- */
.fish-footer {
    text-align: center;
    color: var(--body-text-color-subdued);
    font-size: 0.78rem;
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid var(--border-color-primary);
}
"""

FISH_THEME = gr.themes.Soft(
    primary_hue="rose",
    secondary_hue="orange",
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    block_radius="16px",
    block_shadow="0 2px 14px rgba(190, 18, 60, 0.05)",
    button_primary_background_fill="linear-gradient(90deg, #fb7185 0%, #fb923c 100%)",
    button_primary_background_fill_hover="linear-gradient(90deg, #f43f5e 0%, #f97316 100%)",
    button_primary_text_color="white",
    button_large_radius="14px",
    button_large_padding="14px 24px",
    button_small_padding="8px 16px",
)


def _word_count_display(text: Optional[str]) -> str:
    n = count_words(text) if text and text.strip() else 0
    return f'<span class="word-badge">📝 {n} words</span>'


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
        title="Fish Speech S2 — WebUI v2",
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

        # Primary flow: script -> one reference-audio widget (record or
        # upload, Gradio does both in a single control) -> generate -> output.
        # Everything else (chunking mode, reference id, memory cache, sampling
        # params) lives in one collapsed Advanced settings accordion so the
        # default view stays to four things.
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                with gr.Group(elem_classes=["fish-card"]):
                    gr.Markdown("### 📝 Script", elem_classes=["section-heading"])
                    text_input = gr.Textbox(
                        label=i18n("Input Text"),
                        placeholder="Paste or type text here. For long documents (3000–5000 words), turn on Long-form in Advanced settings.",
                        lines=5,
                        max_lines=30,
                        show_label=False,
                        elem_id="script-input",
                    )
                    word_count = gr.HTML('<span class="word-badge">📝 0 words</span>')
                    text_input.change(
                        fn=_word_count_display,
                        inputs=[text_input],
                        outputs=[word_count],
                    )

                    gr.HTML(EMOTION_TAGS_HTML)

                with gr.Group(elem_classes=["fish-card"]):
                    gr.Markdown("### 🎤 Voice", elem_classes=["section-heading"])
                    reference_audio = gr.Audio(
                        label=i18n("Record or upload your reference voice"),
                        type="filepath",
                        sources=["microphone", "upload"],
                    )

                generate_btn = gr.Button(
                    "🎙️ Generate speech",
                    variant="primary",
                    size="lg",
                    elem_id="generate-btn",
                )

                with gr.Accordion("⚙️ " + i18n("Advanced settings"), open=False):
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
                        placeholder="Leave empty to use the reference audio above",
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

            with gr.Column(scale=2):
                with gr.Group(elem_classes=["fish-card"]):
                    gr.Markdown("### 🔊 Output", elem_classes=["section-heading"])
                    error_out = gr.HTML(label=i18n("Error Message"), value="", elem_id="error-box")
                    audio_out = gr.Audio(
                        label=i18n("Generated Audio"),
                        type="numpy",
                        interactive=False,
                        autoplay=True,
                        elem_id="audio-output",
                    )

        def dispatch(
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
            mode_radio,
            progress=gr.Progress(),
        ):
            if mode_radio == "Long-form (chunked)":
                return run_long_form(
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
            return run_single(
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

        generate_btn.click(
            fn=dispatch,
            inputs=[
                text_input,
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
                mode,
            ],
            outputs=[audio_out, error_out],
            concurrency_limit=1,
        )

        transcribe_btn.click(
            fn=whisper_transcribe_fn,
            inputs=[reference_audio],
            outputs=[reference_text],
        )

        gr.Examples(
            label="✨ Examples",
            examples=[
                ["Hello! This is a short test of Fish Speech S2.", "Single shot"],
                ["[laughing] I can't believe it! This model supports emotion tags.", "Single shot"],
                [
                    "First paragraph of your long document.\n\nSecond paragraph here.\n\nThird paragraph.",
                    "Long-form (chunked)",
                ],
            ],
            inputs=[text_input, mode],
        )

        gr.HTML(FOOTER_HTML)

    return app
