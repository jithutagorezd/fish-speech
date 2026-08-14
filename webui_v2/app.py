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
  <div class="fish-badges">
    <span class="badge">🎙️ Voice Cloning</span>
    <span class="badge">📜 Long-form (5,000+ words)</span>
    <span class="badge">🎭 Emotion Tags</span>
    <span class="badge">🌍 Multilingual</span>
  </div>
</div>
"""

EMOTION_TAGS_HTML = """
<div class="callout">
  <span class="callout-icon">🎭</span>
  <div>
    <strong>Emotion tags</strong> — drop these inline into your script:
    <code>[laugh]</code> <code>[whisper]</code> <code>[excited]</code> <code>[sad]</code> <code>[angry]</code>,
    or go free-form: <code>[whisper in a small voice]</code> · <code>[professional broadcast tone]</code>
  </div>
</div>
"""

TIPS_HTML = """
<div class="fish-card tips-card">
  <div class="tips-title">💡 Quick tips</div>
  <ul>
    <li>Upload <b>5–10s</b> of clean reference audio for the best voice clone.</li>
    <li>Switch to <b>Long-form</b> for scripts over ~500 words — it auto-splits on sentence boundaries.</li>
    <li>Lower <b>Temperature</b> for calmer, more consistent delivery; raise it for expressive variation.</li>
    <li>Leave <b>Reference ID</b> empty to synthesize with your uploaded reference audio directly.</li>
  </ul>
</div>
"""

FOOTER_HTML = """
<div class="fish-footer">🐟 Powered by <b>Fish Speech S2</b></div>
"""

CUSTOM_CSS = """
:root {
    --fish-cyan: #06b6d4;
    --fish-blue: #3b82f6;
    --fish-coral: #fb7185;
}

.gradio-container {
    max-width: 1440px !important;
    margin: 0 auto !important;
}

/* ---- Header banner ---- */
.fish-header {
    background: linear-gradient(135deg, #0f172a 0%, #0e7490 55%, #0891b2 100%);
    border-radius: 20px;
    padding: 28px 36px;
    margin-bottom: 20px;
    color: #f8fafc;
    box-shadow: 0 12px 32px rgba(8, 145, 178, 0.25);
}
.fish-header-row {
    display: flex;
    align-items: center;
    gap: 16px;
}
.fish-logo {
    font-size: 2.6rem;
    line-height: 1;
    filter: drop-shadow(0 2px 6px rgba(0,0,0,0.25));
}
.fish-title {
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #f8fafc !important;
}
.fish-title-accent {
    background: linear-gradient(90deg, #67e8f9, #93c5fd);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent !important;
}
.fish-subtitle {
    font-size: 0.95rem;
    color: #cbd5e1 !important;
    margin-top: 2px;
}
.fish-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 18px;
}
.fish-badges .badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.25);
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.8rem;
    color: #f1f5f9 !important;
    backdrop-filter: blur(4px);
}

/* ---- Section headings ---- */
.section-heading {
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    margin-bottom: 2px !important;
}

/* ---- Cards ---- */
.fish-card {
    border-radius: 16px !important;
    box-shadow: 0 2px 14px rgba(15, 23, 42, 0.06);
    transition: box-shadow .2s ease;
}
.fish-card:hover {
    box-shadow: 0 8px 28px rgba(15, 23, 42, 0.1);
}

/* ---- Callout (emotion tags) ---- */
.callout {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    background: rgba(6, 182, 212, 0.08);
    border: 1px solid rgba(6, 182, 212, 0.25);
    border-radius: 12px;
    padding: 10px 14px;
    font-size: 0.88rem;
    margin: 6px 0 4px 0;
}
.callout-icon { font-size: 1.1rem; }
.callout code {
    background: rgba(6, 182, 212, 0.15);
    padding: 1px 6px;
    border-radius: 6px;
    font-size: 0.82rem;
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
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    height: 54px !important;
    border-radius: 14px !important;
    letter-spacing: .2px;
    margin-top: 10px;
    transition: transform .15s ease, box-shadow .15s ease;
}
#generate-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(8, 145, 178, .35);
}

/* ---- Output panel ---- */
#audio-output {
    border-radius: 14px !important;
}
#error-box:empty { display: none; }

/* ---- Tips card ---- */
.tips-card {
    padding: 16px 18px;
    margin-top: 4px;
}
.tips-title {
    font-weight: 700;
    margin-bottom: 8px;
    font-size: 0.95rem;
}
.tips-card ul {
    margin: 0;
    padding-left: 18px;
    font-size: 0.86rem;
    line-height: 1.6;
    color: var(--body-text-color-subdued);
}

/* ---- Footer ---- */
.fish-footer {
    text-align: center;
    color: var(--body-text-color-subdued);
    font-size: 0.82rem;
    margin-top: 22px;
    padding-top: 14px;
    border-top: 1px solid var(--border-color-primary);
}
"""

FISH_THEME = gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    block_radius="16px",
    block_shadow="0 2px 14px rgba(15, 23, 42, 0.06)",
    button_primary_background_fill="linear-gradient(90deg, #06b6d4 0%, #3b82f6 100%)",
    button_primary_background_fill_hover="linear-gradient(90deg, #0891b2 0%, #2563eb 100%)",
    button_primary_text_color="white",
    button_large_radius="14px",
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

        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                with gr.Group(elem_classes=["fish-card"]):
                    gr.Markdown("### 📝 Script", elem_classes=["section-heading"])
                    text_input = gr.Textbox(
                        label=i18n("Input Text"),
                        placeholder="Paste or type text here. For long documents (3000–5000 words), use Long-form mode below.",
                        lines=14,
                        max_lines=30,
                        show_label=False,
                    )
                    word_count = gr.HTML('<span class="word-badge">📝 0 words</span>')
                    text_input.change(
                        fn=_word_count_display,
                        inputs=[text_input],
                        outputs=[word_count],
                    )

                    gr.HTML(EMOTION_TAGS_HTML)

                with gr.Group(elem_classes=["fish-card"]):
                    with gr.Accordion("⚙️ Mode & long-form", open=True):
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

                    with gr.Accordion("🎤 " + i18n("Reference Audio"), open=False):
                        gr.Markdown(i18n("5 to 10 seconds of reference audio, useful for specifying speaker."))
                        reference_id = gr.Textbox(
                            label=i18n("Reference ID"),
                            placeholder="Leave empty to use uploaded references",
                        )
                        use_memory_cache = gr.Radio(
                            label=i18n("Use Memory Cache"),
                            choices=["on", "off"],
                            value="on",
                        )
                        reference_audio = gr.Audio(label=i18n("Reference Audio"), type="filepath")
                        transcribe_btn = gr.Button("🎤 " + i18n("Auto-transcribe with Whisper"), variant="secondary")
                        reference_text = gr.Textbox(
                            label=i18n("Reference Text"),
                            placeholder="Transcription of the reference audio, or use Auto-transcribe.",
                            lines=3,
                        )

                    with gr.Accordion("🛠️ " + i18n("Advanced Config"), open=False):
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

                generate_btn = gr.Button(
                    "🎙️ Generate speech",
                    variant="primary",
                    size="lg",
                    elem_id="generate-btn",
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

                gr.HTML(TIPS_HTML)

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
                ["[laugh] I can't believe it! This model supports emotion tags.", "Single shot"],
                [
                    "First paragraph of your long document.\n\nSecond paragraph here.\n\nThird paragraph.",
                    "Long-form (chunked)",
                ],
            ],
            inputs=[text_input, mode],
        )

        gr.HTML(FOOTER_HTML)

    return app
