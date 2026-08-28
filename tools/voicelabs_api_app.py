"""
Standalone Gradio front-end for the *hosted* Fish Audio API (api.fish.audio) —
separate from webui_v2/app.py, which talks to the self-hosted EC2 engine.

Lets you either pick one of your already-cloned voices (by voice ID, fetched
live from your account) or supply a fresh reference recording/upload, type
text with emotion tags, and generate speech via the REST API using the key
in .ENV. Defaults to the s2.1-pro-free model so testing doesn't spend credit.
"""

import base64
import os
import tempfile
from pathlib import Path
from typing import Optional

import gradio as gr
import requests

API_BASE = "https://api.fish.audio"
ENV_PATH = Path(__file__).resolve().parents[2] / ".ENV"

MODEL_CHOICES = [
    ("s2.1-pro-free — $0 (Recommended for testing)", "s2.1-pro-free"),
    ("s2.1-pro — $15 / 1M bytes", "s2.1-pro"),
    ("s2-pro — $15 / 1M bytes", "s2-pro"),
    ("s1 — $15 / 1M bytes", "s1"),
]

QUICK_TAGS = ["excited", "angry", "sad", "delight", "surprised", "shocked"]


def _load_api_key() -> str:
    key = os.environ.get("FISH_TTS")
    if key:
        return key
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line.startswith("FISH_TTS="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError(f"FISH_TTS not found in environment or {ENV_PATH}")


def fetch_my_voices():
    """GET /model?self=true — returns [(label, voice_id), ...] for your own cloned voices."""
    try:
        api_key = _load_api_key()
        resp = requests.get(
            f"{API_BASE}/model",
            headers={"Authorization": f"Bearer {api_key}"},
            params={"self": "true"},
            timeout=20,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        choices = [(f"{it['title']} — {it['_id']}", it["_id"]) for it in items]
        status = f"Loaded {len(choices)} voice(s) from your account." if choices else "No cloned voices found on your account yet."
        return gr.update(choices=choices, value=choices[0][1] if choices else None), status
    except Exception as e:
        return gr.update(choices=[], value=None), f"⚠️ Could not load voices: {e}"


def generate(text, voice_id, ref_audio_path, model_name, progress=gr.Progress()):
    if not text or not text.strip():
        return None, "⚠️ Please enter text to synthesize."

    try:
        api_key = _load_api_key()
    except Exception as e:
        return None, f"⚠️ {e}"

    body = {"text": text.strip(), "format": "mp3"}

    if voice_id:
        body["reference_id"] = voice_id
    elif ref_audio_path:
        with open(ref_audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("ascii")
        body["references"] = [{"audio": audio_b64, "text": ""}]
    else:
        return None, "⚠️ Select a voice ID or upload/record a reference audio first."

    progress(0.2, "Calling Fish Audio API…")
    try:
        resp = requests.post(
            f"{API_BASE}/v1/tts",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "model": model_name,
            },
            json=body,
            timeout=60,
        )
    except requests.RequestException as e:
        return None, f"⚠️ Request failed: {e}"

    if resp.status_code != 200:
        try:
            msg = resp.json().get("message", resp.text)
        except Exception:
            msg = resp.text
        return None, f"⚠️ API error {resp.status_code}: {msg}"

    progress(0.9, "Saving audio…")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.write(resp.content)
    tmp.close()
    return tmp.name, ""


INSERT_TAG_JS = """
() => {
  document.addEventListener('click', (e) => {
    const pill = e.target.closest('.qtag[data-tag]');
    if (!pill) return;
    e.preventDefault();
    const ta = document.querySelector('#api-text-input textarea');
    if (!ta) return;
    const insert = '[' + pill.dataset.tag + '] ';
    const start = ta.selectionStart ?? ta.value.length;
    const end = ta.selectionEnd ?? ta.value.length;
    ta.value = ta.value.slice(0, start) + insert + ta.value.slice(end);
    ta.dispatchEvent(new Event('input', { bubbles: true }));
    ta.focus();
    ta.setSelectionRange(start + insert.length, start + insert.length);
  });
}
"""

CSS = """
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
.qtags { display: flex; flex-wrap: wrap; gap: 6px; margin: 6px 0; }
.qtag {
    font-size: 0.8rem; font-weight: 600; padding: 6px 14px; border-radius: 999px;
    background: #f8fafc; border: 1px solid #e2e8f0; color: #0f172a; cursor: pointer;
}
.qtag:hover { border-color: #2563eb; }
#voices-status { font-size: 0.8rem; color: #64748b; }
"""

with gr.Blocks(title="Fish Audio API Console", css=CSS) as app:
    gr.Markdown("## 🐟 Fish Audio API Console\nTest the *hosted* api.fish.audio API — separate from the self-hosted EC2 app.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Voice Input")
            with gr.Tab("Use a saved voice"):
                voice_dropdown = gr.Dropdown(label="Your voices (name — voice ID)", choices=[], value=None)
                refresh_btn = gr.Button("🔄 Refresh voice list", size="sm")
                voices_status = gr.Markdown("", elem_id="voices-status")
            with gr.Tab("Use fresh reference audio"):
                ref_audio = gr.Audio(label="Record or upload a reference voice", type="filepath")

        with gr.Column(scale=1):
            gr.Markdown("### Generate Speech")
            text_input = gr.Textbox(
                label="Text", lines=4, elem_id="api-text-input",
                placeholder="Type text here. Use the tags below or type [tag] manually.",
            )
            gr.HTML(
                '<div class="qtags">'
                + "".join(f'<button type="button" class="qtag" data-tag="{t}">{t}</button>' for t in QUICK_TAGS)
                + "</div>"
            )
            model_radio = gr.Radio(
                label="Model", choices=MODEL_CHOICES, value="s2.1-pro-free",
            )
            generate_btn = gr.Button("🎙️ Generate speech", variant="primary")
            error_out = gr.Markdown("")
            audio_out = gr.Audio(label="Generated Audio", type="filepath")

    app.load(fn=fetch_my_voices, inputs=None, outputs=[voice_dropdown, voices_status])
    app.load(None, None, js=INSERT_TAG_JS)
    refresh_btn.click(fn=fetch_my_voices, inputs=None, outputs=[voice_dropdown, voices_status])

    generate_btn.click(
        fn=generate,
        inputs=[text_input, voice_dropdown, ref_audio, model_radio],
        outputs=[audio_out, error_out],
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", share=True)
