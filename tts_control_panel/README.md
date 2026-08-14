# Cadence — TTS Control Panel

A FastAPI + Jinja2 server-rendered UI for Fish Speech S2, built from the
"TTS Control Panel" design. Vanilla JS/HTMX on the front end (no React); the
backend loads the real Fish Speech S2 model the same way
`tools/run_webui_v2.py` does (llama checkpoint + VQ-GAN decoder + warmup
inference), and adds Whisper-based auto-transcribe for reference audio.

There is no simulated/random audio anywhere in this app: waveforms are drawn
from real decoded PCM, playback is a real `<audio>` element, "Record
reference" uses the real microphone (`MediaRecorder`), and Generate /
Auto-transcribe call the real Fish Speech / Whisper backend.

This app lives inside the `fish-speech` repo (`fish-speech/tts_control_panel/`)
— one git checkout, one deployment, no separate repo to keep in sync.

## Layout

```
fish-speech/                  (repo root)
  fish_speech/, tools/, webui_v2/, checkpoints/, ...   existing fish-speech code
  tts_control_panel/
    main.py               FastAPI app, page route, model-load lifespan hook
    run_server.py          CLI entry point (mirrors run_webui_v2.py's flags)
    backend/
      config.py            Settings from CLI-set env vars (TCP_*)
      model_state.py        Loads TTSInferenceEngine (llama + decoder + warmup)
      audio_io.py            WAV encode + upload-to-tempfile helpers
      routes.py              /api/status, /api/generate, /api/transcribe
    templates/
      base.html, index.html
      partials/               header, error banner, input panel, mode toggle,
                              reference panel, advanced panel, output panel,
                              generate button
    static/
      css/styles.css
      js/app.js               all client-side state + real audio wiring
    scripts/
      install-service.sh       generates + starts the systemd unit
      cadence-tts.service       hand-editable systemd unit template
      start.sh / stop.sh / status.sh   PID-file based alternative to systemd
```

## Setup

This app imports `fish_speech`, `webui_v2`, and `tools.webui` directly from
the parent repo (via `sys.path`), so both dependency sets need to live in the
**same** Python environment. Requires a CUDA GPU (a `g4.2xlarge` or similar)
for practical inference speed — Fish Speech S2's llama checkpoint is large
enough that CPU inference is not practical.

```bash
# from the fish-speech/ repo root
cd tts_control_panel
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt          # web layer: fastapi, uvicorn, jinja2, ...
pip install -e ..                        # Fish Speech S2 + its own deps (torch, etc.)
pip install openai-whisper                # only needed for Auto-transcribe
```

Checkpoints (not included in this repo):

- `../checkpoints/s2-pro/` — the llama + `codec.pth` decoder checkpoint.
  Follow the main `fish-speech/README.md` to download it.
- `../checkpoints/whisper-small-pt/` — local dir for the Whisper "small"
  weights used by Auto-transcribe. If missing, the first Auto-transcribe
  request downloads the standard OpenAI "small" checkpoint (~461 MB) into
  that directory automatically.

## Run

Equivalent to the `run_webui_v2.py` launch command, just for this app
(defaults already point at `../checkpoints/...`, so you can omit the
checkpoint flags if your layout matches the one above):

```bash
python run_server.py \
    --llama-checkpoint-path ../checkpoints/s2-pro \
    --decoder-checkpoint-path ../checkpoints/s2-pro/codec.pth \
    --whisper-model-dir ../checkpoints/whisper-small-pt \
    --device cuda \
    --host 0.0.0.0 --port 8731
```

The server starts immediately; model loading (checkpoint read + warmup
inference) happens in a background thread. The header's status dot polls
`GET /api/status` and shows "loading model…" → "model warm", or "model
error" with the failure message if loading fails. `Generate` and
`Auto-transcribe` return a clear 503 until the model is ready.

Other useful flags: `--half`, `--compile`, `--quantization {int8,int4}`,
`--fish-speech-dir` (if this checkout isn't nested directly under the
fish-speech repo). All flags also have `TCP_*` env var equivalents (see
`backend/config.py`) if you'd rather configure via environment than CLI
args — handy for the systemd unit.

## Running as a service

**systemd (recommended for a GPU instance):**

`scripts/install-service.sh` generates the unit file from your actual
checkout (paths, user, python, checkpoints) and starts it — no manual
editing needed:

```bash
sudo scripts/install-service.sh                      # auto-detects everything
sudo scripts/install-service.sh --device cuda --port 8731
sudo scripts/install-service.sh --no-start            # install + enable only
```

Re-run it any time (e.g. after moving checkpoints) to regenerate and restart.
Then manage it with systemctl as usual:

```bash
sudo systemctl start cadence-tts
sudo systemctl stop cadence-tts
sudo systemctl restart cadence-tts
sudo systemctl status cadence-tts
journalctl -u cadence-tts -f          # tail logs
```

If you'd rather write the unit file by hand, `scripts/cadence-tts.service` is
a template with the same fields commented — copy it to
`/etc/systemd/system/`, edit the paths, then `daemon-reload` + `enable --now`.

**Plain scripts (no systemd/root needed):**

```bash
scripts/start.sh --device cuda --host 0.0.0.0 --port 8731   # any run_server.py args
scripts/status.sh
scripts/stop.sh
```

`start.sh` writes a PID file + log to `run/`; `status.sh` checks both the
process and `/api/status`; `stop.sh` kills it and clears the PID file.
`run/` is gitignored at the repo root.

## API

- `GET /` — the control panel page.
- `GET /api/status` — `{ready, loading, error, device}`.
- `POST /api/generate` — form fields `text`, `mode` (`single`|`long`),
  `reference_id`, `reference_text`, `memory_cache` (`on`|`off`),
  `chunk_words`, optional file `reference_audio`. Returns `audio/wav` bytes,
  or a JSON `{detail}` error (400/422/503).
- `POST /api/transcribe` — file field `reference_audio`. Returns
  `{text}`, or a JSON `{detail}` error.
