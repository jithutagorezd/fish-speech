(function () {
  "use strict";

  var config = window.APP_CONFIG || {};
  var DEFAULT_CHUNK_WORDS = config.defaultChunkWords || 250;
  var INITIAL_TEXT = config.initialText || "";
  var MAX_WORDS = config.maxWords || 5000;

  var SPEEDS = [1, 1.25, 1.5, 2];
  var REF_BUCKETS = 36;
  var OUT_BUCKETS = 50;

  var state = {
    text: INITIAL_TEXT,
    mode: "single",
    chunkWords: DEFAULT_CHUNK_WORDS,
    referenceOpen: true,
    referenceId: "",
    memoryCache: true,

    audioUploaded: false,
    isRecording: false,
    recordSeconds: 0,
    referenceObjectUrl: null,
    referenceAudioBuffer: null,
    refDuration: 0,
    refPercent: 0,
    refPlaying: false,
    refPeakHeights: [],
    trimStart: 0,
    trimEnd: 100,

    mediaRecorder: null,
    mediaStream: null,

    transcribing: false,
    referenceText: "",

    advancedOpen: false,

    outputState: "empty", // empty | loading | ready | error
    errorMessage: "",
    generating: false,

    outObjectUrl: null,
    outAudioBuffer: null,
    outDuration: 0,
    outPercent: 0,
    outPlaying: false,
    outPeakHeights: [],
    outSpeed: 1,

    modelReady: false,
    modelLoading: true,
    modelError: null
  };

  var timers = { record: null, statusPoll: null };
  var audioCtx = null;

  function getAudioContext() {
    if (!audioCtx) {
      var Ctx = window.AudioContext || window.webkitAudioContext;
      audioCtx = new Ctx();
    }
    return audioCtx;
  }

  /* deterministic (non-random) placeholder shapes for pre-audio decorative states */
  function deterministicHeights(count, min, span) {
    var out = [];
    for (var i = 0; i < count; i++) {
      var t = count > 1 ? i / (count - 1) : 0;
      var wave = (Math.sin(t * Math.PI * 3) + 1) / 2;
      out.push(Math.round(min + wave * span));
    }
    return out;
  }

  var flatBarHeights = deterministicHeights(OUT_BUCKETS, 4, 6);

  function fmtTime(sec) {
    if (!isFinite(sec) || sec < 0) sec = 0;
    var m = Math.floor(sec / 60);
    var s = Math.round(sec % 60);
    return m + ":" + String(s).padStart(2, "0");
  }

  function $(id) { return document.getElementById(id); }

  var els = {};

  function cacheEls() {
    [
      "model-status-dot", "model-status-label",
      "error-banner", "error-title", "error-message", "error-dismiss",
      "word-count", "editor-mirror", "script-input",
      "emotion-tags-common", "emotion-tags-grouped", "emotion-tags-expand", "emotion-tags-collapse",
      "mode-single", "mode-long", "chunk-slider-row", "chunk-slider", "chunk-slider-value",
      "reference-toggle", "reference-chevron", "reference-body",
      "ref-id", "memory-cache-toggle",
      "audio-uploaded-block", "audio-empty-block", "ref-waveform", "ref-audio",
      "ref-play-toggle", "ref-time-label", "ref-replace",
      "trim-label", "trim-start", "trim-end",
      "upload-trigger", "upload-input", "record-toggle", "record-dot", "record-label",
      "transcribe-btn", "ref-text",
      "advanced-toggle", "advanced-chevron", "advanced-body",
      "output-estimate", "output-empty", "output-loading", "output-ready",
      "flat-waveform", "pulse-waveform", "out-waveform", "out-audio",
      "out-play-toggle", "out-time-label", "out-speed-toggle", "out-download",
      "generate-btn", "generate-btn-loading", "generate-btn-label"
    ].forEach(function (id) { els[id] = $(id); });
  }

  /* ---------- audio decoding / encoding ---------- */

  function decodeBlobToBuffer(blob) {
    return blob.arrayBuffer().then(function (arrayBuffer) {
      return getAudioContext().decodeAudioData(arrayBuffer.slice(0));
    });
  }

  function computePeaks(buffer, bucketCount) {
    var data = buffer.getChannelData(0);
    var blockSize = Math.max(1, Math.floor(data.length / bucketCount));
    var peaks = [];
    for (var i = 0; i < bucketCount; i++) {
      var start = i * blockSize;
      var end = Math.min(data.length, start + blockSize);
      var max = 0;
      for (var j = start; j < end; j++) {
        var v = Math.abs(data[j]);
        if (v > max) max = v;
      }
      peaks.push(max);
    }
    return peaks;
  }

  function peaksToHeights(peaks, minPx, maxPx) {
    return peaks.map(function (p) { return Math.round(minPx + p * (maxPx - minPx)); });
  }

  function audioBufferToWavBlob(buffer, startFrac, endFrac) {
    var sampleRate = buffer.sampleRate;
    var length = buffer.length;
    var startSample = Math.max(0, Math.min(length, Math.floor(length * startFrac)));
    var endSample = Math.max(startSample, Math.min(length, Math.floor(length * endFrac)));
    var frameCount = endSample - startSample;

    var channelData = [];
    for (var c = 0; c < buffer.numberOfChannels; c++) channelData.push(buffer.getChannelData(c));

    var mono = new Float32Array(frameCount);
    for (var i = 0; i < frameCount; i++) {
      var sum = 0;
      for (var c2 = 0; c2 < channelData.length; c2++) sum += channelData[c2][startSample + i];
      mono[i] = sum / channelData.length;
    }

    var bytesPerSample = 2;
    var dataSize = frameCount * bytesPerSample;
    var arrayBuffer = new ArrayBuffer(44 + dataSize);
    var view = new DataView(arrayBuffer);

    function writeString(offset, str) {
      for (var k = 0; k < str.length; k++) view.setUint8(offset + k, str.charCodeAt(k));
    }

    writeString(0, "RIFF");
    view.setUint32(4, 36 + dataSize, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * bytesPerSample, true);
    view.setUint16(32, bytesPerSample, true);
    view.setUint16(34, 16, true);
    writeString(36, "data");
    view.setUint32(40, dataSize, true);

    var offset = 44;
    for (var n = 0; n < frameCount; n++) {
      var s = Math.max(-1, Math.min(1, mono[n]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }

    return new Blob([view], { type: "audio/wav" });
  }

  /* ---------- bar rendering ---------- */

  function buildFlexBars(container, heights, playedFraction) {
    container.innerHTML = "";
    var frag = document.createDocumentFragment();
    heights.forEach(function (h, i) {
      var bar = document.createElement("div");
      bar.className = "bar-flex";
      bar.style.height = h + "px";
      if ((i / heights.length) * 100 <= playedFraction) bar.classList.add("is-played");
      frag.appendChild(bar);
    });
    container.appendChild(frag);
  }

  function updateFlexBarsProgress(container, heights, playedFraction) {
    var bars = container.children;
    for (var i = 0; i < bars.length; i++) {
      bars[i].classList.toggle("is-played", (i / heights.length) * 100 <= playedFraction);
    }
  }

  function buildFixedBars(container, heights, animated) {
    container.innerHTML = "";
    var frag = document.createDocumentFragment();
    heights.forEach(function (h) {
      var bar = document.createElement("div");
      bar.className = "bar-fixed";
      bar.style.height = h + "px";
      if (!animated) bar.style.animation = "none";
      frag.appendChild(bar);
    });
    container.appendChild(frag);
  }

  /* ---------- error banner ---------- */

  function showError(title, message) {
    els["error-title"].textContent = title;
    els["error-message"].textContent = message;
    els["error-banner"].classList.remove("is-hidden");
  }

  function hideError() {
    els["error-banner"].classList.add("is-hidden");
  }

  function extractErrorMessage(res) {
    return res
      .json()
      .catch(function () { return {}; })
      .then(function (body) {
        return body.detail || "Request failed with status " + res.status + ".";
      });
  }

  /* ---------- model status ---------- */

  function renderModelStatus() {
    var dot = els["model-status-dot"];
    var label = els["model-status-label"];
    dot.classList.remove("is-loading", "is-error");
    if (state.modelError) {
      dot.classList.add("is-error");
      label.textContent = "model error";
      label.title = state.modelError;
    } else if (!state.modelReady) {
      dot.classList.add("is-loading");
      label.textContent = state.modelLoading ? "loading model…" : "model offline";
      label.title = "";
    } else {
      label.textContent = "model warm";
      label.title = "";
    }
  }

  function pollStatus() {
    fetch("/api/status")
      .then(function (res) { return res.json(); })
      .then(function (data) {
        state.modelReady = !!data.ready;
        state.modelLoading = !!data.loading;
        state.modelError = data.error || null;
        renderModelStatus();
        timers.statusPoll = setTimeout(pollStatus, state.modelReady ? 20000 : 3000);
      })
      .catch(function () {
        timers.statusPoll = setTimeout(pollStatus, 8000);
      });
  }

  /* ---------- script editor ---------- */

  function renderEditor() {
    var text = state.text;
    els["script-input"].value = text;

    var wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
    els["word-count"].textContent = wordCount.toLocaleString() + " / " + MAX_WORDS.toLocaleString() + " words";

    var mirror = els["editor-mirror"];
    mirror.innerHTML = "";
    if (text.length === 0) {
      var placeholder = document.createElement("span");
      placeholder.className = "placeholder";
      placeholder.textContent = "Type your script here. Insert tags like [whisper] or [pause] from the patch bay below.";
      mirror.appendChild(placeholder);
    } else {
      var segments = text.split(/(\[[a-zA-Z ]+\])/g).filter(function (t) { return t.length > 0; });
      var frag = document.createDocumentFragment();
      segments.forEach(function (t) {
        var isTag = /^\[[a-zA-Z ]+\]$/.test(t);
        var span = document.createElement("span");
        if (isTag) span.className = "tag";
        span.textContent = t;
        frag.appendChild(span);
      });
      mirror.appendChild(frag);
    }

    updateOutputEstimate(wordCount);
    updateChunksEstimate(wordCount);
  }

  function syncMirrorScroll() {
    els["editor-mirror"].scrollTop = els["script-input"].scrollTop;
  }

  function insertTag(tagName) {
    var ta = els["script-input"];
    var insert = "[" + tagName + "] ";
    var start = ta.selectionStart, end = ta.selectionEnd;
    if (start == null || end == null) {
      state.text = state.text + insert;
      renderEditor();
      return;
    }
    state.text = state.text.slice(0, start) + insert + state.text.slice(end);
    renderEditor();
    ta.focus();
    var pos = start + insert.length;
    ta.setSelectionRange(pos, pos);
  }

  /* ---------- mode toggle ---------- */

  function renderMode() {
    els["mode-single"].classList.toggle("is-active", state.mode === "single");
    els["mode-long"].classList.toggle("is-active", state.mode === "long");
    els["chunk-slider-row"].classList.toggle("is-hidden", state.mode !== "long");
    updateGenerateLabel();
  }

  function updateChunksEstimate(wordCount) {
    var chunks = Math.max(1, Math.ceil(wordCount / state.chunkWords));
    els["chunk-slider-value"].textContent = state.chunkWords + " · ≈" + chunks + " chunks";
  }

  function updateGenerateLabel() {
    if (state.generating) return;
    els["generate-btn-label"].textContent = state.mode === "long" ? "Generate speech (chunked)" : "Generate speech";
  }

  /* ---------- collapsible sections ---------- */

  function renderReferenceCollapse() {
    els["reference-body"].classList.toggle("is-hidden", !state.referenceOpen);
    els["reference-chevron"].classList.toggle("is-open", state.referenceOpen);
  }

  function renderAdvancedCollapse() {
    els["advanced-body"].classList.toggle("is-hidden", !state.advancedOpen);
    els["advanced-chevron"].classList.toggle("is-open", state.advancedOpen);
  }

  /* ---------- memory cache switch ---------- */

  function renderMemoryCache() {
    var track = els["memory-cache-toggle"].querySelector(".switch-track");
    track.classList.toggle("is-on", state.memoryCache);
    els["memory-cache-toggle"].setAttribute("aria-checked", String(state.memoryCache));
  }

  /* ---------- reference audio ---------- */

  function loadReferenceBlob(blob) {
    var url = URL.createObjectURL(blob);
    return decodeBlobToBuffer(blob).then(function (buffer) {
      if (state.referenceObjectUrl) URL.revokeObjectURL(state.referenceObjectUrl);
      state.referenceObjectUrl = url;
      state.referenceAudioBuffer = buffer;
      state.refDuration = buffer.duration;
      state.refPeakHeights = peaksToHeights(computePeaks(buffer, REF_BUCKETS), 4, 40);
      state.audioUploaded = true;
      state.refPercent = 0;
      state.refPlaying = false;
      state.trimStart = 0;
      state.trimEnd = 100;
      els["ref-audio"].src = url;
      renderAudioSection();
    }).catch(function (err) {
      URL.revokeObjectURL(url);
      showError("Couldn't read audio", (err && err.message) || "That file doesn't look like a supported audio format.");
    });
  }

  function renderAudioSection() {
    els["audio-uploaded-block"].classList.toggle("is-hidden", !state.audioUploaded);
    els["audio-empty-block"].classList.toggle("is-hidden", state.audioUploaded);

    if (state.audioUploaded) {
      buildFlexBars(els["ref-waveform"], state.refPeakHeights, state.refPercent);
      updateRefTimeLabel();
      updateTrimLabel();
      renderRefPlayIcon();
      els["trim-start"].value = state.trimStart;
      els["trim-end"].value = state.trimEnd;
    }

    renderRecordButton();
    renderTranscribeButton();
  }

  function renderRefPlayIcon() {
    els["ref-play-toggle"].textContent = state.refPlaying ? "❚❚" : "▶";
  }

  function updateRefTimeLabel() {
    var audio = els["ref-audio"];
    var current = audio && !isNaN(audio.currentTime) ? audio.currentTime : 0;
    var duration = state.refDuration || 0;
    els["ref-time-label"].textContent = fmtTime(current) + " / " + fmtTime(duration);
  }

  function updateTrimLabel() {
    var duration = state.refDuration || 0;
    var startSec = ((duration * state.trimStart) / 100).toFixed(1);
    var endSec = ((duration * state.trimEnd) / 100).toFixed(1);
    els["trim-label"].textContent = "Trim — " + startSec + "s – " + endSec + "s";
  }

  function renderRecordButton() {
    els["record-toggle"].classList.toggle("is-recording", state.isRecording);
    els["record-dot"].classList.toggle("is-hidden", !state.isRecording);
    els["record-label"].textContent = state.isRecording
      ? "Recording… 0:" + String(state.recordSeconds).padStart(2, "0")
      : "● Record reference";
  }

  function renderTranscribeButton() {
    var enabled = state.audioUploaded && !state.transcribing;
    var btn = els["transcribe-btn"];
    btn.disabled = !enabled;
    btn.classList.toggle("is-enabled", enabled);
    btn.textContent = state.transcribing ? "Transcribing…" : "Auto-transcribe";
  }

  /* ---------- output panel ---------- */

  function updateOutputEstimate(wordCount) {
    els["output-estimate"].textContent = "~" + fmtTime(Math.max(2, wordCount / 2.5)) + " est.";
  }

  function renderOutputState() {
    var s = state.outputState;
    els["output-empty"].classList.toggle("is-hidden", s !== "empty");
    els["output-loading"].classList.toggle("is-hidden", s !== "loading");
    els["output-ready"].classList.toggle("is-hidden", s !== "ready");
    els["output-estimate"].classList.toggle("is-hidden", s !== "empty");

    if (s === "empty") buildFixedBars(els["flat-waveform"], flatBarHeights, false);
    if (s === "loading") buildFixedBars(els["pulse-waveform"], flatBarHeights, true);
    if (s === "ready") {
      buildFlexBars(els["out-waveform"], state.outPeakHeights, state.outPercent);
      updateOutTimeLabel();
      renderOutPlayIcon();
      els["out-speed-toggle"].textContent = state.outSpeed + "×";
    }
  }

  function renderOutPlayIcon() {
    els["out-play-toggle"].textContent = state.outPlaying ? "❚❚" : "▶";
  }

  function updateOutTimeLabel() {
    var audio = els["out-audio"];
    var current = audio && !isNaN(audio.currentTime) ? audio.currentTime : 0;
    var duration = state.outDuration || 0;
    els["out-time-label"].textContent = fmtTime(current) + " / " + fmtTime(duration);
  }

  function renderGenerateButton() {
    els["generate-btn"].disabled = state.generating;
    els["generate-btn-loading"].classList.toggle("is-hidden", !state.generating);
    els["generate-btn-label"].classList.toggle("is-hidden", state.generating);
    if (!state.generating) updateGenerateLabel();
  }

  /* ---------- event handlers: script / mode / collapsibles ---------- */

  function onTextChange(e) {
    state.text = e.target.value;
    renderEditor();
  }

  function onModeSingle() { state.mode = "single"; renderMode(); }
  function onModeLong() { state.mode = "long"; renderMode(); }

  function onChunkWordsChange(e) {
    state.chunkWords = Number(e.target.value);
    var wordCount = state.text.trim() ? state.text.trim().split(/\s+/).length : 0;
    updateChunksEstimate(wordCount);
  }

  function onToggleReference() {
    state.referenceOpen = !state.referenceOpen;
    renderReferenceCollapse();
  }

  function onToggleAdvanced() {
    state.advancedOpen = !state.advancedOpen;
    renderAdvancedCollapse();
  }

  function onReferenceIdChange(e) { state.referenceId = e.target.value; }

  function onToggleMemoryCache() {
    state.memoryCache = !state.memoryCache;
    renderMemoryCache();
  }

  function onReferenceTextChange(e) { state.referenceText = e.target.value; }

  /* ---------- event handlers: reference audio ---------- */

  function onTriggerUpload() { els["upload-input"].click(); }

  function onFileChange(e) {
    var file = e.target.files && e.target.files[0];
    if (file) loadReferenceBlob(file);
  }

  function onResetAudio() {
    els["ref-audio"].pause();
    els["ref-audio"].removeAttribute("src");
    if (state.referenceObjectUrl) URL.revokeObjectURL(state.referenceObjectUrl);
    state.referenceObjectUrl = null;
    state.referenceAudioBuffer = null;
    state.refDuration = 0;
    state.refPeakHeights = [];
    state.audioUploaded = false;
    state.refPercent = 0;
    state.refPlaying = false;
    state.trimStart = 0;
    state.trimEnd = 100;
    els["upload-input"].value = "";
    renderAudioSection();
  }

  function stopTimer(name) {
    if (timers[name]) {
      clearInterval(timers[name]);
      timers[name] = null;
    }
  }

  function onToggleRecord() {
    if (state.isRecording) {
      if (state.mediaRecorder && state.mediaRecorder.state !== "inactive") {
        state.mediaRecorder.stop();
      }
      return;
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || typeof MediaRecorder === "undefined") {
      showError("Recording unavailable", "This browser does not support microphone recording.");
      return;
    }
    navigator.mediaDevices.getUserMedia({ audio: true }).then(function (stream) {
      state.mediaStream = stream;
      var chunks = [];
      var recorder;
      try {
        recorder = new MediaRecorder(stream);
      } catch (err) {
        stream.getTracks().forEach(function (t) { t.stop(); });
        showError("Recording unavailable", "Couldn't start the recorder: " + err.message);
        return;
      }
      state.mediaRecorder = recorder;
      recorder.addEventListener("dataavailable", function (e) {
        if (e.data && e.data.size) chunks.push(e.data);
      });
      recorder.addEventListener("stop", function () {
        stream.getTracks().forEach(function (t) { t.stop(); });
        stopTimer("record");
        state.isRecording = false;
        state.recordSeconds = 0;
        state.mediaRecorder = null;
        state.mediaStream = null;
        renderRecordButton();
        var blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
        if (blob.size > 0) loadReferenceBlob(blob);
      });
      recorder.start();
      state.isRecording = true;
      state.recordSeconds = 0;
      renderRecordButton();
      timers.record = setInterval(function () {
        state.recordSeconds += 1;
        renderRecordButton();
      }, 1000);
    }).catch(function (err) {
      showError("Microphone blocked", "Couldn't access the microphone: " + (err.message || err.name || "permission denied"));
    });
  }

  function onToggleRefPlay() {
    var audio = els["ref-audio"];
    if (!audio.src) return;
    if (state.refPlaying) {
      audio.pause();
    } else {
      if (audio.ended || state.refPercent >= 100) audio.currentTime = 0;
      audio.play();
    }
  }

  function onSeekRef(e) {
    var audio = els["ref-audio"];
    if (!audio.src || !audio.duration) return;
    var rect = els["ref-waveform"].getBoundingClientRect();
    var frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    audio.currentTime = frac * audio.duration;
  }

  function onTrimStartChange(e) {
    state.trimStart = Math.min(Number(e.target.value), state.trimEnd - 5);
    els["trim-start"].value = state.trimStart;
    updateTrimLabel();
  }

  function onTrimEndChange(e) {
    state.trimEnd = Math.max(Number(e.target.value), state.trimStart + 5);
    els["trim-end"].value = state.trimEnd;
    updateTrimLabel();
  }

  function onTranscribe() {
    if (!state.audioUploaded || state.transcribing || !state.referenceAudioBuffer) return;
    state.transcribing = true;
    renderTranscribeButton();

    var wavBlob = audioBufferToWavBlob(state.referenceAudioBuffer, 0, 1);
    var formData = new FormData();
    formData.append("reference_audio", wavBlob, "reference.wav");

    fetch("/api/transcribe", { method: "POST", body: formData })
      .then(function (res) {
        if (!res.ok) return extractErrorMessage(res).then(function (msg) { throw new Error(msg); });
        return res.json();
      })
      .then(function (data) {
        state.transcribing = false;
        state.referenceText = data.text || "";
        els["ref-text"].value = state.referenceText;
        renderTranscribeButton();
      })
      .catch(function (err) {
        state.transcribing = false;
        renderTranscribeButton();
        showError("Transcription failed", err.message || "Whisper transcription error.");
      });
  }

  /* ---------- generate ---------- */

  function onDismissError() {
    hideError();
    if (state.outputState === "error") {
      state.outputState = "empty";
      renderOutputState();
    }
  }

  function onGenerate() {
    if (state.generating) return;

    if (!state.text.trim()) {
      state.outputState = "error";
      showError("Generation failed", "Input text is empty. Add a script before generating.");
      renderOutputState();
      return;
    }

    var wordCount = state.text.trim().split(/\s+/).length;
    if (wordCount > MAX_WORDS) {
      state.outputState = "error";
      showError(
        "Generation failed",
        "Script exceeds " + MAX_WORDS.toLocaleString() + " words. Switch to long-form mode or trim the script."
      );
      renderOutputState();
      return;
    }

    if (!state.modelReady) {
      state.outputState = "error";
      showError(
        "Generation failed",
        state.modelError
          ? "Model failed to load: " + state.modelError
          : "The Fish Speech model is still loading. Try again in a moment."
      );
      renderOutputState();
      return;
    }

    hideError();
    state.generating = true;
    state.outputState = "loading";
    renderGenerateButton();
    renderOutputState();

    var formData = new FormData();
    formData.append("text", state.text);
    formData.append("mode", state.mode);
    formData.append("reference_id", state.referenceId);
    formData.append("reference_text", state.referenceText);
    formData.append("memory_cache", state.memoryCache ? "on" : "off");
    formData.append("chunk_words", String(state.chunkWords));

    if (state.referenceAudioBuffer) {
      var wavBlob = audioBufferToWavBlob(state.referenceAudioBuffer, state.trimStart / 100, state.trimEnd / 100);
      formData.append("reference_audio", wavBlob, "reference.wav");
    }

    fetch("/api/generate", { method: "POST", body: formData })
      .then(function (res) {
        if (!res.ok) return extractErrorMessage(res).then(function (msg) { throw new Error(msg); });
        return res.blob();
      })
      .then(function (blob) {
        return decodeBlobToBuffer(blob).then(function (buffer) { return { blob: blob, buffer: buffer }; });
      })
      .then(function (result) {
        if (state.outObjectUrl) URL.revokeObjectURL(state.outObjectUrl);
        state.outObjectUrl = URL.createObjectURL(result.blob);
        state.outAudioBuffer = result.buffer;
        state.outDuration = result.buffer.duration;
        state.outPeakHeights = peaksToHeights(computePeaks(result.buffer, OUT_BUCKETS), 4, 52);
        state.outPercent = 0;
        state.outPlaying = false;
        state.outSpeed = 1;
        els["out-audio"].src = state.outObjectUrl;
        els["out-audio"].playbackRate = 1;

        state.generating = false;
        state.outputState = "ready";
        renderGenerateButton();
        renderOutputState();
      })
      .catch(function (err) {
        state.generating = false;
        state.outputState = "error";
        renderGenerateButton();
        showError("Generation failed", err.message || "Something went wrong while generating audio.");
        renderOutputState();
      });
  }

  function onToggleOutPlay() {
    var audio = els["out-audio"];
    if (!audio.src) return;
    if (state.outPlaying) {
      audio.pause();
    } else {
      if (audio.ended || state.outPercent >= 100) audio.currentTime = 0;
      audio.play();
    }
  }

  function onSeekOut(e) {
    var audio = els["out-audio"];
    if (!audio.src || !audio.duration) return;
    var rect = els["out-waveform"].getBoundingClientRect();
    var frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    audio.currentTime = frac * audio.duration;
  }

  function onCycleSpeed() {
    var idx = SPEEDS.indexOf(state.outSpeed);
    state.outSpeed = SPEEDS[(idx + 1) % SPEEDS.length];
    els["out-audio"].playbackRate = state.outSpeed;
    els["out-speed-toggle"].textContent = state.outSpeed + "×";
  }

  function onDownloadOut() {
    if (!state.outObjectUrl) return;
    var a = document.createElement("a");
    a.href = state.outObjectUrl;
    a.download = "speech-" + Date.now() + ".wav";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  /* ---------- wiring ---------- */

  function buildEmotionTags() {
    var common = els["emotion-tags-common"];
    var grouped = els["emotion-tags-grouped"];

    [common, grouped].forEach(function (wrap) {
      Array.prototype.forEach.call(wrap.querySelectorAll(".emotion-tag[data-tag]"), function (btn) {
        btn.addEventListener("click", function () { insertTag(btn.dataset.tag); });
      });
    });

    els["emotion-tags-expand"].addEventListener("click", function () {
      common.classList.add("is-hidden");
      grouped.classList.remove("is-hidden");
    });
    els["emotion-tags-collapse"].addEventListener("click", function () {
      grouped.classList.add("is-hidden");
      common.classList.remove("is-hidden");
    });
  }

  function wireRefAudioElement() {
    var audio = els["ref-audio"];
    audio.addEventListener("play", function () { state.refPlaying = true; renderRefPlayIcon(); });
    audio.addEventListener("pause", function () { state.refPlaying = false; renderRefPlayIcon(); });
    audio.addEventListener("ended", function () {
      state.refPlaying = false;
      state.refPercent = 0;
      renderRefPlayIcon();
      updateFlexBarsProgress(els["ref-waveform"], state.refPeakHeights, 0);
      updateRefTimeLabel();
    });
    audio.addEventListener("timeupdate", function () {
      if (audio.duration) state.refPercent = (audio.currentTime / audio.duration) * 100;
      updateFlexBarsProgress(els["ref-waveform"], state.refPeakHeights, state.refPercent);
      updateRefTimeLabel();
    });
  }

  function wireOutAudioElement() {
    var audio = els["out-audio"];
    audio.addEventListener("play", function () { state.outPlaying = true; renderOutPlayIcon(); });
    audio.addEventListener("pause", function () { state.outPlaying = false; renderOutPlayIcon(); });
    audio.addEventListener("ended", function () {
      state.outPlaying = false;
      state.outPercent = 0;
      renderOutPlayIcon();
      updateFlexBarsProgress(els["out-waveform"], state.outPeakHeights, 0);
      updateOutTimeLabel();
    });
    audio.addEventListener("timeupdate", function () {
      if (audio.duration) state.outPercent = (audio.currentTime / audio.duration) * 100;
      updateFlexBarsProgress(els["out-waveform"], state.outPeakHeights, state.outPercent);
      updateOutTimeLabel();
    });
  }

  function init() {
    cacheEls();

    els["script-input"].addEventListener("input", onTextChange);
    els["script-input"].addEventListener("scroll", syncMirrorScroll);
    buildEmotionTags();

    els["mode-single"].addEventListener("click", onModeSingle);
    els["mode-long"].addEventListener("click", onModeLong);
    els["chunk-slider"].addEventListener("input", onChunkWordsChange);

    els["reference-toggle"].addEventListener("click", onToggleReference);
    els["advanced-toggle"].addEventListener("click", onToggleAdvanced);

    els["ref-id"].addEventListener("input", onReferenceIdChange);
    els["memory-cache-toggle"].addEventListener("click", onToggleMemoryCache);

    els["upload-trigger"].addEventListener("click", onTriggerUpload);
    els["upload-input"].addEventListener("change", onFileChange);
    els["record-toggle"].addEventListener("click", onToggleRecord);
    els["ref-replace"].addEventListener("click", onResetAudio);
    els["ref-play-toggle"].addEventListener("click", onToggleRefPlay);
    els["ref-waveform"].addEventListener("click", onSeekRef);
    els["trim-start"].addEventListener("input", onTrimStartChange);
    els["trim-end"].addEventListener("input", onTrimEndChange);
    wireRefAudioElement();

    els["transcribe-btn"].addEventListener("click", onTranscribe);
    els["ref-text"].addEventListener("input", onReferenceTextChange);

    els["error-dismiss"].addEventListener("click", onDismissError);

    els["out-play-toggle"].addEventListener("click", onToggleOutPlay);
    els["out-waveform"].addEventListener("click", onSeekOut);
    els["out-speed-toggle"].addEventListener("click", onCycleSpeed);
    els["out-download"].addEventListener("click", onDownloadOut);
    wireOutAudioElement();

    els["generate-btn"].addEventListener("click", onGenerate);

    renderEditor();
    renderMode();
    renderReferenceCollapse();
    renderAdvancedCollapse();
    renderMemoryCache();
    renderAudioSection();
    renderOutputState();
    hideError();
    renderGenerateButton();
    renderModelStatus();
    pollStatus();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
