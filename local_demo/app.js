const inputText = document.getElementById("inputText");
const sourceLang = document.getElementById("sourceLang");
const targetLang = document.getElementById("targetLang");
const ttsBackend = document.getElementById("ttsBackend");
const directMode = document.getElementById("directMode");
const autoPlay = document.getElementById("autoPlay");

const translateBtn = document.getElementById("translateBtn");
const recordBtn = document.getElementById("recordBtn");
const voiceTranslateBtn = document.getElementById("voiceTranslateBtn");
const refreshHealthBtn = document.getElementById("refreshHealth");

const apiDot = document.getElementById("apiDot");
const apiStatus = document.getElementById("apiStatus");
const heroHealth = document.getElementById("heroHealth");

const statusText = document.getElementById("statusText");
const sourceText = document.getElementById("sourceText");
const outputText = document.getElementById("outputText");
const audioPlayer = document.getElementById("audioPlayer");
const audioHint = document.getElementById("audioHint");
const errorText = document.getElementById("errorText");

const latencyBadge = document.getElementById("latencyBadge");
const vadMs = document.getElementById("vadMs");
const sttMs = document.getElementById("sttMs");
const trMs = document.getElementById("trMs");
const ttsMs = document.getElementById("ttsMs");
const progressBar = document.getElementById("progressBar");

const recordState = document.getElementById("recordState");
const recordDuration = document.getElementById("recordDuration");

const state = {
  isBusy: false,
  mediaRecorder: null,
  mediaStream: null,
  recordChunks: [],
  recordBlob: null,
  recordStartedAt: 0,
  recordTimer: null,
  audioUrl: "",
};

function formatSeconds(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  return value.toFixed(2) + "s";
}

function formatClock(totalSeconds) {
  const safe = Math.max(0, Math.floor(totalSeconds));
  const mins = String(Math.floor(safe / 60)).padStart(2, "0");
  const secs = String(safe % 60).padStart(2, "0");
  return mins + ":" + secs;
}

function setStatus(message) {
  statusText.textContent = message;
}

function setError(message) {
  errorText.textContent = message || "";
}

function setApiStatus(mode, message) {
  apiDot.classList.remove("dot-ok", "dot-warn", "dot-error");
  if (mode === "ok") {
    apiDot.classList.add("dot-ok");
  } else if (mode === "error") {
    apiDot.classList.add("dot-error");
  } else {
    apiDot.classList.add("dot-warn");
  }
  apiStatus.textContent = message;
  heroHealth.textContent = message;
}

function setBusy(isBusy) {
  state.isBusy = isBusy;
  translateBtn.disabled = isBusy;
  refreshHealthBtn.disabled = isBusy;
  recordBtn.disabled = isBusy && !isRecording();
  voiceTranslateBtn.disabled = isBusy || !state.recordBlob;
}

function resetProgress() {
  progressBar.classList.remove("is-running", "is-complete");
  progressBar.style.width = "0%";
}

function runProgress() {
  progressBar.classList.add("is-running");
  progressBar.style.width = "70%";
}

function completeProgress() {
  progressBar.classList.remove("is-running");
  progressBar.classList.add("is-complete");
  progressBar.style.width = "100%";
}

function updateMetrics(metrics) {
  const safe = metrics || {};
  vadMs.textContent = formatSeconds(safe.vadSeconds);
  sttMs.textContent = formatSeconds(safe.sttSeconds);
  trMs.textContent = formatSeconds(safe.translateSeconds);
  ttsMs.textContent = formatSeconds(safe.ttsSeconds);
  latencyBadge.textContent = "Total: " + formatSeconds(safe.totalSeconds);
}

function updateAudio(audioBase64, backendLabel) {
  if (state.audioUrl) {
    URL.revokeObjectURL(state.audioUrl);
    state.audioUrl = "";
  }

  if (!audioBase64) {
    audioPlayer.removeAttribute("src");
    audioPlayer.load();
    audioHint.textContent = "No synthesized audio returned.";
    return;
  }

  try {
    const binary = atob(audioBase64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    const blob = new Blob([bytes], { type: "audio/wav" });
    state.audioUrl = URL.createObjectURL(blob);
    audioPlayer.src = state.audioUrl;
    audioHint.textContent = "Synthesized with backend: " + (backendLabel || "unknown");

    if (autoPlay.checked) {
      audioPlayer.play().catch(() => {
        audioHint.textContent = "Audio ready. Press play to listen.";
      });
    }
  } catch (error) {
    audioHint.textContent = "Audio decode failed.";
  }
}

function isRecording() {
  return state.mediaRecorder && state.mediaRecorder.state === "recording";
}

function stopRecordingTimer() {
  if (state.recordTimer) {
    clearInterval(state.recordTimer);
    state.recordTimer = null;
  }
}

function startRecordingTimer() {
  stopRecordingTimer();
  state.recordStartedAt = Date.now();
  recordDuration.textContent = "00:00";
  state.recordTimer = setInterval(() => {
    const elapsed = (Date.now() - state.recordStartedAt) / 1000;
    recordDuration.textContent = formatClock(elapsed);
  }, 250);
}

function finalizeRecorderState() {
  stopRecordingTimer();
  recordBtn.textContent = "Start recording";
}

async function fetchHealth() {
  setApiStatus("warn", "Checking backend connection...");
  setError("");
  try {
    const response = await fetch("/api/health");
    if (!response.ok) {
      throw new Error("health endpoint returned " + response.status);
    }

    const data = await response.json();
    if (!data.apiKeyConfigured) {
      setApiStatus("warn", "Backend online, API key missing");
      setError("Set SARVAM_API_KEY in your environment to enable live translation.");
      return;
    }

    setApiStatus("ok", "Backend online and ready");
  } catch (error) {
    setApiStatus("error", "Backend unreachable");
    setError("Could not reach /api/health. Start the server with: python serve_demo.py --port 8080");
  }
}

async function runTextTranslate() {
  const text = inputText.value.trim();
  if (!text) {
    setError("Enter text before translating.");
    return;
  }

  setError("");
  setBusy(true);
  resetProgress();
  runProgress();
  setStatus("Translating text...");

  try {
    const response = await fetch("/api/text-translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        sourceLang: sourceLang.value,
        targetLang: targetLang.value,
        ttsBackend: ttsBackend.value,
        includeAudio: true,
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "text translation failed");
    }

    sourceText.textContent = data.sourceText || text;
    outputText.textContent = data.translatedText || "(empty response)";
    updateMetrics(data.metrics);
    updateAudio(data.tts && data.tts.audioWavBase64, data.tts && data.tts.selected);
    setStatus("Done");
    completeProgress();
  } catch (error) {
    setStatus("Error");
    resetProgress();
    setError(String(error.message || error));
  } finally {
    setBusy(false);
  }
}

async function startRecording() {
  if (!navigator.mediaDevices || !window.MediaRecorder) {
    setError("This browser does not support audio recording.");
    return;
  }

  setError("");
  state.recordBlob = null;
  voiceTranslateBtn.disabled = true;

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });

    const preferredTypes = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
    ];
    let mimeType = "";
    for (const candidate of preferredTypes) {
      if (MediaRecorder.isTypeSupported(candidate)) {
        mimeType = candidate;
        break;
      }
    }

    state.recordChunks = [];
    state.mediaStream = stream;
    state.mediaRecorder = mimeType
      ? new MediaRecorder(stream, { mimeType })
      : new MediaRecorder(stream);

    state.mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        state.recordChunks.push(event.data);
      }
    };

    state.mediaRecorder.onstop = () => {
      const resolvedType = state.mediaRecorder && state.mediaRecorder.mimeType
        ? state.mediaRecorder.mimeType
        : "audio/webm";
      state.recordBlob = new Blob(state.recordChunks, { type: resolvedType });
      const sizeKb = (state.recordBlob.size / 1024).toFixed(0);
      recordState.textContent = "Recording ready (" + sizeKb + " KB)";
      voiceTranslateBtn.disabled = state.isBusy || !state.recordBlob;

      if (state.mediaStream) {
        state.mediaStream.getTracks().forEach((track) => track.stop());
      }
      state.mediaStream = null;
      finalizeRecorderState();
    };

    state.mediaRecorder.start();
    recordBtn.textContent = "Stop recording";
    recordState.textContent = "Recording in progress...";
    startRecordingTimer();
  } catch (error) {
    setError("Unable to access microphone: " + String(error.message || error));
  }
}

function stopRecording() {
  if (isRecording()) {
    state.mediaRecorder.stop();
  } else {
    finalizeRecorderState();
  }
}

async function runVoiceTranslate() {
  if (!state.recordBlob) {
    setError("Record audio first, then run voice translation.");
    return;
  }

  setBusy(true);
  setError("");
  resetProgress();
  runProgress();
  setStatus("Uploading audio...");

  try {
    const extension = state.recordBlob.type.includes("ogg") ? "ogg" : "webm";
    const form = new FormData();
    form.append("audio", state.recordBlob, "recording." + extension);
    form.append("targetLang", targetLang.value);
    form.append("directTranslate", directMode.checked ? "true" : "false");
    form.append("ttsBackend", ttsBackend.value);

    const response = await fetch("/api/voice-translate", {
      method: "POST",
      body: form,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "voice translation failed");
    }

    sourceText.textContent = data.englishText || "(speech input)";
    outputText.textContent = data.translatedText || "(empty response)";
    updateMetrics(data.metrics);
    updateAudio(data.tts && data.tts.audioWavBase64, data.tts && data.tts.selected);
    setStatus("Done");
    completeProgress();
  } catch (error) {
    setStatus("Error");
    resetProgress();
    setError(String(error.message || error));
  } finally {
    setBusy(false);
  }
}

translateBtn.addEventListener("click", runTextTranslate);

recordBtn.addEventListener("click", async () => {
  if (isRecording()) {
    stopRecording();
    return;
  }
  await startRecording();
});

voiceTranslateBtn.addEventListener("click", runVoiceTranslate);
refreshHealthBtn.addEventListener("click", fetchHealth);

const chips = document.querySelectorAll(".chip");
chips.forEach((chip) => {
  chip.addEventListener("click", () => {
    inputText.value = chip.getAttribute("data-phrase") || "";
    inputText.focus();
  });
});

window.addEventListener("beforeunload", () => {
  stopRecordingTimer();
  if (state.audioUrl) {
    URL.revokeObjectURL(state.audioUrl);
  }
  if (state.mediaStream) {
    state.mediaStream.getTracks().forEach((track) => track.stop());
  }
});

resetProgress();
setBusy(false);
fetchHealth();
