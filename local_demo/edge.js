const edgeText = document.getElementById("edgeText");
const edgeSource = document.getElementById("edgeSource");
const edgeTarget = document.getElementById("edgeTarget");
const edgeTts = document.getElementById("edgeTts");
const edgeOutput = document.getElementById("edgeOutput");
const edgeGain = document.getElementById("edgeGain");

const edgeStart = document.getElementById("edgeStart");
const edgeStop = document.getElementById("edgeStop");
const edgeSend = document.getElementById("edgeSend");

const edgeHealth = document.getElementById("edgeHealth");
const edgeStatus = document.getElementById("edgeStatus");
const edgeTranslated = document.getElementById("edgeTranslated");
const edgeMessage = document.getElementById("edgeMessage");
const edgeError = document.getElementById("edgeError");
const edgeNote = document.getElementById("edgeNote");

let recognition = null;
let isDictating = false;

const langMap = {
  english: "en-IN",
  hindi: "hi-IN",
  tamil: "ta-IN",
  telugu: "te-IN",
};

function setStatus(text) {
  edgeStatus.textContent = text;
}

function setError(text) {
  edgeError.textContent = text || "";
}

function setMessage(text) {
  edgeMessage.textContent = text || "--";
}

async function fetchHealth() {
  edgeHealth.textContent = "Checking...";
  try {
    const response = await fetch("/api/health");
    if (!response.ok) {
      throw new Error("health check failed");
    }
    const data = await response.json();
    if (!data.apiKeyConfigured) {
      edgeHealth.textContent = "API key missing";
      edgeNote.textContent = "Set SARVAM_API_KEY on the Pi to enable translation.";
      return;
    }
    edgeHealth.textContent = "Online";
  } catch (error) {
    edgeHealth.textContent = "Offline";
    edgeNote.textContent = "Start the server: python serve_demo.py --port 8080";
  }
}

function setupDictation() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    edgeStart.disabled = true;
    edgeStop.disabled = true;
    edgeNote.textContent = "Dictation not supported. Use the keyboard mic instead.";
    return;
  }

  recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = langMap[edgeSource.value] || "en-IN";

  recognition.onresult = (event) => {
    let transcript = "";
    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      transcript += event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        transcript += " ";
      }
    }
    edgeText.value = transcript.trim();
  };

  recognition.onend = () => {
    isDictating = false;
    edgeStart.disabled = false;
    edgeStop.disabled = true;
    setStatus("Idle");
  };

  recognition.onerror = (event) => {
    setError(event.error || "Dictation error");
  };
}

edgeSource.addEventListener("change", () => {
  if (recognition) {
    recognition.lang = langMap[edgeSource.value] || "en-IN";
  }
});

edgeStart.addEventListener("click", () => {
  if (!recognition) {
    return;
  }
  setError("");
  isDictating = true;
  edgeStart.disabled = true;
  edgeStop.disabled = false;
  setStatus("Dictating...");
  recognition.lang = langMap[edgeSource.value] || "en-IN";
  recognition.start();
});

edgeStop.addEventListener("click", () => {
  if (!recognition) {
    return;
  }
  recognition.stop();
});

edgeSend.addEventListener("click", async () => {
  const text = edgeText.value.trim();
  if (!text) {
    setError("Enter or dictate some text first.");
    return;
  }

  setError("");
  setStatus("Sending...");
  setMessage("--");
  edgeTranslated.textContent = "--";

  const payload = {
    text,
    sourceLang: edgeSource.value,
    targetLang: edgeTarget.value,
    ttsBackend: edgeTts.value,
    outputDevice: edgeOutput.value ? Number(edgeOutput.value) : null,
    playbackGain: edgeGain.value ? Number(edgeGain.value) : 1.0,
  };

  try {
    const response = await fetch("/api/speak", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Speak request failed");
    }

    edgeTranslated.textContent = data.translatedText || "(empty)";
    setMessage("Queued on Pi using " + (data.ttsBackend || "auto"));
    setStatus("Queued");
  } catch (error) {
    setStatus("Error");
    setError(String(error.message || error));
  }
});

fetchHealth();
setupDictation();
