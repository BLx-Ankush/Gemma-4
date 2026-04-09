const els = {
  modePill: document.getElementById("mode-pill"),
  modelPill: document.getElementById("model-pill"),
  queryForm: document.getElementById("query-form"),
  queryText: document.getElementById("query-text"),
  queryLanguage: document.getElementById("query-language"),
  queryImage: document.getElementById("query-image"),
  querySubmit: document.getElementById("query-submit"),
  responseBox: document.getElementById("response-box"),
  responseAudio: document.getElementById("response-audio"),
  metaBox: document.getElementById("meta-box"),
  startSession: document.getElementById("start-session"),
  endSession: document.getElementById("end-session"),
  sessionId: document.getElementById("session-id"),
  recordToggle: document.getElementById("record-toggle"),
  voiceLanguage: document.getElementById("voice-language"),
  historyBox: document.getElementById("history-box"),
};

let activeSessionId = null;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;


async function fetchHealth() {
  try {
    const response = await fetch("/status");
    const payload = await response.json();
    if (!payload.ok) {
      throw new Error("health check failed");
    }

    const mode = payload.compute?.mode || "unknown";
    const hasVision = payload.model?.has_vision ? "yes" : "no";
    const modelName = payload.model?.active_backend || (payload.model?.loaded ? "ready" : "cold");
    els.modePill.textContent = `Mode: ${mode}`;
    els.modelPill.textContent = `Backend: ${modelName}, vision: ${hasVision}`;
  } catch (_err) {
    els.modePill.textContent = "Mode: unavailable";
    els.modelPill.textContent = "Model: unavailable";
  }
}


function setResponse(text, metaText = "", audioDataUrl = "") {
  els.responseBox.textContent = text || "No response";
  els.metaBox.textContent = metaText || "";

  if (audioDataUrl) {
    els.responseAudio.hidden = false;
    els.responseAudio.src = audioDataUrl;
  } else {
    els.responseAudio.hidden = true;
    els.responseAudio.removeAttribute("src");
  }
}


function renderHistory(history) {
  if (!Array.isArray(history) || history.length === 0) {
    els.historyBox.textContent = "Session history will appear here.";
    return;
  }

  const html = history
    .map((turn, idx) => {
      const label = idx + 1;
      const user = turn.user_text || "(no transcript)";
      const assistant = turn.assistant_text || "(no response)";
      const language = turn.language || "Unknown";
      return `
        <div class="turn">
          <strong>Turn ${label} - ${language}</strong><br>
          <span><strong>User:</strong> ${user}</span><br>
          <span><strong>AEGIS:</strong> ${assistant}</span>
        </div>
      `;
    })
    .join("");

  els.historyBox.innerHTML = html;
}


async function refreshHistory() {
  if (!activeSessionId) {
    renderHistory([]);
    return;
  }

  const response = await fetch(`/api/voice/history/${encodeURIComponent(activeSessionId)}`);
  const payload = await response.json();
  renderHistory(payload.history || []);
}


els.queryForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  els.querySubmit.disabled = true;
  setResponse("Working on your query...");

  try {
    const formData = new FormData();
    formData.append("text", els.queryText.value.trim());
    formData.append("language", els.queryLanguage.value);
    if (els.queryImage.files[0]) {
      formData.append("image", els.queryImage.files[0]);
    }

    const response = await fetch("/veda", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();

    if (!payload.ok) {
      throw new Error(payload.error || "Failed to fetch response");
    }

    const data = payload.data;
    const mirror = data.mirror || {};
    const meta = [
      `Language: ${data.detected_language || "Unknown"}`,
      `MIRROR verdict: ${mirror.verdict || "n/a"}`,
      `MIRROR confidence: ${mirror.confidence_score ?? "n/a"}`,
      `Time: ${data.processing_time_ms || 0} ms`,
    ].join(" | ");

    setResponse(data.response_text, meta, data.audio_response_data_url || "");
  } catch (err) {
    setResponse(`Error: ${err.message}`);
  } finally {
    els.querySubmit.disabled = false;
  }
});


els.startSession.addEventListener("click", async () => {
  const response = await fetch("/api/voice/session", { method: "POST" });
  const payload = await response.json();
  if (payload.ok) {
    activeSessionId = payload.session_id;
    els.sessionId.textContent = `Session: ${activeSessionId}`;
    els.recordToggle.disabled = false;
    await refreshHistory();
  }
});


els.endSession.addEventListener("click", async () => {
  if (!activeSessionId) {
    return;
  }

  await fetch(`/api/voice/session/${encodeURIComponent(activeSessionId)}`, { method: "DELETE" });
  activeSessionId = null;
  els.sessionId.textContent = "No active voice session";
  els.recordToggle.disabled = true;
  els.recordToggle.textContent = "Start Recording";
  renderHistory([]);
});


async function ensureRecorder() {
  if (mediaRecorder) {
    return;
  }

  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);

  mediaRecorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) {
      audioChunks.push(event.data);
    }
  };

  mediaRecorder.onstop = async () => {
    const blob = new Blob(audioChunks, { type: "audio/webm" });
    audioChunks = [];
    await sendVoiceTurn(blob);
  };
}


async function sendVoiceTurn(blob) {
  if (!activeSessionId) {
    setResponse("Voice session is not active.");
    return;
  }

  setResponse("Transcribing and processing voice query...");

  const formData = new FormData();
  formData.append("session_id", activeSessionId);
  formData.append("language", els.voiceLanguage.value);
  formData.append("audio", blob, "turn.webm");

  try {
    const response = await fetch("/voice", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();

    if (!payload.ok) {
      throw new Error(payload.error || "Voice query failed");
    }

    const data = payload.data;
    const mirror = data.mirror || {};
    const meta = [
      `Session: ${data.session_id}`,
      `MIRROR verdict: ${mirror.verdict || "n/a"}`,
      `Turn count: ${data.turn_count || 0}`,
      `Time: ${data.processing_time_ms || 0} ms`,
    ].join(" | ");

    setResponse(data.response_text, meta, data.audio_response_data_url || "");
    await refreshHistory();
  } catch (err) {
    setResponse(`Error: ${err.message}`);
  }
}


els.recordToggle.addEventListener("click", async () => {
  if (!activeSessionId) {
    setResponse("Start a session before recording.");
    return;
  }

  try {
    await ensureRecorder();
  } catch (_err) {
    setResponse("Microphone access failed. Please allow microphone permission.");
    return;
  }

  if (!isRecording) {
    isRecording = true;
    audioChunks = [];
    mediaRecorder.start();
    els.recordToggle.textContent = "Stop Recording";
    setResponse("Recording... click again to stop.");
    return;
  }

  isRecording = false;
  els.recordToggle.textContent = "Start Recording";
  mediaRecorder.stop();
});


fetchHealth();