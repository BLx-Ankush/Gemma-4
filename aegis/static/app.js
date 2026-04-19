/**
 * AEGIS Frontend — Unified one-panel assistant flow
 * - Optional photo capture
 * - Center speak button for voice query
 * - Auto language detection from transcription
 * - Response text + audio playback
 * - Local history drawer from top-right icon
 */

"use strict";

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.getRegistrations().then((registrations) => {
      registrations.forEach((registration) => {
        registration.unregister();
      });
    }).catch(() => {
      // Ignore service worker cleanup failures.
    });
  });
}

const $ = (id) => document.getElementById(id);

// Status + runtime badges
const statusDot = $("status-dot");
const statusText = $("status-text");
const modelText = $("model-text");
const modePill = $("mode-pill");
const cooldownPill = $("cooldown-pill");
const cooldownText = $("cooldown-text");

// History drawer
const historyToggle = $("history-toggle");
const historyCount = $("history-count");
const historyDrawer = $("history-drawer");
const historyClose = $("history-close");
const historyClear = $("history-clear");
const historyList = $("history-list");

// Assistant panel
const assistantImage = $("assistant-image");
const assistantImagePreviewWrap = $("assistant-image-preview-wrap");
const assistantImagePreview = $("assistant-image-preview");
const assistantImageRemove = $("assistant-image-remove");
const captureLabel = $("capture-label");

const recordBtn = $("record-btn");
const recordRing = $("record-ring");
const recordIconMic = document.querySelector(".record-icon-mic");
const recordIconStop = document.querySelector(".record-icon-stop");
const recordHint = $("record-hint");
const waveCanvas = $("waveform-canvas");

const assistantResponseArea = $("assistant-response-area");
const assistantTranscriptWrap = $("assistant-transcript-wrap");
const assistantTranscript = $("assistant-transcript");
const assistantResponseText = $("assistant-response-text");
const assistantResponseAudio = $("assistant-response-audio");
const assistantLangChip = $("assistant-lang-chip");
const assistantModeChip = $("assistant-mode-chip");
const assistantResponseMeta = $("assistant-response-meta");

const ONLINE_REQUEST_TIMEOUT_MS = 60000;
const OFFLINE_REQUEST_TIMEOUT_MS = 180000;
const REQUEST_TIMEOUT_BUFFER_MS = 5000;
const HISTORY_STORAGE_KEY = "aegis.assistant.history.v1";
const HISTORY_LIMIT = 80;

let currentMode = "unknown";
let cloudCooldownUntilMs = 0;
let activeRequestController = null;
let serverVoiceTimeoutMs = 0;

let activeSessionId = null;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let audioCtx = null;
let analyserNode = null;
let waveAnimFrame = null;
let micStream = null;

function escHtml(value) {
  return String(value || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;");
}

function getCloudCooldownRemainingSec() {
  return Math.max(0, Math.ceil((cloudCooldownUntilMs - Date.now()) / 1000));
}

function isLikelyLocalProcessing() {
  return currentMode === "offline" || getCloudCooldownRemainingSec() > 0;
}

function getRequestTimeoutMs() {
  const baseTimeoutMs = isLikelyLocalProcessing() ? OFFLINE_REQUEST_TIMEOUT_MS : ONLINE_REQUEST_TIMEOUT_MS;
  if (serverVoiceTimeoutMs > 0) {
    return Math.max(baseTimeoutMs, serverVoiceTimeoutMs + REQUEST_TIMEOUT_BUFFER_MS);
  }
  return baseTimeoutMs;
}

function renderCooldownPill() {
  if (!cooldownPill || !cooldownText) {
    return;
  }
  const remainingSec = getCloudCooldownRemainingSec();
  if (remainingSec > 0) {
    cooldownPill.hidden = false;
    cooldownText.textContent = `Cloud cooldown: ${remainingSec}s`;
  } else {
    cooldownPill.hidden = true;
  }
}

async function fetchWithTimeout(url, options = {}, timeoutMs = getRequestTimeoutMs()) {
  const controller = new AbortController();
  activeRequestController = controller;
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
    if (activeRequestController === controller) {
      activeRequestController = null;
    }
  }
}

async function fetchStatus() {
  try {
    const res = await fetch("/status");
    const data = await res.json();
    if (!data.ok) {
      throw new Error("status not ok");
    }

    const mode = data.compute?.mode || "unknown";
    const online = mode === "online";
    const backend = data.model?.active_backend || (data.model?.loaded ? "ready" : "cold");
    const vision = data.model?.has_vision ? " · vision ✓" : "";

    const rawCooldown = Number(data.model?.cloud_cooldown_remaining_sec || 0);
    const cooldownSec = Number.isFinite(rawCooldown) ? Math.max(0, Math.round(rawCooldown)) : 0;
    const rawVoiceTimeoutSec = Number(data.runtime?.timeouts?.voice_request_sec || 0);

    currentMode = mode;
    cloudCooldownUntilMs = Date.now() + (cooldownSec * 1000);
    serverVoiceTimeoutMs = Number.isFinite(rawVoiceTimeoutSec) && rawVoiceTimeoutSec > 0
      ? Math.round(rawVoiceTimeoutSec * 1000)
      : 0;

    statusDot.dataset.mode = mode;
    statusText.textContent = online
      ? (cooldownSec > 0 ? "Online — Cloud Cooling Down" : "Online — Cloud")
      : "Offline — Local";
    modelText.textContent = `${backend}${vision}`;
    modePill.dataset.mode = mode;
    renderCooldownPill();
  } catch {
    currentMode = "unknown";
    cloudCooldownUntilMs = 0;
    serverVoiceTimeoutMs = 0;
    statusDot.dataset.mode = "error";
    statusText.textContent = "Server unreachable";
    modelText.textContent = "";
    renderCooldownPill();
  }
}

function getHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveHistory(entries) {
  localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(entries.slice(0, HISTORY_LIMIT)));
}

function updateHistoryCount() {
  if (!historyCount) {
    return;
  }
  historyCount.textContent = String(getHistory().length);
}

function renderHistoryList() {
  if (!historyList) {
    return;
  }

  const entries = getHistory();
  if (!entries.length) {
    historyList.innerHTML = "<p class='empty-note'>No history yet.</p>";
    return;
  }

  historyList.innerHTML = entries.map((entry, index) => {
    const ts = entry.timestamp ? new Date(entry.timestamp).toLocaleString() : "";
    const photoTag = entry.hasImage ? "Photo" : "Voice only";
    return `
      <article class="history-item">
        <div class="history-item-head">
          <span class="history-item-index">#${entries.length - index}</span>
          <span class="history-item-time">${escHtml(ts)}</span>
        </div>
        <p class="history-item-line"><strong>Input:</strong> ${escHtml(entry.input)}</p>
        <p class="history-item-line"><strong>Response:</strong> ${escHtml(entry.output)}</p>
        <div class="history-item-tags">
          <span class="history-tag">${escHtml(entry.detectedLanguage || "Unknown")}</span>
          <span class="history-tag">${escHtml((entry.mode || "unknown").toUpperCase())}</span>
          <span class="history-tag">${photoTag}</span>
        </div>
      </article>
    `;
  }).join("");
}

function addHistoryEntry(entry) {
  const entries = getHistory();
  entries.unshift(entry);
  saveHistory(entries);
  updateHistoryCount();
  renderHistoryList();
}

function openHistoryDrawer() {
  renderHistoryList();
  historyDrawer.hidden = false;
  document.body.classList.add("history-open");
  historyToggle.setAttribute("aria-expanded", "true");
}

function closeHistoryDrawer() {
  historyDrawer.hidden = true;
  document.body.classList.remove("history-open");
  historyToggle.setAttribute("aria-expanded", "false");
}

function toggleHistoryDrawer() {
  if (historyDrawer.hidden) {
    openHistoryDrawer();
  } else {
    closeHistoryDrawer();
  }
}

function drawWaveform() {
  if (!analyserNode) {
    return;
  }

  const ctx = waveCanvas.getContext("2d");
  const width = waveCanvas.width;
  const height = waveCanvas.height;
  const data = new Uint8Array(analyserNode.frequencyBinCount);

  function frame() {
    if (!isRecording) {
      return;
    }

    waveAnimFrame = requestAnimationFrame(frame);
    analyserNode.getByteTimeDomainData(data);

    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = "#0db89e";
    ctx.lineWidth = 2;
    ctx.beginPath();

    const step = width / data.length;
    data.forEach((value, index) => {
      const x = index * step;
      const y = (value / 128) * (height / 2);
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    ctx.stroke();
  }

  frame();
}

async function ensureVoiceSession() {
  if (activeSessionId) {
    return activeSessionId;
  }

  const res = await fetch("/api/voice/session", { method: "POST" });
  const data = await res.json();

  if (!data.ok || !data.session_id) {
    throw new Error(data.error || "Could not create voice session");
  }

  activeSessionId = data.session_id;
  return activeSessionId;
}

function stopMicResources() {
  cancelAnimationFrame(waveAnimFrame);
  micStream?.getTracks().forEach((track) => track.stop());
  audioCtx?.close();

  waveCanvas.hidden = true;
  const ctx = waveCanvas.getContext("2d");
  ctx.clearRect(0, 0, waveCanvas.width, waveCanvas.height);
}

async function startRecording() {
  if (isRecording) {
    return;
  }

  recordBtn.disabled = true;
  recordHint.textContent = "Preparing microphone...";

  try {
    await ensureVoiceSession();
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    alert(err?.message || "Microphone access denied. Please allow microphone permission in your browser.");
    recordBtn.disabled = false;
    recordHint.textContent = "Tap to speak";
    return;
  }

  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  analyserNode = audioCtx.createAnalyser();
  analyserNode.fftSize = 256;
  audioCtx.createMediaStreamSource(micStream).connect(analyserNode);

  const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
    ? "audio/webm;codecs=opus"
    : (MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "");

  mediaRecorder = new MediaRecorder(micStream, mimeType ? { mimeType } : {});
  audioChunks = [];

  mediaRecorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) {
      audioChunks.push(event.data);
    }
  };

  mediaRecorder.onstop = async () => {
    const blob = new Blob(audioChunks, { type: mimeType || "audio/webm" });
    audioChunks = [];
    await sendVoiceTurn(blob);
  };

  mediaRecorder.start(200);
  isRecording = true;

  recordBtn.classList.add("recording");
  recordBtn.setAttribute("aria-pressed", "true");
  recordIconMic.hidden = true;
  recordIconStop.hidden = false;
  recordHint.textContent = "Tap again to stop recording";
  waveCanvas.hidden = false;
  recordBtn.disabled = false;

  drawWaveform();
}

function stopRecording() {
  if (!mediaRecorder || mediaRecorder.state === "inactive") {
    return;
  }

  isRecording = false;
  mediaRecorder.stop();
  stopMicResources();

  recordBtn.classList.remove("recording");
  recordBtn.setAttribute("aria-pressed", "false");
  recordIconMic.hidden = false;
  recordIconStop.hidden = true;
  recordHint.textContent = "Processing...";
  recordBtn.disabled = true;
}

async function sendVoiceTurn(blob) {
  const requestTimeoutMs = getRequestTimeoutMs();

  const formData = new FormData();
  formData.append("session_id", activeSessionId);
  formData.append("language", "English");
  formData.append("audio", blob, "turn.webm");

  if (assistantImage.files[0]) {
    formData.append("image", assistantImage.files[0]);
  }

  try {
    const res = await fetchWithTimeout("/voice", { method: "POST", body: formData }, requestTimeoutMs);
    const data = await res.json();

    if (!data.ok) {
      throw new Error(data.error || "Voice query failed");
    }

    const payload = data.data || {};
    const transcript = payload.transcript || payload.query_text || "";
    const responseText = payload.response_text || "";

    if (transcript) {
      assistantTranscript.textContent = transcript;
      assistantTranscriptWrap.hidden = false;
    } else {
      assistantTranscriptWrap.hidden = true;
    }

    assistantResponseText.textContent = responseText;
    assistantLangChip.textContent = payload.detected_language || "Detected";
    assistantModeChip.textContent = (payload.mode || currentMode || "unknown").toUpperCase();
    assistantModeChip.dataset.mode = payload.mode || currentMode || "unknown";
    assistantResponseMeta.textContent = `Processing: ${payload.processing_time_ms || 0}ms`;

    if (payload.audio_response_data_url) {
      assistantResponseAudio.src = payload.audio_response_data_url;
      assistantResponseAudio.hidden = false;
      assistantResponseAudio.play().catch(() => {});
    } else {
      assistantResponseAudio.hidden = true;
      assistantResponseAudio.src = "";
    }

    assistantResponseArea.hidden = false;

    addHistoryEntry({
      timestamp: Date.now(),
      input: transcript,
      output: responseText,
      detectedLanguage: payload.detected_language || "Unknown",
      mode: payload.mode || currentMode,
      hasImage: Boolean(assistantImage.files[0]),
    });
  } catch (err) {
    if (err?.name === "AbortError") {
      assistantResponseText.textContent = `Error: Voice request timed out after ${Math.round(requestTimeoutMs / 1000)}s. Please retry.`;
    } else {
      assistantResponseText.textContent = `Error: ${err?.message || "Unknown error"}`;
    }
    assistantResponseAudio.hidden = true;
    assistantResponseMeta.textContent = "";
    assistantResponseArea.hidden = false;
  } finally {
    recordBtn.disabled = false;
    recordHint.textContent = "Tap to speak";
  }
}

assistantImage.addEventListener("change", () => {
  const file = assistantImage.files[0];
  if (!file) {
    return;
  }

  const objectUrl = URL.createObjectURL(file);
  assistantImagePreview.src = objectUrl;
  assistantImagePreviewWrap.hidden = false;
  captureLabel.classList.add("has-image");
});

assistantImageRemove.addEventListener("click", () => {
  assistantImage.value = "";
  assistantImagePreview.src = "";
  assistantImagePreviewWrap.hidden = true;
  captureLabel.classList.remove("has-image");
});

recordBtn.addEventListener("click", () => {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording().catch((err) => {
      alert(err?.message || "Failed to start recording");
      recordBtn.disabled = false;
      recordHint.textContent = "Tap to speak";
    });
  }
});

historyToggle.addEventListener("click", (event) => {
  event.preventDefault();
  toggleHistoryDrawer();
});

historyClose.addEventListener("click", (event) => {
  event.preventDefault();
  closeHistoryDrawer();
});

historyClear.addEventListener("click", () => {
  saveHistory([]);
  updateHistoryCount();
  renderHistoryList();
});

historyDrawer.addEventListener("click", (event) => {
  if (event.target === historyDrawer) {
    closeHistoryDrawer();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeHistoryDrawer();
  }
});

window.addEventListener("beforeunload", () => {
  if (activeRequestController) {
    activeRequestController.abort();
  }

  stopMicResources();

  if (activeSessionId) {
    fetch(`/api/voice/session/${encodeURIComponent(activeSessionId)}`, { method: "DELETE" }).catch(() => {});
  }
});

renderHistoryList();
updateHistoryCount();
historyToggle.setAttribute("aria-expanded", "false");
recordBtn.disabled = false;

fetchStatus();
setInterval(fetchStatus, 20000);
setInterval(renderCooldownPill, 1000);

window.AEGIS_ASSISTANT = {
  getHistory,
  openHistoryDrawer,
  closeHistoryDrawer,
  clearHistory: () => {
    saveHistory([]);
    updateHistoryCount();
    renderHistoryList();
  },
};
