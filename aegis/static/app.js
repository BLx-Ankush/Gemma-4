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
    navigator.serviceWorker.register("/sw.js").catch(() => {
      // Service workers are optional for local dev.
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
const installBtn = $("install-btn");
const networkPill = $("network-pill");
const networkText = $("network-text");
const cachePill = $("cache-pill");
const cacheText = $("cache-text");
const benchmarkPill = $("benchmark-pill");
const benchmarkText = $("benchmark-text");

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
const assistantMirrorPanel = $("assistant-mirror-panel");
const assistantMirrorScore = $("assistant-mirror-score");
const assistantMirrorScoreRing = $("assistant-mirror-score-ring");
const assistantMirrorVerdict = $("assistant-mirror-verdict");
const assistantMirrorFlags = $("assistant-mirror-flags");
const assistantMirrorReasoning = $("assistant-mirror-reasoning");
const assistantMirrorViewPanel = $("assistant-mirror-view-panel");
const assistantMirrorViewSummary = $("assistant-mirror-view-summary");
const assistantMirrorViewList = $("assistant-mirror-view-list");
const mirrorViewClear = $("mirror-view-clear");

const ONLINE_REQUEST_TIMEOUT_MS = 60000;
const OFFLINE_REQUEST_TIMEOUT_MS = 180000;
const REQUEST_TIMEOUT_BUFFER_MS = 5000;
const HISTORY_STORAGE_KEY = "aegis.assistant.history.v1";
const HISTORY_LIMIT = 80;
const MIRROR_VIEW_STORAGE_KEY = "aegis.assistant.mirror.view.v1";
const MIRROR_VIEW_LIMIT = 80;

let currentMode = "unknown";
let cloudCooldownUntilMs = 0;
let activeRequestController = null;
let serverVoiceTimeoutMs = 0;
let deferredInstallPrompt = null;

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

function formatUtcShort(isoValue) {
  if (!isoValue) {
    return "n/a";
  }
  const parsed = new Date(isoValue);
  if (Number.isNaN(parsed.getTime())) {
    return "n/a";
  }
  return parsed.toLocaleString();
}

function renderCachePill(cacheInfo) {
  if (!cacheText || !cachePill) {
    return;
  }
  if (!cacheInfo || !cacheInfo.cache_exists) {
    cacheText.textContent = "Drug cache: unavailable";
    return;
  }

  const count = Number(cacheInfo.cache_entries || 0);
  const modified = formatUtcShort(cacheInfo.cache_last_modified);
  cacheText.textContent = `Drug cache: ${count} entries · updated ${modified}`;
}

function renderBenchmarkPill(benchmark) {
  if (!benchmarkPill || !benchmarkText) {
    return;
  }
  if (!benchmark || !benchmark.available) {
    benchmarkPill.hidden = true;
    return;
  }

  const p95 = Number(benchmark.p95_ms || 0);
  const caught = Number(benchmark.safety_caught || 0);
  const dangerous = Number(benchmark.dangerous_scenarios || 0);
  benchmarkText.textContent = `Benchmark p95: ${p95}ms · Safety: ${caught}/${dangerous}`;
  benchmarkPill.hidden = false;
}

function renderNetworkPill() {
  if (!networkPill || !networkText) {
    return;
  }
  const online = navigator.onLine;
  networkPill.dataset.state = online ? "online" : "offline";
  networkText.textContent = online ? "Browser: online" : "Browser: offline";
}

function setupInstallPrompt() {
  if (!installBtn) {
    return;
  }

  window.addEventListener("beforeinstallprompt", (event) => {
    event.preventDefault();
    deferredInstallPrompt = event;
    installBtn.hidden = false;
  });

  window.addEventListener("appinstalled", () => {
    deferredInstallPrompt = null;
    installBtn.hidden = true;
  });

  installBtn.addEventListener("click", async () => {
    if (!deferredInstallPrompt) {
      return;
    }

    deferredInstallPrompt.prompt();
    try {
      await deferredInstallPrompt.userChoice;
    } catch {
      // Ignore install choice errors.
    }
    deferredInstallPrompt = null;
    installBtn.hidden = true;
  });
}

function _scoreRingColor(score) {
  if (score >= 75) {
    return "#0db89e";
  }
  if (score >= 45) {
    return "#e07b20";
  }
  return "#c6432d";
}

function renderMirrorPanel(mirrorData) {
  if (!assistantMirrorPanel) {
    return;
  }
  if (!mirrorData || typeof mirrorData !== "object") {
    assistantMirrorPanel.hidden = true;
    return;
  }

  const score = Math.max(0, Math.min(100, Number(mirrorData.confidence_score || 0)));
  const verdict = String(mirrorData.verdict || "warn").toLowerCase();
  const flags = Array.isArray(mirrorData.flags) ? mirrorData.flags : [];
  const reasoning = String(mirrorData.reasoning_trace || "No additional notes.");

  assistantMirrorScore.textContent = String(Math.round(score));
  assistantMirrorScoreRing.setAttribute("stroke-dasharray", `${score} ${100 - score}`);
  assistantMirrorScoreRing.setAttribute("stroke", _scoreRingColor(score));

  assistantMirrorVerdict.dataset.verdict = verdict;
  assistantMirrorVerdict.textContent = verdict.toUpperCase();

  if (flags.length) {
    assistantMirrorFlags.innerHTML = flags
      .map((flag) => `<span class="flag-tag">${escHtml(String(flag))}</span>`)
      .join("");
  } else {
    assistantMirrorFlags.innerHTML = "<span class=\"flag-ok\">No risk flags</span>";
  }

  assistantMirrorReasoning.textContent = reasoning;
  assistantMirrorPanel.hidden = false;
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
    const cacheInfo = data.drug_cache || null;
    const benchmarkInfo = data.benchmark || null;

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
    renderCachePill(cacheInfo);
    renderBenchmarkPill(benchmarkInfo);
    renderNetworkPill();
  } catch {
    currentMode = "unknown";
    cloudCooldownUntilMs = 0;
    serverVoiceTimeoutMs = 0;
    statusDot.dataset.mode = "error";
    statusText.textContent = "Server unreachable";
    modelText.textContent = "";
    renderCooldownPill();
    if (cacheText) {
      cacheText.textContent = "Drug cache: status unavailable";
    }
    if (benchmarkPill) {
      benchmarkPill.hidden = true;
    }
    renderNetworkPill();
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
    const mirrorTag = entry.mirrorVerdict ? `MIRROR: ${String(entry.mirrorVerdict).toUpperCase()}` : "";
    const confidenceTag = Number.isFinite(Number(entry.mirrorConfidence))
      ? `Confidence: ${Math.round(Number(entry.mirrorConfidence))}`
      : "";
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
          ${mirrorTag ? `<span class="history-tag">${escHtml(mirrorTag)}</span>` : ""}
          ${confidenceTag ? `<span class="history-tag">${escHtml(confidenceTag)}</span>` : ""}
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

function getMirrorViewEntries() {
  try {
    const raw = localStorage.getItem(MIRROR_VIEW_STORAGE_KEY);
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveMirrorViewEntries(entries) {
  localStorage.setItem(MIRROR_VIEW_STORAGE_KEY, JSON.stringify(entries.slice(0, MIRROR_VIEW_LIMIT)));
}

function renderMirrorViewPanel() {
  if (!assistantMirrorViewPanel || !assistantMirrorViewSummary || !assistantMirrorViewList) {
    return;
  }

  const entries = getMirrorViewEntries();
  if (!entries.length) {
    assistantMirrorViewPanel.hidden = true;
    assistantMirrorViewSummary.innerHTML = "";
    assistantMirrorViewList.innerHTML = "";
    return;
  }

  const passCount = entries.filter((entry) => String(entry.verdict || "").toLowerCase() === "pass").length;
  const warnCount = entries.filter((entry) => String(entry.verdict || "").toLowerCase() === "warn").length;
  const blockCount = entries.filter((entry) => String(entry.verdict || "").toLowerCase() === "block").length;

  assistantMirrorViewSummary.innerHTML = `
    <div class="summary-card neutral"><span class="summary-num">${entries.length}</span>Total</div>
    <div class="summary-card pass"><span class="summary-num">${passCount}</span>Pass</div>
    <div class="summary-card warn"><span class="summary-num">${warnCount}</span>Warn</div>
    <div class="summary-card block"><span class="summary-num">${blockCount}</span>Block</div>
  `;

  assistantMirrorViewList.innerHTML = entries.map((entry, index) => {
    const verdict = String(entry.verdict || "warn").toLowerCase();
    const score = Math.max(0, Math.min(100, Number(entry.confidenceScore || 0)));
    const flags = Array.isArray(entry.flags) ? entry.flags : [];
    const ts = entry.timestamp ? new Date(entry.timestamp).toLocaleString() : "";
    const flagsHtml = flags.length
      ? flags.map((flag) => `<span class="flag-tag">${escHtml(String(flag))}</span>`).join("")
      : "<span class=\"flag-ok\">No risk flags</span>";

    return `
      <article class="log-entry" data-verdict="${escHtml(verdict)}">
        <div class="log-entry-header">
          <span class="verdict-badge" data-verdict="${escHtml(verdict)}">${escHtml(verdict.toUpperCase())}</span>
          <span class="log-score">Confidence ${Math.round(score)}</span>
          <span class="log-backend">${escHtml(String((entry.mode || "unknown").toUpperCase()))}</span>
          <span class="log-time">${escHtml(ts)}</span>
        </div>
        <p class="log-body-line"><strong>Request:</strong> ${escHtml(entry.input || "")}</p>
        <p class="log-body-line"><strong>Draft:</strong> ${escHtml(entry.draftResponse || "")}</p>
        <p class="log-action">${escHtml(entry.rectificationAction || "No rectification details available.")}</p>
        <div class="log-flags">${flagsHtml}</div>
        <p class="log-reason">${escHtml(entry.reasoning || "No additional notes.")}</p>
        <p class="log-body-line"><strong>Final answer:</strong> ${escHtml(entry.finalResponse || "")}</p>
      </article>
    `;
  }).join("");

  assistantMirrorViewPanel.hidden = false;
}

function addMirrorViewEntry(entry) {
  const entries = getMirrorViewEntries();
  entries.unshift(entry);
  saveMirrorViewEntries(entries);
  renderMirrorViewPanel();
}

function buildMirrorViewEntry(payload, transcript, hasImage) {
  const mirror = payload.mirror || {};
  const mirrorView = payload.mirror_view || {};

  const draftResponse = String(mirrorView.draft_response || payload.raw_response || payload.response_text || "");
  const finalResponse = String(mirrorView.final_response || payload.response_text || "");
  const rectified = Boolean(mirrorView.rectified || draftResponse.trim() !== finalResponse.trim());
  const defaultAction = rectified
    ? "MIRROR rectified the draft response before final delivery."
    : "Draft response passed MIRROR checks; no rectification was needed.";

  return {
    timestamp: Date.now(),
    input: transcript || payload.query_text || "",
    draftResponse,
    finalResponse,
    rectified,
    rectificationAction: String(mirrorView.rectification_action || defaultAction),
    verdict: String(mirrorView.verdict || mirror.verdict || "warn").toLowerCase(),
    confidenceScore: Number(mirrorView.confidence_score ?? mirror.confidence_score ?? 0),
    flags: Array.isArray(mirrorView.flags)
      ? mirrorView.flags
      : (Array.isArray(mirror.flags) ? mirror.flags : []),
    reasoning: String(mirrorView.reasoning_trace || mirror.reasoning_trace || mirror.notes || ""),
    mode: payload.mode || currentMode || "unknown",
    hasImage: Boolean(hasImage),
  };
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
    const mirror = payload.mirror || null;
    const drugLookup = payload.drug_lookup || {};

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

    const totalMs = Math.max(0, Number(payload.processing_time_ms || 0));
    const modelMs = Math.max(0, Number(payload.model_latency_ms || 0));
    const auditMs = Math.max(0, Number(mirror?.audit_time_ms || 0));
    const sourceText = Array.isArray(drugLookup.sources) && drugLookup.sources.length
      ? drugLookup.sources.join(", ")
      : "none";
    assistantResponseMeta.textContent = `Total ${Math.round(totalMs)}ms · Model ${Math.round(modelMs)}ms · MIRROR ${Math.round(auditMs)}ms · Drug source: ${sourceText}`;
    renderMirrorPanel(mirror);

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
      mirrorVerdict: mirror?.verdict || "",
      mirrorConfidence: mirror?.confidence_score,
    });
    addMirrorViewEntry(buildMirrorViewEntry(payload, transcript, Boolean(assistantImage.files[0])));
  } catch (err) {
    if (err?.name === "AbortError") {
      assistantResponseText.textContent = `Error: Voice request timed out after ${Math.round(requestTimeoutMs / 1000)}s. Please retry.`;
    } else {
      assistantResponseText.textContent = `Error: ${err?.message || "Unknown error"}`;
    }
    assistantResponseAudio.hidden = true;
    assistantResponseMeta.textContent = "";
    renderMirrorPanel(null);
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

if (mirrorViewClear) {
  mirrorViewClear.addEventListener("click", () => {
    saveMirrorViewEntries([]);
    renderMirrorViewPanel();
  });
}

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
renderMirrorViewPanel();
historyToggle.setAttribute("aria-expanded", "false");
recordBtn.disabled = false;
setupInstallPrompt();
renderNetworkPill();
renderMirrorPanel(null);

window.addEventListener("online", renderNetworkPill);
window.addEventListener("offline", renderNetworkPill);

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
