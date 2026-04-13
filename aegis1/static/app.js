/**
 * AEGIS Frontend — Vanilla JS
 * No framework, no build step. Runs directly from Termux Flask server.
 *
 * Covers:
 *  - Tab navigation
 *  - VEDA: text + image medical query
 *  - VOICE: MediaRecorder session with waveform visualiser
 *  - MIRROR: audit log viewer with verdict stats
 *  - Online/offline status badge (polls /status every 20s)
 */

"use strict";

// ─────────────────────────────────────────────────────────────────────────────
// DOM refs
// ─────────────────────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

const tabs        = document.querySelectorAll(".tab");
const tabPanels   = document.querySelectorAll(".tab-panel");

// Status bar
const statusDot   = $("status-dot");
const statusText  = $("status-text");
const modelText   = $("model-text");

// VEDA
const vedaForm      = $("veda-form");
const vedaText      = $("veda-text");
const vedaLanguage  = $("veda-language");
const vedaImage     = $("veda-image");
const vedaSubmit    = $("veda-submit");
const imgPreviewWrap = $("img-preview-wrap");
const imgPreview    = $("img-preview");
const imgRemove     = $("img-remove");
const captureLabel  = $("capture-label");
const vedaResponseArea = $("veda-response-area");
const vedaResponseText = $("veda-response-text");
const vedaAudio        = $("veda-audio");
const vedaLangChip     = $("veda-lang-chip");
const vedaModeChip     = $("veda-mode-chip");
const vedaMirrorPanel  = $("veda-mirror-panel");
const vedaScoreArc     = $("veda-score-arc");
const vedaScoreNum     = $("veda-score-num");
const vedaVerdict      = $("veda-verdict");
const vedaFlags        = $("veda-flags");
const vedaReasoning    = $("veda-reasoning");
const vedaMeta         = $("veda-meta");

// VOICE
const btnStartSession = $("btn-start-session");
const btnEndSession   = $("btn-end-session");
const sessionChip     = $("session-chip");
const recordBtn       = $("record-btn");
const recordRing      = $("record-ring");
const recordIconMic   = document.querySelector(".record-icon-mic");
const recordIconStop  = document.querySelector(".record-icon-stop");
const recordHint      = $("record-hint");
const waveCanvas      = $("waveform-canvas");
const voiceLanguage   = $("voice-language");
const voiceResponseArea = $("voice-response-area");
const voiceTranscriptBubble = $("voice-transcript-bubble");
const voiceTranscriptText   = $("voice-transcript-text");
const voiceResponseText = $("voice-response-text");
const voiceAudio        = $("voice-audio");
const voiceLangChip     = $("voice-lang-chip");
const voiceModeChip     = $("voice-mode-chip");
const voiceMirrorPanel  = $("voice-mirror-panel");
const voiceScoreArc     = $("voice-score-arc");
const voiceScoreNum     = $("voice-score-num");
const voiceVerdict      = $("voice-verdict");
const voiceFlags        = $("voice-flags");
const voiceReasoning    = $("voice-reasoning");
const voiceMeta         = $("voice-meta");
const historySection    = $("history-section");
const historyList       = $("history-list");

// MIRROR log
const mirrorSummary    = $("mirror-summary");
const mirrorLogList    = $("mirror-log-list");
const btnRefreshLog    = $("btn-refresh-log");

// Spinner
const spinner    = $("spinner");
const spinnerMsg = $("spinner-msg");


// ─────────────────────────────────────────────────────────────────────────────
// Tab navigation
// ─────────────────────────────────────────────────────────────────────────────
tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const target = tab.dataset.tab;
    tabs.forEach((t) => t.classList.remove("active"));
    tabPanels.forEach((p) => {
      p.classList.remove("active");
      p.hidden = true;
    });
    tab.classList.add("active");
    const panel = $("panel-" + target);
    panel.classList.add("active");
    panel.hidden = false;

    // Refresh MIRROR log when switching to that tab
    if (target === "mirror") loadMirrorLog();
  });
});


// ─────────────────────────────────────────────────────────────────────────────
// Spinner helpers
// ─────────────────────────────────────────────────────────────────────────────
function showSpinner(msg = "Processing…") {
  spinnerMsg.textContent = msg;
  spinner.hidden = false;
}
function hideSpinner() {
  spinner.hidden = true;
}


// ─────────────────────────────────────────────────────────────────────────────
// Status polling — every 20 seconds
// ─────────────────────────────────────────────────────────────────────────────
async function fetchStatus() {
  try {
    const res  = await fetch("/status");
    const data = await res.json();
    if (!data.ok) throw new Error();

    const mode    = data.compute?.mode || "unknown";
    const online  = mode === "online";
    const backend = data.model?.active_backend || (data.model?.loaded ? "ready" : "cold");
    const vision  = data.model?.has_vision ? " · vision ✓" : "";

    statusDot.dataset.mode = mode;   // CSS uses this for colour
    statusText.textContent = online ? "Online — Cloud" : "Offline — Local";
    modelText.textContent  = `${backend}${vision}`;

    $("mode-pill").dataset.mode = mode;
  } catch {
    statusDot.dataset.mode = "error";
    statusText.textContent = "Server unreachable";
    modelText.textContent  = "";
  }
}

fetchStatus();
setInterval(fetchStatus, 20000);


// ─────────────────────────────────────────────────────────────────────────────
// MIRROR panel renderer (shared by VEDA + VOICE)
// ─────────────────────────────────────────────────────────────────────────────
function renderMirror(mirrorData, { panel, arc, num, verdict, flags, reasoning }) {
  if (!mirrorData) return;

  const score   = mirrorData.confidence_score ?? 0;
  const verd    = (mirrorData.verdict || "warn").toLowerCase();
  const flagArr = mirrorData.flags || [];
  const reason  = mirrorData.reasoning_trace || "";

  // Score ring: circumference = 2π×18 ≈ 113
  const offset = 113 - (score / 100) * 113;
  arc.style.strokeDashoffset = offset;
  arc.style.stroke = verd === "pass" ? "#0db89e" : verd === "warn" ? "#f59d34" : "#c6432d";
  num.textContent = score;

  // Verdict badge
  const verdLabel = { pass: "✓ PASS", warn: "⚠ WARN", block: "✗ BLOCK" }[verd] || verd.toUpperCase();
  verdict.textContent = verdLabel;
  verdict.dataset.verdict = verd;

  // Flags
  flags.innerHTML = flagArr.length
    ? flagArr.map((f) => `<span class="flag-tag">${f.replace(/_/g, " ")}</span>`).join("")
    : '<span class="flag-ok">No flags</span>';

  // Reasoning
  reasoning.textContent = reason;

  panel.hidden = false;
}


// ─────────────────────────────────────────────────────────────────────────────
// VEDA — text + image query
// ─────────────────────────────────────────────────────────────────────────────

// Image preview
vedaImage.addEventListener("change", () => {
  const file = vedaImage.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  imgPreview.src = url;
  imgPreviewWrap.hidden = false;
  captureLabel.classList.add("has-image");
});

imgRemove.addEventListener("click", () => {
  vedaImage.value = "";
  imgPreview.src = "";
  imgPreviewWrap.hidden = true;
  captureLabel.classList.remove("has-image");
});

vedaForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  const text = vedaText.value.trim();
  if (!text && !vedaImage.files[0]) {
    vedaText.focus();
    return;
  }

  showSpinner("Sending to AEGIS…");
  vedaSubmit.disabled = true;

  try {
    const fd = new FormData();
    fd.append("text", text);
    fd.append("language", vedaLanguage.value);
    if (vedaImage.files[0]) fd.append("image", vedaImage.files[0]);

    const res  = await fetch("/veda", { method: "POST", body: fd });
    const data = await res.json();

    if (!data.ok) throw new Error(data.error || "Request failed");

    const d = data.data;

    // Response text
    vedaResponseText.textContent = d.response_text || "";

    // Chips
    vedaLangChip.textContent = d.detected_language || "";
    vedaModeChip.textContent = (d.mode || "").toUpperCase();
    vedaModeChip.dataset.mode = d.mode || "";

    // Audio
    if (d.audio_response_data_url) {
      vedaAudio.src = d.audio_response_data_url;
      vedaAudio.hidden = false;
      vedaAudio.play().catch(() => {});
    } else {
      vedaAudio.hidden = true;
    }

    // MIRROR
    renderMirror(d.mirror, {
      panel: vedaMirrorPanel, arc: vedaScoreArc, num: vedaScoreNum,
      verdict: vedaVerdict, flags: vedaFlags, reasoning: vedaReasoning,
    });

    // Meta bar
    const ms = d.processing_time_ms || 0;
    const auditMs = d.mirror?.audit_time_ms || 0;
    vedaMeta.textContent = `Inference: ${ms}ms · Audit: ${auditMs}ms · Drug context: ${d.drug_context ? "yes" : "none"}`;

    vedaResponseArea.hidden = false;
    vedaResponseArea.scrollIntoView({ behavior: "smooth", block: "nearest" });

  } catch (err) {
    vedaResponseText.textContent = "Error: " + err.message;
    vedaMirrorPanel.hidden = true;
    vedaResponseArea.hidden = false;
  } finally {
    hideSpinner();
    vedaSubmit.disabled = false;
  }
});


// ─────────────────────────────────────────────────────────────────────────────
// VOICE — session + MediaRecorder + waveform
// ─────────────────────────────────────────────────────────────────────────────
let activeSessionId  = null;
let mediaRecorder    = null;
let audioChunks      = [];
let isRecording      = false;
let audioCtx         = null;
let analyserNode     = null;
let waveAnimFrame    = null;
let micStream        = null;

// Waveform drawing
function drawWaveform() {
  if (!analyserNode) return;
  const ctx = waveCanvas.getContext("2d");
  const W = waveCanvas.width;
  const H = waveCanvas.height;
  const buf = new Uint8Array(analyserNode.frequencyBinCount);

  function frame() {
    if (!isRecording) return;
    waveAnimFrame = requestAnimationFrame(frame);
    analyserNode.getByteTimeDomainData(buf);
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = "#0db89e";
    ctx.lineWidth = 2;
    ctx.beginPath();
    const step = W / buf.length;
    buf.forEach((v, i) => {
      const x = i * step;
      const y = (v / 128) * (H / 2);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  frame();
}

async function startRecording() {
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    alert("Microphone access denied. Please allow microphone permission in your browser.");
    return;
  }

  // Web Audio waveform
  audioCtx    = new (window.AudioContext || window.webkitAudioContext)();
  analyserNode = audioCtx.createAnalyser();
  analyserNode.fftSize = 256;
  audioCtx.createMediaStreamSource(micStream).connect(analyserNode);

  // MediaRecorder — prefer audio/webm, fallback to any supported
  const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
    ? "audio/webm;codecs=opus"
    : MediaRecorder.isTypeSupported("audio/webm")
    ? "audio/webm"
    : "";

  mediaRecorder = new MediaRecorder(micStream, mimeType ? { mimeType } : {});
  audioChunks = [];

  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) audioChunks.push(e.data);
  };

  mediaRecorder.onstop = async () => {
    const blob = new Blob(audioChunks, { type: mimeType || "audio/webm" });
    audioChunks = [];
    await sendVoiceTurn(blob);
  };

  mediaRecorder.start(200);   // collect data every 200ms
  isRecording = true;

  // UI state
  recordBtn.classList.add("recording");
  recordBtn.setAttribute("aria-pressed", "true");
  recordIconMic.hidden = true;
  recordIconStop.hidden = false;
  recordHint.textContent = "Tap again to stop recording";
  waveCanvas.hidden = false;
  drawWaveform();
}

function stopRecording() {
  if (!mediaRecorder || mediaRecorder.state === "inactive") return;
  isRecording = false;
  cancelAnimationFrame(waveAnimFrame);

  mediaRecorder.stop();
  micStream?.getTracks().forEach((t) => t.stop());
  audioCtx?.close();

  // UI state
  recordBtn.classList.remove("recording");
  recordBtn.setAttribute("aria-pressed", "false");
  recordIconMic.hidden = false;
  recordIconStop.hidden = true;
  recordHint.textContent = "Processing…";
  waveCanvas.hidden = true;
  const wCtx = waveCanvas.getContext("2d");
  wCtx.clearRect(0, 0, waveCanvas.width, waveCanvas.height);
}

recordBtn.addEventListener("click", () => {
  if (!activeSessionId) return;
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
});

// Session management
btnStartSession.addEventListener("click", async () => {
  const res  = await fetch("/api/voice/session", { method: "POST" });
  const data = await res.json();
  if (!data.ok) { alert("Failed to start session"); return; }

  activeSessionId = data.session_id;
  sessionChip.textContent = "Session: " + activeSessionId.slice(0, 8) + "…";
  btnEndSession.disabled = false;
  btnStartSession.disabled = true;
  recordBtn.disabled = false;
  recordHint.textContent = "Tap to speak";
  historySection.hidden = false;
});

btnEndSession.addEventListener("click", async () => {
  if (!activeSessionId) return;
  await fetch(`/api/voice/session/${encodeURIComponent(activeSessionId)}`, { method: "DELETE" });
  activeSessionId = null;
  sessionChip.textContent = "No active session";
  btnEndSession.disabled = true;
  btnStartSession.disabled = false;
  recordBtn.disabled = true;
  recordHint.textContent = "Start a session first";
  historyList.innerHTML = "";
  historySection.hidden = true;
  voiceResponseArea.hidden = true;
});

// Send recorded audio to backend
async function sendVoiceTurn(blob) {
  showSpinner("Transcribing and processing…");

  const fd = new FormData();
  fd.append("session_id", activeSessionId);
  fd.append("language", voiceLanguage.value);
  fd.append("audio", blob, "turn.webm");

  try {
    const res  = await fetch("/voice", { method: "POST", body: fd });
    const data = await res.json();

    if (!data.ok) throw new Error(data.error || "Voice query failed");

    const d = data.data;

    // Transcript bubble
    if (d.transcript || d.query_text) {
      voiceTranscriptText.textContent = d.transcript || d.query_text;
      voiceTranscriptBubble.hidden = false;
    }

    // Response
    voiceResponseText.textContent = d.response_text || "";

    // Chips
    voiceLangChip.textContent = d.detected_language || "";
    voiceModeChip.textContent = (d.mode || "").toUpperCase();
    voiceModeChip.dataset.mode = d.mode || "";

    // Audio — autoplay (key demo moment)
    if (d.audio_response_data_url) {
      voiceAudio.src = d.audio_response_data_url;
      voiceAudio.hidden = false;
      voiceAudio.play().catch(() => {});
    } else {
      voiceAudio.hidden = true;
    }

    // MIRROR
    renderMirror(d.mirror, {
      panel: voiceMirrorPanel, arc: voiceScoreArc, num: voiceScoreNum,
      verdict: voiceVerdict, flags: voiceFlags, reasoning: voiceReasoning,
    });

    voiceMeta.textContent = `Turn ${d.turn_count || 1} · ${d.processing_time_ms || 0}ms`;

    voiceResponseArea.hidden = false;
    voiceResponseArea.scrollIntoView({ behavior: "smooth", block: "nearest" });

    // Update history
    await refreshHistory();

  } catch (err) {
    voiceResponseText.textContent = "Error: " + err.message;
    voiceResponseArea.hidden = false;
  } finally {
    hideSpinner();
    recordHint.textContent = "Tap to speak";
  }
}

// Fetch and render turn history
async function refreshHistory() {
  if (!activeSessionId) return;
  const res  = await fetch(`/api/voice/history/${encodeURIComponent(activeSessionId)}`);
  const data = await res.json();
  const turns = data.history || [];

  if (!turns.length) {
    historyList.innerHTML = "<p class='empty-note'>No turns yet.</p>";
    return;
  }

  historyList.innerHTML = turns.map((t, i) => `
    <div class="history-turn">
      <div class="history-turn-header">Turn ${i + 1} &middot; <span class="hist-lang">${t.language || ""}</span></div>
      <div class="history-user"><strong>You:</strong> ${escHtml(t.user_text || "(no transcript)")}</div>
      <div class="history-aegis"><strong>AEGIS:</strong> ${escHtml(t.assistant_text || "")}</div>
    </div>
  `).join("");
}


// ─────────────────────────────────────────────────────────────────────────────
// MIRROR audit log
// ─────────────────────────────────────────────────────────────────────────────
async function loadMirrorLog() {
  const res  = await fetch("/mirror/log?limit=50");
  const data = await res.json();
  const entries = data.data || [];

  if (!entries.length) {
    mirrorLogList.innerHTML = "<p class='empty-note'>No audit entries yet.</p>";
    mirrorSummary.innerHTML = "";
    return;
  }

  // Summary stats
  const total   = entries.length;
  const passes  = entries.filter((e) => e.report?.verdict === "pass").length;
  const warns   = entries.filter((e) => e.report?.verdict === "warn").length;
  const blocks  = entries.filter((e) => e.report?.verdict === "block").length;
  const avgConf = Math.round(
    entries.reduce((s, e) => s + (e.report?.confidence_score || 0), 0) / total
  );

  mirrorSummary.innerHTML = `
    <div class="summary-grid">
      <div class="summary-card pass"><span class="summary-num">${passes}</span><span>PASS</span></div>
      <div class="summary-card warn"><span class="summary-num">${warns}</span><span>WARN</span></div>
      <div class="summary-card block"><span class="summary-num">${blocks}</span><span>BLOCK</span></div>
      <div class="summary-card neutral"><span class="summary-num">${avgConf}</span><span>Avg Confidence</span></div>
    </div>
  `;

  // Log entries (newest first — already reversed by backend)
  mirrorLogList.innerHTML = entries.map((e) => {
    const r    = e.report || {};
    const verd = (r.verdict || "warn").toLowerCase();
    const ts   = e.timestamp ? new Date(e.timestamp * 1000).toLocaleTimeString() : "";
    const q    = escHtml((e.query || "").slice(0, 120));
    const resp = escHtml((e.response || "").slice(0, 160));
    const flags = (r.flags || []).map((f) => `<span class="flag-tag">${f.replace(/_/g, " ")}</span>`).join("");
    return `
      <div class="log-entry" data-verdict="${verd}">
        <div class="log-entry-header">
          <span class="verdict-badge" data-verdict="${verd}">${verd.toUpperCase()}</span>
          <span class="log-score">Confidence: ${r.confidence_score ?? "—"}</span>
          <span class="log-time">${ts}</span>
          <span class="log-backend">${escHtml(e.backend || "")}</span>
        </div>
        <div class="log-query"><strong>Query:</strong> ${q}</div>
        <div class="log-resp"><strong>Response:</strong> ${resp}</div>
        <div class="log-flags">${flags}</div>
        ${r.reasoning_trace ? `<div class="log-reason">${escHtml(r.reasoning_trace)}</div>` : ""}
      </div>
    `;
  }).join("");
}

btnRefreshLog.addEventListener("click", loadMirrorLog);


// ─────────────────────────────────────────────────────────────────────────────
// Utility
// ─────────────────────────────────────────────────────────────────────────────
function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
