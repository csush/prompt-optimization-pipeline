// Trace review interface (no dependencies). Loads a rollouts.jsonl written by
// the pipeline, renders one rollout at a time, and collects Pass/Fail/Defer
// labels + notes. Labels auto-save to localStorage and export as labels.jsonl.

const els = {
  fileInput: document.getElementById("file-input"),
  runBadge: document.getElementById("run-badge"),
  counter: document.getElementById("counter"),
  counts: document.getElementById("counts"),
  jumpId: document.getElementById("jump-id"),
  jumpBtn: document.getElementById("jump-btn"),
  export: document.getElementById("export"),
  empty: document.getElementById("empty"),
  card: document.getElementById("trace-card"),
  controls: document.getElementById("controls"),
  rolloutBadge: document.getElementById("rollout-badge"),
  phaseBadge: document.getElementById("phase-badge"),
  verdictBadge: document.getElementById("verdict-badge"),
  promptDetails: document.getElementById("prompt-details"),
  promptText: document.getElementById("prompt-text"),
  question: document.getElementById("question"),
  pred: document.getElementById("pred"),
  gold: document.getElementById("gold"),
  studentBlock: document.getElementById("student-block"),
  studentOut: document.getElementById("student-out"),
  feedbackBlock: document.getElementById("feedback-block"),
  feedback: document.getElementById("feedback"),
  prev: document.getElementById("prev"),
  next: document.getElementById("next"),
  pass: document.getElementById("pass"),
  fail: document.getElementById("fail"),
  defer: document.getElementById("defer"),
  notes: document.getElementById("notes"),
};

const state = {
  traces: [],
  idx: 0,
  runId: null,
  labels: {},      // key -> { verdict, notes, ts }
  undo: [],        // [{ key, before }] where before is previous label or null
};

// ---------- storage ----------

function storageKey() {
  return state.runId ? `trace-labels::${state.runId}` : null;
}

function loadLabels() {
  const k = storageKey();
  if (!k) { state.labels = {}; return; }
  try {
    state.labels = JSON.parse(localStorage.getItem(k) || "{}");
  } catch {
    state.labels = {};
  }
}

function saveLabels() {
  const k = storageKey();
  if (!k) return;
  localStorage.setItem(k, JSON.stringify(state.labels));
}

function traceKey(t) {
  // Defensive: use rollout_id when present, else index, to form a stable key.
  const rid = t.rollout_id ?? t.idx ?? state.idx;
  return String(t.phase ?? "trace") + "#" + rid + ":" + (t.question || "").slice(0, 40);
}

// ---------- parsing ----------

function parseJsonl(text) {
  const out = [];
  for (const line of text.split(/\r?\n/)) {
    const s = line.trim();
    if (!s) continue;
    try { out.push(JSON.parse(s)); } catch { /* skip malformed line */ }
  }
  return out;
}

async function loadFile(file) {
  const text = await file.text();
  let records;
  if (file.name.endsWith(".jsonl") || file.name.endsWith(".txt")) {
    records = parseJsonl(text);
  } else {
    // Single meta.json: only run context, no traces. Show run banner if present.
    try {
      const obj = JSON.parse(text);
      state.runId = obj.run_id || null;
      state.traces = [];
      loadLabels();
      renderRunBadge();
      showEmpty("Loaded meta.json — open the matching rollouts.jsonl to review traces.");
      return;
    } catch {
      records = parseJsonl(text);
    }
  }
  if (records.length === 0) {
    showEmpty("No trace records found in that file.");
    return;
  }
  state.traces = records;
  state.runId = state.traces[0].run_id || null;
  state.idx = 0;
  state.undo = [];
  loadLabels();
  els.empty.hidden = true;
  els.card.hidden = false;
  els.controls.hidden = false;
  els.export.disabled = false;
  renderRunBadge();
  render();
}

function renderRunBadge() {
  if (state.runId) {
    els.runBadge.textContent = state.runId;
    els.runBadge.hidden = false;
  } else {
    els.runBadge.hidden = true;
  }
}

function showEmpty(msg) {
  els.empty.hidden = false;
  els.card.hidden = true;
  els.controls.hidden = true;
  els.export.disabled = true;
  if (msg) {
    const p = els.empty.querySelector("p");
    if (p) p.textContent = msg;
  }
  els.counter.textContent = "no traces loaded";
  els.counts.textContent = "";
}

// ---------- rendering ----------

function escapeHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// Highlight the GSM8K '#### <number>' answer line so reviewers' eyes land on it.
function renderStudentOut(text) {
  const safe = escapeHtml(text);
  return safe.replace(/(####\s*[\-?]?\d[\d,]*\.?\d*)/g,
    '<span class="answer-line">$1</span>');
}

function renderVerdictBadge(t) {
  const correct = t.correct === true;
  els.verdictBadge.className = "badge verdict " + (correct ? "correct" : "incorrect");
  els.verdictBadge.textContent = correct ? "✓ correct" : "✗ incorrect";
}

function renderPredGold(t) {
  const pred = t.pred ?? "—";
  els.pred.textContent = pred;
  els.gold.textContent = t.gold ?? "—";
  const match = t.correct === true || (t.pred != null && t.pred === t.gold);
  els.pred.className = "value mono " + (t.pred == null ? "" : (match ? "match" : "mismatch"));
}

function render() {
  if (state.idx < 0 || state.idx >= state.traces.length) return;
  const t = state.traces[state.idx];

  const rid = t.rollout_id ?? "—";
  const phase = t.phase ?? "trace";
  els.rolloutBadge.textContent = `rollout ${rid}`;
  els.phaseBadge.textContent = phase;
  renderVerdictBadge(t);
  renderPredGold(t);

  els.promptText.textContent = t.prompt ?? "(no prompt recorded)";
  els.promptDetails.open = false;
  els.question.textContent = t.question ?? "";

  if (t.student_out) {
    els.studentBlock.hidden = false;
    els.studentOut.innerHTML = renderStudentOut(t.student_out);
  } else {
    els.studentBlock.hidden = true;
  }

  if (t.feedback && t.feedback.trim()) {
    els.feedbackBlock.hidden = false;
    els.feedback.textContent = t.feedback;
  } else {
    els.feedbackBlock.hidden = true;
  }

  // Restore label state for this trace.
  const key = traceKey(t);
  const label = state.labels[key];
  els.notes.value = label?.notes ?? "";
  highlightVerdict(label?.verdict);

  updateCounter();
}

function highlightVerdict(verdict) {
  els.pass.classList.toggle("active", verdict === "pass");
  els.fail.classList.toggle("active", verdict === "fail");
  els.defer.classList.toggle("active", verdict === "defer");
}

function updateCounter() {
  const total = state.traces.length;
  if (total === 0) return;
  els.counter.textContent = `${state.idx + 1} of ${total}`;
  let p = 0, f = 0, d = 0;
  for (const t of state.traces) {
    const v = state.labels[traceKey(t)]?.verdict;
    if (v === "pass") p++;
    else if (v === "fail") f++;
    else if (v === "defer") d++;
  }
  const remaining = total - p - f - d;
  els.counts.innerHTML =
    `<span class="pass-c">Pass ${p}</span> · ` +
    `<span class="fail-c">Fail ${f}</span> · ` +
    `<span class="defer-c">Defer ${d}</span> · ` +
    `<span>${total - p - f - d} unlabeled</span>`;
  els.prev.disabled = state.idx <= 0;
  els.next.disabled = state.idx >= total - 1;
}

// ---------- actions ----------

function setVerdict(verdict, { advance = true } = {}) {
  if (state.traces.length === 0) return;
  const t = state.traces[state.idx];
  const key = traceKey(t);
  const before = state.labels[key] ?? null;
  state.labels[key] = {
    verdict,
    notes: els.notes.value,
    ts: new Date().toISOString(),
  };
  state.undo.push({ key, before });
  saveLabels();
  highlightVerdict(verdict);
  flashCard();
  updateCounter();
  if (advance) next();
}

function saveNotes() {
  if (state.traces.length === 0) return;
  const t = state.traces[state.idx];
  const key = traceKey(t);
  const existing = state.labels[key];
  if (!existing && !els.notes.value) return;
  state.labels[key] = {
    verdict: existing?.verdict ?? null,
    notes: els.notes.value,
    ts: new Date().toISOString(),
  };
  saveLabels();
  updateCounter();
}

function undo() {
  const last = state.undo.pop();
  if (!last) return;
  if (last.before == null) delete state.labels[last.key];
  else state.labels[last.key] = last.before;
  saveLabels();
  render();
}

function next() {
  if (state.idx >= state.traces.length - 1) return;
  state.idx++;
  render();
}

function prev() {
  if (state.idx <= 0) return;
  state.idx--;
  render();
}

function jumpToId() {
  const q = els.jumpId.value.trim();
  if (!q) return;
  const n = Number(q);
  for (let i = 0; i < state.traces.length; i++) {
    const rid = state.traces[i].rollout_id;
    if (rid != null && (String(rid) === q || rid === n)) {
      state.idx = i;
      els.jumpId.value = "";
      render();
      return;
    }
  }
  // Fallback: treat as 1-based trace position.
  const pos = parseInt(q, 10);
  if (!Number.isNaN(pos) && pos >= 1 && pos <= state.traces.length) {
    state.idx = pos - 1;
    els.jumpId.value = "";
    render();
  }
}

function flashCard() {
  els.card.classList.remove("flash");
  void els.card.offsetWidth;
  els.card.classList.add("flash");
}

function exportLabels() {
  const lines = [];
  for (const t of state.traces) {
    const key = traceKey(t);
    const label = state.labels[key];
    if (!label) continue;
    lines.push(JSON.stringify({
      run_id: state.runId,
      rollout_id: t.rollout_id ?? null,
      phase: t.phase ?? null,
      question: t.question ?? "",
      gold: t.gold ?? null,
      pred: t.pred ?? null,
      correct: t.correct ?? null,
      verdict: label.verdict,
      notes: label.notes ?? "",
      ts: label.ts,
    }));
  }
  const blob = new Blob([lines.join("\n") + (lines.length ? "\n" : "")], { type: "application/x-jsonlines" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = (state.runId ? state.runId + "-" : "") + "labels.jsonl";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ---------- events ----------

els.fileInput.addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (f) loadFile(f);
});

els.pass.addEventListener("click", () => setVerdict("pass"));
els.fail.addEventListener("click", () => setVerdict("fail"));
els.defer.addEventListener("click", () => setVerdict("defer", { advance: false }));
els.prev.addEventListener("click", prev);
els.next.addEventListener("click", next);
els.jumpBtn.addEventListener("click", jumpToId);
els.export.addEventListener("click", exportLabels);
els.notes.addEventListener("input", saveNotes);
els.jumpId.addEventListener("keydown", (e) => { if (e.key === "Enter") jumpToId(); });

document.addEventListener("keydown", (e) => {
  // Don't hijack typing in inputs/textareas.
  const inField = e.target.matches("input, textarea");
  if (inField) {
    // Allow Cmd+S / Cmd+Enter even while typing notes.
    if ((e.metaKey || e.ctrlKey) && (e.key === "s" || e.key === "Enter")) {
      e.preventDefault();
      if (e.key === "s") exportLabels();
      else { saveNotes(); next(); }
    }
    return;
  }
  if (state.traces.length === 0) return;
  if ((e.metaKey || e.ctrlKey) && e.key === "s") { e.preventDefault(); exportLabels(); return; }
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); saveNotes(); next(); return; }
  switch (e.key) {
    case "ArrowLeft": prev(); break;
    case "ArrowRight": next(); break;
    case "1": setVerdict("pass"); break;
    case "2": setVerdict("fail"); break;
    case "d": case "D": setVerdict("defer", { advance: false }); break;
    case "u": case "U": undo(); break;
  }
});