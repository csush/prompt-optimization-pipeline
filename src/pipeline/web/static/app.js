/* Explorer UI — variant A, folded into the real /runs route.
   Rail = list of runs from /api/runs; detail = config + iterations chart +
   live log stream + seed/best prompt diff; SSE at /api/runs/{id}/events. */

const state = { runs: [], selectedRunId: null, es: null, liveIters: [] };

const API = "/api/runs";
const SEED_LABEL_OF = {
  student_model: "student_model", better_model: "better_model",
  train_size: "train_size", val_size: "val_size", test_size: "test_size",
  max_metric_calls: "max_metric_calls", reflection_minibatch_size: "reflection_minibatch_size",
  max_workers: "max_workers", seed: "seed",
};

const formFields = [
  "name", "student_model", "better_model", "train_size", "val_size",
  "test_size", "max_metric_calls", "reflection_minibatch_size", "max_workers", "seed",
];

/* ── HTTP ───────────────────────────────────────────────── */
async function fetchRuns() {
  const res = await fetch(API);
  state.runs = await res.json();
  renderRail();
}

async function fetchRun(id) {
  const res = await fetch(`${API}/${id}`);
  if (!res.ok) throw new Error(`run ${id} not found`);
  return res.json();
}

/* ── SVG chart ──────────────────────────────────────────── */
function svgLine(scores, w, h, opts = {}) {
  const pad = opts.pad ?? 8;
  if (scores.length === 0) return `<svg width="${w}" height="${h}" xmlns="http://www.w3.org/2000/svg"></svg>`;
  const x = scores.length <= 1 ? 0 : (i) => (i / (scores.length - 1)) * (w - pad * 2) + pad;
  const y = (v) => h - pad - v * (h - pad * 2);
  const pts = scores.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const circles = scores.map((v, i) =>
    `<circle cx="${x(i).toFixed(1)}" cy="${y(v).toFixed(1)}" r="${opts.r ?? 2.5}" fill="${opts.fill ?? '#5b9dff'}" />`
  ).join("");
  return `<svg width="${w}" height="${h}" xmlns="http://www.w3.org/2000/svg">
    <polyline points="${pts}" fill="none" stroke="${opts.stroke ?? '#5b9dff'}" stroke-width="${opts.sw ?? 1.5}" />
    ${circles}
  </svg>`;
}

function delta(r) { return (r.optimized_accuracy ?? 0) - r.baseline_accuracy; }
function fmtPct(v) { return v == null ? "—" : `${(v * 100).toFixed(1)}%`; }

/* ── render ─────────────────────────────────────────────── */
function statusClass(s) { return s === "completed" ? "ok" : s === "failed" ? "fail" : "run"; }

function renderRail() {
  const el = document.getElementById("rail-list");
  document.getElementById("rail-count").textContent = state.runs.length;
  if (state.runs.length === 0) {
    el.innerHTML = `<div class="empty-run">No runs yet. Start one.</div>`;
    return;
  }
  el.innerHTML = state.runs.map(r => {
    const sc = statusClass(r.status);
    const sub = r.optimized_accuracy == null
      ? (r.status === "running" ? "running…" : r.status)
      : `Δ ${(delta(r) * 100).toFixed(1)}pp`;
    return `<div class="rail-item ${r.id === state.selectedRunId ? "active" : ""}" data-run="${r.id}">
      <div class="ri-name">${escapeHtml(r.name)}</div>
      <div class="ri-meta"><span class="${sc}">${r.status}</span> · ${sub} · ${r.created_at}</div>
    </div>`;
  }).join("");
  el.querySelectorAll(".rail-item").forEach(node => {
    node.addEventListener("click", () => selectRun(node.dataset.run));
  });
}

function renderDetail(r, iters, logLines) {
  const detail = document.getElementById("detail");
  detail.innerHTML = `
    <h1>${escapeHtml(r.name)}</h1>
    <div class="status"><span class="${statusClass(r.status)}">${r.status}</span> · created ${r.created_at}${r.error ? `<span class="err">${escapeHtml(r.error)}</span>` : ""}</div>
    ${renderDelta(r)}
    <section class="block"><h3>Config</h3><table>${renderConfig(r.config)}</table></section>
    <section class="block chart"><h3>Iteration score (${iters.length})</h3>${svgLine(iters.map(i => i.score), 560, 150, { r: 3, stroke: "#5b9dff", fill: "#5b9dff" })}</section>
    <section class="block"><h3>Log</h3><div class="log" id="log">${logLines.join("")}</div></section>
    <section class="block"><h3>Prompts</h3><div class="prompts">
      <div><h3>Seed prompt</h3><pre>${escapeHtml(r.seed_prompt ?? "—")}</pre></div>
      <div><h3>Best prompt</h3><pre>${escapeHtml(r.best_prompt ?? "—")}</pre></div>
    </div></section>`;
  const log = document.getElementById("log");
  if (log) log.scrollTop = log.scrollHeight;
}

function renderDelta(r) {
  const running = r.optimized_accuracy == null && r.status === "running";
  if (running) {
    const latest = r.iterations.at(-1)?.score;
    return `<div class="delta-bar"><span class="big" style="color:#f0c060">running…</span>` +
      `<span class="sub">${r.iterations.length} iters${latest != null ? ` · latest ${fmtPct(latest)}` : ""}</span></div>`;
  }
  if (r.optimized_accuracy == null) {
    return `<div class="delta-bar"><span class="big">—</span><span class="sub">no result yet</span></div>`;
  }
  const d = delta(r);
  const dCls = d >= 0 ? "pos" : "neg";
  return `<div class="delta-bar"><span class="big ${dCls}">${d >= 0 ? "+" : ""}${(d * 100).toFixed(1)}pp</span>` +
    `<span class="sub">${fmtPct(r.baseline_accuracy)} → ${fmtPct(r.optimized_accuracy)} on ${r.n_test} held-out</span></div>`;
}

function renderConfig(cfg) {
  const keys = ["student_model", "better_model", "train_size", "val_size", "test_size",
                "max_metric_calls", "reflection_minibatch_size", "max_workers", "seed"];
  return keys.filter(k => cfg && k in cfg).map(k => `<tr><td>${k}</td><td>${escapeHtml(String(cfg[k]))}</td></tr>`).join("");
}

function logLineHtml(evt) {
  return `<div><span class="ts">[${evt.ts || ""}]</span> <span class="tag tag-${evt.tag}">[${evt.tag}]</span> ${escapeHtml(evt.msg)}</div>`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ── select + SSE ───────────────────────────────────────── */
async function selectRun(id) {
  if (state.es) { state.es.close(); state.es = null; }
  state.selectedRunId = id;
  state.liveIters = [];
  renderRail();
  let r;
  try { r = await fetchRun(id); }
  catch { return; }
  renderDetail(r, r.iterations, []);
  openSSE(id, r);
}

function openSSE(id, r) {
  const logLines = [];
  const es = new EventSource(`${API}/${id}/events`);
  state.es = es;
  es.onmessage = (e) => {
    const evt = JSON.parse(e.data);
    const d = document.getElementById("detail");
    if (!d || state.selectedRunId !== id) { es.close(); return; }
    if (evt.type === "log") {
      logLines.push(logLineHtml(evt));
      const log = document.getElementById("log");
      if (log) {
        log.insertAdjacentHTML("beforeend", logLineHtml(evt));
        log.scrollTop = log.scrollHeight;
      }
    } else if (evt.type === "iteration") {
      state.liveIters.push(evt);
      refreshChart(r);
    } else if (evt.type === "status") {
      updateStatusBadge(evt.status);
    } else if (evt.type === "done") {
      es.close();
      state.es = null;
      fetchRuns().then(() => fetchRun(id).then(latest => renderDetail(latest, latest.iterations, logLines)));
    }
  };
  es.onerror = () => { /* keepalive/breaks — let EventSource retry */ };
}

function refreshChart(r) {
  const all = [...(r?.iterations ?? []), ...state.liveIters];
  const chart = document.querySelector(".chart");
  if (chart) {
    chart.innerHTML = `<h3>Iteration score (${all.length})</h3>` + svgLine(all.map(i => i.score), 560, 150, { r: 3, stroke: "#5b9dff", fill: "#5b9dff" });
  }
}

function updateStatusBadge(status) {
  const badge = document.querySelector(".status span:first-child");
  if (badge) {
    badge.className = statusClass(status);
    badge.textContent = status;
  }
}

/* ── create run ─────────────────────────────────────────── */
async function startRun(ev) {
  ev.preventDefault();
  const errEl = document.getElementById("start-err");
  errEl.textContent = "";
  const body = { name: document.getElementById("f-name").value.trim(), overrides: {} };
  for (const k of formFields) {
    if (k === "name") continue;
    const v = document.getElementById(`f-${k}`).value;
    if (v === "") continue;
    body.overrides[k] = v;
  }
  let res;
  try {
    res = await fetch(API, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  } catch { errEl.textContent = "network error"; return; }
  if (res.status === 409) { errEl.textContent = "another run is already active"; return; }
  if (!res.ok) { errEl.textContent = `error ${res.status}`; return; }
  const run = await res.json();
  document.getElementById("new-run").classList.add("hidden");
  await fetchRuns();
  selectRun(run.id);
}

/* ── init ───────────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("new-run-btn").addEventListener("click", () =>
    document.getElementById("new-run").classList.toggle("hidden"));
  document.getElementById("start-run").addEventListener("click", startRun);
  fetchRuns().then(() => {
    if (state.runs.length) selectRun(state.runs[0].id);
  });
});