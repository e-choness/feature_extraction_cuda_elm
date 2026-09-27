"use strict";

// ---------------------------------------------------------------------------------------------
// Helpers

const $ = (id) => document.getElementById(id);
const tooltip = $("tooltip");
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

async function api(path, options = {}) {
  const resp = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let body = null;
  try {
    body = await resp.json();
  } catch {
    body = { status: "error", message: `HTTP ${resp.status}` };
  }
  if (!resp.ok) {
    throw new Error(body?.message || `HTTP ${resp.status}`);
  }
  return body;
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "text") node.textContent = value;
    else if (key === "class") node.className = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) node.append(child);
  return node;
}

function svg(tag, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}

function showTooltip(event, title, rows) {
  tooltip.replaceChildren(el("div", { class: "tt-title", text: title }));
  for (const [label, value, color] of rows) {
    const row = el("div", { class: "tt-row" });
    if (color) row.append(el("span", { class: "swatch", style: `background:${color}` }));
    row.append(label, el("b", { text: value }));
    tooltip.append(row);
  }
  tooltip.hidden = false;
  const pad = 14;
  const { innerWidth: w, innerHeight: h } = window;
  const rect = tooltip.getBoundingClientRect();
  let x = event.clientX + pad;
  let y = event.clientY + pad;
  if (x + rect.width > w - 8) x = event.clientX - rect.width - pad;
  if (y + rect.height > h - 8) y = event.clientY - rect.height - pad;
  tooltip.style.left = `${x}px`;
  tooltip.style.top = `${y}px`;
}

function hideTooltip() {
  tooltip.hidden = true;
}

const fmtMs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : ms >= 10 ? `${ms.toFixed(0)} ms` : `${ms.toFixed(1)} ms`);
const fmtPct = (x) => `${(x * 100).toFixed(1)}%`;

// ---------------------------------------------------------------------------------------------
// Status

let gpuEnabled = false;

async function refreshStatus() {
  const status = $("status");
  try {
    const health = await api("/api/health");
    gpuEnabled = health.gpu_enabled;
    status.className = `status ${gpuEnabled ? "gpu" : "cpu"}`;
    $("statusText").textContent = gpuEnabled
      ? `GPU · ${health.device}`
      : health.gpu_available
        ? `CPU (GPU ${health.device} present but disabled)`
        : "CPU only";
    $("version").textContent = `v${health.version}`;
    if (health.max_hidden) {
      hiddenInput.max = String(health.max_hidden);
      if (Number(hiddenInput.value) > health.max_hidden) hiddenInput.value = String(health.max_hidden);
      $("hiddenOut").textContent = hiddenInput.value;
    }
    const gpuRadio = $("be-gpu");
    gpuRadio.disabled = !gpuEnabled;
    gpuRadio.parentElement.title = gpuEnabled ? "" : "This instance has no CUDA device";
    if (gpuEnabled) gpuRadio.checked = true;
  } catch (err) {
    status.className = "status down";
    $("statusText").textContent = "Server unreachable";
  }
}

// ---------------------------------------------------------------------------------------------
// Drawing pad

const SAMPLES = {
  0: [0,0,5,13,9,1,0,0,0,0,13,15,10,15,5,0,0,3,15,2,0,11,8,0,0,4,12,0,0,8,8,0,0,5,8,0,0,9,8,0,0,4,11,0,1,12,7,0,0,2,14,5,10,12,0,0,0,0,6,13,10,0,0,0],
  1: [0,0,0,12,13,5,0,0,0,0,0,11,16,9,0,0,0,0,3,15,16,6,0,0,0,7,15,16,16,2,0,0,0,0,1,16,16,3,0,0,0,0,1,16,16,6,0,0,0,0,1,16,16,6,0,0,0,0,0,11,16,10,0,0],
  2: [0,0,0,4,15,12,0,0,0,0,3,16,15,14,0,0,0,0,8,13,8,16,0,0,0,0,1,6,15,11,0,0,0,1,8,13,15,1,0,0,0,9,16,16,5,0,0,0,0,3,13,16,16,11,5,0,0,0,0,3,11,16,9,0],
  3: [0,0,7,15,13,1,0,0,0,8,13,6,15,4,0,0,0,2,1,13,13,0,0,0,0,0,2,15,11,1,0,0,0,0,0,1,12,12,1,0,0,0,0,0,1,10,8,0,0,0,8,4,5,14,9,0,0,0,7,13,13,9,0,0],
  4: [0,0,0,1,11,0,0,0,0,0,0,7,8,0,0,0,0,0,1,13,6,2,2,0,0,0,7,15,0,9,8,0,0,5,16,10,0,16,6,0,0,4,15,16,13,16,1,0,0,0,0,3,15,10,0,0,0,0,0,2,16,4,0,0],
  5: [0,0,12,10,0,0,0,0,0,0,14,16,16,14,0,0,0,0,13,16,15,10,1,0,0,0,11,16,16,7,0,0,0,0,0,4,7,16,7,0,0,0,0,0,4,16,9,0,0,0,5,4,12,16,4,0,0,0,9,16,16,10,0,0],
  6: [0,0,0,12,13,0,0,0,0,0,5,16,8,0,0,0,0,0,13,16,3,0,0,0,0,0,14,13,0,0,0,0,0,0,15,12,7,2,0,0,0,0,13,16,13,16,3,0,0,0,7,16,11,15,8,0,0,0,1,9,15,11,3,0],
  7: [0,0,7,8,13,16,15,1,0,0,7,7,4,11,12,0,0,0,0,0,8,13,1,0,0,4,8,8,15,15,6,0,0,2,11,15,15,4,0,0,0,0,0,16,5,0,0,0,0,0,9,15,1,0,0,0,0,0,13,5,0,0,0,0],
  8: [0,0,9,14,8,1,0,0,0,0,12,14,14,12,0,0,0,0,9,10,0,15,4,0,0,0,3,16,12,14,2,0,0,0,4,16,16,2,0,0,0,3,16,8,10,13,2,0,0,1,15,1,3,16,8,0,0,0,11,16,15,11,1,0],
  9: [0,0,11,12,0,0,0,0,0,2,16,16,16,13,0,0,0,3,16,12,10,14,0,0,0,1,16,1,12,15,0,0,0,0,13,16,9,15,2,0,0,0,0,3,0,9,11,0,0,0,0,0,9,15,4,0,0,0,9,12,13,3,0,0],
};

const pad = $("pad");
const ctx = pad.getContext("2d");
const pixels = new Array(64).fill(0);
let painting = false;
let lastCell = -1;
let classifyTimer = null;

function renderPad() {
  const size = pad.width / 8;
  ctx.clearRect(0, 0, pad.width, pad.height);
  const ink = css("--pad-ink") || "#eaffd0";
  for (let i = 0; i < 64; i++) {
    const x = (i % 8) * size;
    const y = Math.floor(i / 8) * size;
    ctx.globalAlpha = 1;
    ctx.fillStyle = "rgba(255,255,255,0.04)";
    ctx.fillRect(x + 1, y + 1, size - 2, size - 2);
    if (pixels[i] > 0) {
      ctx.globalAlpha = Math.min(1, pixels[i] / 16);
      ctx.fillStyle = ink;
      ctx.fillRect(x + 1, y + 1, size - 2, size - 2);
    }
  }
  ctx.globalAlpha = 1;
}

function cellAt(event) {
  const rect = pad.getBoundingClientRect();
  const col = Math.floor(((event.clientX - rect.left) / rect.width) * 8);
  const row = Math.floor(((event.clientY - rect.top) / rect.height) * 8);
  if (col < 0 || col > 7 || row < 0 || row > 7) return -1;
  return row * 8 + col;
}

function paint(event) {
  const cell = cellAt(event);
  if (cell < 0 || cell === lastCell) return;
  lastCell = cell;
  const row = Math.floor(cell / 8);
  const col = cell % 8;
  // A soft brush: strong centre, light spill into the 4-neighbourhood, like the scanned digits.
  const bump = (r, c, amount) => {
    if (r < 0 || r > 7 || c < 0 || c > 7) return;
    const i = r * 8 + c;
    pixels[i] = Math.min(16, pixels[i] + amount);
  };
  bump(row, col, 12);
  bump(row - 1, col, 3);
  bump(row + 1, col, 3);
  bump(row, col - 1, 3);
  bump(row, col + 1, 3);
  renderPad();
  scheduleClassify();
}

function scheduleClassify() {
  clearTimeout(classifyTimer);
  classifyTimer = setTimeout(classify, 60);
}

async function classify() {
  if (pixels.every((p) => p === 0)) {
    $("predDigit").textContent = "?";
    renderScores(null);
    return;
  }
  try {
    const result = await api("/api/classify", { method: "POST", body: JSON.stringify({ pixels }) });
    $("predDigit").textContent = result.digit;
    renderScores(result.scores, result.digit);
  } catch (err) {
    $("predDigit").textContent = "!";
  }
}

function renderScores(scores, top) {
  const container = $("scoreBars");
  if (!container.childElementCount) {
    for (let d = 0; d < 10; d++) {
      container.append(
        el("div", { class: "bar-row", "data-digit": d }, [
          el("span", { text: String(d) }),
          el("div", { class: "bar-track" }, [el("div", { class: "bar-fill" })]),
          el("span", { class: "bar-value", text: "—" }),
        ]),
      );
    }
  }
  // ELM outputs are unbounded regression scores; show them clipped to [0, 1] for the bars.
  [...container.children].forEach((row, d) => {
    const value = scores ? scores[d] : 0;
    row.classList.toggle("top", scores != null && d === top);
    row.querySelector(".bar-fill").style.width = `${Math.max(0, Math.min(1, value)) * 100}%`;
    row.querySelector(".bar-value").textContent = scores ? value.toFixed(2) : "—";
  });
}

pad.addEventListener("pointerdown", (event) => {
  painting = true;
  lastCell = -1;
  pad.setPointerCapture(event.pointerId);
  paint(event);
});
pad.addEventListener("pointermove", (event) => painting && paint(event));
pad.addEventListener("pointerup", () => {
  painting = false;
});
$("clearPad").addEventListener("click", () => {
  pixels.fill(0);
  $("sampleSelect").value = "";
  renderPad();
  classify();
});
for (let d = 0; d < 10; d++) $("sampleSelect").append(el("option", { value: d, text: `Digit ${d}` }));
$("sampleSelect").addEventListener("change", (event) => {
  const sample = SAMPLES[event.target.value];
  if (!sample) return;
  sample.forEach((v, i) => (pixels[i] = v));
  renderPad();
  classify();
});

// ---------------------------------------------------------------------------------------------
// Train & evaluate

const history = [];
const form = $("evalForm");
const hiddenInput = form.elements.namedItem("hidden");
hiddenInput.addEventListener("input", () => ($("hiddenOut").textContent = hiddenInput.value));

function setTile(id, text) {
  const node = $(id);
  node.textContent = text;
  node.classList.remove("flash");
  void node.offsetWidth;
  node.classList.add("flash");
}

function renderConfusion(matrix) {
  const container = $("confusion");
  container.replaceChildren();
  const max = Math.max(1, ...matrix.flat());
  container.append(el("span", { class: "axis" }));
  for (let c = 0; c < 10; c++) container.append(el("span", { class: "axis", text: String(c) }));
  for (let r = 0; r < 10; r++) {
    container.append(el("span", { class: "axis", text: String(r) }));
    for (let c = 0; c < 10; c++) {
      const value = matrix[r][c];
      // Sequential blue ramp: 0 stays on the surface, the rest steps 1..7 by share of the max.
      const step = value === 0 ? 0 : 1 + Math.min(6, Math.floor((value / max) * 6.999));
      const ink = step >= 4 ? " dark-ink" : step >= 1 ? " light-ink" : "";
      const cell = el("span", { class: `cell${ink}`, role: "cell", text: value ? String(value) : "" });
      cell.style.background = `var(--seq-${step})`;
      cell.addEventListener("pointermove", (event) =>
        showTooltip(event, r === c ? `Correct: ${r}` : `True ${r} → predicted ${c}`, [["Samples", String(value)]]),
      );
      cell.addEventListener("pointerleave", hideTooltip);
      container.append(cell);
    }
  }
}

function renderHistory() {
  const body = $("historyBody");
  body.replaceChildren();
  history.slice(0, 8).forEach((run, index) => {
    const tr = el("tr", index === 0 ? { class: "fresh" } : {}, [
      el("td", { text: run.model }),
      el("td", { class: "num", text: String(run.hidden) }),
      el("td", {}, [
        el("span", { class: "swatch", style: `background:var(--series-${run.backend})` }),
        run.backend.toUpperCase(),
      ]),
      el("td", { text: run.precision }),
      el("td", { class: "num", text: fmtPct(run.test_accuracy) }),
      el("td", { class: "num", text: run.train_ms.toFixed(1) }),
    ]);
    body.append(tr);
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = $("evalButton");
  const error = $("evalError");
  error.hidden = true;
  button.disabled = true;
  button.textContent = "Training…";
  const payload = {
    model: form.model.value,
    hidden: Number(hiddenInput.value),
    activation: form.activation.value,
    precision: form.precision.value,
    backend: form.backend.value,
  };
  try {
    const result = await api("/api/evaluate", { method: "POST", body: JSON.stringify(payload) });
    setTile("tAcc", fmtPct(result.test_accuracy));
    setTile("tTrain", fmtMs(result.train_ms));
    setTile("tPredict", fmtMs(result.predict_ms));
    setTile("tBackend", result.backend.toUpperCase());
    renderConfusion(result.confusion);
    history.unshift(result);
    renderHistory();
  } catch (err) {
    error.textContent = err.message;
    error.hidden = false;
  } finally {
    button.disabled = false;
    button.textContent = "Train";
  }
});

// ---------------------------------------------------------------------------------------------
// Scaling chart: one y-axis (train time, log scale), one line per backend.

let benchRows = [];

function renderBenchChart() {
  const container = $("benchChart");
  const legend = $("benchLegend");
  container.replaceChildren();
  legend.replaceChildren();
  const W = 720;
  const H = 300;
  const m = { top: 30, right: 64, bottom: 40, left: 64 };
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Training time versus hidden nodes" });
  container.append(root);
  if (!benchRows.length) {
    const text = svg("text", { x: W / 2, y: H / 2, "text-anchor": "middle", class: "empty-msg" });
    text.textContent = "Run the sweep to compare backends.";
    root.append(text);
    return;
  }

  const series = [{ key: "cpu", label: "CPU", color: css("--series-cpu") }];
  if (benchRows.some((r) => r.gpu_train_ms != null)) series.push({ key: "gpu", label: "GPU", color: css("--series-gpu") });
  if (series.length > 1) {
    for (const s of series) legend.append(el("span", {}, [el("span", { class: "swatch", style: `background:${s.color}` }), s.label]));
  }

  const values = benchRows.flatMap((r) => series.map((s) => r[`${s.key}_train_ms`]).filter((v) => v != null && v > 0));
  const lo = Math.pow(10, Math.floor(Math.log10(Math.min(...values))));
  const hi = Math.pow(10, Math.ceil(Math.log10(Math.max(...values))));
  const x = (i) => m.left + (i / Math.max(1, benchRows.length - 1)) * (W - m.left - m.right);
  const y = (v) => m.top + (1 - (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * (H - m.top - m.bottom);

  for (let p = Math.log10(lo); p <= Math.log10(hi) + 1e-9; p++) {
    const v = Math.pow(10, p);
    root.append(svg("line", { x1: m.left, x2: W - m.right, y1: y(v), y2: y(v), class: "gridline" }));
    const label = svg("text", { x: m.left - 8, y: y(v) + 4, "text-anchor": "end", class: "axis-label" });
    label.textContent = fmtMs(v);
    root.append(label);
  }
  benchRows.forEach((row, i) => {
    const label = svg("text", { x: x(i), y: H - m.bottom + 18, "text-anchor": "middle", class: "axis-label" });
    label.textContent = row.hidden;
    root.append(label);
  });
  const xTitle = svg("text", { x: (m.left + W - m.right) / 2, y: H - 4, "text-anchor": "middle", class: "axis-label" });
  xTitle.textContent = "hidden nodes";
  root.append(xTitle);
  const yTitle = svg("text", { x: m.left - 8, y: 12, "text-anchor": "end", class: "axis-label" });
  yTitle.textContent = "train time (log)";
  root.append(yTitle);

  for (const s of series) {
    const points = benchRows.map((r, i) => [x(i), r[`${s.key}_train_ms`]]).filter(([, v]) => v != null);
    const d = points.map(([px, v], i) => `${i ? "L" : "M"}${px},${y(v)}`).join("");
    const path = svg("path", { d, class: "series-line draw", stroke: s.color });
    root.append(path);
    const len = path.getTotalLength ? path.getTotalLength() : 1000;
    path.style.setProperty("--len", len);
    for (const [px, v] of points) root.append(svg("circle", { cx: px, cy: y(v), r: 4.5, fill: s.color, class: "marker" }));
    const [lx, lv] = points[points.length - 1];
    const label = svg("text", { x: lx + 10, y: y(lv) + 4, class: "direct-label" });
    label.textContent = s.label;
    root.append(label);
  }

  // Hover layer: a crosshair snapped to the nearest hidden size.
  const cross = svg("line", { y1: m.top, y2: H - m.bottom, class: "crosshair", visibility: "hidden" });
  root.append(cross);
  const hit = svg("rect", { x: m.left - 20, y: 0, width: W - m.left - m.right + 40, height: H, fill: "transparent" });
  root.append(hit);
  hit.addEventListener("pointermove", (event) => {
    const box = root.getBoundingClientRect();
    const px = ((event.clientX - box.left) / box.width) * W;
    const i = Math.max(0, Math.min(benchRows.length - 1, Math.round(((px - m.left) / (W - m.left - m.right)) * (benchRows.length - 1))));
    const row = benchRows[i];
    cross.setAttribute("x1", x(i));
    cross.setAttribute("x2", x(i));
    cross.setAttribute("visibility", "visible");
    const rows = series
      .filter((s) => row[`${s.key}_train_ms`] != null)
      .map((s) => [`${s.label} ${fmtPct(row[`${s.key}_accuracy`])}`, fmtMs(row[`${s.key}_train_ms`]), s.color]);
    if (row.gpu_train_ms) rows.push(["Speed-up", `${(row.cpu_train_ms / row.gpu_train_ms).toFixed(1)}×`]);
    showTooltip(event, `${row.hidden} hidden nodes`, rows);
  });
  hit.addEventListener("pointerleave", () => {
    cross.setAttribute("visibility", "hidden");
    hideTooltip();
  });
}

function renderBenchTable() {
  const hasGpu = benchRows.some((r) => r.gpu_train_ms != null);
  const head = ["Hidden", "CPU train", "CPU accuracy"].concat(hasGpu ? ["GPU train", "GPU accuracy", "Speed-up"] : []);
  const table = el("table", {}, [
    el("thead", {}, [el("tr", {}, head.map((h, i) => el("th", { class: i ? "num" : "", text: h })))]),
    el(
      "tbody",
      {},
      benchRows.map((r) =>
        el(
          "tr",
          {},
          [
            el("td", { text: String(r.hidden) }),
            el("td", { class: "num", text: fmtMs(r.cpu_train_ms) }),
            el("td", { class: "num", text: fmtPct(r.cpu_accuracy) }),
          ].concat(
            hasGpu
              ? [
                  el("td", { class: "num", text: r.gpu_train_ms != null ? fmtMs(r.gpu_train_ms) : "—" }),
                  el("td", { class: "num", text: r.gpu_accuracy != null ? fmtPct(r.gpu_accuracy) : "—" }),
                  el("td", { class: "num", text: r.gpu_train_ms ? `${(r.cpu_train_ms / r.gpu_train_ms).toFixed(1)}×` : "—" }),
                ]
              : [],
          ),
        ),
      ),
    ),
  ]);
  $("benchTable").replaceChildren(table);
}

$("benchButton").addEventListener("click", async () => {
  const button = $("benchButton");
  button.disabled = true;
  button.textContent = "Running…";
  $("benchNote").textContent = "Training 5 hidden sizes per backend…";
  try {
    const result = await api("/api/benchmark", { method: "POST", body: "{}" });
    benchRows = result.rows;
    $("benchNote").textContent = result.gpu_enabled
      ? `float32 Batch ELM on ${result.device}`
      : "float32 Batch ELM · this instance has no GPU, so only the CPU line is shown";
    renderBenchChart();
    renderBenchTable();
  } catch (err) {
    $("benchNote").textContent = err.message;
  } finally {
    button.disabled = false;
    button.textContent = "Run sweep";
  }
});

$("benchTableToggle").addEventListener("click", (event) => {
  const table = $("benchTable");
  table.hidden = !table.hidden;
  event.target.setAttribute("aria-pressed", String(!table.hidden));
  event.target.textContent = table.hidden ? "Show table" : "Hide table";
});

// ---------------------------------------------------------------------------------------------
// Snapshots

async function loadSnapshotList() {
  const select = $("snapshotSelect");
  try {
    const { snapshots } = await api("/api/benchmarks");
    select.replaceChildren(...snapshots.map((s) => el("option", { value: s.name, text: s.name })));
    if (snapshots.length) showSnapshot(snapshots[0].name);
  } catch {
    select.replaceChildren(el("option", { text: "No snapshots" }));
  }
}

async function showSnapshot(name) {
  const target = $("snapshotTable");
  try {
    const data = await api(`/api/benchmarks/${encodeURIComponent(name)}`);
    const rows = (data.benchmarks || []).slice(0, 40);
    const context = data.context || {};
    target.replaceChildren(
      el("p", { class: "muted", text: `${context.host_name ? context.host_name + " · " : ""}${context.num_cpus ?? "?"} CPUs · ${context.date ?? ""}` }),
      el("table", {}, [
        el("thead", {}, [el("tr", {}, ["Benchmark", "Real time", "Items / s"].map((h, i) => el("th", { class: i ? "num" : "", text: h })))]),
        el(
          "tbody",
          {},
          rows.map((b) =>
            el("tr", {}, [
              el("td", { text: b.name }),
              el("td", { class: "num", text: `${Number(b.real_time).toFixed(1)} ${b.time_unit}` }),
              el("td", { class: "num", text: b.items_per_second ? Math.round(b.items_per_second).toLocaleString() : "—" }),
            ]),
          ),
        ),
      ]),
    );
  } catch (err) {
    target.replaceChildren(el("p", { class: "error", text: err.message }));
  }
}

$("snapshotSelect").addEventListener("change", (event) => showSnapshot(event.target.value));

// ---------------------------------------------------------------------------------------------

renderPad();
renderScores(null);
renderConfusion(Array.from({ length: 10 }, () => new Array(10).fill(0)));
renderBenchChart();
refreshStatus();
loadSnapshotList();
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  renderPad();
  renderBenchChart();
});
