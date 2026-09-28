#!/usr/bin/env node
// Generates images/banner.svg: a 16:9 animated banner (input digit -> random hidden layer ->
// class scores, plus a CPU-vs-GPU training race). The race numbers are read from
// the committed benchmark snapshot (bench_datasets.json, else bench_elm.json) so the banner stays
// in step with measured numbers.
//
//   node scripts/gen_banner.mjs

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const W = 1920;
const H = 1080;
const LOOP = 6; // seconds per animation cycle

// --- benchmark numbers -------------------------------------------------------------------------
// Prefer the full MNIST run (60k images, 4,096 hidden units) from bench_datasets.json; fall back to
// the 2,048-sample micro-benchmark in bench_elm.json when the dataset suite has not been run.
const latest = join(root, "data/benchmarks/latest");
const loadRuns = (file) => {
  try {
    return JSON.parse(readFileSync(join(latest, file), "utf8")).benchmarks.filter((b) => !b.error_occurred);
  } catch {
    return [];
  }
};
const toMs = (run) => run.real_time * { ns: 1e-6, us: 1e-3, ms: 1, s: 1e3 }[run.time_unit];
const pick = (runs, cpuName, gpuName) => {
  const cpu = runs.find((b) => b.name === cpuName);
  const gpu = runs.find((b) => b.name === gpuName);
  return cpu && gpu ? { cpuMs: toMs(cpu), gpuMs: toMs(gpu) } : null;
};
const HIDDEN = 4096;
const race =
  (() => {
    const r = pick(loadRuns("bench_datasets.json"), `BatchElmTrain/mnist_cpu/${HIDDEN}/iterations:1/real_time`,
                   `BatchElmTrain/mnist_gpu/${HIDDEN}/iterations:1/real_time`);
    return r && { ...r, label: `MNIST 60K · ${HIDDEN.toLocaleString("en-US")} HIDDEN` };
  })() ||
  (() => {
    const r = pick(loadRuns("bench_elm.json"), "BenchmarkElmTrainCpu/2048/real_time",
                   "BenchmarkElmTrainGpu/2048/real_time");
    return r && { ...r, label: "TRAINING, 2,048 HIDDEN" };
  })();
if (!race) throw new Error("no CPU/GPU benchmark pair found; run scripts/run_benchmarks.sh on a GPU host");
const { cpuMs, gpuMs } = race;
const speedup = Math.round(cpuMs / gpuMs);
const fmt = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`);

// --- scene data -------------------------------------------------------------------------------
// A real "7" from the bundled UCI digits dataset (intensities 0..16).
const digit = [
  0, 0, 7, 8, 13, 16, 15, 1, 0, 0, 7, 7, 4, 11, 12, 0, 0, 0, 0, 0, 8, 13, 1, 0, 0, 4, 8, 8, 15, 15,
  6, 0, 0, 2, 11, 15, 15, 4, 0, 0, 0, 0, 0, 16, 5, 0, 0, 0, 0, 0, 9, 15, 1, 0, 0, 0, 0, 0, 13, 5, 0,
  0, 0, 0,
];
const scores = [0.05, 0.12, 0.28, 0.33, 0.06, 0.15, 0.02, 0.92, 0.1, 0.18]; // illustrative, 7 wins

const px = 34; // digit cell size
const digitX = 150;
const digitY = 496; // grid centre aligned with the hidden column centre (632)
const hidden = Array.from({ length: 9 }, (_, i) => ({ x: 720, y: 400 + i * 58 }));
const outX = 1010;
const outY0 = 402;
const outStep = 52;
const labelY = 366; // shared baseline for the four section labels

const out = [];
const add = (s) => out.push(s);

add(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-labelledby="t d">
  <title id="t">Feature ELM</title>
  <desc id="d">Extreme Learning Machines in C++20 and CUDA. An 8 by 8 digit passes through a random hidden layer to class scores, and Batch ELM training (${race.label.toLowerCase()}) runs about ${speedup} times faster on an RTX 4080 than on a 32-thread CPU.</desc>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#0a0f09"/><stop offset="1" stop-color="#061219"/>
    </linearGradient>
    <linearGradient id="brand" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#76b900"/><stop offset="1" stop-color="#00b4d8"/>
    </linearGradient>
    <radialGradient id="glowL" cx="0.25" cy="0.62" r="0.42">
      <stop offset="0" stop-color="#76b900" stop-opacity="0.20"/><stop offset="1" stop-color="#76b900" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="glowR" cx="0.82" cy="0.3" r="0.4">
      <stop offset="0" stop-color="#00b4d8" stop-opacity="0.14"/><stop offset="1" stop-color="#00b4d8" stop-opacity="0"/>
    </radialGradient>
    <pattern id="grid" width="48" height="48" patternUnits="userSpaceOnUse">
      <path d="M48 0H0V48" fill="none" stroke="#fff" stroke-opacity="0.03"/>
    </pattern>
    <style>
      text { font-family: "Segoe UI", Inter, Helvetica, Arial, sans-serif; }
      .title { font-size: 112px; font-weight: 800; fill: url(#brand); letter-spacing: -2px; }
      .tag { font-size: 34px; fill: #dfe8d6; }
      .muted { font-size: 22px; fill: #8f9d88; }
      .label { font-size: 22px; fill: #b9c6b1; letter-spacing: 2px; }
      .chip { font-size: 22px; fill: #c3cfbb; }
      .chip-bg { fill: #fff; fill-opacity: 0.05; stroke: #fff; stroke-opacity: 0.12; }
      .cell { fill: #eaffd0; opacity: 0; animation: cell ${LOOP}s ease-out infinite; }
      @keyframes cell { 0% { opacity: 0; } 12% { opacity: var(--o); } 88% { opacity: var(--o); } 100% { opacity: 0; } }
      .edge { stroke: #7fd0a0; stroke-opacity: 0.10; stroke-width: 1.3; }
      .flow { fill: none; stroke: url(#brand); stroke-width: 3; stroke-linecap: round;
              stroke-dasharray: 16 420; stroke-dashoffset: 436; animation: flow ${LOOP}s linear infinite; }
      @keyframes flow { 0%, 15% { stroke-dashoffset: 436; opacity: 0; } 17% { opacity: 1; } 45% { stroke-dashoffset: 0; opacity: 1; } 50%, 100% { stroke-dashoffset: 0; opacity: 0; } }
      .node { fill: #0f1a12; stroke: url(#brand); stroke-width: 3; animation: node ${LOOP}s ease-in-out infinite; }
      @keyframes node { 0%, 25% { fill: #0f1a12; } 40% { fill: #3d7a10; } 60%, 100% { fill: #0f1a12; } }
      .bar-track { fill: #fff; fill-opacity: 0.05; }
      .score { fill: #3987e5; transform-box: fill-box; transform-origin: left center; transform: scaleX(0); animation: score ${LOOP}s ease-out infinite; }
      .score.win { fill: #76b900; }
      @keyframes score { 0%, 45% { transform: scaleX(0); } 65%, 92% { transform: scaleX(1); } 100% { transform: scaleX(0); } }
      .digit-label { font-size: 24px; fill: #9fb096; font-weight: 600; }
      .win-label { fill: #ffffff; animation: win ${LOOP}s ease-out infinite; }
      @keyframes win { 0%, 60% { fill: #9fb096; } 68%, 92% { fill: #ffffff; } 100% { fill: #9fb096; } }
      .pred { font-size: 150px; font-weight: 800; fill: #fff; opacity: 0; animation: pred ${LOOP}s ease-out infinite; }
      @keyframes pred { 0%, 62% { opacity: 0; } 70%, 92% { opacity: 1; } 100% { opacity: 0; } }
      .race-cpu { fill: #3987e5; transform-box: fill-box; transform-origin: left center; animation: raceCpu ${LOOP}s linear infinite; }
      .race-gpu { fill: #76b900; transform-box: fill-box; transform-origin: left center; animation: raceGpu ${LOOP}s ease-out infinite; }
      @keyframes raceCpu { 0% { transform: scaleX(0); } 80%, 100% { transform: scaleX(1); } }
      @keyframes raceGpu { 0% { transform: scaleX(0); } 3%, 100% { transform: scaleX(1); } }
      .race-val { font-size: 30px; font-weight: 700; fill: #fff; }
      .race-late { opacity: 0; animation: late ${LOOP}s ease-out infinite; }
      @keyframes late { 0%, 78% { opacity: 0; } 84%, 100% { opacity: 1; } }
      .speed { font-size: 120px; font-weight: 800; fill: url(#brand); }
      @media (prefers-reduced-motion: reduce) {
        .cell { animation: none; opacity: var(--o); }
        .flow { animation: none; stroke-dasharray: none; stroke-opacity: 0.25; }
        .node, .score, .win-label, .pred, .race-cpu, .race-gpu, .race-late { animation: none; }
        .score, .race-cpu, .race-gpu { transform: none; }
        .pred, .race-late { opacity: 1; }
      }
    </style>
  </defs>
  <rect width="${W}" height="${H}" fill="url(#bg)"/>
  <rect width="${W}" height="${H}" fill="url(#grid)"/>
  <rect width="${W}" height="${H}" fill="url(#glowL)"/>
  <rect width="${W}" height="${H}" fill="url(#glowR)"/>`);

// Title block
add(`  <text x="150" y="215" class="title">Feature ELM</text>
  <text x="154" y="275" class="tag">Extreme Learning Machines in C++20 · trained in one solve, on CPU or CUDA</text>`);

// Section labels
add(`  <text x="${digitX}" y="${labelY}" class="label">INPUT 8×8</text>
  <text x="${hidden[0].x - 70}" y="${labelY}" class="label">RANDOM HIDDEN LAYER</text>
  <text x="${outX}" y="${labelY}" class="label">CLASS SCORES</text>`);

// Edges digit -> hidden (from the right edge of the digit grid), hidden -> outputs
const gridRight = digitX + 8 * px;
const edges = [];
for (let r = 0; r < 8; r += 1) {
  for (const h of hidden) edges.push(`M${gridRight + 10} ${digitY + r * px + px / 2} ${h.x} ${h.y}`);
}
for (const h of hidden) {
  for (let d = 0; d < 10; d += 1) edges.push(`M${h.x} ${h.y} ${outX - 40} ${outY0 + d * outStep}`);
}
add(`  <path class="edge" d="${edges.join(" ")}"/>`);

// Travelling signals: a handful of input -> hidden -> output paths, staggered.
const flows = [
  [1, 1, 7], [3, 3, 7], [4, 5, 3], [5, 6, 7], [2, 0, 2], [6, 7, 7], [0, 2, 9], [7, 8, 7], [3, 4, 5],
];
flows.forEach(([row, hi, dOut], i) => {
  const h = hidden[hi];
  const d = `M${gridRight + 10} ${digitY + row * px + px / 2} L${h.x} ${h.y} L${outX - 40} ${outY0 + dOut * outStep}`;
  add(`  <path class="flow" d="${d}" style="animation-delay:${(i * 0.09).toFixed(2)}s"/>`);
});

// Digit cells
digit.forEach((v, i) => {
  const x = digitX + (i % 8) * px;
  const y = digitY + Math.floor(i / 8) * px;
  add(`  <rect x="${x + 1}" y="${y + 1}" width="${px - 2}" height="${px - 2}" rx="3" fill="#fff" fill-opacity="0.04"/>`);
  if (v > 0) {
    add(`  <rect class="cell" x="${x + 1}" y="${y + 1}" width="${px - 2}" height="${px - 2}" rx="3" style="--o:${(v / 16).toFixed(2)};animation-delay:${((i / 64) * 0.5).toFixed(2)}s"/>`);
  }
});

// Hidden nodes
hidden.forEach((h, i) => add(`  <circle class="node" cx="${h.x}" cy="${h.y}" r="15" style="animation-delay:${(i * 0.06).toFixed(2)}s"/>`));

// Output bars
const barW = 250;
scores.forEach((s, d) => {
  const y = outY0 + d * outStep;
  const win = d === 7;
  add(`  <text x="${outX - 22}" y="${y + 8}" text-anchor="end" class="digit-label${win ? " win-label" : ""}">${d}</text>
  <rect class="bar-track" x="${outX}" y="${y - 12}" width="${barW}" height="24" rx="6"/>
  <rect class="score${win ? " win" : ""}" x="${outX}" y="${y - 12}" width="${Math.round(barW * s)}" height="24" rx="6"/>`);
});
add(`  <text x="${outX + barW + 45}" y="${outY0 + 5.2 * outStep}" class="pred">7</text>`);

// CPU vs GPU race (right column)
const rx = 1500;
const ry = labelY + 70;
const trackW = 290;
const gpuW = Math.max(6, Math.round((trackW * gpuMs) / cpuMs));
add(`  <text x="${rx}" y="${ry - 70}" class="label">${race.label}</text>
  <text x="${rx}" y="${ry}" class="chip">CPU</text>
  <rect class="bar-track" x="${rx}" y="${ry + 16}" width="${trackW}" height="30" rx="8"/>
  <rect class="race-cpu" x="${rx}" y="${ry + 16}" width="${trackW}" height="30" rx="8"/>
  <text x="${rx}" y="${ry + 86}" class="race-val race-late">${fmt(cpuMs)}</text>
  <text x="${rx}" y="${ry + 150}" class="chip">GPU</text>
  <rect class="bar-track" x="${rx}" y="${ry + 166}" width="${trackW}" height="30" rx="8"/>
  <rect class="race-gpu" x="${rx}" y="${ry + 166}" width="${gpuW}" height="30" rx="8"/>
  <text x="${rx}" y="${ry + 236}" class="race-val">${fmt(gpuMs)}</text>
  <text x="${rx - 6}" y="${ry + 380}" class="speed race-late">~${speedup}×</text>
  <text x="${rx}" y="${ry + 425}" class="muted race-late">RTX 4080 vs 32-thread CPU</text>`);

// Chips along the bottom
const chips = ["C++20", "CUDA 13.4", "cuBLAS · cuSOLVER", "OS-ELM · ML-ELM · RBF", "Docker", "MIT"];
let cx = 150;
const chipY = 960;
for (const c of chips) {
  const w = 30 + c.length * 12.5;
  add(`  <rect class="chip-bg" x="${cx}" y="${chipY}" width="${w.toFixed(0)}" height="48" rx="24"/>
  <text x="${(cx + w / 2).toFixed(0)}" y="${chipY + 32}" text-anchor="middle" class="chip">${c}</text>`);
  cx += w + 16;
}
add(`  <text x="${W - 150}" y="${chipY + 32}" text-anchor="end" class="muted">github.com/e-choness/feature_extraction_cuda_elm</text>`);

add("</svg>\n");
writeFileSync(join(root, "images/banner.svg"), out.join("\n"));
console.log(`images/banner.svg: CPU ${fmt(cpuMs)} vs GPU ${fmt(gpuMs)} (~${speedup}x) at ${HIDDEN} hidden`);
