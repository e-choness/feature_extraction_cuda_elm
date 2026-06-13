async function fetchJson(path, options = {}) {
  const resp = await fetch(path, options);
  return resp.text();
}

async function refreshHealth() {
  const body = await fetchJson("/health");
  document.getElementById("health").textContent = body;
}

async function loadBenchmarks() {
  const body = await fetchJson("/benchmark-snapshots");
  document.getElementById("benchmarks").textContent = body;
}

async function runInference() {
  const payload = { input: [1.0, 2.0, 3.0, 4.0] };
  const body = await fetchJson("/run-inference", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  document.getElementById("inferenceResult").textContent = body;
}

async function runBenchmark() {
  const body = await fetchJson("/run-benchmark", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  });
  document.getElementById("benchmarkResult").textContent = body;
}

document.getElementById("loadBenchmarks").addEventListener("click", loadBenchmarks);
document.getElementById("runInference").addEventListener("click", runInference);
document.getElementById("runBenchmark").addEventListener("click", runBenchmark);
window.addEventListener("load", refreshHealth);
