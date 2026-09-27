#!/usr/bin/env bash
set -euo pipefail

# gen_benchmark_badge.sh - Generate the shields.io endpoint JSON and the README benchmark table
# from the Google Benchmark JSON in data/benchmarks/latest/ (written by run_benchmarks.sh).
# Outputs: docs/badges/benchmark.json and the table between the BENCHMARK_TABLE markers in README.md.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 - "${repo_root}" <<'PYEOF'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
latest = root / "data" / "benchmarks" / "latest"
badge_dir = root / "docs" / "badges"
readme = root / "README.md"


def load(name):
    path = latest / name
    if not path.exists():
        return {}, {}
    data = json.loads(path.read_text())
    runs = {b["name"]: b for b in data.get("benchmarks", []) if not b.get("error_occurred")}
    return runs, data.get("context", {})


def ms(run):
    if run is None:
        return None
    scale = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}[run.get("time_unit", "ns")]
    return run["real_time"] * scale


def fmt(value):
    if value is None:
        return "—"
    return f"{value:,.2f} ms" if value < 10 else f"{value:,.1f} ms"


elm, context = load("bench_elm.json")
solvers, _ = load("bench_solvers.json")
maps, _ = load("bench_feature_maps.json")
ml, _ = load("bench_ml_elm.json")

rows = []
for hidden in (256, 512, 1024, 2048):
    rows.append((f"Batch ELM train, {hidden} hidden", "", ms(elm.get(f"BenchmarkElmTrainCpu/{hidden}/real_time")),
                 "", ms(elm.get(f"BenchmarkElmTrainGpu/{hidden}/real_time"))))
for hidden in (1024, 4096):
    rows.append((f"Hidden-layer transform, {hidden} hidden", "", ms(elm.get(f"BenchmarkHiddenTransformCpu/{hidden}/real_time")),
                 "", ms(elm.get(f"BenchmarkHiddenTransformGpu/{hidden}/real_time"))))
rows.append(("Ridge solve, 256 features", "Cholesky", ms(solvers.get("BenchmarkRidgeSolveCholeskyPrimal/256")),
             "cuSOLVER QR", ms(solvers.get("BenchmarkRidgeSolveGpuQr/256"))))
rows.append(("RBF map transform, 2048", "", ms(maps.get("BenchmarkRbfMapTransform/2048")), "", None))
rows.append(("ML-ELM fit, 1024", "", ms(ml.get("BenchmarkMlElmFit/1024")), "", None))
rows.append(("RLS update, 256", "", ms(solvers.get("BenchmarkRlsUpdate/256")), "", None))

lines = ["| Benchmark | CPU | GPU |", "|---|---:|---:|"]
for name, cpu_note, cpu, gpu_note, gpu in rows:
    cpu_cell = fmt(cpu) + (f" ({cpu_note})" if cpu is not None and cpu_note else "")
    gpu_cell = fmt(gpu) + (f" ({gpu_note})" if gpu is not None and gpu_note else "")
    lines.append(f"| {name} | {cpu_cell} | {gpu_cell} |")
date = context.get("date", "")[:10]
cpus = context.get("num_cpus", "?")
lines.append("")
lines.append(f"<sub>Wall time per call (2048 samples, 64 inputs, float32), lower is better. Recorded {date} "
             f"with {cpus} CPU threads and an RTX 4080; the CPU reference is single-threaded.</sub>")
table = "\n".join(lines)

badge_dir.mkdir(parents=True, exist_ok=True)
gpu_train = ms(elm.get("BenchmarkElmTrainGpu/512/real_time"))
badge = {
    "schemaVersion": 1,
    "label": "GPU ELM train (512 hidden, 2048 samples)",
    "message": fmt(gpu_train) if gpu_train else "n/a",
    "color": "76b900",
}
(badge_dir / "benchmark.json").write_text(json.dumps(badge, indent=2) + "\n")

if readme.exists():
    content = readme.read_text(encoding="utf-8")
    pattern = r"<!-- BENCHMARK_TABLE_START -->.*?<!-- BENCHMARK_TABLE_END -->"
    replacement = f"<!-- BENCHMARK_TABLE_START -->\n{table}\n<!-- BENCHMARK_TABLE_END -->"
    updated = re.sub(pattern, lambda _: replacement, content, flags=re.DOTALL)
    if updated != content:
        readme.write_text(updated, encoding="utf-8")
        print("Updated README.md benchmark table")
    else:
        print("README.md markers not found or table unchanged")
print(f"Wrote {badge_dir / 'benchmark.json'}")
PYEOF
