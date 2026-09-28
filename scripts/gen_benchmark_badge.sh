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
             "cuSOLVER Cholesky", ms(solvers.get("BenchmarkRidgeSolveGpu/256"))))
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
             f"with {cpus} CPU threads (OpenMP) and an RTX 4080.</sub>")
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

# Full-dataset table (bench_datasets.json): one row per model/dataset/hidden size.
datasets_path = latest / "bench_datasets.json"
dataset_table = None
if datasets_path.exists():
    runs = {b["name"]: b for b in json.loads(datasets_path.read_text()).get("benchmarks", [])
            if not b.get("error_occurred")}
    labels = {"mnist": "MNIST", "fashion": "Fashion-MNIST"}
    models = {"BatchElmTrain": "Batch ELM", "OsElmStream": "OS-ELM (stream)"}
    rows = ["| Model | Dataset | Hidden | CPU | GPU | Speed-up | Test accuracy |",
            "|---|---|---:|---:|---:|---:|---:|"]
    seen = set()
    for name in runs:
        fn, variant, hidden = name.split("/")[:3]
        dataset_key = variant.rsplit("_", 1)[0]
        key = (fn, dataset_key, hidden)
        if key in seen:
            continue
        seen.add(key)
        suffix = "/iterations:1/real_time"
        cpu = runs.get(f"{fn}/{dataset_key}_cpu/{hidden}{suffix}")
        gpu = runs.get(f"{fn}/{dataset_key}_gpu/{hidden}{suffix}")
        cpu_ms, gpu_ms = ms(cpu), ms(gpu)
        speed = f"{cpu_ms / gpu_ms:.0f}×" if cpu_ms and gpu_ms else "—"
        accuracy = (gpu or cpu or {}).get("accuracy")
        acc = f"{accuracy * 100:.1f}%" if accuracy is not None else "—"
        fmt_s = lambda v: "—" if v is None else (f"{v / 1000:.2f} s" if v >= 1000 else f"{v:.0f} ms")
        rows.append(f"| {models.get(fn, fn)} | {labels.get(dataset_key, dataset_key)} | {int(hidden):,} "
                    f"| {fmt_s(cpu_ms)} | {fmt_s(gpu_ms)} | {speed} | {acc} |")
    rows.append("")
    rows.append("<sub>60,000 training images (784 inputs), float32, trained once; accuracy on the "
                "10,000 test images. CPU uses all threads (OpenMP).</sub>")
    dataset_table = "\n".join(rows)

if readme.exists():
    content = readme.read_text(encoding="utf-8")
    for marker, body in (("BENCHMARK_TABLE", table), ("DATASET_TABLE", dataset_table)):
        if body is None:
            continue
        pattern = rf"<!-- {marker}_START -->.*?<!-- {marker}_END -->"
        replacement = f"<!-- {marker}_START -->\n{body}\n<!-- {marker}_END -->"
        content = re.sub(pattern, lambda _: replacement, content, flags=re.DOTALL)
    readme.write_text(content, encoding="utf-8")
    print("Updated README.md benchmark tables")
print(f"Wrote {badge_dir / 'benchmark.json'}")
PYEOF
