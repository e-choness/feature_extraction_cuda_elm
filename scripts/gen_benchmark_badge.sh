#!/usr/bin/env bash
set -euo pipefail

# gen_benchmark_badge.sh - Generate shields.io endpoint JSON and README benchmark table
# Reads benchmark JSON from data/benchmarks/latest/ and data/benchmarks/snapshots/
# Outputs: docs/badges/benchmark.json (shields endpoint) and updates README table

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
latest_dir="${repo_root}/data/benchmarks/latest"
snapshot_dir="${repo_root}/data/benchmarks/snapshots"
badge_dir="${repo_root}/docs/badges"
readme="${repo_root}/README.md"

mkdir -p "${badge_dir}"

# Run Python script to process benchmark JSON
updated_table=$(python3 - "${latest_dir}" "${snapshot_dir}" "${badge_dir}" "${readme}" << 'PYEOF'
import json
import os
import sys
from pathlib import Path
import re

def extract_max_throughput(json_path, key_substring):
    """Extract the maximum items_per_second for benchmarks matching key_substring."""
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        throughputs = []
        for bench in data.get('benchmarks', []):
            if key_substring in bench.get('name', ''):
                ips = bench.get('items_per_second', 0)
                if ips and ips > 0:
                    throughputs.append(ips)
        return max(throughputs) if throughputs else None
    except (json.JSONDecodeError, IOError):
        return None

latest_dir = Path(sys.argv[1])
snapshot_dir = Path(sys.argv[2])
badge_dir = Path(sys.argv[3])
readme = Path(sys.argv[4])

# Extract CPU throughputs
cpu_additive = extract_max_throughput(latest_dir / 'bench_feature_maps.json', 'AdditiveMapTransform')
cpu_rbf = extract_max_throughput(latest_dir / 'bench_feature_maps.json', 'RbfMapTransform')
cpu_ml_elm = extract_max_throughput(latest_dir / 'bench_ml_elm.json', 'MlElm')
cpu_ridge = extract_max_throughput(latest_dir / 'bench_solvers.json', 'RidgeSolveCholesky')

# Extract GPU throughputs from snapshots (or None)
gpu_additive = extract_max_throughput(snapshot_dir / 'bench_feature_maps.json', 'AdditiveMapTransform')
gpu_rbf = extract_max_throughput(snapshot_dir / 'bench_feature_maps.json', 'RbfMapTransform')
gpu_ml_elm = extract_max_throughput(snapshot_dir / 'bench_ml_elm.json', 'MlElm')
gpu_ridge = extract_max_throughput(snapshot_dir / 'bench_solvers.json', 'RidgeSolveGpuQr')

def fmt(val):
    if val is None:
        return "N/A"
    return f"{val:,.0f}"

# Generate shields.io endpoint JSON
badge_data = {
    "schemaVersion": 1,
    "label": "benchmark",
    "message": f"ridge:{fmt(cpu_ridge)}/s",
    "color": "blue"
}
with open(badge_dir / 'benchmark.json', 'w') as f:
    json.dump(badge_data, f, indent=2)

# Generate benchmark table
table = f"""| Benchmark | CPU (items/sec) | GPU (items/sec) |
|-----------|-----------------|-----------------|
| Additive Map Transform | {fmt(cpu_additive)} | {fmt(gpu_additive)} |
| RBF Map Transform | {fmt(cpu_rbf)} | {fmt(gpu_rbf)} |
| ML-ELM Fit | {fmt(cpu_ml_elm)} | {fmt(gpu_ml_elm)} |
| Ridge Solve (Cholesky) | {fmt(cpu_ridge)} | {fmt(gpu_ridge)} |"""

# Update README.md in-place if it has the markers
if readme.exists():
    content = readme.read_text()
    start_marker = "<!-- BENCHMARK_TABLE_START -->"
    end_marker = "<!-- BENCHMARK_TABLE_END -->"
    
    pattern = re.escape(start_marker) + r'.*?' + re.escape(end_marker)
    replacement = f"{start_marker}\n{table}\n{end_marker}"
    
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    if new_content != content:
        readme.write_text(new_content)
        print("Updated README.md benchmark table")
    else:
        print("README.md markers not found or table unchanged")

print(f"Generated benchmark badge at {badge_dir}/benchmark.json")
PYEOF
)

echo "${updated_table}"