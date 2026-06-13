#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${FEATURE_ELM_BUILD_DIR:-${repo_root}/build}"
output_dir="${repo_root}/data/benchmarks/latest"

mkdir -p "${build_dir}"
cd "${repo_root}"

cmake -S "${repo_root}" -B "${build_dir}" -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --target bench_elm_core --target bench_elm_cuda

mkdir -p "${output_dir}"
"${build_dir}/bench/bench_elm_core" --benchmark_format=json --benchmark_out="${output_dir}/bench_elm_core.json"
"${build_dir}/bench/bench_elm_cuda" --benchmark_format=json --benchmark_out="${output_dir}/bench_elm_cuda.json"
