#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${FEATURE_ELM_BUILD_DIR:-${repo_root}/build}"
output_dir="${repo_root}/data/benchmarks/latest"

mkdir -p "${build_dir}"
cd "${repo_root}"

cmake -S "${repo_root}" -B "${build_dir}" -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" \
  --target bench_feature_maps \
  --target bench_solvers \
  --target bench_ml_elm \
  --target bench_elm_cuda

mkdir -p "${output_dir}"
"${build_dir}/bench/bench_feature_maps" --benchmark_format=json --benchmark_out="${output_dir}/bench_feature_maps.json"
"${build_dir}/bench/bench_solvers" --benchmark_format=json --benchmark_out="${output_dir}/bench_solvers.json"
"${build_dir}/bench/bench_ml_elm" --benchmark_format=json --benchmark_out="${output_dir}/bench_ml_elm.json"
"${build_dir}/bench/bench_elm_cuda" --benchmark_format=json --benchmark_out="${output_dir}/bench_elm_cuda.json"
