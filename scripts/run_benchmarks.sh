#!/usr/bin/env bash
# Build and run the Google Benchmark suites, writing JSON to data/benchmarks/latest/.
#
# The full-dataset suite (bench_datasets: MNIST / Fashion-MNIST, CPU vs GPU) runs when the data has
# been fetched with `python3 scripts/fetch_datasets.py`; its CPU cases take several minutes. Set
# FEATURE_ELM_BENCH_DATASETS=0 to skip it.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${FEATURE_ELM_BUILD_DIR:-${repo_root}/build}"
output_dir="${repo_root}/data/benchmarks/latest"

mkdir -p "${build_dir}"
cd "${repo_root}"

cmake -S "${repo_root}" -B "${build_dir}" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-native}"
cmake --build "${build_dir}" \
  --target bench_feature_maps \
  --target bench_solvers \
  --target bench_ml_elm \
  --target bench_elm \
  --target bench_datasets

mkdir -p "${output_dir}"
"${build_dir}/bench/bench_feature_maps" --benchmark_format=json --benchmark_out="${output_dir}/bench_feature_maps.json"
"${build_dir}/bench/bench_solvers" --benchmark_format=json --benchmark_out="${output_dir}/bench_solvers.json"
"${build_dir}/bench/bench_ml_elm" --benchmark_format=json --benchmark_out="${output_dir}/bench_ml_elm.json"
"${build_dir}/bench/bench_elm" --benchmark_format=json --benchmark_out="${output_dir}/bench_elm.json"

if [[ "${FEATURE_ELM_BENCH_DATASETS:-1}" != "0" ]]; then
  if [[ -f "${repo_root}/.cache/datasets/mnist/train-images-idx3-ubyte" ]]; then
    "${build_dir}/bench/bench_datasets" --benchmark_format=json --benchmark_out="${output_dir}/bench_datasets.json"
  else
    echo "bench_datasets: skipped (run 'python3 scripts/fetch_datasets.py' first)"
  fi
fi
