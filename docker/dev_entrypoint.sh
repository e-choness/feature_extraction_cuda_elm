#!/usr/bin/env bash
set -euo pipefail

repo_root="${FEATURE_ELM_SOURCE_DIR:-/workspace}"
build_dir="${FEATURE_ELM_BUILD_DIR:-/tmp/feature_elm_build}"
build_type="${CMAKE_BUILD_TYPE:-Debug}"

# Compile device code only for the GPU that is present (fast), or for sm_75 when the container has
# no GPU (CI). Release images build the full fat binary instead; see docker/Dockerfile.demo.gpu.
if [[ -z "${CMAKE_CUDA_ARCHITECTURES:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    CMAKE_CUDA_ARCHITECTURES=native
  else
    CMAKE_CUDA_ARCHITECTURES=75
  fi
fi

configure_and_build() {
  cmake -S "${repo_root}" -B "${build_dir}" -G Ninja \
    -DCMAKE_BUILD_TYPE="${build_type}" \
    -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  cmake --build "${build_dir}"
}

case "${1:-}" in
  ctest)
    configure_and_build
    cd "${build_dir}"
    exec ctest "${@:2}"
    ;;
  build)
    configure_and_build
    ;;
  ./*.sh)
    export CMAKE_CUDA_ARCHITECTURES
    exec bash "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
