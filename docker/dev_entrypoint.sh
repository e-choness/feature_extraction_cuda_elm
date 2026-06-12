#!/usr/bin/env bash
set -euo pipefail

repo_root="${FEATURE_ELM_SOURCE_DIR:-/workspace}"
build_dir="${FEATURE_ELM_BUILD_DIR:-/tmp/feature_elm_build}"
build_type="${CMAKE_BUILD_TYPE:-Debug}"

configure_and_build() {
  cmake -S "${repo_root}" -B "${build_dir}" -G Ninja \
    -DCMAKE_BUILD_TYPE="${build_type}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  cmake --build "${build_dir}"
}

if [[ "${1:-}" == "ctest" ]]; then
  configure_and_build
  cd "${build_dir}"
  exec ctest "${@:2}"
fi

if [[ "${1:-}" == "./scripts/style_check.sh" ]]; then
  exec bash "$@"
fi

exec "$@"
