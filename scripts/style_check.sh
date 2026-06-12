#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${FEATURE_ELM_BUILD_DIR:-/tmp/feature_elm_build}"
build_type="${CMAKE_BUILD_TYPE:-Debug}"

cd "${repo_root}"

mapfile -t format_files < <(
  find src tests -type f \
    \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' \
    -o -name '*.h' -o -name '*.hpp' -o -name '*.hh' -o -name '*.hxx' \
    -o -name '*.cu' -o -name '*.cuh' \) | sort
)

if ((${#format_files[@]} > 0)); then
  clang-format --dry-run --Werror "${format_files[@]}"
fi

cmake -S "${repo_root}" -B "${build_dir}" -G Ninja \
  -DCMAKE_BUILD_TYPE="${build_type}" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON >/dev/null
cmake --build "${build_dir}" >/dev/null

mapfile -t tidy_files < <(
  find src -type f \
    \( -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' \) | sort
)

if ((${#tidy_files[@]} > 0)); then
  clang-tidy "${tidy_files[@]}" -p "${build_dir}"
fi
