#!/usr/bin/env bash
set -eu

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${FEATURE_ELM_BUILD_DIR:-/tmp/feature_elm_build}"
build_type="${CMAKE_BUILD_TYPE:-Debug}"

cd "${repo_root}"

format_files=$(find src tests -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' -o -name '*.h' -o -name '*.hpp' -o -name '*.hh' -o -name '*.hxx' -o -name '*.cu' -o -name '*.cuh' \) | sort)

for f in ${format_files}; do
  clang-format --dry-run --Werror "${f}"
done

cmake -S "${repo_root}" -B "${build_dir}" -G Ninja \
  -DCMAKE_BUILD_TYPE="${build_type}" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON >/dev/null
cmake --build "${build_dir}" >/dev/null

tidy_files=$(find src -type f \( -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' \) | sort)

for f in ${tidy_files}; do
  clang-tidy "${f}" -p "${build_dir}"
done