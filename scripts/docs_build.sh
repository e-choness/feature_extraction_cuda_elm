#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

./scripts/docs_check.sh

rm -rf docs/generated/api/reference/html site
mkdir -p docs/generated/api/reference
doxygen Doxyfile
mkdocs build --strict
