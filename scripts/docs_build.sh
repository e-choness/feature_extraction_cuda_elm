#!/usr/bin/env bash
# Build the documentation site: Markdown link and Mermaid checks, then VitePress.
# Requires python3 and Node.js >= 20.19 on PATH.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

./scripts/docs_check.sh

if [[ ! -d node_modules ]]; then
  npm ci
fi
npm run docs:build
