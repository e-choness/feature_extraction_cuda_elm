#!/usr/bin/env bash
# Print the CHANGELOG.md section for one version (the lines under "## [X.Y.Z]"), failing if it is
# missing or empty. Used by .github/workflows/release.yml for the GitHub release notes.
#
#   scripts/release_notes.sh 0.2.1
set -euo pipefail

version="${1:?usage: $0 X.Y.Z}"
changelog="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/CHANGELOG.md"

notes="$(awk -v h="## [$version]" '/^## \[/{p = (index($0, h) == 1); next} p' "$changelog")"
if [[ -z "${notes//[[:space:]]/}" ]]; then
  echo "error: CHANGELOG.md has no '## [$version]' section (cut releases with scripts/release.sh)" >&2
  exit 1
fi
printf '%s\n' "$notes"
