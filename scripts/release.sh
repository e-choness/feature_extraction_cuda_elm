#!/usr/bin/env bash
# Cut a release: bump the version, date the changelog, commit, tag and (optionally) push.
#
#   scripts/release.sh 0.2.1          # prepare the commit and annotated tag locally
#   scripts/release.sh 0.2.1 --push   # ...and push master and the tag together
#
# What it changes:
#   - CMakeLists.txt         project(VERSION ...)
#   - src/core/version.hpp   kVersion and kVersionText
#   - docs/deployment.md     pinned image tags (<flavor>-X.Y.Z)
#   - CHANGELOG.md           "## [Unreleased]" entries move under "## [X.Y.Z] - <today>"
#
# Pushing the tag runs .github/workflows/release.yml, which publishes the images and a GitHub
# release whose notes are the CHANGELOG section for that version.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

die() { echo "error: $*" >&2; exit 1; }

version="${1:-}"
push=false
[[ "${2:-}" == "--push" ]] && push=true
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "usage: $0 X.Y.Z [--push]"
IFS=. read -r major minor patch <<<"$version"
tag="v$version"

branch="$(git rev-parse --abbrev-ref HEAD)"
[[ "$branch" == "master" ]] || die "releases are cut from master (on '$branch')"
[[ -z "$(git status --porcelain)" ]] || die "working tree is not clean"
git rev-parse -q --verify "refs/tags/$tag" >/dev/null && die "tag $tag already exists (git tag -d $tag if it was never pushed)"

current="$(sed -nE 's/^  VERSION ([0-9]+\.[0-9]+\.[0-9]+)$/\1/p' CMakeLists.txt)"
[[ -n "$current" ]] || die "could not read the project version from CMakeLists.txt"
[[ "$current" != "$version" ]] || die "version is already $version"
newest="$(printf '%s\n%s\n' "$current" "$version" | sort -t. -k1,1n -k2,2n -k3,3n | tail -n1)"
[[ "$newest" == "$version" ]] || die "$version is older than the current version $current"

# The Unreleased section must have entries, or the release notes would be empty.
unreleased="$(awk '/^## \[/{p = ($0 == "## [Unreleased]"); next} p && NF' CHANGELOG.md)"
[[ -n "$unreleased" ]] || die "CHANGELOG.md has nothing under ## [Unreleased]"

# rewrite FILE AWK_PROGRAM [awk args...]: apply an awk program to a file in place.
rewrite() {
  local file="$1"; shift
  local tmp; tmp="$(mktemp)"
  awk "$@" "$file" >"$tmp" && cat "$tmp" >"$file" && rm -f "$tmp"
}

rewrite CMakeLists.txt -v v="$version" '/^  VERSION [0-9.]+$/ && !done {print "  VERSION " v; done = 1; next} {print}'
rewrite src/core/version.hpp -v v="$version" -v a="$major" -v b="$minor" -v c="$patch" '
  /^inline constexpr Version kVersion\{/ {print "inline constexpr Version kVersion{" a ", " b ", " c "};"; next}
  /^inline constexpr std::string_view kVersionText\{/ {print "inline constexpr std::string_view kVersionText{\"" v "\"};"; next}
  {print}'
rewrite docs/deployment.md -v old="$current" -v new="$version" '{
  out = ""; s = $0
  while ((i = index(s, "-" old)) > 0) {
    pre = substr(s, 1, i - 1)
    out = out pre "-" ((pre ~ /(cpu|gpu)$/) ? new : old)
    s = substr(s, i + length(old) + 1)
  }
  print out s
}'
rewrite CHANGELOG.md -v v="$version" -v d="$(date +%Y-%m-%d)" '
  $0 == "## [Unreleased]" {print; print ""; print "## [" v "] - " d; next} {print}'

grep -q "kVersionText{\"$version\"}" src/core/version.hpp || die "failed to update src/core/version.hpp"

git add CMakeLists.txt src/core/version.hpp docs/deployment.md CHANGELOG.md
git commit -q -m "Release $tag"
git tag -a "$tag" -m "Release $tag"
echo "Created commit $(git rev-parse --short HEAD) and tag $tag."

if $push; then
  git push --atomic origin master "$tag"
else
  echo "Review it, then publish with:  git push --atomic origin master $tag"
fi
