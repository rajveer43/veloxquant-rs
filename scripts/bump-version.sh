#!/usr/bin/env bash
# Bumps workspace.package.version in the root Cargo.toml, refreshes
# Cargo.lock, and moves the CHANGELOG's [Unreleased] section under the new
# version heading.
#
# Usage: scripts/bump-version.sh <new-version>
# Example: scripts/bump-version.sh 0.2.0
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <new-version>" >&2
    exit 1
fi

new_version="$1"
if [[ ! "$new_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "error: '$new_version' is not a valid semver x.y.z version" >&2
    exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

current_version="$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)"
if [[ -z "$current_version" ]]; then
    echo "error: could not find workspace.package.version in Cargo.toml" >&2
    exit 1
fi

if [[ "$current_version" == "$new_version" ]]; then
    echo "error: new version ($new_version) matches current version" >&2
    exit 1
fi

echo "bumping workspace version: $current_version -> $new_version"

# Bump the workspace package version (the single `version = "..."` line
# under [workspace.package] in the root Cargo.toml), and every
# workspace-internal path dependency's pinned `version = "..."` across all
# crate manifests (each `veloxquant-* = { path = ..., version = "..." }`
# line, whether in [workspace.dependencies] or a crate's own [dependencies]).
sed -i.bak -E "s/^version = \"$current_version\"\$/version = \"$new_version\"/" Cargo.toml
rm -f Cargo.toml.bak

while IFS= read -r -d '' manifest; do
    sed -i.bak -E "s/(veloxquant[a-z-]* = \{ path = \"[^\"]+\", version = )\"$current_version\"/\1\"$new_version\"/g" "$manifest"
    rm -f "$manifest.bak"
done < <(find . -name Cargo.toml -not -path "./target/*" -print0)

# Update the changelog: move [Unreleased] content under a new version
# heading, and leave a fresh empty [Unreleased] section above it.
release_date="$(date +%Y-%m-%d)"
if grep -q '^## \[Unreleased\]$' CHANGELOG.md; then
    awk -v version="$new_version" -v date="$release_date" '
        /^## \[Unreleased\]$/ {
            print
            print ""
            print "## [" version "] - " date
            found=1
            next
        }
        { print }
    ' CHANGELOG.md > CHANGELOG.md.new
    mv CHANGELOG.md.new CHANGELOG.md
else
    echo "warning: no [Unreleased] section found in CHANGELOG.md; skipping changelog update" >&2
fi

cargo update --workspace --offline 2>/dev/null || cargo update --workspace

echo "done. review the diff, then commit and tag:"
echo "  git add -A && git commit -m \"release: v$new_version\""
echo "  git tag v$new_version"
echo "  git push && git push --tags"
