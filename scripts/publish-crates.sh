#!/usr/bin/env bash
# Publishes every publishable workspace crate to crates.io in dependency
# order, skipping any crate whose current version is already on crates.io.
#
# - Order is derived from `cargo metadata` (a topological sort over
#   workspace path dependencies, normal + build kinds only), never
#   hardcoded. The v0.2.1 release failed partway because a hardcoded list
#   published veloxquant-runtime before veloxquant-openai, which it had
#   started depending on.
# - Crates with `publish = false` (veloxquant-cli) are excluded.
# - Idempotent: a re-run after a partial failure picks up where it left
#   off, and independently-versioned crates (veloxquant-rig) are skipped on
#   workspace releases that don't bump them.
# - Each crate is verified (`cargo publish` without `--no-verify`), and
#   cargo (>= 1.66) waits for each upload to reach the index before
#   returning, so no fixed `sleep` is needed between crates.
#
# Usage:
#   scripts/publish-crates.sh            # publish (needs CARGO_REGISTRY_TOKEN)
#   scripts/publish-crates.sh --plan     # print the order + skip decisions only
set -euo pipefail

mode="publish"
case "${1:-}" in
    "") ;;
    --plan) mode="plan" ;;
    *)
        echo "usage: $0 [--plan]" >&2
        exit 1
        ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Prints "name version" lines for publishable crates, dependencies first.
publish_order() {
    cargo metadata --no-deps --format-version 1 | python3 -c '
import json, sys

meta = json.load(sys.stdin)
members = set(meta["workspace_members"])
pkgs = {p["name"]: p for p in meta["packages"] if p["id"] in members}

def publishable(p):
    # `publish = false` shows up as an empty registry list.
    return p.get("publish") is None or len(p["publish"]) > 0

deps = {
    name: sorted(
        d["name"]
        for d in p["dependencies"]
        if d.get("path") and d["kind"] in (None, "build") and d["name"] in pkgs
    )
    for name, p in pkgs.items()
}

order, state = [], {}
def visit(name, stack=()):
    if state.get(name) == "done":
        return
    if state.get(name) == "visiting":
        sys.exit("dependency cycle: " + " -> ".join(stack + (name,)))
    state[name] = "visiting"
    for dep in deps[name]:
        visit(dep, stack + (name,))
    state[name] = "done"
    order.append(name)

for name in sorted(pkgs):
    visit(name)

for name in order:
    p = pkgs[name]
    if not publishable(p):
        continue
    bad = [d for d in deps[name] if not publishable(pkgs[d])]
    if bad:
        sys.exit(f"{name} depends on unpublishable crate(s): {bad}")
    print(name, p["version"])
'
}

# Succeeds if crates.io already has this exact crate version.
already_published() {
    local name="$1" version="$2" status
    status="$(curl -s -o /dev/null -w '%{http_code}' \
        -A "veloxquant-rs release script (https://github.com/rajveer43/veloxquant-rs)" \
        "https://crates.io/api/v1/crates/${name}/${version}")"
    case "$status" in
        200) return 0 ;;
        404) return 1 ;;
        *)
            echo "error: crates.io returned HTTP $status for ${name} ${version}" >&2
            exit 1
            ;;
    esac
}

order="$(publish_order)"
if [[ -z "$order" ]]; then
    echo "error: no publishable crates found" >&2
    exit 1
fi

if [[ "$mode" == "publish" && -z "${CARGO_REGISTRY_TOKEN:-}" ]]; then
    echo "error: CARGO_REGISTRY_TOKEN is not set" >&2
    exit 1
fi

published=0
skipped=0
while read -r name version; do
    if already_published "$name" "$version"; then
        echo "skip     $name $version (already on crates.io)"
        skipped=$((skipped + 1))
        continue
    fi
    if [[ "$mode" == "plan" ]]; then
        echo "publish  $name $version"
        continue
    fi
    echo "publish  $name $version"
    cargo publish -p "$name" --locked
    published=$((published + 1))
done <<< "$order"

if [[ "$mode" == "publish" ]]; then
    echo "done: $published published, $skipped already on crates.io"
fi
