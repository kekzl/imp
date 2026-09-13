#!/bin/bash
# ab_base_image.sh [ref] (default origin/main): builds the base arm of the paired perf gate
# in a throwaway worktree, tagged imp:ab-<sha8>/imp:ab-base (reused if it exists, one build per sha).
# Uses the worktree's own scripts/dep_build_args.sh pins, not this tree's. Prints the tag on the last line.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
REF="${1:-origin/main}"
SHA="$(git rev-parse --short=8 "$REF")"
TAG="imp:ab-$SHA"

if docker image inspect "$TAG" >/dev/null 2>&1; then
    echo "ab-base: $TAG exists, reused ($REF)" >&2
    docker tag "$TAG" imp:ab-base
    echo "$TAG"
    exit 0
fi

WT="$(mktemp -d "${TMPDIR:-/tmp}/imp-ab-XXXXXX")"
cleanup() {
    git worktree remove --force "$WT" >/dev/null 2>&1 || true
    rm -rf "$WT"
}
trap cleanup EXIT
git worktree add --detach --quiet "$WT" "$SHA"

DEP_ARGS="$(bash "$WT/scripts/dep_build_args.sh")"
LOG="${TMPDIR:-/tmp}/ab_base_build_${SHA}.log"
echo "ab-base: building $REF ($SHA) -> $TAG (log: $LOG)" >&2
# shellcheck disable=SC2086
if ! docker build --build-arg IMP_BUILD_TESTS=ON $DEP_ARGS -t "$TAG" "$WT" >"$LOG" 2>&1; then
    echo "ab-base: build of $SHA failed:" >&2
    tail -20 "$LOG" >&2
    exit 1
fi
docker tag "$TAG" imp:ab-base
echo "$TAG"
