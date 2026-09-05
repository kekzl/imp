#!/bin/bash
# ab_base_image.sh - build the base arm of the paired perf gate.
#
# <ref> (default origin/main) is checked out into a throwaway git worktree and
# built with the same Dockerfile arguments `make build` uses, tagged
# imp:ab-<sha8> and imp:ab-base. The tag is reused when it exists, so one main
# sha costs one 3.5-minute build however many pushes are gated against it.
# The worktree's own scripts/dep_build_args.sh supplies the dependency pins, so
# the base arm is built with ITS pins, not this tree's.
#
# Usage: scripts/ab_base_image.sh [ref]      # prints the tag on the last line
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
