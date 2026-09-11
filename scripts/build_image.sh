#!/bin/bash
# build_image.sh <tag> [docker build args...]
#
# `docker build` unless <tag> already carries this exact tree. The image is
# labelled imp.tree=<fingerprint> at build time; the fingerprint is the index
# tree (`git write-tree`) plus the build arguments, and it exists only when the
# working tree IS the index: no unstaged change to a tracked file, no untracked
# file that .gitignore does not cover. Anything else builds unconditionally,
# because `COPY . .` would ship what the index does not describe.
#
# Measured cost of the rebuild this skips: ~6 min cold, 10-40 s from a warm
# ccache (2026-09-11), per `make build`, which the
# pre-commit hook (test-gpu), the pre-push hook (verify-fast) and verify-ab
# each reached for the same tree.
#
# IMP_FORCE_BUILD=1 builds regardless.
set -eu
TAG="${1:?usage: build_image.sh <tag> [docker build args...]}"
shift
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

fingerprint() {
    git diff --quiet || return 0
    [ -z "$(git ls-files --others --exclude-standard)" ] || return 0
    local tree
    tree="$(git write-tree)"
    printf '%s\n%s\n' "$tree" "$*" | sha256sum | cut -c1-16
}

FP="$(fingerprint "$@")"
if [ "${IMP_FORCE_BUILD:-0}" != "1" ] && [ -n "$FP" ]; then
    HAVE="$(docker image inspect "$TAG" --format '{{index .Config.Labels "imp.tree"}}' 2>/dev/null || true)"
    if [ "$HAVE" = "$FP" ]; then
        echo "build: $TAG already carries this tree (imp.tree=$FP), skipping docker build (IMP_FORCE_BUILD=1 overrides)"
        exit 0
    fi
fi
if [ -z "$FP" ]; then
    echo "build: working tree differs from the index (unstaged or untracked files), building without a fingerprint"
fi
exec docker build "$@" --build-arg "IMP_TREE_ID=${FP}" -t "$TAG" .
