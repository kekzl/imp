#!/bin/bash
# build_image.sh <tag> [docker build args...]: docker build unless <tag> already carries this
# exact tree and these build args. Labels: imp.tree (scripts/image_tag.sh tree, dirty trees
# included), imp.build (sha of the args), imp.worktree (toplevel). IMP_FORCE_BUILD=1 builds anyway.
set -eu
TAG="${1:?usage: build_image.sh <tag> [docker build args...]}"
shift
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

TREE="$(bash "$(dirname "$(readlink -f "$0")")/image_tag.sh" tree)"
ARGS_ID="$(printf '%s\n' "$*" | sha256sum | cut -c1-16)"
if [ "${IMP_FORCE_BUILD:-0}" != "1" ]; then
    HAVE="$(docker image inspect "$TAG" \
        --format '{{index .Config.Labels "imp.tree"}} {{index .Config.Labels "imp.build"}}' 2>/dev/null || true)"
    if [ "$HAVE" = "$TREE $ARGS_ID" ]; then
        echo "build: $TAG already carries this tree (imp.tree=$TREE), skipping docker build (IMP_FORCE_BUILD=1 overrides)"
        exit 0
    fi
fi
echo "build: $TAG from $ROOT (imp.tree=$TREE)"
exec docker build "$@" --build-arg "IMP_TREE_ID=${TREE}" \
    --label "imp.tree=${TREE}" --label "imp.build=${ARGS_ID}" --label "imp.worktree=${ROOT}" \
    -t "$TAG" .
