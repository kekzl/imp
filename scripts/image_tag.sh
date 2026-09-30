#!/bin/bash
# Per-worktree test image: the main checkout keeps imp:test, a linked worktree gets
# imp:test-<dir>-<sha256(toplevel)[:8]>, so two worktrees never build into one tag.
# Usage:
#   image_tag.sh [tag]      print the tag ($DOCKER_IMG wins when set)
#   image_tag.sh tree       print the tree id of the working tree (index + unstaged + untracked)
#   image_tag.sh check IMG  exit 1 unless IMG's label imp.tree equals `tree`
#                           (IMP_ALLOW_FOREIGN_IMAGE=1 overrides)
set -uo pipefail

derive_tag() {
    local top gd cd base
    top="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo "imp:test"; return; }
    gd="$(git rev-parse --absolute-git-dir)"
    cd="$(cd "$(git rev-parse --git-common-dir)" && pwd -P)"
    if [ "$(cd "$gd" && pwd -P)" = "$cd" ]; then echo "imp:test"; return; fi
    base="$(printf '%s' "${top##*/}" | tr -c 'A-Za-z0-9_.-' '-' | cut -c1-40)"
    echo "imp:test-${base}-$(printf '%s' "$top" | sha256sum | cut -c1-8)"
}

# Tree id of what `docker build .` sees: a scratch copy of the index plus every non-ignored
# change. Equals `git write-tree` on a clean tree; the real index is never touched.
tree_id() {
    local top idx rc
    top="$(git rev-parse --show-toplevel 2>/dev/null)" || return 1
    idx="$(mktemp)"
    # -p keeps the index mtime: git's racy check hashes entries stat'd in the index's own second.
    # A fresh mtime hid a same-second, same-size edit (#2287).
    cp -p "$(git -C "$top" rev-parse --path-format=absolute --git-path index)" "$idx" 2>/dev/null || : > "$idx"
    GIT_INDEX_FILE="$idx" git -C "$top" add -A -- . >/dev/null 2>&1 \
        && GIT_INDEX_FILE="$idx" git -C "$top" write-tree
    rc=$?
    rm -f "$idx"
    return $rc
}

label_of() {  # label_of <image> <label>
    docker image inspect "$1" --format "{{index .Config.Labels \"$2\"}}" 2>/dev/null
}

check_image() {
    local img="$1" have want from
    if ! docker image inspect "$img" >/dev/null 2>&1; then
        echo "image check: $img does not exist; run 'make build'" >&2
        return 1
    fi
    have="$(label_of "$img" imp.tree)"
    want="$(tree_id)" || { echo "image check: not a git tree, skipped" >&2; return 0; }
    [ "$have" = "$want" ] && return 0
    from="$(label_of "$img" imp.worktree)"
    if [ "${IMP_ALLOW_FOREIGN_IMAGE:-0}" = "1" ]; then
        echo "image check: $img imp.tree='$have' != this tree '$want', allowed by IMP_ALLOW_FOREIGN_IMAGE=1" >&2
        return 0
    fi
    echo "image check: refusing to run $img: it was not built from this tree." >&2
    echo "  image imp.tree='${have}' imp.worktree='${from}'" >&2
    echo "  this  tree=${want} worktree=$(git rev-parse --show-toplevel)" >&2
    echo "  Rebuild with 'make build', or override with IMP_ALLOW_FOREIGN_IMAGE=1." >&2
    return 1
}

case "${1:-tag}" in
    tag) if [ -n "${DOCKER_IMG:-}" ]; then echo "$DOCKER_IMG"; else derive_tag; fi ;;
    tree) tree_id ;;
    check) check_image "${2:?usage: image_tag.sh check <image>}" ;;
    *) echo "usage: image_tag.sh [tag|tree|check <image>]" >&2; exit 2 ;;
esac
