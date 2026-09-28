#!/usr/bin/env bash
# shellcheck disable=SC2016,SC2034 # check() evals its condition later; vars are read there
# scripts/image_tag.sh and scripts/build_image.sh against throwaway git repos and a docker stub
# on PATH (no Docker daemon, no GPU). Wired into scripts/ci_static_gates.sh (group gpulock).
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
TAG_SH="$PWD/scripts/image_tag.sh"
BUILD_SH="$PWD/scripts/build_image.sh"

T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
PASS=0 FAIL=0
ok()  { PASS=$((PASS + 1)); echo "  ok    $1"; }
bad() { FAIL=$((FAIL + 1)); echo "  FAIL  $1"; }
check() { if eval "$2"; then ok "$1"; else bad "$1"; fi; }

# docker stub: labels of image X live in $STUB/X.labels as "key=value" lines; a build records
# its argv and writes the --label values as the new image's labels.
STUB="$T/stub"; mkdir -p "$STUB/bin"
cat > "$STUB/bin/docker" <<'EOF'
#!/usr/bin/env bash
db="$(dirname "$0")/.."
key() { printf '%s' "$1" | tr ':/' '__'; }
if [ "$1 $2" = "image inspect" ]; then
    f="$db/$(key "$3").labels"; [ -f "$f" ] || exit 1
    fmt="${5:-}"; out="$fmt"
    for k in imp.tree imp.build imp.worktree; do
        v="$(sed -n "s/^$k=//p" "$f")"
        out="${out//\{\{index .Config.Labels \"$k\"\}\}/$v}"
    done
    printf '%s\n' "$out"; exit 0
fi
if [ "$1" = "build" ]; then
    shift; echo "$*" >> "$db/build.log"; tag="" labels=""
    while [ $# -gt 0 ]; do
        case "$1" in -t) tag="$2"; shift ;; --label) labels="$labels$2"$'\n' ;; esac; shift
    done
    printf '%s' "$labels" > "$db/$(key "$tag").labels"; exit 0
fi
exit 1
EOF
chmod +x "$STUB/bin/docker"
export PATH="$STUB/bin:$PATH"
unset DOCKER_IMG IMP_ALLOW_FOREIGN_IMAGE
g() { git -c user.name=t -c user.email=t@t -c init.defaultBranch=main "$@"; }

g init -q "$T/imp" && echo a > "$T/imp/a.txt" && g -C "$T/imp" add a.txt && g -C "$T/imp" commit -qm init
g -C "$T/imp" worktree add -q "$T/imp-wt-one" -b one 2>/dev/null
g -C "$T/imp" worktree add -q "$T/imp-wt-two" -b two 2>/dev/null

echo "== tag derivation =="
t_main=$(cd "$T/imp" && bash "$TAG_SH")
t_one=$(cd "$T/imp-wt-one" && bash "$TAG_SH")
t_two=$(cd "$T/imp-wt-two" && bash "$TAG_SH")
t_sub=$(cd "$T/imp-wt-one" && mkdir -p sub && cd sub && bash "$TAG_SH")
echo "  main=$t_main one=$t_one two=$t_two"
check "main checkout -> imp:test" '[ "$t_main" = imp:test ]'
check "worktree one -> imp:test-imp-wt-one-<hash8>" '[[ "$t_one" =~ ^imp:test-imp-wt-one-[0-9a-f]{8}$ ]]'
check "two worktrees -> two tags" '[ "$t_one" != "$t_two" ] && [ "$t_two" != imp:test ]'
check "same worktree from a subdirectory -> same tag" '[ "$t_sub" = "$t_one" ]'
check "DOCKER_IMG wins" '[ "$(cd "$T/imp-wt-one" && DOCKER_IMG=imp:mine bash "$TAG_SH")" = imp:mine ]'

echo "== tree id =="
clean=$(cd "$T/imp-wt-one" && git write-tree)
check "clean tree: tree id == git write-tree" '[ "$(cd "$T/imp-wt-one" && bash "$TAG_SH" tree)" = "$clean" ]'
echo b > "$T/imp-wt-one/new.txt"
dirty=$(cd "$T/imp-wt-one" && bash "$TAG_SH" tree)
check "untracked file changes the tree id" '[ "$dirty" != "$clean" ]'
check "the real index is untouched" '[ -z "$(git -C "$T/imp-wt-one" diff --cached --name-only)" ]'
rm "$T/imp-wt-one/new.txt"; rmdir "$T/imp-wt-one/sub"

echo "== build labels + skip =="
( cd "$T/imp-wt-one" && bash "$BUILD_SH" "$t_one" --build-arg X=1 >/dev/null )
lab="$STUB/$(printf '%s' "$t_one" | tr ':/' '__').labels"
check "build labels imp.tree=<tree>" 'grep -qx "imp.tree=$clean" "$lab"'
check "build labels imp.worktree=<toplevel>" 'grep -qx "imp.worktree=$T/imp-wt-one" "$lab"'
check "build passes IMP_TREE_ID=<tree>" 'grep -q -- "--build-arg IMP_TREE_ID=$clean" "$STUB/build.log"'
out=$(cd "$T/imp-wt-one" && bash "$BUILD_SH" "$t_one" --build-arg X=1)
check "same tree + args: build skipped" 'grep -q "skipping docker build" <<<"$out" && [ "$(wc -l < "$STUB/build.log")" = 1 ]'
( cd "$T/imp-wt-one" && bash "$BUILD_SH" "$t_one" --build-arg X=2 >/dev/null )
check "changed build args: rebuilt" '[ "$(wc -l < "$STUB/build.log")" = 2 ]'

echo "== runner refuses a foreign image =="
check "image built from this tree: accepted" '(cd "$T/imp-wt-one" && bash "$TAG_SH" check "$t_one")'
check "worktree two, byte-identical tree: accepted (label is content, not path)" \
    '(cd "$T/imp-wt-two" && bash "$TAG_SH" check "$t_one")'
echo x > "$T/imp-wt-two/b.txt"
out=$(cd "$T/imp-wt-two" && bash "$TAG_SH" check "$t_one" 2>&1); rc=$?
while IFS= read -r l; do echo "  | $l"; done <<<"$out"
check "image from worktree one, run in worktree two (other tree): refused (exit $rc)" \
    '[ "$rc" = 1 ] && grep -q "refusing to run" <<<"$out" && grep -q "imp.worktree=.$T/imp-wt-one" <<<"$out"'
printf 'imp.tree=\n' > "$STUB/imp_test.labels"
out=$(cd "$T/imp" && bash "$TAG_SH" check imp:test 2>&1); rc=$?
check "empty imp.tree label: refused (exit $rc)" '[ "$rc" = 1 ] && grep -q "refusing to run" <<<"$out"'
check "IMP_ALLOW_FOREIGN_IMAGE=1 overrides" '(cd "$T/imp" && IMP_ALLOW_FOREIGN_IMAGE=1 bash "$TAG_SH" check imp:test 2>/dev/null)'
echo y > "$T/imp-wt-one/a.txt"
out=$(cd "$T/imp-wt-one" && bash "$TAG_SH" check "$t_one" 2>&1); rc=$?
check "same worktree edited after build: refused (exit $rc)" '[ "$rc" = 1 ]'
out=$(cd "$T/imp" && bash "$TAG_SH" check imp:none 2>&1); rc=$?
check "missing image: refused (exit $rc)" '[ "$rc" = 1 ] && grep -q "does not exist" <<<"$out"'

echo "image_tag: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
