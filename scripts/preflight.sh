#!/usr/bin/env bash
# make preflight: the CPU-only gates CI runs, locally, before push. No GPU, no build.
#   static gates (scripts/ci_static_gates.sh), clang-tidy on changed TUs, git clang-format on
#   changed lines, actionlint when .github/ changed. Exit 1 if any row is FAIL.
# Host half resolves the base and starts imp:lint; `--inner` runs inside it.
# Base: IMP_GATE_BASE, else merge-base HEAD origin/main. Diff = working tree + untracked vs base,
# or the staged tree with PREFLIGHT_STAGED=1 (pre-commit hook).
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1

OUT="${PREFLIGHT_DIR:-build-preflight}"
RES="$OUT/results.tsv"   # label \t ok|FAIL|skip \t seconds \t note

changed_files() {  # files changed vs $1 in the working tree, plus untracked
    { git diff --name-only --diff-filter=d "$1"; git ls-files --others --exclude-standard; } | sort -u
}

row() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "${4:-}" >> "$RES"; }

inner() {
    local base="$1" t0 rc files cpp cu fmt
    mkdir -p "$OUT"; : > "$RES"
    mapfile -t files < <(changed_files "$base")

    echo "== static gates (scripts/ci_static_gates.sh) =="
    t0=$SECONDS
    IMP_GATE_BASE="$base" bash scripts/ci_static_gates.sh > "$OUT/static-gates.log" 2>&1; rc=$?
    if [ "$rc" -eq 0 ]; then
        row "static gates" ok $((SECONDS - t0)) "$(grep -c '^  ok ' "$OUT/static-gates.log") ok"
    else
        # A failing gate prints its diagnostics before its own `FAIL` line: keep everything but `ok`.
        grep -v '^  ok ' "$OUT/static-gates.log"
        while IFS= read -r l; do row "static: ${l#  FAIL  }" FAIL $((SECONDS - t0)); done \
            < <(grep '^  FAIL  ' "$OUT/static-gates.log")
        grep -q '^  FAIL  ' "$OUT/static-gates.log" || row "static gates" FAIL $((SECONDS - t0)) "exit $rc"
    fi

    echo "== clang-tidy (changed .cpp + src .cu host side) =="
    t0=$SECONDS
    mapfile -t cpp < <(printf '%s\n' "${files[@]}" | grep -E '^(src|tools)/.*\.cpp$')
    mapfile -t cu < <(printf '%s\n' "${files[@]}" | grep -E '^src/.*\.cu$')
    if [ $(( ${#cpp[@]} + ${#cu[@]} )) -eq 0 ]; then
        row "clang-tidy" skip 0 "no changed src/tools .cpp or src .cu"
    else
        # Same configure as CI's tidy job (cmake --preset ci); a reconfigure costs ~3 s.
        if cmake --preset ci -B "$OUT/build" \
                -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/deps/googletest \
                -DFETCHCONTENT_SOURCE_DIR_CUTLASS=/deps/cutlass \
                -DFETCHCONTENT_SOURCE_DIR_HTTPLIB=/deps/httplib \
                -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=/deps/json > "$OUT/configure.log" 2>&1 \
           && cmake -DIN=tools/imp-server/webui/index.html -DOUT="$OUT/build/generated/webui_asset.h" \
                -P cmake/embed_webui.cmake >> "$OUT/configure.log" 2>&1; then
            TIDY_BUILD="$OUT/build" bash scripts/tidy_lane.sh "${cpp[@]}" "${cu[@]}" > "$OUT/tidy.log" 2>&1; rc=$?
            if [ "$rc" -eq 0 ]; then
                row "clang-tidy" ok $((SECONDS - t0)) "$(( ${#cpp[@]} + ${#cu[@]} )) changed file(s)"
            else
                cat "$OUT/tidy.log"
                row "clang-tidy" FAIL $((SECONDS - t0)) "see $OUT/tidy.log"
            fi
        else
            tail -20 "$OUT/configure.log"
            row "clang-tidy" FAIL $((SECONDS - t0)) "cmake configure failed, see $OUT/configure.log"
        fi
    fi

    echo "== clang-format (changed lines) =="
    t0=$SECONDS
    fmt="$(git clang-format --style=file --diff "$base" 2>&1)"
    if [ -z "$fmt" ] || grep -qiE 'did not modify|no modified files' <<< "$fmt"; then
        row "clang-format" ok $((SECONDS - t0)) "changed lines clean"
    else
        printf '%s\n' "$fmt" | head -100
        row "clang-format" FAIL $((SECONDS - t0)) "fix: git clang-format --style=file $base"
    fi
}

if [ "${1:-}" = "--inner" ]; then
    inner "$2"
    exit 0
fi

# ---- host half ----
T0=$SECONDS
IMG="${PREFLIGHT_IMG:-imp:lint}"
ACTIONLINT_IMG="${ACTIONLINT_IMG:-rhysd/actionlint:1.7.12}"
BASE="${IMP_GATE_BASE:-}"
if [ -z "$BASE" ]; then
    BASE="$(git merge-base HEAD origin/main)" || { echo "preflight: no IMP_GATE_BASE and no origin/main"; exit 2; }
fi
COMMON="$(git rev-parse --path-format=absolute --git-common-dir)"

# PREFLIGHT_STAGED=1 (pre-commit): check the tree being committed, not the working tree.
# write-tree honours the hook's GIT_INDEX_FILE (commit -a, commit <paths>); the check runs in a
# throwaway worktree under $COMMON holding HEAD + that tree, removed on exit.
if [ "${PREFLIGHT_STAGED:-0}" = "1" ]; then
    TREE="$(git write-tree)" || { echo "preflight: git write-tree failed (unmerged index?)"; exit 2; }
    unset GIT_INDEX_FILE GIT_DIR GIT_WORK_TREE GIT_PREFIX PREFLIGHT_STAGED
    # 755: the actionlint image runs as user guest, mktemp's 700 hides the tree from it.
    STAGED="$(mktemp -d "$COMMON/preflight-staged.XXXXXX")" && chmod 755 "$STAGED" || exit 2
    # shellcheck disable=SC2064 # expand STAGED now
    trap "git worktree remove --force '$STAGED' >/dev/null 2>&1 || rm -rf '$STAGED'; git worktree prune" EXIT
    git worktree add -q --detach --no-checkout "$STAGED" HEAD \
        && git -C "$STAGED" read-tree -u --reset "$TREE" \
        || { echo "preflight: could not build the staged tree in $STAGED"; exit 2; }
    echo "preflight: staged tree $(git rev-parse --short "$TREE") (unstaged and untracked files ignored)"
    (cd "$STAGED" && IMP_GATE_BASE="$BASE" bash scripts/preflight.sh)
    exit $?
fi

echo "preflight: base $(git rev-parse --short "$BASE"), $(changed_files "$BASE" | wc -l) changed file(s)"

# Worktree and the main .git at their host paths: a linked worktree's .git file points there.
# Host uid/gid: no root-owned build-preflight/ or __pycache__/ left behind (#2501).
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$PWD:$PWD" -w "$PWD" -v "$COMMON:$COMMON" \
    -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0='*' \
    -e PREFLIGHT_DIR="$OUT" "$IMG" bash scripts/preflight.sh --inner "$BASE"
[ -f "$RES" ] || { echo "preflight: container produced no $RES"; exit 2; }

echo "== actionlint (.github changed) =="
t0=$SECONDS
if changed_files "$BASE" | grep -q '^\.github/'; then
    if docker run --rm -v "$PWD:/repo" -w /repo "$ACTIONLINT_IMG" > "$OUT/actionlint.log" 2>&1; then
        row "actionlint" ok $((SECONDS - t0)) "workflows clean"
    else
        cat "$OUT/actionlint.log"
        row "actionlint" FAIL $((SECONDS - t0)) "see $OUT/actionlint.log"
    fi
else
    row "actionlint" skip 0 "no .github change"
fi

echo
printf '%-58s %-5s %5s  %s\n' CHECK RESULT SEC NOTE
awk -F'\t' '{ printf "%-58s %-5s %5s  %s\n", substr($1, 1, 58), $2, $3, $4 }' "$RES"
nfail=$(awk -F'\t' '$2 == "FAIL"' "$RES" | wc -l)
echo "preflight: $nfail failure(s), $((SECONDS - T0)) s wall"
[ "$nfail" -eq 0 ]
