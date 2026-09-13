#!/bin/sh
# Guards the pre-commit hook's path filter against over/under-gating: a path-prefix-only
# filter used to run the full GPU suite on a Markdown edit under tools/ or tests/ (which also
# hold CLAUDE.md and gate scripts). Under-gating is worse: a C++ change skipping the suite is
# gated nowhere (CI has no GPU). Filter is READ OUT OF the hook, not copied; also guards
# pre-push's same filter.

set -eu

ROOT="${1:?usage: check_precommit_filter.sh <repo-root>}"
HOOK="$ROOT/scripts/pre-commit.hook"

if [ ! -f "$HOOK" ]; then
    echo "check_precommit_filter: $HOOK not found" >&2
    exit 2
fi

# The two grep expressions the hook chains, pulled from the hook itself.
KEEP=$(grep -m1 -- "| grep -E '\^(src/" "$HOOK" | sed "s/.*grep -E '//; s/' *\\\\*$//")
DROP=$(grep -m1 -- "| grep -vE" "$HOOK" | sed "s/.*grep -vE '//; s/' *\\\\*$//")

if [ -z "$KEEP" ] || [ -z "$DROP" ]; then
    echo "check_precommit_filter: could not read both filter expressions out of $HOOK" >&2
    echo "  keep='$KEEP' drop='$DROP'" >&2
    exit 1
fi

# Left column: does the hook run the GPU suite for this staged set?
#   gate = must run it. skip = must not.
cases='gate|src/model/jinja.cpp
gate|include/core/logging.h
gate|tools/imp-server/handlers_chat.cpp
gate|tests/test_jinja.cpp
gate|CMakeLists.txt
gate|cmake/imp-deps.cmake
gate|tools/imp-server/webui/index.html
gate|tests/refs/ppl_corpus.txt
gate|tests/refs/chat_template_goldens.h
gate|tests/CLAUDE.md src/model/jinja.cpp
skip|tests/CLAUDE.md
skip|tools/imp-server/CLAUDE.md
skip|src/compute/CLAUDE.md
skip|tools/kernel_resources.py
skip|tests/refs/gen_chat_goldens.py
skip|tools/kernel_resources.py tests/CLAUDE.md
skip|README.md
skip|docs/internals/SM120.md
skip|CHANGELOG.md
skip|tools/mutation/catalogue.json
skip|tools/mutation/run.py
gate|tools/mutation/catalogue.json src/model/jinja.cpp'

# The loop runs in a subshell, so it cannot set a variable the parent reads:
# it prints one line per mismatch and the count IS the verdict.
bad=$(echo "$cases" | while IFS='|' read -r want files; do
    [ -z "$want" ] && continue
    got=$(printf '%s\n' $files | grep -E "$KEEP" | grep -vE "$DROP" || true)
    if [ -n "$got" ]; then have=gate; else have=skip; fi
    if [ "$have" != "$want" ]; then
        echo "check_precommit_filter: '$files' -> $have, expected $want" >&2
        echo x
    fi
done | wc -l)

if [ "$bad" -ne 0 ]; then
    echo "check_precommit_filter: $bad case(s) on the wrong side of the pre-commit filter" >&2
    exit 1
fi

# --- the pre-push hook strips the same extensions -------------------------
PUSH="$ROOT/scripts/pre-push.hook"
if [ -f "$PUSH" ]; then
    PDROP=$(grep -m1 -- "grep -vE '\\\\." "$PUSH" | sed "s/.*grep -vE '//; s/' .*//")
    if [ -z "$PDROP" ]; then
        echo "check_precommit_filter: could not read the extension filter out of $PUSH" >&2
        exit 1
    fi
    for f in tests/CLAUDE.md tools/kernel_resources.py tests/api/test_chat.py \
             tools/mutation/catalogue.json; do
        if printf '%s\n' "$f" | grep -qE "$PDROP"; then :; else
            echo "check_precommit_filter: pre-push would gate '$f' (docs/scripts cannot move a number)" >&2
            exit 1
        fi
    done
    for f in src/model/jinja.cpp tools/imp-server/webui/index.html; do
        if printf '%s\n' "$f" | grep -qE "$PDROP"; then
            echo "check_precommit_filter: pre-push would SKIP '$f' — that is buildable source" >&2
            exit 1
        fi
    done
fi

# --- scripts/*.sh: only the ones verify.sh can reach may gate ---------------
# The hook delegates that question to verify_touches.sh; this checks both sides
# of it against the tree rather than against a copied list.
TOUCHES="$ROOT/scripts/verify_touches.sh"
if [ -x "$TOUCHES" ]; then
    for f in scripts/verify.sh scripts/check_det_suite_filter.sh \
             scripts/check_e2e_lane_split.sh scripts/smoke_test.sh; do
        if ! "$TOUCHES" "$f" >/dev/null 2>&1; then
            echo "check_precommit_filter: pre-push would SKIP '$f' - verify.sh or a" >&2
            echo "  registered ctest runs it, so an edit there can change the gate's own outcome" >&2
            exit 1
        fi
    done
    for f in scripts/ci_static_gates.sh scripts/check-release.sh scripts/bench_gate.sh; do
        if "$TOUCHES" "$f" >/dev/null 2>&1; then
            echo "check_precommit_filter: pre-push would gate '$f' on the GPU, but verify.sh" >&2
            echo "  never runs it. Either a call was added (then say so here) or the test is wrong." >&2
            exit 1
        fi
    done
fi

# make install-hooks copies scripts/pre-commit.hook into .git/hooks/; editing the repo copy
# changes nothing until that reruns (#1723 filter stayed stale locally after merge).
# Absent (fresh clone, CI) is fine; present and different is not.
for h in pre-commit pre-push; do
    INSTALLED="$ROOT/.git/hooks/$h"
    [ -f "$INSTALLED" ] || continue
    if ! cmp -s "$INSTALLED" "$ROOT/scripts/$h.hook"; then
        echo "check_precommit_filter: .git/hooks/$h differs from scripts/$h.hook — what runs is" >&2
        echo "  not what this tree says. Either the installed copy predates a merged hook change" >&2
        echo "  (the case this exists for: #1723 shipped and kept gating .py locally), or you are" >&2
        echo "  on a branch whose hook edit is not installed. Both resolve the same way:" >&2
        echo "    make install-hooks" >&2
        exit 1
    fi
done

echo "check_precommit_filter: $(echo "$cases" | wc -l) pre-commit case(s), 5 pre-push case(s), 7 scripts/*.sh case(s), installed hooks current"
exit 0
