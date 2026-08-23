#!/bin/sh
# Guard the pre-commit hook's path filter against over- and under-gating.
#
# The hook runs the FULL GPU suite, so its filter decides minutes of a shared
# card per commit. It used to select on the path prefix alone, and `tools/` and
# `tests/` also hold the CLAUDE.md tree and the gate/generator scripts - so a
# Markdown edit paid an image build plus the whole suite for a result that
# cannot move. Excluding too much is the worse failure: a C++ change that skips
# the suite is gated nowhere, because CI has no GPU runner.
#
# The filter is READ OUT OF the hook rather than copied here. A guard with its
# own copy of the expression guards the copy.
#
# Usage: check_precommit_filter.sh <repo-root>
# Exit 0 = every case below lands on the expected side.

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
skip|CHANGELOG.md'

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

echo "check_precommit_filter: $(echo "$cases" | wc -l) case(s) match the pre-commit filter"
exit 0
