#!/bin/sh
# Guard the pre-push gate's gtest filter against silent suite-rename drift
# (#1586).
#
# scripts/verify.sh carries a 12-pattern gtest filter for the fast lane. It is
# the only place several of those suites ever run: VramBudget* and
# ForwardPassTest.* are SKIP_IF_NO_CUDA, so CI structurally cannot execute
# them, and DecodeLogitsInvariantToBatchComposition is the tree's only assert
# that a sequence's logits do not depend on its batch neighbours (#1314, the
# class #1044/#1045 came from).
#
# A pattern that matches nothing is not an error in gtest. It runs zero tests
# and reports success, so renaming a suite out of the filter silently deletes
# the only coverage it had. Two sibling filters are already guarded
# (guard_e2e_lane_split, guard_det_suite_filter); this one was not.
#
# The filter is READ OUT OF verify.sh rather than copied here. A guard with its
# own copy of the string guards the copy.
#
# Usage: check_verify_filter.sh <path-to-imp-tests> [<path-to-verify.sh>]
# Exit 0 = every pattern matches at least one test.

set -eu

BIN="${1:?usage: check_verify_filter.sh <imp-tests> [verify.sh]}"
VERIFY="${2:-$(dirname "$0")/verify.sh}"

if [ ! -x "$BIN" ]; then
    echo "check_verify_filter: $BIN not executable — skipping" >&2
    exit 0   # fail-open: the GPU test binary is not built in every configuration
fi
if [ ! -f "$VERIFY" ]; then
    echo "check_verify_filter: $VERIFY not found" >&2
    exit 2
fi

FILTER=$(grep -m1 '^ *FILTER="' "$VERIFY" | sed 's/^ *FILTER="//; s/"$//')
if [ -z "$FILTER" ]; then
    echo "check_verify_filter: no FILTER= line in $VERIFY — the gate lost its filter" >&2
    exit 1
fi

# Can the binary run at all? On a host without the CUDA runtime it exits before
# main and prints nothing, and then EVERY pattern looks empty. A guard that
# reports twelve findings when it in fact could not measure is worse than one
# that stays quiet: it teaches the reader to ignore it.
TOTAL=$("$BIN" --gtest_list_tests 2>/dev/null | grep -c '^  ' || true)
if [ "$TOTAL" -eq 0 ]; then
    echo "check_verify_filter: SKIP ($BIN lists no tests at all — no CUDA runtime here?)"
    exit 0
fi

EMPTY=""
PATTERNS=$(printf '%s' "$FILTER" | tr ':' '\n')
for pat in $PATTERNS; do
    [ -z "$pat" ] && continue
    # --gtest_list_tests runs no test body, so this needs no GPU.
    COUNT=$("$BIN" --gtest_list_tests --gtest_filter="$pat" 2>/dev/null |
            grep -c '^  ' || true)
    if [ "$COUNT" -eq 0 ]; then
        EMPTY="$EMPTY $pat"
    fi
done

if [ -n "$EMPTY" ]; then
    echo "check_verify_filter: FAIL" >&2
    echo "" >&2
    echo "These patterns in the pre-push filter match no test:" >&2
    for pat in $EMPTY; do echo "  $pat" >&2; done
    echo "" >&2
    echo "gtest reports success for a filter that matches nothing, so the suite" >&2
    echo "was not skipped - it was silently dropped from the only gate that runs" >&2
    echo "it. Fix the pattern in scripts/verify.sh, or delete it deliberately." >&2
    exit 1
fi

echo "check_verify_filter: PASS ($(printf '%s' "$PATTERNS" | grep -c .) patterns, all non-empty)"
