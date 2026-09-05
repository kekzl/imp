#!/bin/sh
# Every `--gtest_filter="..."` literal in the Makefile must match at least one
# test in the binary the same line names (AUDIT_arch_2026 I-7).
#
# gtest treats a pattern that matches nothing as success: it runs zero tests
# and reports PASSED. The repo has paid for that three times (#1299 DetEval,
# #1575 the pre-commit "full suite", #1586 `AttentionTest.*` for four months)
# and built a guard per filter each time (guard_e2e_lane_split,
# guard_det_suite_filter, guard_verify_filter). The Makefile's own filters had
# none, and one was live: `test-text --gtest_filter="*Qwen38TokenizerParity*"`
# named a suite that lives in test-e2e, so `make test-gpu` ran "0 tests,
# PASSED" for it on every commit.
#
# Reads the literals OUT OF the Makefile (a copy would guard the copy), maps
# the binary named before `--gtest_filter` to the build dir, and lists tests
# with --gtest_list_tests, which runs no test body and needs no GPU. Splits
# each filter on `:` so one dead pattern inside a long filter is still caught.
#
# Usage: check_makefile_filters.sh <build-dir> [<Makefile>]
# Exit 0 = every pattern matches; 2 = usage; 1 = a dead pattern.
# Fail-open per binary: a binary this configuration did not build is skipped
# with a note, never counted as a match.
set -eu
BUILD="${1:?usage: check_makefile_filters.sh <build-dir> [Makefile]}"
MAKEFILE="${2:-$(dirname "$0")/../Makefile}"
if [ ! -f "$MAKEFILE" ]; then
    echo "check_makefile_filters: $MAKEFILE not found" >&2
    exit 2
fi

DEAD=""
CHECKED=0
SKIPPED=""
# One line per occurrence: "<binary> <filter>". The binary is the last word
# before --gtest_filter; `$(DOCKER_IMG)` precedes it on every line.
grep -E -- '--gtest_filter="[^"]+"' "$MAKEFILE" |
    sed -E 's/.*[[:space:]]([A-Za-z0-9_-]+)[[:space:]]+--gtest_filter="([^"]+)".*/\1 \2/' |
    sort -u > /tmp/imp_makefile_filters.$$
while read -r bin filter; do
    [ -z "$bin" ] && continue
    exe="$BUILD/$bin"
    if [ ! -x "$exe" ]; then
        SKIPPED="$SKIPPED $bin"
        continue
    fi
    # A binary that lists nothing at all cannot be measured (no CUDA runtime on
    # this host): skip it rather than report every pattern dead.
    TOTAL=$("$exe" --gtest_list_tests 2>/dev/null | grep -c '^  ' || true)
    if [ "$TOTAL" -eq 0 ]; then
        SKIPPED="$SKIPPED $bin(no-list)"
        continue
    fi
    for pat in $(printf '%s' "$filter" | tr ':' '\n'); do
        [ -z "$pat" ] && continue
        CHECKED=$((CHECKED + 1))
        COUNT=$("$exe" --gtest_list_tests --gtest_filter="$pat" 2>/dev/null | grep -c '^  ' || true)
        if [ "$COUNT" -eq 0 ]; then
            DEAD="$DEAD $bin:$pat"
        fi
    done
done < /tmp/imp_makefile_filters.$$
rm -f /tmp/imp_makefile_filters.$$

if [ -n "$DEAD" ]; then
    echo "check_makefile_filters: FAIL" >&2
    echo "" >&2
    echo "These Makefile --gtest_filter patterns match no test in the binary the line names:" >&2
    for d in $DEAD; do echo "  $d" >&2; done
    echo "" >&2
    echo "gtest reports PASSED for a pattern that matches nothing. Either the suite was" >&2
    echo "renamed, or the line names the wrong binary (the Qwen38 parity line named" >&2
    echo "test-text for a suite in test-e2e). Fix the Makefile line." >&2
    exit 1
fi
echo "check_makefile_filters: OK ($CHECKED pattern(s) resolve${SKIPPED:+; skipped:$SKIPPED})"
