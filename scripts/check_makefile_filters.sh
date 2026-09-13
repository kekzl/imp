#!/bin/sh
# AUDIT_arch_2026 I-7: every --gtest_filter literal in the Makefile must match >=1 test in
# the binary named on the same line. gtest treats a matchless pattern as success (0 tests,
# PASSED); this repo paid for that three times (#1299, #1575, #1586).
# Reads literals out of the Makefile, lists tests via --gtest_list_tests (no GPU needed).
# Fail-open per binary a given configuration did not build.
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
