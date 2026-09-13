#!/bin/sh
# DetEvalE2ETest is value-parameterised over model env vars (#1299): tests are named
# Models/DetEvalE2ETest.<Case>/<label>, not DetEvalE2ETest.<Case>, so a naive filter matches
# zero tests and gtest reports PASSED. --gtest_list_tests needs no GPU; CPU-lane check.

set -eu

BIN="${1:?usage: check_det_suite_filter.sh <test-e2e> <filter>}"
FILTER="${2:?missing filter}"

if [ ! -x "$BIN" ]; then
    echo "check_det_suite_filter: $BIN not executable" >&2
    exit 2
fi

# Model rows that must be present. Keep in sync with det_models() in
# tests/test_determinism_e2e.cpp.
EXPECTED_LABELS="moe dense"

LISTED=$("$BIN" --gtest_filter="$FILTER" --gtest_list_tests 2>/dev/null | awk '
    /^[^[:space:]].*\.$/ { fixture=$1; next }
    /^[[:space:]]+[A-Za-z]/ { name=$1; sub(/#.*/, "", name); gsub(/[[:space:]]/, "", name); if (name != "") print fixture name }
')

COUNT=$(printf '%s\n' "$LISTED" | grep -c . || true)
if [ "$COUNT" -eq 0 ]; then
    echo "check_det_suite_filter: FAIL — filter '$FILTER' matches no test." >&2
    echo "gtest reports '[  PASSED  ] 0 tests' for that, which looks green and is not." >&2
    echo "DetEvalE2ETest is a TEST_P suite: its names are Models/DetEvalE2ETest.<Case>/<label>." >&2
    exit 1
fi

for label in $EXPECTED_LABELS; do
    if ! printf '%s\n' "$LISTED" | grep -q "/$label\$"; then
        echo "check_det_suite_filter: FAIL — no test matched for model row '$label'." >&2
        echo "Filter '$FILTER' resolved to:" >&2
        printf '%s\n' "$LISTED" | sed 's/^/  /' >&2
        echo "Either the instantiation lost a row, or det_models() and" >&2
        echo "EXPECTED_LABELS in this script have drifted apart." >&2
        exit 1
    fi
done

echo "check_det_suite_filter: OK — '$FILTER' resolves to $COUNT tests across: $EXPECTED_LABELS"
exit 0
