#!/bin/sh
# Guard the deterministic-mode E2E suite against a filter that matches nothing.
#
# `DetEvalE2ETest` is value-parameterised over the model env vars (#1299), so
# its tests are named `Models/DetEvalE2ETest.<Case>/<label>` — NOT
# `DetEvalE2ETest.<Case>`. A `--gtest_filter='DetEvalE2ETest.*'` therefore
# matches zero tests and gtest reports `[  PASSED  ] 0 tests`, which reads as
# green. This suite has already been invisible once: it was gated on an env var
# nothing set, so it skipped from #542 until #1299 found it red.
#
# This asserts the filter used by `make test-e2e` resolves to a non-empty set
# covering every instantiated model row. It needs no GPU — `--gtest_list_tests`
# runs no test bodies — so it belongs in the CPU unit lane.
#
# Usage: check_det_suite_filter.sh <path-to-test-e2e> "<gtest_filter>"

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
