#!/bin/sh
# Guard the test-e2e unit/gpu lane split against silent rename drift (R5/#580).
#
# test-e2e bundles CPU-only stub/API tests AND GPU-backed E2E tests in one
# binary; the unit lane is carved out by a gtest_filter (_unit_e2e_filter in
# CMakeLists.txt). A test rename used to silently shift a test into the wrong
# lane — a CPU test renamed out of the filter would start running in the GPU
# lane (skipped in CI, lost coverage) with no error.
#
# This script asserts the filter resolves to EXACTLY the expected set of unit
# tests. Renaming/removing/adding a unit test now fails this guard loudly until
# both the filter and the expected list below are updated together.
#
# Usage: check_e2e_lane_split.sh <path-to-test-e2e> "<gtest_filter>"
# Exit 0 = lanes match expectation; non-zero = drift (prints the diff).

set -eu

BIN="${1:?usage: check_e2e_lane_split.sh <test-e2e> <unit_filter>}"
FILTER="${2:?missing unit filter}"

if [ ! -x "$BIN" ]; then
    echo "check_e2e_lane_split: $BIN not executable" >&2
    exit 2
fi

# Expected unit-lane tests (fully-qualified). Keep in sync with _unit_e2e_filter
# in CMakeLists.txt. These are the CPU-only stub/API tests in test-e2e; the rest
# of the binary (EndToEndModelTest.*, StubModelTest GPU subtests, GPUBatchTest.*)
# is the GPU lane.
EXPECTED=$(cat <<'EOF'
BatchBuilderTest.MultipleDecodeSequences
BatchBuilderTest.PrefillSequence
BatchBuilderTest.PrefillWithStartPos
BatchBuilderTest.ResetClearsPreviousData
BatchBuilderTest.SingleDecodeSequence
BatchBuilderTest.SingleToken
BatchBuilderTest.SixteenDecodeSequences
EndToEndTest.ConfigDefault
EndToEndTest.ErrorStrings
EndToEndTest.GenerateParamsDefault
EndToEndTest.LoadNonexistentModel
EndToEndTest.NullArguments
EndToEndTest.VersionString
RequestTest.ContextLen
RequestTest.DefaultState
RequestTest.StatusTransitions
SchedulerTest.ACancelledQueueSchedulesNothing
SchedulerTest.AddRemoveRapidly
SchedulerTest.AllRequestsTooLargeForMemory
SchedulerTest.BasicPrefillThenDecode
SchedulerTest.DoesNotPromoteARequestCancelledWhileQueued
SchedulerTest.BatchedDecodeWithMidBatchCompletion
SchedulerTest.ChunkedPrefillCompleteThenDecode
SchedulerTest.ChunkedPrefillRescheduling
SchedulerTest.DecodeBatchSizeLimit
SchedulerTest.EmptyBatch
SchedulerTest.EmptyScheduler
SchedulerTest.FullLifecycle
SchedulerTest.HandlesCancel
SchedulerTest.MaxBatchSize
SchedulerTest.MaxBatchSizeLimit
SchedulerTest.MemoryAwareScheduling
SchedulerTest.MemoryAwareSkipsLargeAdmitsSmall
SchedulerTest.NewPrefillWhileDecoding
SchedulerTest.PrefillPriorityOverDecode
SchedulerTest.RemovesFinishedRequests
SchedulerTest.ShortestInputFirst
StubModelTest.LoadStubModel
StubModelTest.TokenizeStub
EOF
)

# Parse `--gtest_list_tests` output: a "Fixture." line followed by indented
# "  TestName" lines. Reconstruct fully-qualified names, sort for a stable diff.
ACTUAL=$("$BIN" --gtest_filter="$FILTER" --gtest_list_tests 2>/dev/null | awk '
    /^[^[:space:]].*\.$/ { fixture=$1; next }
    /^[[:space:]]+[A-Za-z]/ { name=$1; sub(/#.*/, "", name); gsub(/[[:space:]]/, "", name); if (name != "") print fixture name }
')

EXP_SORTED=$(printf '%s\n' "$EXPECTED" | sort)
ACT_SORTED=$(printf '%s\n' "$ACTUAL" | sort)

if [ "$EXP_SORTED" = "$ACT_SORTED" ]; then
    n=$(printf '%s\n' "$ACT_SORTED" | grep -c .)
    echo "check_e2e_lane_split: OK — unit lane resolves to $n expected tests"
    exit 0
fi

echo "check_e2e_lane_split: FAIL — test-e2e unit lane drifted from expectation." >&2
echo "The _unit_e2e_filter in CMakeLists.txt no longer matches the frozen unit set." >&2
echo "If a test was intentionally renamed/added/removed, update BOTH the filter" >&2
echo "and the EXPECTED list in this script. Diff (< expected, > actual):" >&2
# POSIX sh: no process substitution — compare via comm on temp files.
_exp_tmp=$(mktemp); _act_tmp=$(mktemp)
trap 'rm -f "$_exp_tmp" "$_act_tmp"' EXIT
printf '%s\n' "$EXP_SORTED" >"$_exp_tmp"
printf '%s\n' "$ACT_SORTED" >"$_act_tmp"
comm -23 "$_exp_tmp" "$_act_tmp" | sed 's/^/< missing from lane: /' >&2
comm -13 "$_exp_tmp" "$_act_tmp" | sed 's/^/> unexpectedly in lane: /' >&2
exit 1
