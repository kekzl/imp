#!/usr/bin/env bash
# GPU acceptance for #2195: FP8 FMHA prefill runs for head_dim 64.
# Prints PASS/FAIL per criterion, exit 0 only if every criterion passes. Needs a GPU and the
# worktree image (make build).
# Usage: bash scripts/accept_2195.sh
# Env: IMP_TEST_IMG (default: this tree's image tag).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

# Hold the card for the whole run (scripts/gpu_lock.sh); refuse a busy one.
if [ -z "${IMP_ACCEPT_2195_LOCKED:-}" ]; then
    bash scripts/require_free_gpu.sh "accept_2195" || exit 1
    IMP_ACCEPT_2195_LOCKED=1 exec bash scripts/gpu_lock.sh run "accept_2195" -- bash "$0" "$@"
fi

IMG="${IMP_TEST_IMG:-$(bash scripts/image_tag.sh)}"
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FAIL setup: image $IMG missing (make build)"; exit 1; }
bash scripts/image_tag.sh check "$IMG" || { echo "FAIL setup: $IMG is not this tree (make build)"; exit 1; }

LOG="$(mktemp -d "${TMPDIR:-/tmp}/accept_2195.XXXXXX")"
trap 'rm -rf "$LOG"' EXIT

RESULTS=()
verdict() {  # verdict <id> <PASS|FAIL> <detail>
    RESULTS+=("$2 $1: $3")
    echo "$2 $1: $3"
}

gpu_test() {  # gpu_test <log> <gtest args...>; returns the test binary's exit code
    local out="$1"; shift
    docker run --rm --gpus all "$IMG" test-attention "$@" >"$out" 2>&1
}

# C1: every FmhaFP8Test case passes, none skipped.
gpu_test "$LOG/fp8.log" --gtest_filter='FmhaFP8Test.*'
rc=$?
ran=$(grep -oE '^\[==========\] [0-9]+ tests? from [0-9]+ test suites? ran' "$LOG/fp8.log" | grep -oE '[0-9]+' | head -1)
passed=$(grep -oE '^\[  PASSED  \] [0-9]+' "$LOG/fp8.log" | grep -oE '[0-9]+$')
skipped=$(grep -cE '^\[  SKIPPED \] FmhaFP8Test\.' "$LOG/fp8.log")
failed=$(grep -cE '^\[  FAILED  \] FmhaFP8Test\.' "$LOG/fp8.log")
detail="exit=$rc ran=${ran:-?} passed=${passed:-?} skipped=$skipped failed=$failed"
if [ "$rc" = 0 ] && [ -n "$ran" ] && [ "$ran" = "${passed:-x}" ] && [ "$skipped" = 0 ] && [ "$failed" = 0 ]; then
    verdict C1-fp8-suite PASS "$detail"
else
    verdict C1-fp8-suite FAIL "$detail"
fi

# C2: HD64 ran and its error vs the CPU reference is inside the test tolerance (max_rel < 0.05).
hd64=$(grep -E '^\[fp8-fmha\] HD=64 ' "$LOG/fp8.log" | head -1)
hd64_ok=$(grep -cE '^\[       OK \] FmhaFP8Test\.HD64 ' "$LOG/fp8.log")
rel=$(printf '%s\n' "$hd64" | grep -oE 'max_rel=[0-9.]+' | cut -d= -f2)
if [ "$hd64_ok" = 1 ] && [ -n "$rel" ] && awk -v r="$rel" 'BEGIN { exit !(r < 0.05) }'; then
    verdict C2-hd64-accuracy PASS "$hd64"
else
    verdict C2-hd64-accuracy FAIL "OK lines=$hd64_ok line='${hd64:-missing}'"
fi

# C3: cross-path HD64 agreement, now that the FP8 path runs there instead of declining.
gpu_test "$LOG/xpath.log" --gtest_filter='AttentionCrossPathTest.HeadDim64'
rc=$?
if [ "$rc" = 0 ] && grep -qE '^\[       OK \] AttentionCrossPathTest\.HeadDim64 ' "$LOG/xpath.log"; then
    verdict C3-crosspath-hd64 PASS "exit=0"
else
    verdict C3-crosspath-hd64 FAIL "exit=$rc: $(grep -E 'Failure|FAILED|fmha_fp8' "$LOG/xpath.log" | head -5 | tr '\n' ' ')"
fi

# C4: microbench line, HD64 FP8 vs FP16 FMHA at Sq=Skv=2048, 32 heads.
gpu_test "$LOG/bench.log" --gtest_also_run_disabled_tests --gtest_filter='FmhaFP8Test.DISABLED_BenchHD64VsFp16'
rc=$?
bench=$(grep -E '^\[fp8-fmha-bench\] ' "$LOG/bench.log" | head -1)
if [ "$rc" = 0 ] && [ -n "$bench" ]; then
    verdict C4-hd64-bench PASS "$bench"
else
    verdict C4-hd64-bench FAIL "exit=$rc line='${bench:-missing}'"
fi

echo
echo "== accept_2195 summary (image $IMG) =="
printf '%s\n' "${RESULTS[@]}"
fails=$(printf '%s\n' "${RESULTS[@]}" | grep -c '^FAIL')
[ ${#RESULTS[@]} -eq 4 ] || { echo "FAIL: ${#RESULTS[@]}/4 criteria reported"; exit 1; }
[ "$fails" = 0 ] && { echo "ALL PASS"; exit 0; }
echo "$fails FAIL"
exit 1
