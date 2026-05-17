#!/bin/bash
# Smoke test: quick validation of the entire imp stack.
# Runs in Docker, needs GPU + models mounted.
# Usage: docker run --gpus all -v ./models:/models imp:test bash /scripts/smoke_test.sh
set -uo pipefail

PASS=0
FAIL=0
SKIP=0

run_test() {
    local name="$1"
    shift
    echo -n "  $name... "
    if output=$("$@" 2>&1); then
        echo "PASS"
        ((PASS++))
    else
        echo "FAIL"
        echo "    Output: $(echo "$output" | tail -3)"
        ((FAIL++))
    fi
}

skip_test() {
    echo "  $1... SKIP ($2)"
    ((SKIP++))
}

echo "=== imp Smoke Test ==="
echo ""

echo "[1/6] Unit tests (GTest)..."
run_test "GTest suite" imp-tests --gtest_filter="TensorTest.*:GgufLoaderTest.*:TokenizerSPMTest.*:ChatTemplateApplyTest.ChatMLBasicStructure"

echo ""
echo "[2/6] GPU kernel tests..."
run_test "Attention kernels" imp-tests --gtest_filter="AttentionTest.*"
run_test "GEMM kernels" imp-tests --gtest_filter="GemmTest.*:FP8GemmTest.*"
run_test "Sampling" imp-tests --gtest_filter="SamplingTest.*"
run_test "Softmax" imp-tests --gtest_filter="SoftmaxTest.*"
run_test "KV Cache" imp-tests --gtest_filter="KVCacheTest.*:KVCacheManagerTest.*"
run_test "GDN kernel" test-gdn

echo ""
echo "[3/6] E2E stub model..."
run_test "Stub load+infer" imp-tests --gtest_filter="StubModelTest.*"

echo ""
echo "[4/6] Inference (Qwen3-4B)..."
MODEL=/models/Qwen3-4B-Instruct-2507-Q8_0.gguf
if [ -f "$MODEL" ]; then
    run_test "Greedy decode" bash -c "imp-cli --model $MODEL --prompt '2+2=' --max-tokens 3 --temperature 0 --chat-template none 2>&1 | grep -q 'tok='"
    run_test "Benchmark" bash -c "imp-cli --model $MODEL --bench --bench-pp 128 --bench-reps 1 --max-tokens 32 --temperature 0 2>&1 | grep -q '^tg '"
else
    skip_test "Greedy decode" "model not found"
    skip_test "Benchmark" "model not found"
fi

echo ""
echo "[5/6] Full test suite..."
FULL=$(imp-tests 2>&1 | tail -3)
TOTAL=$(echo "$FULL" | grep "PASSED" | grep -oP '\d+' | head -1)
FAILED=$(echo "$FULL" | grep "FAILED" | grep -oP '\d+' | head -1 || echo 0)
echo "  GTest: $TOTAL passed, ${FAILED:-0} failed"
if [ "${FAILED:-0}" = "0" ]; then
    ((PASS++))
else
    ((FAIL++))
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed, $SKIP skipped ==="
[ "$FAIL" -eq 0 ] && echo "ALL SMOKE TESTS PASSED" || echo "SOME TESTS FAILED"
exit "$FAIL"
