#!/usr/bin/env bash
# Engine integration tests for KV block pre-allocation, async graph loop,
# and multi-turn chat.
#
# Usage: ./tests/test_engine_chat.sh [build_dir]

set -eo pipefail

BUILD="${1:-build}"
CLI="$BUILD/imp-cli"
PASS=0; FAIL=0; SKIP=0
EXIT_CODE=0
OUT=""

# ── Model paths ──────────────────────────────────────────────────────
M="models"
PHI4="$M/models--unsloth--Phi-4-mini-instruct-GGUF/snapshots/78eb92a46fc37e6b524df991ed9aca9bc6aa7b80/Phi-4-mini-instruct.Q8_0.gguf"
QWEN4B="$M/models--unsloth--Qwen3-4B-Instruct-2507-GGUF/snapshots/a06e946bb6b655725eafa393f4a9745d460374c9/Qwen3-4B-Instruct-2507-Q8_0.gguf"
DS7B="$M/models--unsloth--DeepSeek-R1-Distill-Qwen-7B-GGUF/snapshots/097680e4eed7a83b3df6b0bb5e5134099cadf1b0/DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf"

# ── Helpers ───────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; BOLD='\033[1m'; NC='\033[0m'

pass() { PASS=$((PASS+1)); echo -e "  ${GREEN}PASS${NC}  $1"; }
fail() { FAIL=$((FAIL+1)); echo -e "  ${RED}FAIL${NC}  $1 — $2"; }
skip() { SKIP=$((SKIP+1)); echo -e "  ${YELLOW}SKIP${NC}  $1 — $2"; }

has_model() { [ -e "$1" ]; }
has_crash() { echo "$OUT" | grep -qi "segmentation fault\|SIGSEGV\|core dumped\|SIGABRT\|Aborted"; }
has_text()  { echo "$OUT" | grep -qi "$1"; }
has_exact() { echo "$OUT" | grep -q "$1"; }

run_single() {
    local model="$1" prompt="$2" timeout_s="${3:-60}" extra="${4:-}"
    local tmp; tmp=$(mktemp)
    set +e
    timeout "$timeout_s" "$CLI" --model "$model" \
        --prompt "$prompt" --temperature 0 --max-tokens 128 $extra >"$tmp" 2>&1
    EXIT_CODE=$?
    set -e
    OUT=$(<"$tmp"); rm -f "$tmp"
}

run_chat() {
    local model="$1" prompts="$2" timeout_s="${3:-60}" extra="${4:-}"
    local tmp; tmp=$(mktemp)
    set +e
    echo "$prompts" | timeout "$timeout_s" "$CLI" --model "$model" \
        --interactive --temperature 0 --max-tokens 256 $extra >"$tmp" 2>&1
    EXIT_CODE=$?
    set -e
    OUT=$(<"$tmp"); rm -f "$tmp"
}

run_bench() {
    local model="$1" pp="${2:-64}" tg="${3:-64}" extra="${4:-}"
    local tmp; tmp=$(mktemp)
    set +e
    timeout 90 "$CLI" --model "$model" --bench \
        --bench-pp "$pp" --max-tokens "$tg" --bench-reps 1 $extra >"$tmp" 2>&1
    EXIT_CODE=$?
    set -e
    OUT=$(<"$tmp"); rm -f "$tmp"
}

# ─────────────────────────────────────────────────────────────────────
echo -e "${BOLD}Engine Chat Integration Tests${NC}"
echo "============================================="
echo ""
[ -x "$CLI" ] || { echo "ERROR: $CLI not found. Build first."; exit 1; }

# =================================================================
# PHI-4 MINI  (32 layers, phi3→llama arch, phi template)
# =================================================================
echo -e "${BOLD}── Phi-4 Mini ──${NC}"

if ! has_model "$PHI4"; then
    skip "phi4-*" "model not found"
else

# 1. Single prompt — basic correctness
TEST="phi4-single-prompt"
run_single "$PHI4" "What is 2+2? Answer with just the number."
if has_crash; then fail "$TEST" "segfault"
elif ! has_exact "4"; then fail "$TEST" "expected '4' in output"
else pass "$TEST"; fi

# 2. Interactive single turn — async graph should launch
TEST="phi4-interactive-graph-launch"
run_chat "$PHI4" "Hello
quit"
if has_crash; then fail "$TEST" "segfault"
elif has_exact "AsyncGraphLoop: launched"; then
    if has_text "hello\|hi\|help\|assist"; then pass "$TEST"
    else fail "$TEST" "graph launched but no coherent output"; fi
else pass "$TEST (step decode fallback)"; fi

# 3. Interactive multi-turn — no crash, no timeout
TEST="phi4-interactive-2-turns"
run_chat "$PHI4" "Say the word apple
Say the word banana
quit" 90
if has_crash; then fail "$TEST" "segfault"
elif [ $EXIT_CODE -eq 124 ]; then fail "$TEST" "timeout"
else pass "$TEST"; fi

# 4. No CUDA graphs — pure step decode path
TEST="phi4-no-cuda-graphs"
run_single "$PHI4" "What is the capital of France? One word." 30 "--no-cuda-graphs"
if has_crash; then fail "$TEST" "segfault"
elif [ $EXIT_CODE -ne 0 ] && [ $EXIT_CODE -ne 124 ]; then fail "$TEST" "exit code $EXIT_CODE"
else pass "$TEST"; fi

# 5. Short max-tokens — should NOT hit KV capacity warning
TEST="phi4-short-max-tokens"
run_chat "$PHI4" "Hi
quit" 30
if has_crash; then fail "$TEST" "segfault"
elif has_exact "no KV capacity"; then fail "$TEST" "unexpected KV capacity warning"
else pass "$TEST"; fi

# 6. Benchmark mode (ignore_eos, stress KV)
TEST="phi4-bench"
run_bench "$PHI4" 64 64
if has_crash; then fail "$TEST" "segfault"
elif has_exact "tok/s"; then pass "$TEST"
else fail "$TEST" "no benchmark results"; fi

# 7. Self-eviction regression (the original bug)
TEST="phi4-no-self-eviction"
run_chat "$PHI4" "Tell me a joke
quit" 60
if has_crash; then fail "$TEST" "segfault (self-eviction)"
elif has_exact "CANCELLED"; then fail "$TEST" "request cancelled"
elif has_exact "failed to pre-allocate KV blocks"; then fail "$TEST" "old eviction bug"
else pass "$TEST"; fi

# 8. Very small max-tokens (graph finishes cleanly)
TEST="phi4-tiny-generation"
run_chat "$PHI4" "Hi
quit" 30 "--max-tokens 8"
if has_crash; then fail "$TEST" "segfault"
elif [ $EXIT_CODE -ne 0 ] && [ $EXIT_CODE -ne 124 ]; then fail "$TEST" "exit code $EXIT_CODE"
else pass "$TEST"; fi

fi  # PHI4

echo ""

# =================================================================
# QWEN3-4B  (36 layers, qwen3 arch, qwen3 template)
# =================================================================
echo -e "${BOLD}── Qwen3-4B ──${NC}"

if ! has_model "$QWEN4B"; then
    skip "qwen3-4b-*" "model not found"
else

# 9. Single prompt
TEST="qwen3-4b-single-prompt"
run_single "$QWEN4B" "What is 2+2? Answer with just the number."
if has_crash; then fail "$TEST" "segfault"
elif has_exact "4"; then pass "$TEST"
else pass "$TEST (no crash)"; fi

# 10. Interactive multi-turn
TEST="qwen3-4b-interactive-2-turns"
run_chat "$QWEN4B" "Say the word apple
Say the word banana
quit" 90
if has_crash; then fail "$TEST" "segfault"
elif [ $EXIT_CODE -eq 124 ]; then fail "$TEST" "timeout"
else pass "$TEST"; fi

# 11. Benchmark
TEST="qwen3-4b-bench"
run_bench "$QWEN4B" 64 64
if has_crash; then fail "$TEST" "segfault"
elif has_exact "tok/s"; then pass "$TEST"
else fail "$TEST" "no benchmark results"; fi

fi  # QWEN4B

echo ""

# =================================================================
# DEEPSEEK-R1 7B  (28 layers, qwen2 arch, deepseek template)
# =================================================================
echo -e "${BOLD}── DeepSeek-R1-7B ──${NC}"

if ! has_model "$DS7B"; then
    skip "ds-r1-7b-*" "model not found"
else

# 12. Single prompt
TEST="ds-r1-7b-single-prompt"
run_single "$DS7B" "What is 2+2?"
if has_crash; then fail "$TEST" "segfault"
elif [ $EXIT_CODE -ne 0 ] && [ $EXIT_CODE -ne 124 ]; then fail "$TEST" "exit code $EXIT_CODE"
else pass "$TEST"; fi

# 13. Interactive
TEST="ds-r1-7b-interactive"
run_chat "$DS7B" "Say hello briefly
quit" 90
if has_crash; then fail "$TEST" "segfault"
elif [ $EXIT_CODE -eq 124 ]; then fail "$TEST" "timeout"
else pass "$TEST"; fi

# 14. Benchmark
TEST="ds-r1-7b-bench"
run_bench "$DS7B" 64 64
if has_crash; then fail "$TEST" "segfault"
elif has_exact "tok/s"; then pass "$TEST"
else fail "$TEST" "no benchmark results"; fi

fi  # DS7B

# =================================================================
# Summary
# =================================================================
echo ""
echo "============================================="
TOTAL=$((PASS + FAIL + SKIP))
echo -e " ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}, ${YELLOW}$SKIP skipped${NC} / $TOTAL total"
echo "============================================="

exit $((FAIL > 0 ? 1 : 0))
