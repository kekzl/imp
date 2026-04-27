#!/bin/bash
# verify.sh — pre-commit/pre-push verification for imp.
#
# Runs the three-step gate from CLAUDE.md:
#   1. Build (incremental)
#   2. Tests (gtest filter or full suite)
#   3. Perf vs. tests/perf_baseline.json (decode regression > 3% = fail)
#   4. Smoke prompts on real models (degeneration detector)
#
# Modes:
#   verify.sh fast    Unit tests + perf baseline + 1 smoke prompt   (~90s)
#   verify.sh full    Full ctest + perf baseline + 2 smoke prompts  (~5min)
#
# Env overrides:
#   IMP_VERIFY_BIN=build/imp-cli
#   IMP_VERIFY_TESTS=build/imp-tests
#   IMP_VERIFY_MODELS=models
#   IMP_VERIFY_BASELINE=tests/perf_baseline.json
#   IMP_VERIFY_SKIP_BUILD=1   skip cmake build step
#
set -uo pipefail

MODE="${1:-fast}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BIN="${IMP_VERIFY_BIN:-build/imp-cli}"
TESTS_BIN="${IMP_VERIFY_TESTS:-build/imp-tests}"
MODELS="${IMP_VERIFY_MODELS:-models}"
BASELINE="${IMP_VERIFY_BASELINE:-tests/perf_baseline.json}"

RED=$'\033[0;31m'; GRN=$'\033[0;32m'; YLW=$'\033[0;33m'; RST=$'\033[0m'
FAIL=0
section() { echo; echo "${YLW}== $* ==${RST}"; }
pass()    { echo "${GRN}PASS${RST} $*"; }
fail()    { echo "${RED}FAIL${RST} $*"; FAIL=$((FAIL+1)); }
skip()    { echo "${YLW}SKIP${RST} $*"; }

# -------------------------------------------------------------------- 1. build
section "build"
if [ "${IMP_VERIFY_SKIP_BUILD:-0}" = "1" ]; then
    skip "build (IMP_VERIFY_SKIP_BUILD=1)"
elif [ ! -d build ]; then
    fail "no build/ directory — run 'cmake -B build -DCMAKE_BUILD_TYPE=Release' first"
else
    if cmake --build build -j"$(nproc)" >/tmp/imp_verify_build.log 2>&1; then
        pass "incremental build"
    else
        fail "build (see /tmp/imp_verify_build.log)"
        tail -20 /tmp/imp_verify_build.log
        exit 1
    fi
fi

# -------------------------------------------------------------------- 2. tests
section "tests"
if [ ! -x "$TESTS_BIN" ]; then
    fail "$TESTS_BIN not found"
else
    if [ "$MODE" = "fast" ]; then
        FILTER="TensorTest.*:GgufLoaderTest.*:Tokenizer*:ChatTemplate*:KVCache*:GemmTest.*:FP8GemmTest.*:SamplingTest.*:SoftmaxTest.*:AttentionTest.*"
        if "$TESTS_BIN" --gtest_filter="$FILTER" >/tmp/imp_verify_tests.log 2>&1; then
            pass "fast gtest filter"
        else
            fail "gtest (see /tmp/imp_verify_tests.log)"
            tail -30 /tmp/imp_verify_tests.log
        fi
    else
        if "$TESTS_BIN" >/tmp/imp_verify_tests.log 2>&1; then
            pass "full gtest suite"
        else
            fail "gtest (see /tmp/imp_verify_tests.log)"
            grep -E "FAIL|fatal" /tmp/imp_verify_tests.log | head -20
        fi
    fi
fi

# --------------------------------------------------------------------- 3. perf
section "perf vs baseline"
if [ ! -f "$BASELINE" ]; then
    skip "no $BASELINE — run scripts/gen_perf_baseline.sh to create one"
elif [ ! -x "$BIN" ]; then
    fail "$BIN not found"
else
    BL_MODEL=$(jq -r '.model' "$BASELINE")
    BL_TG=$(jq -r '.metrics.decode_tps.tg128' "$BASELINE")
    BL_PP=$(jq -r '.metrics.prefill_tps.pp512' "$BASELINE")
    DEC_THR=$(jq -r '.thresholds.decode_regression_pct' "$BASELINE")
    PRE_THR=$(jq -r '.thresholds.prefill_regression_pct' "$BASELINE")
    MODEL_PATH="$MODELS/$BL_MODEL"

    if [ ! -f "$MODEL_PATH" ]; then
        skip "baseline model $MODEL_PATH not present"
    else
        REPS=3
        ERR=$(mktemp)
        "$BIN" --model "$MODEL_PATH" --bench --bench-pp 512 --bench-reps $REPS \
              --max-tokens 128 --temperature 0 >/dev/null 2>"$ERR"
        # Bench lines (stderr) have variable spacing inside parens for short numbers:
        #   "pp   512 tokens  avg    38.47 ms  (13310.12 tok/s)  [3 reps]"
        #   "tg   128 tokens  avg   861.50 ms  ( 148.58 tok/s)  [3 reps]"
        PP=$(grep -oP '^pp\s+512\s.*\(\s*\K[0-9.]+(?=\s+tok/s)' "$ERR" | head -1)
        TG=$(grep -oP '^tg\s+128\s.*\(\s*\K[0-9.]+(?=\s+tok/s)' "$ERR" | head -1)
        if [ -z "$PP" ] || [ -z "$TG" ]; then
            fail "could not parse bench output (see $ERR)"
            tail -15 "$ERR"
        else
            rm -f "$ERR"
            DEC_DELTA=$(awk -v cur="$TG" -v base="$BL_TG" 'BEGIN{printf "%.2f", (cur-base)/base*100}')
            PRE_DELTA=$(awk -v cur="$PP" -v base="$BL_PP" 'BEGIN{printf "%.2f", (cur-base)/base*100}')
            DEC_REG=$(awk -v d="$DEC_DELTA" -v t="$DEC_THR" 'BEGIN{print (-d > t) ? 1 : 0}')
            PRE_REG=$(awk -v d="$PRE_DELTA" -v t="$PRE_THR" 'BEGIN{print (-d > t) ? 1 : 0}')

            printf "  decode  tg128 = %7.2f tok/s  (baseline %7.2f, delta %+s%%)\n" "$TG" "$BL_TG" "$DEC_DELTA"
            printf "  prefill pp512 = %7.2f tok/s  (baseline %7.2f, delta %+s%%)\n" "$PP" "$BL_PP" "$PRE_DELTA"

            if [ "$DEC_REG" = "1" ]; then
                fail "decode regression > ${DEC_THR}%  (if expected: ./scripts/gen_perf_baseline.sh $MODEL_PATH)"
            else
                pass "decode within ${DEC_THR}% threshold"
            fi
            if [ "$PRE_REG" = "1" ]; then
                # prefill is noisy (cuBLAS autotuning), warn only
                echo "${YLW}WARN${RST} prefill regression > ${PRE_THR}% (often cuBLAS variance, not real)"
            else
                pass "prefill within ${PRE_THR}% threshold"
            fi
        fi
    fi
fi

# ------------------------------------------------------------- 4. smoke prompts
section "smoke prompts (degeneration check)"

# Greedy decode on a known-deterministic prompt.
# Quality gate: output must contain expected substring AND last 32 tokens must
# have at least 8 distinct tokens (catches "own own own" stuck-token failures).
smoke_prompt() {
    local label="$1" model="$2" prompt="$3" expect="$4"
    if [ ! -f "$MODELS/$model" ]; then
        skip "$label ($model not present)"
        return
    fi
    local ERR; ERR=$(mktemp)
    OUT=$("$BIN" --model "$MODELS/$model" --prompt "$prompt" \
          --max-tokens 64 --temperature 0 --chat-template none 2>"$ERR")
    # Token markers '[tok=NNNN ' word']' land on stderr.
    TOKS=$(grep -oP '\[tok=\K[0-9]+' "$ERR" | tail -32)
    DISTINCT=$(echo "$TOKS" | sort -u | wc -l)
    # Generated word stream (stripped of token id prefix) for NaN/Inf scan
    WORDS=$(grep -oP "\[tok=[0-9]+ '\K[^']*" "$ERR" | tr '\n' ' ')
    rm -f "$ERR"

    if echo " $WORDS " | grep -qiE ' (nan|inf|-inf|-nan) '; then
        fail "$label — NaN/Inf token in output"
        echo "  words: $WORDS"
        return
    fi
    if [ "$DISTINCT" -lt 8 ]; then
        fail "$label — degenerate (only $DISTINCT distinct tokens in last 32)"
        echo "  tokens: $(echo "$TOKS" | tr '\n' ' ')"
        return
    fi
    # Generated text appears interleaved with logs on stdout — substring match works
    if ! echo "$OUT$WORDS" | grep -q "$expect"; then
        fail "$label — expected '$expect' in output"
        echo "  words: $WORDS"
        return
    fi
    pass "$label (distinct=$DISTINCT, contains '$expect')"
}

smoke_prompt "Qwen3-4B Q8_0 (dense)" \
    "Qwen3-4B-Instruct-2507-Q8_0.gguf" \
    "The capital of France is" \
    "Paris"

if [ "$MODE" = "full" ]; then
    smoke_prompt "Qwen3.5-4B Q8_0 (GDN)" \
        "Qwen3.5-4B-Q8_0.gguf" \
        "The capital of France is" \
        "Paris"
fi

# ------------------------------------------------------------------- summary
echo
if [ "$FAIL" -eq 0 ]; then
    echo "${GRN}=== verify $MODE: OK ===${RST}"
    exit 0
else
    echo "${RED}=== verify $MODE: $FAIL failure(s) ===${RST}"
    exit 1
fi
