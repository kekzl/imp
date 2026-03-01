#!/usr/bin/env bash
set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IMP_CLI="$ROOT_DIR/build/imp-cli"
TIMEOUT=120

# Colors
if [[ -t 1 ]]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[0;33m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    DIM='\033[2m'
    RESET='\033[0m'
else
    GREEN='' RED='' YELLOW='' CYAN='' BOLD='' DIM='' RESET=''
fi

# ─── Prompts ─────────────────────────────────────────────────────────────────
PROMPT_NAMES=(
    "tiny"
    "short"
    "medium"
    "long"
    "code"
    "reasoning"
)
PROMPT_TEXTS=(
    "Hi"
    "What is 2+2? Answer in one word."
    "Explain how a CPU executes a single instruction, from fetch to writeback. Cover the role of the program counter, instruction register, ALU, and control unit in this process."
    "You are a computer architecture expert. Explain the memory hierarchy of a modern processor in detail. Start with registers, then L1, L2, and L3 caches, then main memory (DRAM), and finally disk storage. For each level, describe the typical size, latency in CPU cycles, and bandwidth. Explain why this hierarchy exists in terms of the trade-off between speed and cost per bit. Then discuss how cache coherence protocols like MESI work in multi-core systems and why they are necessary. Finally, describe how TLBs and virtual memory interact with this hierarchy."
    "Write a Python function that takes a list of integers and returns the second largest unique value. Handle edge cases."
    "A farmer has 3 fields. Field A produces 20 bushels per acre, field B produces 35 bushels per acre, and field C produces 15 bushels per acre. He has 100 acres total and must plant at least 10 acres of each crop. If he wants to maximize total bushels, how should he allocate his acres? Show your reasoning step by step."
)
PROMPT_MAX_TOKENS=(
    16
    32
    64
    128
    96
    64
)

# ─── Model discovery (reused from benchmark.sh) ─────────────────────────────
HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}/hub"

find_gguf() {
    local pattern="$1"
    local result=""
    result=$(find "$ROOT_DIR/models" "$HF_CACHE" \( -name "$pattern" \) \( -type f -o -type l \) 2>/dev/null | head -1)
    if [[ -n "$result" ]]; then
        result=$(readlink -f "$result")
    fi
    echo "$result"
}

discover_models() {
    declare -g -a MODEL_NAMES=(
        "Phi-4-Mini-Instruct"
        "Qwen3-4B-Instruct"
        "DeepSeek-R1-Distill-Qwen-7B"
        "Gemma-3-12B-IT"
        "DeepSeek-R1-Distill-Qwen-14B"
        "Qwen3-Coder-30B-A3B-Instruct"
        "Nemotron-3-Nano-30B-A3B"
    )
    declare -g -a MODEL_PATHS=(
        "$(find_gguf '*Phi-4-mini-instruct*Q8_0.gguf')"
        "$(find_gguf '*Qwen3-4B*Q8_0.gguf')"
        "$(find_gguf '*DeepSeek-R1-Distill-Qwen-7B*Q8_0.gguf')"
        "$(find_gguf '*gemma-3-12b*Q8_0.gguf')"
        "$(find_gguf '*DeepSeek-R1-Distill-Qwen-14B*Q6_K.gguf')"
        "$(find_gguf '*Qwen3-Coder-30B*Q6_K.gguf')"
        "$(find_gguf '*Nemotron-3-Nano*Q6_K.gguf')"
    )
}

# ─── Parse arguments ────────────────────────────────────────────────────────
SINGLE_MODEL=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            SINGLE_MODEL="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ─── Preflight ───────────────────────────────────────────────────────────────
if [[ ! -x "$IMP_CLI" ]]; then
    echo -e "${RED}ERROR: imp-cli not found at $IMP_CLI${RESET}"
    echo "Build with: cmake --build build -j\$(nproc)"
    exit 1
fi

# Set up model list
if [[ -n "$SINGLE_MODEL" ]]; then
    if [[ ! -f "$SINGLE_MODEL" ]]; then
        echo -e "${RED}ERROR: model file not found: $SINGLE_MODEL${RESET}"
        exit 1
    fi
    MODEL_NAMES=("$(basename "$SINGLE_MODEL")")
    MODEL_PATHS=("$(readlink -f "$SINGLE_MODEL")")
else
    discover_models
fi

# ─── Counters ────────────────────────────────────────────────────────────────
TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0

# Results for summary table
declare -a RESULTS=()

# ─── Temp directory ──────────────────────────────────────────────────────────
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# ─── Run tests ───────────────────────────────────────────────────────────────
echo -e "${BOLD}═══ imp prompt tests ═══${RESET}"
echo ""

for m in "${!MODEL_NAMES[@]}"; do
    model_name="${MODEL_NAMES[$m]}"
    model_path="${MODEL_PATHS[$m]}"

    if [[ -z "$model_path" || ! -f "$model_path" ]]; then
        echo -e "${YELLOW}━━━ $model_name: SKIP (not found) ━━━${RESET}"
        for p in "${!PROMPT_NAMES[@]}"; do
            TOTAL=$((TOTAL + 1))
            SKIPPED=$((SKIPPED + 1))
            RESULTS+=("$model_name|${PROMPT_NAMES[$p]}|SKIP|-|-")
        done
        echo ""
        continue
    fi

    echo -e "${BOLD}━━━ $model_name ━━━${RESET}"
    echo -e "${DIM}    $model_path${RESET}"

    for p in "${!PROMPT_NAMES[@]}"; do
        prompt_name="${PROMPT_NAMES[$p]}"
        prompt_text="${PROMPT_TEXTS[$p]}"
        max_tokens="${PROMPT_MAX_TOKENS[$p]}"
        TOTAL=$((TOTAL + 1))

        stdout_file="$TMPDIR/stdout_${m}_${p}"
        stderr_file="$TMPDIR/stderr_${m}_${p}"

        printf "  %-12s " "$prompt_name"

        # Run with timeout
        exit_code=0
        timeout "$TIMEOUT" "$IMP_CLI" \
            --model "$model_path" \
            --prompt "$prompt_text" \
            --max-tokens "$max_tokens" \
            --temperature 0 \
            --seed 42 \
            "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
            > "$stdout_file" 2> "$stderr_file" || exit_code=$?

        # Extract timing from stderr
        pp_toks=$(grep "^pp " "$stderr_file" 2>/dev/null | sed 's/.*(\s*\([0-9.]*\) tok\/s).*/\1/' || echo "")
        tg_toks=$(grep "^tg " "$stderr_file" 2>/dev/null | sed 's/.*(\s*\([0-9.]*\) tok\/s).*/\1/' || echo "")

        # Check pass criteria: exit 0, non-empty stdout, stderr contains tok/s
        stdout_content=$(cat "$stdout_file" 2>/dev/null || echo "")
        has_toks=$(grep -c "tok/s" "$stderr_file" 2>/dev/null || echo "0")

        if [[ $exit_code -eq 0 && -n "$stdout_content" && "$has_toks" -gt 0 ]]; then
            PASSED=$((PASSED + 1))
            pp_display="${pp_toks:-?}"
            tg_display="${tg_toks:-?}"
            echo -e "${GREEN}PASS${RESET}  pp=${pp_display} tok/s  tg=${tg_display} tok/s"
            RESULTS+=("$model_name|$prompt_name|PASS|$pp_display|$tg_display")
        else
            FAILED=$((FAILED + 1))
            echo -e "${RED}FAIL${RESET}  exit=$exit_code stdout_len=${#stdout_content} has_timing=$has_toks"
            if [[ $exit_code -eq 124 ]]; then
                echo -e "         ${DIM}(timeout after ${TIMEOUT}s)${RESET}"
            elif [[ $exit_code -ne 0 ]]; then
                tail -3 "$stderr_file" 2>/dev/null | while read -r line; do
                    echo -e "         ${DIM}$line${RESET}"
                done
            fi
            RESULTS+=("$model_name|$prompt_name|FAIL|-|-")
        fi
    done
    echo ""
done

# ─── Summary ─────────────────────────────────────────────────────────────────
echo -e "${BOLD}═══ Summary ═══${RESET}"
echo ""

# Print table header
printf "  ${BOLD}%-35s %-12s %-6s %10s %10s${RESET}\n" "Model" "Prompt" "Result" "pp tok/s" "tg tok/s"
printf "  %-35s %-12s %-6s %10s %10s\n" "-----------------------------------" "------------" "------" "----------" "----------"

for result in "${RESULTS[@]}"; do
    IFS='|' read -r rmodel rprompt rstatus rpp rtg <<< "$result"
    case "$rstatus" in
        PASS) color="$GREEN" ;;
        FAIL) color="$RED" ;;
        *)    color="$YELLOW" ;;
    esac
    printf "  %-35s %-12s ${color}%-6s${RESET} %10s %10s\n" "$rmodel" "$rprompt" "$rstatus" "$rpp" "$rtg"
done

echo ""
echo -e "  Total: $TOTAL  ${GREEN}Passed: $PASSED${RESET}  ${RED}Failed: $FAILED${RESET}  ${YELLOW}Skipped: $SKIPPED${RESET}"
echo ""

# Exit code
if [[ $FAILED -gt 0 ]]; then
    exit 1
fi
exit 0
