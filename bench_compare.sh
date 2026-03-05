#!/usr/bin/env bash
# Comparable benchmarks: imp vs llama.cpp
# Identical parameters: pp=512, tg=128, reps=5, all layers on GPU
set -euo pipefail

IMP_CLI="/home/kekz/github.com/kekzl/imp/build/imp-cli"
LLAMA_BENCH="/home/kekz/llama.cpp/build/bin/llama-bench"
MODELS_DIR="/home/kekz/github.com/kekzl/imp/models"

PP=512
TG=128
REPS=5

declare -A MODEL_PATHS=(
    ["Qwen3-4B Q8_0"]="$MODELS_DIR/models--unsloth--Qwen3-4B-Instruct-2507-GGUF/snapshots/a06e946bb6b655725eafa393f4a9745d460374c9/Qwen3-4B-Instruct-2507-Q8_0.gguf"
    ["DS-R1-7B Q8_0"]="$MODELS_DIR/models--unsloth--DeepSeek-R1-Distill-Qwen-7B-GGUF/snapshots/097680e4eed7a83b3df6b0bb5e5134099cadf1b0/DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf"
    ["DS-R1-14B Q6_K"]="$MODELS_DIR/models--unsloth--DeepSeek-R1-Distill-Qwen-14B-GGUF/snapshots/7b05b58b41f623e66fc74cd27b35475267b2f3e3/DeepSeek-R1-Distill-Qwen-14B-Q6_K.gguf"
    ["Qwen3-Coder-30B-A3B Q6_K"]="$MODELS_DIR/models--unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF/snapshots/b17cb02dd882d5b6ab62fc777ad2995f19668350/Qwen3-Coder-30B-A3B-Instruct-Q6_K.gguf"
)

# Ordered list (bash associative arrays are unordered)
MODEL_ORDER=(
    "Qwen3-4B Q8_0"
    "DS-R1-7B Q8_0"
    "DS-R1-14B Q6_K"
    "Qwen3-Coder-30B-A3B Q6_K"
)

RESULTS_FILE="bench_results.txt"
> "$RESULTS_FILE"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RESULTS_FILE"; }

log "============================================"
log "  imp vs llama.cpp — Comparable Benchmarks"
log "  pp=$PP  tg=$TG  reps=$REPS"
log "  $(date)"
log "============================================"
log ""

# Collect results: model -> "imp_pp imp_tg llama_pp llama_tg"
declare -A RESULTS

for name in "${MODEL_ORDER[@]}"; do
    model="${MODEL_PATHS[$name]}"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "  Model: $name"
    log "  File:  $(basename "$model")"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # --- imp ---
    log ""
    log ">>> imp-cli --bench"
    imp_output=$("$IMP_CLI" --model "$model" --bench --bench-pp "$PP" --max-tokens "$TG" --bench-reps "$REPS" 2>&1)
    echo "$imp_output" | tee -a "$RESULTS_FILE"

    imp_pp=$(echo "$imp_output" | grep "^pp " | grep -oP '[\d.]+(?= tok/s)')
    imp_tg=$(echo "$imp_output" | grep "^tg " | grep -oP '[\d.]+(?= tok/s)')
    log "  => imp  pp: ${imp_pp} tok/s  tg: ${imp_tg} tok/s"

    # GPU cooldown
    sleep 5

    # --- llama.cpp ---
    log ""
    log ">>> llama-bench"
    llama_output=$("$LLAMA_BENCH" -m "$model" -p "$PP" -n "$TG" -r "$REPS" -ngl 99 2>&1)
    echo "$llama_output" | tee -a "$RESULTS_FILE"

    # llama-bench markdown output: | model | ... | test | t/s |
    # pp line has "pp512", tg line has "tg128"
    # t/s column uses ± (Unicode), parse the number before it
    llama_pp=$(echo "$llama_output" | grep "pp${PP}" | grep -oP '[\d.]+(?= ±)' | tail -1)
    llama_tg=$(echo "$llama_output" | grep "tg${TG}" | grep -oP '[\d.]+(?= ±)' | tail -1)
    # Fallback: parse last numeric column from markdown table
    if [[ -z "$llama_pp" ]]; then
        llama_pp=$(echo "$llama_output" | grep "pp${PP}" | awk -F'|' '{print $(NF-1)}' | grep -oP '^[\s]*[\d.]+' | tr -d ' ' | head -1)
    fi
    if [[ -z "$llama_tg" ]]; then
        llama_tg=$(echo "$llama_output" | grep "tg${TG}" | awk -F'|' '{print $(NF-1)}' | grep -oP '^[\s]*[\d.]+' | tr -d ' ' | head -1)
    fi
    log "  => llama.cpp  pp: ${llama_pp} tok/s  tg: ${llama_tg} tok/s"

    RESULTS["$name"]="$imp_pp $imp_tg $llama_pp $llama_tg"

    log ""
    sleep 5
done

# Summary table
log ""
log "============================================"
log "  SUMMARY — RTX 5090, pp=$PP, tg=$TG, $REPS reps"
log "============================================"
log ""

# Header
printf "| %-28s | %10s | %10s | %8s | %10s | %10s | %8s |\n" \
    "Model" "imp pp" "llama pp" "Δ pp" "imp tg" "llama tg" "Δ tg" | tee -a "$RESULTS_FILE"
printf "|%-30s|%12s|%12s|%10s|%12s|%12s|%10s|\n" \
    "------------------------------" "------------" "------------" "----------" "------------" "------------" "----------" | tee -a "$RESULTS_FILE"

for name in "${MODEL_ORDER[@]}"; do
    read -r imp_pp imp_tg llama_pp llama_tg <<< "${RESULTS[$name]}"

    # Compute deltas
    if [[ -n "$llama_pp" && -n "$imp_pp" ]]; then
        delta_pp=$(awk "BEGIN {printf \"%.1f\", (($imp_pp - $llama_pp) / $llama_pp) * 100}")
    else
        delta_pp="N/A"
    fi
    if [[ -n "$llama_tg" && -n "$imp_tg" ]]; then
        delta_tg=$(awk "BEGIN {printf \"%.1f\", (($imp_tg - $llama_tg) / $llama_tg) * 100}")
    else
        delta_tg="N/A"
    fi

    printf "| %-28s | %8s t/s | %8s t/s | %6s%% | %8s t/s | %8s t/s | %6s%% |\n" \
        "$name" "$imp_pp" "$llama_pp" "$delta_pp" "$imp_tg" "$llama_tg" "$delta_tg" | tee -a "$RESULTS_FILE"
done

log ""
log "Done. Full results saved to $RESULTS_FILE"
