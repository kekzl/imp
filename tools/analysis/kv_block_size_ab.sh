#!/usr/bin/env bash
# KV block size A/B on ONE binary: kv_cache.block_size=<A> vs =<B>, alternating
# arms, spec-off, tg128 per context length (AUDIT_arch_2026 B-5).
#
# The engine's auto rule (32 for n_kv_heads <= 4, else 16) dates from
# 2026-03-23 and carried no measurement. Run this on a model of each class:
# Qwen3-8B-Q8_0 (8 KV heads, auto 16) and Qwen3-14B-NVFP4 (8 KV heads) or
# Qwen3.8-27B-NVFP4-vllm (4 KV heads, auto 32). Block size moves the split-K
# count and the table-walk length, so the effect, if any, grows with context:
# 2048 alone cannot see it (scripts/bench_longctx_ab.sh rule 1).
#
# Usage:
#   tools/analysis/kv_block_size_ab.sh <bs-A> <bs-B> [ctx-list] [model-list] [image]
#   tools/analysis/kv_block_size_ab.sh 16 32 "2048 8192 32768" /models/Qwen3-8B-Q8_0.gguf
set -euo pipefail
BS_A="${1:?usage: $0 <bs-A> <bs-B> [ctx-list] [model-list] [image]}"
BS_B="${2:?usage: $0 <bs-A> <bs-B> [ctx-list] [model-list] [image]}"
CTXS="${3:-2048 8192 32768}"
MODELS="${4:-/models/Qwen3-8B-Q8_0.gguf /models/Qwen3.8-27B-NVFP4-vllm}"
IMG="${5:-imp:test}"
ROUNDS="${AB_ROUNDS:-3}"
MODELS_DIR="${IMP_MODELS_DIR:-$HOME/models}"

if command -v nvidia-smi >/dev/null 2>&1; then
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "${used:-0}" -gt 2000 ]; then
        echo "REFUSING: GPU holds ${used} MiB; free it first (docker ps)." >&2
        exit 2
    fi
fi

run_one() {  # $1=block size  $2=model  $3=ctx
    local bs="$1" model="$2" ctx="$3"
    timeout 1800 docker run --rm --gpus all -v "$MODELS_DIR:/models" "$IMG" \
        imp-cli --model "$model" --bench --bench-pp "$ctx" --bench-reps 3 --max-tokens 128 \
        --set speculative.ngram=false --set kv_cache.block_size="$bs" 2>&1 \
        | grep -oE 'tg +128 tokens.*\( *[0-9.]+ tok/s\)' | grep -oE '[0-9.]+' | tail -1
}

median() { printf '%s\n' "$@" | sort -n | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }

printf '%-42s %8s %10s %10s %9s %s\n' model ctx "bs=$BS_A" "bs=$BS_B" delta pairs
for model in $MODELS; do
    for ctx in $CTXS; do
        a_vals=(); b_vals=(); same=0
        for _ in $(seq "$ROUNDS"); do
            a=$(run_one "$BS_A" "$model" "$ctx"); b=$(run_one "$BS_B" "$model" "$ctx")
            [ -n "$a" ] && [ -n "$b" ] || { echo "no tg128 line for $model ctx=$ctx" >&2; exit 1; }
            a_vals+=("$a"); b_vals+=("$b")
            awk -v a="$a" -v b="$b" 'BEGIN{exit !(b>=a)}' && same=$((same+1))
        done
        ma=$(median "${a_vals[@]}"); mb=$(median "${b_vals[@]}")
        delta=$(awk -v a="$ma" -v b="$mb" 'BEGIN{printf "%+.2f%%", (b-a)/a*100}')
        printf '%-42s %8s %10s %10s %9s %d/%d B>=A\n' "$model" "$ctx" "$ma" "$mb" "$delta" "$same" "$ROUNDS"
    done
done
