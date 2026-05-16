#!/bin/bash
# Sweep M ∈ {1,2,4,8,16,32,64,128,256,512} on Qwen3-32B Q4_K_M and compare
# the dequant+cuBLAS prefill path vs the existing `mmvq_kernel` (warp-per-
# output dp4a). Documents the crossover at M≈16 — above it cuBLAS wins,
# below it mmvq wins. See `q4k_mmvq_crossover_2026_05_15.md` for the full
# finding.
#
# Usage:
#   tools/analysis/bench_q4k_mmvq_crossover.sh [model_path]
#
# Default model: models/Qwen3-32B-Q4_K_M.gguf.

set -e
MODEL_PATH="${1:-/m/Qwen3-32B-Q4_K_M.gguf}"
MODELS_DIR="${MODELS_DIR:-$(pwd)/models}"

CONF_DIR="$(mktemp -d)"
trap "rm -rf '$CONF_DIR'" EXIT
cat > "$CONF_DIR/force_mmvq.conf" << 'EOF'
[gemma4]
force_mmvq = true
EOF

printf "%-5s %12s %12s %s\n" "M" "cuBLAS tok/s" "mmvq tok/s" "winner"
for M in 1 2 4 8 16 32 64 128 256 512; do
    cublas_tps=$(docker run --rm --gpus all \
        -v "$MODELS_DIR":/m imp:test \
        imp-cli --model "$MODEL_PATH" \
        --bench --bench-pp $M --bench-reps 3 --max-tokens 1 --temperature 0 2>&1 \
        | grep "pp" | grep "tokens" | tail -1 \
        | awk '{for(i=1;i<=NF;i++){if($i=="("){print $(i+1); exit}}}')

    mmvq_tps=$(docker run --rm --gpus all \
        -v "$MODELS_DIR":/m -v "$CONF_DIR":/conf imp:test \
        imp-cli --config /conf/force_mmvq.conf --model "$MODEL_PATH" \
        --bench --bench-pp $M --bench-reps 3 --max-tokens 1 --temperature 0 2>&1 \
        | grep "pp" | grep "tokens" | tail -1 \
        | awk '{for(i=1;i<=NF;i++){if($i=="("){print $(i+1); exit}}}')

    winner=$(awk -v a="$cublas_tps" -v b="$mmvq_tps" \
        'BEGIN { if (a+0 > b+0) print "cuBLAS"; else print "mmvq"; }')
    printf "%-5s %12s %12s %s\n" "$M" "$cublas_tps" "$mmvq_tps" "$winner"
done
