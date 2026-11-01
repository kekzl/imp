#!/usr/bin/env bash
# Two-image decode A/B on Ling-3.0-tiny (BF16 MoE): alternating rounds, one process per run.
# Usage: ling_moe_decode_ab.sh <image_a> <image_b> <out_dir> [rounds=3] [model_dir=Ling-3.0-tiny]
# Prints per-run pp512/tg128 tok/s lines (grep "^RESULT") and the per-run graph-capture status.
set -euo pipefail
IMG_A=$1
IMG_B=$2
OUT=$3
ROUNDS=${4:-3}
MODEL=${5:-Ling-3.0-tiny}
mkdir -p "$OUT"
run() {
    local tag=$1 img=$2 round=$3
    local log="$OUT/${tag}_r${round}.log"
    docker run --rm --gpus all -v "$HOME/models:/models:ro" -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        --entrypoint imp-cli "$img" --model "/models/$MODEL" --bench --bench-pp 512 --max-tokens 128 \
        --bench-reps 3 >"$log" 2>&1
    local pp tg capt
    pp=$(grep -oE '^pp +512 tokens .*\(( *[0-9.]+) tok/s' "$log" | grep -oE '[0-9.]+ tok/s' | head -1)
    tg=$(grep -oE '^tg +[0-9]+ tokens .*\(( *[0-9.]+) tok/s' "$log" | grep -oE '[0-9.]+ tok/s' | head -1)
    capt=$(grep -c 'not graph-capturable' "$log" || true)
    echo "RESULT $tag round=$round pp512=${pp:-NA} tg=${tg:-NA} not_capturable_lines=$capt"
}
for r in $(seq 1 "$ROUNDS"); do
    run A "$IMG_A" "$r"
    run B "$IMG_B" "$r"
done
