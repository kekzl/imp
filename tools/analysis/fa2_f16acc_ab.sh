#!/bin/bash
# Issue #597: A/B the f16-accumulate QK^T experiment for the FA2 prefill kernel.
# Fresh ncu (2026-06-09) showed the FP16QK FA2 kernel is tensor-pipe-leaning
# (52.8% pipe active, the busiest unit) running f32-acc QK^T at 1/4 rate on
# GeForce sm_120, occupancy smem-capped at 16.7%. f16-acc lifts the score MMA
# to the full-rate class. This measures prefill throughput + perplexity (quality
# gate) — run once on the baseline image, once on the f16-acc image.
set -uo pipefail
IMG=imp:test
MODELS=$HOME/models
REPO=$HOME/github.com/kekzl/imp
MODEL="${1:-/models/Qwen3-14B-NVFP4}"   # dense hd=128, exercises FP16QK FA2
CORPUS=/work/tools/analysis/ppl_corpus.txt

run() { docker run --rm --gpus all -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
          -v "$MODELS":/models -v "$REPO":/work "$IMG" "$@"; }

echo "######## FA2 f16-acc A/B: $(basename "$MODEL") ########"
echo "[warmup, discarded]"
run imp-cli --model "$MODEL" --bench --bench-pp 2048 --bench-reps 3 --max-tokens 1 \
    --temperature 0 >/dev/null 2>&1

for PP in 512 2048 4096; do
  echo "===== pp$PP (3 isolated trials) ====="
  for t in 1 2 3; do
    echo -n "  trial$t: "
    run imp-cli --model "$MODEL" --bench --bench-pp "$PP" --bench-reps 5 --max-tokens 1 \
        --temperature 0 --seed 42 2>&1 | grep -E '^pp' | tr '\n' ' '; echo
  done
done
echo "===== perplexity (quality gate, same corpus) ====="
run imp-cli --model "$MODEL" --perplexity "$CORPUS" --temperature 0 2>&1 | grep -iE 'perplexity'
echo "DONE"
