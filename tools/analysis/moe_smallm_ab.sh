#!/bin/bash
# Issue #601: A/B the NVFP4 MoE grouped-GEMM paths for prefill. gemm_grouped_nvfp4
# reaches only 41-52% roofline on nvfp4-moe prefill (23% occupancy, 1-wave grid).
# Three arms:
#   A  default          — device-args CUTLASS 3.x grouped (Tier 1, default ON)
#   B  da-off + smallM   — hand-rolled M-aware tile kernel (Tier 2, opt-in)
#   C  da-off + grouped  — host-args CUTLASS grouped (Tier 3)
# The smallM (B) and host-grouped (C) tiers both require device_args=false.
# Result (2026-06-09, Qwen3-30B-A3B-NVFP4): default wins by ~25-32% — smallM
# regresses (confirms the 2026-05-14 device-args +11-39% A/B). Default is the
# best available path; 41% roofline is the small-M grouped structural ceiling.
set -uo pipefail
IMG=imp:test; MODELS=/home/kekz/models; REPO=/home/kekz/github.com/kekzl/imp
MODEL="${1:-/models/Qwen3-30B-A3B-NVFP4-Modelopt}"
CORPUS=/work/tools/analysis/ppl_corpus.txt
R() { docker run --rm --gpus all -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        -v "$MODELS":/models -v "$REPO":/work "$IMG" imp-cli --model "$MODEL" "$@"; }

echo "######## MoE grouped-GEMM A/B: $(basename "$MODEL") ########"
echo "[warmup]"; R --bench --bench-pp 2048 --bench-reps 3 --max-tokens 1 --temperature 0 >/dev/null 2>&1

arm() {  # $1=label  $2..=flags
  local lab="$1"; shift
  echo "-- $lab --"
  for PP in 512 2048; do for t in 1 2 3; do
    echo -n "  pp$PP t$t: "
    R --bench --bench-pp $PP --bench-reps 5 --max-tokens 1 --temperature 0 --seed 42 "$@" 2>&1 \
      | grep -E '^pp' | tr '\n' ' '; echo
  done; done
}
arm "A default (device-args)"
arm "B da-off + smallM" --set moe.nvfp4_device_args=false --set moe.nvfp4_smallM=true --set moe.nvfp4_smallM_threshold=128
arm "C da-off + grouped" --set moe.nvfp4_device_args=false

echo "===== perplexity (quality gate) ====="
echo -n "  default: "; R --perplexity "$CORPUS" --temperature 0 2>&1 | grep -iE 'perplexity'
echo -n "  smallM : "; R --perplexity "$CORPUS" --temperature 0 --set moe.nvfp4_device_args=false --set moe.nvfp4_smallM=true --set moe.nvfp4_smallM_threshold=128 2>&1 | grep -iE 'perplexity'
echo "DONE"
