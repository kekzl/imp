#!/bin/bash
# Issue #600: sweep the NVFP4 MoE-decode rows-per-block knob (moe.mr_nr).
# gemv_nvfp4 reaches only ~30% roofline on nvfp4-moe/tg256 (vs 61% dense).
# Hypothesis: the multi-row tile NR trades block-count (wave depth) for
# per-block work. Default 8 gives ~1.5-2.0 waves on Qwen3-30B-A3B; lowering
# NR raises the wave count. Decode-only A/B (the reliable signal on this box).
#
# Methodology: one process per config (isolation), discarded warmup run for
# clock ramp (>1s), 3 measured trials of 10-rep decode each. Clocks sampled
# during the run to confirm a healthy host day (mem 13801 MHz / ~500 W).
set -uo pipefail

IMG=imp:test
MODELS=$HOME/models
REPO=$HOME/github.com/kekzl/imp
MODEL="${1:-/models/Qwen3-30B-A3B-NVFP4-Modelopt}"

run() {  # NR -> decode bench, print tg lines
  docker run --rm --gpus all -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -v "$MODELS":/models -v "$REPO":/work "$IMG" \
    imp-cli --model "$MODEL" --bench --bench-pp 16 --bench-reps 10 \
    --max-tokens 256 --temperature 0 --set moe.mr_nr="$1" 2>&1 \
    | grep -E '^tg|^pp'
}

echo "######## MoE NR sweep: $(basename "$MODEL") ########"
echo "[warmup, discarded]"; run 8 >/dev/null 2>&1

( for i in 1 2 3 4 5 6; do
    nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw --format=csv,noheader
    sleep 5
  done > /tmp/moe_nr_clocks.log ) &

for NR in 4 8 16 32; do
  echo "===== moe.mr_nr=$NR ====="
  for t in 1 2 3; do
    echo -n "  trial$t: "; run "$NR" | tr '\n' '  '; echo
  done
done
wait
echo "==== clocks sampled during run (sm,mem,power) ===="
cat /tmp/moe_nr_clocks.log
echo "DONE"
