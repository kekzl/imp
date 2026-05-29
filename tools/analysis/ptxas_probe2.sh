#!/bin/bash
# Probe2: the remaining high-value ptxas codegen knobs for fmha_sm120_fa2_kernel
# (maxrregcount already refuted — occupancy is smem-bound). Scoring: end-to-end pp4096.
set -uo pipefail
cd /src/build-ciq
MODEL="${MODEL:-/models/Qwen3-14B-NVFP4}"; PP=4096; REPS=10
OBJ=CMakeFiles/imp.dir/src/compute/attention_fmha_sm120.cu.o
BASE_CMD=$(ninja -t commands "$OBJ" 2>/dev/null | tail -1)
LINK_CMD=$(ninja -t commands imp-cli 2>/dev/null | tail -1)
bench(){ ./imp-cli --model "$MODEL" --bench --bench-pp $PP --bench-reps $REPS --max-tokens 1 \
  --temperature 0 --set attention.fmha_fa2=on 2>&1 | grep -oP '^pp\s+'$PP'\s.*\(\s*\K[0-9.]+' | head -1; }

declare -a KNOBS=(
  "baseline:"
  "expensive-on:-Xptxas --allow-expensive-optimizations=true"
  "loadcache-cg:-Xptxas --def-load-cache=cg"
  "loadcache-cs:-Xptxas --def-load-cache=cs"
  "sched-aggr:-Xptxas --allow-expensive-optimizations=true -Xptxas --def-load-cache=cg"
  "fastmath:--use_fast_math"
)
echo "pp=$PP reps=$REPS  (knob | pp tok/s)"
for kv in "${KNOBS[@]}"; do
  name="${kv%%:*}"; flags="${kv#*:}"
  eval "$BASE_CMD $flags" 2>/tmp/cc.log || { echo "$name | COMPILE FAIL"; tail -2 /tmp/cc.log; continue; }
  eval "$LINK_CMD" 2>/dev/null
  printf "%-14s | %s\n" "$name" "$(bench)"
  sleep 10
done
eval "$BASE_CMD" 2>/dev/null; eval "$LINK_CMD" 2>/dev/null; echo "(restored baseline)"
