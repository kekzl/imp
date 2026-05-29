#!/bin/bash
# A/B the Sawtooth wavefront reordering (PR #456) in flash_attention_blackwell.
# Forces the WMMA fallback path (fp8_fmha=never, fmha_sm120=never) so the
# sawtooth kernel actually runs, then compares pp with IMP_SAWTOOTH 1 vs 0.
set -uo pipefail
cd /src/build-ciq
MODEL="${MODEL:-/models/Qwen3-14B-NVFP4}"
REPS="${REPS:-8}"
OBJ=CMakeFiles/imp.dir/src/compute/attention_blackwell.cu.o
BASE_CMD=$(ninja -t commands "$OBJ" 2>/dev/null | tail -1)
LINK_CMD=$(ninja -t commands imp-cli 2>/dev/null | tail -1)

run() {  # $1 = pp size; prints "pp_tok/s | active_log"
  local pp="$1"
  local out
  out=$(./imp-cli --model "$MODEL" --bench --bench-pp "$pp" --bench-reps "$REPS" \
        --max-tokens 1 --temperature 0 --prefill-chunk-size 0 \
        --set attention.fp8_fmha=never --set attention.fmha_sm120=never \
        --set attention.fmha_prefill_threshold=1 2>&1)
  local tps active
  tps=$(echo "$out" | grep -oP '^pp\s+'"$pp"'\s.*\(\s*\K[0-9.]+' | head -1)
  active=$(echo "$out" | grep -oE "flash_attention_blackwell.*ACTIVE: sawtooth=[01]" | head -1)
  echo "${tps:-FAIL} | ${active:-<blackwell NOT active!>}"
}

for PP in 4096 8192 16384; do
  echo "===== pp=$PP reps=$REPS ====="
  # sawtooth ON (default build)
  eval "$BASE_CMD" 2>/tmp/cc.log && eval "$LINK_CMD" 2>/dev/null || { echo "build ON fail"; tail -3 /tmp/cc.log; break; }
  echo "  sawtooth=ON  : $(run $PP)"
  sleep 10
  # sawtooth OFF
  eval "$BASE_CMD -DIMP_SAWTOOTH=0" 2>/tmp/cc.log && eval "$LINK_CMD" 2>/dev/null || { echo "build OFF fail"; tail -3 /tmp/cc.log; break; }
  echo "  sawtooth=OFF : $(run $PP)"
  sleep 10
done
# restore default
eval "$BASE_CMD" 2>/dev/null; eval "$LINK_CMD" 2>/dev/null; echo "(restored sawtooth=ON build)"
