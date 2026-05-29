#!/bin/bash
# ptxas maxrregcount sweep for the register-resident FA2 prefill kernel
# (fmha_sm120_fa2_kernel, REG:144, SHARED:0 — occupancy is register-bound).
# Recompiles ONLY the attn TU per candidate, relinks imp-cli, benches pp4096.
# Run inside imp:ciq with build-ciq/ present, GPU, and /models mounted.
set -uo pipefail
cd /src/build-ciq

MODEL="${MODEL:-/models/Qwen3-14B-NVFP4}"
PP="${PP:-4096}"
REPS="${REPS:-8}"
OBJ=CMakeFiles/imp.dir/src/compute/attention_fmha_sm120.cu.o

# Capture the canonical compile + link commands once.
BASE_CMD=$(ninja -t commands "$OBJ" 2>/dev/null | tail -1)
LINK_CMD=$(ninja -t commands imp-cli 2>/dev/null | tail -1)
if [ -z "$BASE_CMD" ] || [ -z "$LINK_CMD" ]; then echo "FATAL: could not extract ninja commands"; exit 1; fi

bench() {
  ./imp-cli --model "$MODEL" --bench --bench-pp "$PP" --bench-reps "$REPS" \
     --max-tokens 1 --temperature 0 --set attention.fmha_fa2=on 2>&1 \
     | grep -oP '^pp\s+'"$PP"'\s.*\(\s*\K[0-9.]+' | head -1
}

regcount() {
  cuobjdump -res-usage "$OBJ" 2>/dev/null | grep -A1 "fa2_kernel" | grep -oP 'REG:\K[0-9]+' | head -1
}

echo "model=$MODEL pp=$PP reps=$REPS"
echo "rreg | regs | pp_tok/s"
echo "-----|------|---------"

for RREG in baseline 64 80 96 112 128 160 200; do
  if [ "$RREG" = "baseline" ]; then
    eval "$BASE_CMD" 2>/tmp/cc.log
  else
    eval "$BASE_CMD -Xptxas --maxrregcount=$RREG" 2>/tmp/cc.log
  fi
  if [ $? -ne 0 ]; then echo "$RREG | COMPILE FAIL"; tail -3 /tmp/cc.log; continue; fi
  eval "$LINK_CMD" 2>/dev/null
  R=$(regcount)
  TPS=$(bench)
  printf "%-5s| %-5s| %s\n" "$RREG" "${R:-?}" "${TPS:-FAIL}"
  sleep 8   # cooldown between candidates
done

# Restore baseline .o so the tree is clean afterwards.
eval "$BASE_CMD" 2>/dev/null; eval "$LINK_CMD" 2>/dev/null
echo "(restored baseline build)"
