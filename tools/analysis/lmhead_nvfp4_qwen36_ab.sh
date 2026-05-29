#!/bin/bash
# A/B NVFP4 lm_head on Qwen3.6-35B: decode speed + coherence, FP16 vs NVFP4.
set -uo pipefail
cd /src/build-ciq
M=/models/Qwen3.6-35B-A3B-NVFP4
PROMPT="Explain why the sky is blue, then say why sunsets are red."

decode() { ./imp-cli --model "$M" --bench --bench-pp 16 --bench-reps 3 --max-tokens 128 \
             --temperature 0 --set gemm.nvfp4_lm_head_gdn="$1" 2>&1 | grep -E '^tg'; }

cohere() { ./imp-cli --model "$M" --prompt "$PROMPT" --max-tokens 80 --temperature 0 \
             --set gemm.nvfp4_lm_head_gdn="$1" 2>&1 \
             | grep -oE '\][^[]*' | sed -E 's/^\] ?//' | tr -d '\n'; }

for F in false true; do
  lab=$([ "$F" = true ] && echo "NVFP4 lm_head" || echo "FP16 lm_head")
  echo "===== $lab (gdn=$F) ====="
  echo -n "decode: "; decode "$F" | tail -1
  echo -n "text: "; cohere "$F"; echo
done
echo DONE
