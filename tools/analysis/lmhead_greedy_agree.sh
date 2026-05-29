#!/bin/bash
# Rigor check for flipping gemm.nvfp4_lm_head_gdn default-on: greedy-token
# agreement FP16 vs NVFP4 lm_head across ALL GDN/hybrid models the flag affects.
# lm_head produces the argmax token; identical greedy sequences => argmax-
# preserving => quality-safe. Divergence => keep opt-in.
set -uo pipefail
cd /src/build-ciq
PROMPT="Explain step by step how a transformer neural network processes a sentence, then summarize in one line."
MODELS=(Qwen3.6-35B-A3B-NVFP4 Nemotron-3-Nano-30B-A3B-NVFP4 Nemotron-Labs-3-Elastic-30B-A3B-NVFP4)

ids() { # $1 model, $2 flag -> space-separated token id sequence
  ./imp-cli --model "/models/$1" --prompt "$PROMPT" --max-tokens 64 --temperature 0 \
    --set gemm.nvfp4_lm_head_gdn="$2" 2>&1 \
    | grep -oE 'tok=[0-9]+' | sed 's/tok=//' | tr '\n' ' '
}

for M in "${MODELS[@]}"; do
  echo "===== $M ====="
  A=$(ids "$M" false); B=$(ids "$M" true)
  if [ -z "$A" ] || [ -z "$B" ]; then echo "  (load/gen failed — skip)"; continue; fi
  # compare token-by-token, report first divergence + agreement count
  read -ra AA <<< "$A"; read -ra BB <<< "$B"
  n=${#AA[@]}; [ ${#BB[@]} -lt $n ] && n=${#BB[@]}
  match=0; firstdiv=-1
  for ((i=0;i<n;i++)); do
    if [ "${AA[i]}" = "${BB[i]}" ]; then match=$((match+1)); else [ $firstdiv -lt 0 ] && firstdiv=$i; fi
  done
  echo "  tokens compared=$n  agree=$match  first_divergence_at=$firstdiv"
  echo "  FP16 : ${AA[*]:0:14}"
  echo "  NVFP4: ${BB[*]:0:14}"
done
echo DONE
