#!/bin/bash
# Zoo-wide FA2 default-on coherence check (pre-merge). Long prompt (FA2-active
# for hd=128) -> generate greedy -> eyeball coherent + note FA2 activation.
# Run with imp:test (attention.fmha_fa2=on default), no --set.
set -uo pipefail
F="The annual infrastructure review summarizes network capacity, uptime, and incident response metrics across all regional data centers. "
PROMPT=""; for i in $(seq 1 220); do PROMPT+="$F"; done
PROMPT+=$'\n\nIn one sentence, summarize the text.'

MODELS=(
  "Qwen3-8B-NVFP4-cortecs|dense-NVFP4-hd128"
  "Qwen3-30B-A3B-NVFP4-Modelopt|MoE-NVFP4-hd128"
  "Qwen3.6-35B-A3B-NVFP4|GDN+MoE-hybrid"
  "Phi-4-reasoning-plus-NVFP4|fused-proj"
  "Gemma-4-26B-A4B-it-NVFP4|hd256-fallback"
)
for entry in "${MODELS[@]}"; do
  m="${entry%%|*}"; arch="${entry#*|}"
  echo "===== $m  [$arch] ====="
  out=$(imp-cli --model "/models/$m" --prompt "$PROMPT" --max-tokens 24 --temperature 0 2>&1)
  echo "$out" | grep -qE "register-resident kernel ACTIVE" && echo "  [FA2-ACTIVE]" || echo "  [no-FA2 / fallback]"
  echo "$out" | grep -oE "^pp +[0-9]+ tokens" | head -1
  echo -n "  gen: "; echo "$out" | grep -oE "\[tok=[0-9]+ [^]]*\]" | head -14 | tr '\n' ' '
  echo
done
echo DONE
