#!/bin/bash
# FA2 long-prefill coherence + needle retrieval (fa2 on vs never). ~3600-tok prompt.
set -uo pipefail
cd /src/build-ciq
MODEL="${MODEL:-/models/Qwen3-14B-NVFP4}"
F="The quarterly logistics report covers warehouse throughput and inventory turnover. "
PROMPT=""
for i in $(seq 1 170); do PROMPT+="$F"; done
PROMPT+="CRITICAL: The access code for vault seven is CRIMSON-PELICAN-93. "
for i in $(seq 1 90); do PROMPT+="$F"; done
PROMPT+=$'\n\nWhat is the access code for vault seven?'

for FA2 in on never; do
  echo "===== fmha_fa2=$FA2 ====="
  ./imp-cli --model "$MODEL" --prompt "$PROMPT" --max-tokens 48 --temperature 0 \
    --set attention.fmha_fa2="$FA2" 2>&1 \
    | grep -oE "register-resident kernel ACTIVE|\[tok=[0-9]+ [^]]*\]|pp +[0-9]+ tokens" | tr '\n' ' '
  echo
done
echo DONE
