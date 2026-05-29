#!/bin/bash
# Long-context coherence probe for the FA2-active prefill path (>4096 tokens).
# Needle-in-haystack: embeds a secret in long filler, asks for it. If FA2
# long-prefill attention is correct, the model retrieves it. A/B fa2 on vs never.
set -uo pipefail
cd /src/build-ciq
MODEL="${MODEL:-/models/Qwen3-14B-NVFP4}"

# Build a long prompt (~5000+ tokens): filler before + needle + filler after + question.
FILLER="The quarterly logistics report covers warehouse throughput, fleet scheduling, and inventory turnover across the regional distribution network. "
PROMPT=""
for i in $(seq 1 90); do PROMPT+="$FILLER"; done
PROMPT+="IMPORTANT FACT: The access code for vault seven is CRIMSON-PELICAN-93. "
for i in $(seq 1 90); do PROMPT+="$FILLER"; done
PROMPT+=$'\n\nQuestion: What is the access code for vault seven? Answer with just the code.\nAnswer:'

gen() {  # $1 = never|on
  ./imp-cli --model "$MODEL" --prompt "$PROMPT" --max-tokens 24 --temperature 0 \
     --set attention.fmha_fa2="$1" 2>&1
}

for FA2 in never on; do
  echo "===== fmha_fa2=$FA2 ====="
  out=$(gen "$FA2")
  echo "$out" | grep -qE "FMHA FA2 register-resident kernel ACTIVE" && echo "[FA2-ACTIVE]" || echo "[no-FA2]"
  # print prompt token count + the generated answer (last non-log lines)
  echo "$out" | grep -iE "prompt|prefill.*tokens|^Answer|CRIMSON|tokens\b" | grep -viE "INFO|WARN" | tail -6
  echo "--- raw tail ---"
  echo "$out" | grep -vE "INFO|WARN|^\[" | tail -8
  echo
done
