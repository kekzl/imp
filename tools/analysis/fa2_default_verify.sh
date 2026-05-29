#!/bin/bash
# Verify the FA2 default-on flip end-to-end (no --set): FA2 must activate on a
# long Qwen3 prefill, and Gemma (hd=256) must stay coherent via fp8 fallback.
set -uo pipefail
F="The annual infrastructure review summarizes network capacity, uptime, and incident response metrics across all regional data centers and edge sites. "
PR=""; for i in $(seq 1 260); do PR+="$F"; done
PR+=$'\n\nIn one sentence, what does the text describe?'

echo "===== Qwen3-14B-NVFP4 (default settings — FA2 should be ON now) ====="
imp-cli --model /models/Qwen3-14B-NVFP4 --prompt "$PR" --max-tokens 36 --temperature 0 2>&1 \
  | grep -oE "register-resident kernel ACTIVE|pp +[0-9]+ tokens|\[tok=[0-9]+ [^]]*\]" | tr '\n' ' '
echo; echo

echo "===== Gemma-4-26B-A4B-NVFP4 (hd=256 — FA2 must DECLINE, fp8 fallback, stay coherent) ====="
imp-cli --model /models/Gemma-4-26B-A4B-it-NVFP4 --prompt "What is the capital of France? Answer in one word." \
  --max-tokens 12 --temperature 0 2>&1 \
  | grep -oE "register-resident kernel ACTIVE|\[tok=[0-9]+ [^]]*\]" | tr '\n' ' '
echo; echo DONE
