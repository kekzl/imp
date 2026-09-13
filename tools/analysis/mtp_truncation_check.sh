#!/bin/bash
# Does MTP speculation truncate answers? Runs six prompts and flags any answer stopping early
# (finish_reason=stop well under budget; signature is a ~40-token answer restating the question).
# deterministic_gemm is pinned on purpose: it stabilises the degenerate state rather than
# removing it, making this a repeatable check. Findings: docs/LIMITATIONS.md.
# Usage: bash tools/analysis/mtp_truncation_check.sh 1|0 (MTP on|control).
# MTP_EXTRA_SET='k=v' adds further --set pairs without editing this file.
set -uo pipefail
IMG=${IMP_IMAGE:-imp:test}; MODEL=${MTP_MODEL:-/models/Qwen3.8-27B-NVFP4}; PORT=8095; K=$1
EXTRA=""; for kv in ${MTP_EXTRA_SET:-}; do EXTRA="$EXTRA --set $kv"; done
cleanup(){ docker rm -f prproc >/dev/null 2>&1; }
trap cleanup EXIT
docker rm -f prproc >/dev/null 2>&1
docker run -d --name prproc --gpus all -p $PORT:8080 -v "${MODELS_DIR:-$HOME/models}":/models "$IMG" \
  imp-server --host 0.0.0.0 --port 8080 --model "$MODEL" --think-budget 0 \
    --set speculative.mtp_k=$K --set speculative.ngram=false --set server.prefix_cache=false \
    --set runtime.deterministic_gemm=true $EXTRA >/dev/null
for _ in $(seq 1 200); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  docker ps --format '{{.Names}}'|grep -q '^prproc$' || break; sleep 3; done
while IFS= read -r p; do
  curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$(basename $MODEL)\",\"messages\":[{\"role\":\"user\",\"content\":$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$p")}],\"max_tokens\":400,\"temperature\":0,\"top_k\":1}" \
    | python3 -c "
import json,sys
d=json.load(sys.stdin); ch=d['choices'][0]; c=ch['message'].get('content') or ''
bad = ch.get('finish_reason')=='stop' and len(c)<600
print(f'  k=$K {\"DEGENERATE\" if bad else \"ok        \"} {len(c):5d}B {ch.get(\"finish_reason\"):6s} {c[-38:]!r}')"
done <<'PROMPTS'
Explain how a paged KV cache works in an LLM inference engine, and why block size matters.
Describe the trade-offs between grouped-query attention and multi-head attention, and when each wins.
Walk through how a CUDA kernel launch reaches the GPU, and what the driver does in between.
Explain what makes an LLM inference engine memory-bound at batch size one, and how to tell.
Summarise the difference between prefill and decode in transformer inference, and why they scale differently.
What is speculative decoding, and under what conditions does it fail to pay off?
PROMPTS

# Engine counters read BEFORE the trap removes the container: once it's gone the log is gone,
# and an arm whose knob never engaged looks identical to one whose knob did nothing.
echo "  --- engine counters (an arm with no line here proves nothing) ---"
docker logs prproc 2>&1 | grep -E '\[spec-ngram\]' | sed 's/^/  /' || true
