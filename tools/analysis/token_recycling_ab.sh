#!/bin/bash
# token_recycling on/off A/B, alternating arms, fresh process each.
#
# Usage: bash tools/analysis/token_recycling_ab.sh
#        ROUNDS=3 RECYCLE_MODEL=/models/<other> bash tools/analysis/token_recycling_ab.sh
#
# Prints CSV: arm,round,tokens,ms,tok_s,drafted,accepted,verifies
# Defaults to Qwen3-14B-Q6_K, the model the roadmap's -7% verdict used, so a
# re-run is comparable. Results: docs/roadmap.md, gap 5.
set -uo pipefail
IMG=${IMP_IMAGE:-imp:test}
MODEL=${RECYCLE_MODEL:-/models/Qwen3-14B-Q6_K.gguf}
PORT=8098
ROUNDS=${ROUNDS:-2}
cleanup(){ docker rm -f recyc >/dev/null 2>&1; }
trap cleanup EXIT
start(){ # on|off
  docker rm -f recyc >/dev/null 2>&1
  docker run -d --name recyc --gpus all -p $PORT:8080 -v "${MODELS_DIR:-$HOME/models}":/models "$IMG" \
    imp-server --host 0.0.0.0 --port 8080 --model "$MODEL" --think-budget 0 \
      --set speculative.token_recycling=$1 --set server.prefix_cache=false >/dev/null
  for _ in $(seq 1 200); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    docker ps --format '{{.Names}}'|grep -q '^recyc$' || { docker logs recyc 2>&1|tail -12; return 1; }; sleep 3; done
  return 1
}
PROMPTS=(
 "A farmer has 17 sheep. All but 9 run away. He buys twice as many as remain, then sells 5. Work through it step by step."
 "Explain why merge sort is O(n log n) while insertion sort is O(n^2), deriving both from first principles."
 "Design a rate limiter for an API that must allow bursts but bound the hourly total. Reason about the trade-offs."
)
echo "arm,round,tokens,ms,tok_s,drafted,accepted,verifies"
for r in $(seq 1 $ROUNDS); do
  ARMS="false true"; [ $((r % 2)) -eq 0 ] && ARMS="true false"
  for a in $ARMS; do
    start "$a" || continue
    docker logs recyc 2>&1 | grep -q "Weight upload consumed" && { echo "# POISONED" >&2; continue; }
    tot=0; ms=0
    for p in "${PROMPTS[@]}"; do
      t0=$(date +%s%N)
      n=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$(basename $MODEL)\",\"messages\":[{\"role\":\"user\",\"content\":$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$p")}],\"max_tokens\":600,\"temperature\":0,\"top_k\":1,\"stream\":false}" \
        | python3 -c "import json,sys
try: print(json.load(sys.stdin)['usage']['completion_tokens'])
except Exception: print(0)")
      t1=$(date +%s%N); tot=$((tot+n)); ms=$((ms+ (t1-t0)/1000000 ))
    done
    m=$(curl -s "http://127.0.0.1:$PORT/metrics" | awk '/^imp_spec_(drafted|accepted|verify_steps)_total/{printf "%s ", $2}')
    ts=$(python3 -c "print(f'{$tot/($ms/1000):.2f}' if $ms>0 else 0)")
    echo "$a,$r,$tot,$ms,$ts,$(echo $m | tr ' ' ',')"
  done
done
