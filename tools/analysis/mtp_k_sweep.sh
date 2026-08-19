#!/bin/bash
# MTP k-sweep: throughput, acceptance and cost-per-verify against chain length.
#
# Usage: bash tools/analysis/mtp_k_sweep.sh            # k=0..3, 2 rounds
#        KS="0 1" ROUNDS=1 bash tools/analysis/mtp_k_sweep.sh
#        MTP_MODEL=/models/<other> bash tools/analysis/mtp_k_sweep.sh
#
# Prints CSV: k,round,tokens,ms,tok_s,drafted,accepted,verifies
# Results and how to read them: docs/roadmap.md, "Re-measured on the fixed build".
#
# Controls, each one earned by a past false result:
#   - speculative.ngram=false      MTP head is the ONLY drafter
#   - speculative.mtp_econ_min_emit=0  the guard unbinds after 8 verifies otherwise
#   - server.prefix_cache=false    a hit fakes throughput
#   - --think-budget 0             the default 0.5 disables speculation in a
#                                  think block, and this IS a reasoning model
#   - fresh process per arm, arms ALTERNATED across rounds
#   - tokens from usage.completion_tokens, verifies from /metrics
set -uo pipefail
IMG=${IMP_IMAGE:-imp:test}
MODEL=${MTP_MODEL:-/models/Qwen3.8-27B-NVFP4}
PORT=8099
ROUNDS=${ROUNDS:-2}
KS=${KS:-"0 1 2 3"}

cleanup(){ docker rm -f mtpsweep >/dev/null 2>&1; }
trap cleanup EXIT

start_arm(){ # k
  docker rm -f mtpsweep >/dev/null 2>&1
  docker run -d --name mtpsweep --gpus all -p $PORT:8080 \
    -v /home/kekz/models:/models "$IMG" \
    imp-server --host 0.0.0.0 --port 8080 --model "$MODEL" --think-budget 0 \
      --set speculative.ngram=false \
      --set speculative.mtp_k=$1 \
      --set speculative.mtp_econ_min_emit=0 \
      --set server.prefix_cache=false ${EXTRA:-} >/dev/null
  for _ in $(seq 1 200); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    docker ps --format '{{.Names}}' | grep -q '^mtpsweep$' || { echo "DIED k=$1"; docker logs mtpsweep 2>&1|tail -20; return 1; }
    sleep 3
  done; echo "TIMEOUT k=$1"; return 1
}

# Refuse to measure on a card someone else is on — the load says so since #1476.
poisoned(){ docker logs mtpsweep 2>&1 | grep -q "Weight upload consumed"; }

metrics(){ curl -s "http://127.0.0.1:$PORT/metrics" | awk '/^imp_spec_(drafted|accepted|verify_steps)_total/{print $1"="$2}' | tr '\n' ' '; }

ask(){ # prompt -> "tokens elapsed_ms"
  local p="$1" t0 t1
  t0=$(date +%s%N)
  local r
  r=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$(basename $MODEL)\",\"messages\":[{\"role\":\"user\",\"content\":$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$p")}],\"max_tokens\":700,\"temperature\":0,\"top_k\":1,\"stream\":false}")
  t1=$(date +%s%N)
  python3 -c "
import json,sys
try:
  d=json.loads(sys.argv[1]); print(d['usage']['completion_tokens'], int(($t1-$t0)/1000000))
except Exception as e: print(0,0)
" "$r"
}

PROMPTS=(
 "Explain how a paged KV cache works in an LLM inference engine, and why block size matters."
 "Write a Python function that merges overlapping intervals, then explain its complexity."
 "List the trade-offs between speculative decoding and larger batch sizes on a memory-bound GPU."
)

echo "k,round,tokens,ms,tok_s,drafted,accepted,verifies"
for r in $(seq 1 $ROUNDS); do
  # alternate direction each round so an ordering effect cannot masquerade as a trend
  ORDER="$KS"; [ $((r % 2)) -eq 0 ] && ORDER=$(echo $KS | tr ' ' '\n' | tac | tr '\n' ' ')
  for k in $ORDER; do
    start_arm "$k" || continue
    if poisoned; then echo "# k=$k round=$r POISONED CARD — skipping" >&2; continue; fi
    tot_tok=0; tot_ms=0
    for p in "${PROMPTS[@]}"; do
      read -r tk ms <<< "$(ask "$p")"
      tot_tok=$((tot_tok+tk)); tot_ms=$((tot_ms+ms))
    done
    m=$(metrics)
    d=$(echo "$m" | grep -oP 'drafted_total=\K[0-9.e+]+' || echo 0)
    a=$(echo "$m" | grep -oP 'accepted_total=\K[0-9.e+]+' || echo 0)
    v=$(echo "$m" | grep -oP 'verify_steps_total=\K[0-9.e+]+' || echo 0)
    ts=$(python3 -c "print(f'{$tot_tok/($tot_ms/1000):.2f}' if $tot_ms>0 else '0')")
    echo "$k,$r,$tot_tok,$tot_ms,$ts,$d,$a,$v"
  done
done
