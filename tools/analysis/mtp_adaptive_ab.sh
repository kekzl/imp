#!/bin/bash
# Adaptive MTP chain depth A/B: k=1 vs fixed k=2 vs adaptive k=2, on a
# draft-poor and a draft-rich prompt class. Single image, alternating arms,
# fresh server process per arm; econ guard at its shipping default.
#
# Usage: bash tools/analysis/mtp_adaptive_ab.sh              # 3 rounds
#        ROUNDS=2 CLASSES="poor" bash tools/analysis/mtp_adaptive_ab.sh
#        THINK=1 CLASSES="poor" bash tools/analysis/mtp_adaptive_ab.sh
#        ARMS="k2ad w2ad" THINK=1 bash tools/analysis/mtp_adaptive_ab.sh   # width axis
# THINK=1 keeps the model's thinking on (drops --think-budget 0) and raises
# max_tokens to 1024 - the think-traffic regime where a spec config must also
# be judged (#1796: numbers praising a spec config need a think arm).
#
# Prints CSV: class,arm,round,tokens,ms,tok_s,drafted,accepted,verifies
# Controls inherited from mtp_k_sweep.sh (each earned by a past false result):
# ngram=false (MTP is the only drafter), prefix_cache=false, --think-budget 0,
# fresh process per arm, arms alternated across rounds, tokens from
# usage.completion_tokens, counters from /metrics.
set -uo pipefail
IMG=${IMP_IMAGE:-imp:test}
MODEL=${MTP_MODEL:-/models/Qwen3.8-27B-NVFP4}
PORT=8099
ROUNDS=${ROUNDS:-3}
ARMS=${ARMS:-"k1 k2fix k2ad"}
CLASSES=${CLASSES:-"poor rich"}
OUTDIR=${OUTDIR:-/tmp/mtp_adaptive_ab}
THINK=${THINK:-0}
THINK_FLAG="--think-budget 0"; MAXTOK=700
[ "$THINK" = 1 ] && { THINK_FLAG=""; MAXTOK=1024; }
mkdir -p "$OUTDIR"

cleanup(){ docker rm -f mtpab >/dev/null 2>&1; }
trap cleanup EXIT

arm_flags(){ # arm -> --set flags
  case "$1" in
    k0)    echo "--set speculative.mtp_k=0" ;;
    k1)    echo "--set speculative.mtp_k=1" ;;
    k2fix) echo "--set speculative.mtp_k=2 --set speculative.mtp_adaptive_k=false" ;;
    k2ad)  echo "--set speculative.mtp_k=2" ;;
    # Width axis (roadmap gap 5): W=2 multi-candidate chains on top of the
    # fixed / adaptive depth-2 arms.
    w2fix) echo "--set speculative.mtp_k=2 --set speculative.mtp_adaptive_k=false --set speculative.mtp_tree_width=2" ;;
    w2ad)  echo "--set speculative.mtp_k=2 --set speculative.mtp_tree_width=2" ;;
  esac
}

start_arm(){ # arm
  docker rm -f mtpab >/dev/null 2>&1
  # shellcheck disable=SC2046
  docker run -d --name mtpab --gpus all -p $PORT:8080 \
    -v "${MODELS_DIR:-$HOME/models}":/models "$IMG" \
    imp-server --host 0.0.0.0 --port 8080 --model "$MODEL" $THINK_FLAG \
      --set speculative.ngram=false \
      --set server.prefix_cache=false $(arm_flags "$1") ${EXTRA:-} >/dev/null
  for _ in $(seq 1 200); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    docker ps --format '{{.Names}}' | grep -q '^mtpab$' || { echo "DIED $1"; docker logs mtpab 2>&1|tail -20; return 1; }
    sleep 3
  done; echo "TIMEOUT $1"; return 1
}

# Herestring dodges the pipefail/EPIPE race (see mtp_k_sweep.sh).
poisoned(){ grep -q "Weight upload consumed" <<< "$(docker logs mtpab 2>&1)"; }

metrics(){ curl -s "http://127.0.0.1:$PORT/metrics" | awk '/^imp_spec_(drafted|accepted|verify_steps)_total/{print $1"="$2}' | tr '\n' ' '; }

ask(){ # prompt outfile -> "tokens elapsed_ms"
  local p="$1" of="$2" t0 t1 r
  t0=$(date +%s%N)
  r=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$(basename $MODEL)\",\"messages\":[{\"role\":\"user\",\"content\":$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$p")}],\"max_tokens\":$MAXTOK,\"temperature\":0,\"top_k\":1,\"stream\":false}")
  t1=$(date +%s%N)
  python3 -c "
import json,sys
try:
  d=json.loads(sys.argv[1])
  open(sys.argv[2],'a').write(d['choices'][0]['message'].get('content') or '')
  print(d['usage']['completion_tokens'], int(($t1-$t0)/1000000))
except Exception: print(0,0)
" "$r" "$of"
}

POOR=(
 "Explain how a paged KV cache works in an LLM inference engine, and why block size matters."
 "Write a Python function that merges overlapping intervals, then explain its complexity."
 "List the trade-offs between speculative decoding and larger batch sizes on a memory-bound GPU."
)
PARA="The quick brown fox jumps over the lazy dog while the river runs quietly past the old mill, and the miller counts his sacks of grain one by one before the sun sets behind the hills."
RICH=(
 "Repeat the following paragraph exactly, word for word, five times in a row, with no commentary: $PARA"
 "Count from 1 to 150, one number per line, digits only, no commentary."
 "Print the line 'the wheels on the bus go round and round' exactly 30 times, numbered 1. to 30."
)

echo "class,arm,round,tokens,ms,tok_s,drafted,accepted,verifies"
for r in $(seq 1 $ROUNDS); do
  ORDER="$ARMS"; [ $((r % 2)) -eq 0 ] && ORDER=$(echo $ARMS | tr ' ' '\n' | tac | tr '\n' ' ')
  for arm in $ORDER; do
    for cls in $CLASSES; do
      start_arm "$arm" || continue
      if poisoned; then echo "# $arm/$cls round=$r POISONED CARD" >&2; continue; fi
      tot_tok=0; tot_ms=0
      of="$OUTDIR/${cls}_${arm}_r${r}.txt"; : > "$of"
      if [ "$cls" = poor ]; then PS=("${POOR[@]}"); else PS=("${RICH[@]}"); fi
      for p in "${PS[@]}"; do
        read -r tk ms <<< "$(ask "$p" "$of")"
        tot_tok=$((tot_tok+tk)); tot_ms=$((tot_ms+ms))
      done
      m=$(metrics)
      d=$(echo "$m" | grep -oP 'drafted_total=\K[0-9.e+]+' || echo 0)
      a=$(echo "$m" | grep -oP 'accepted_total=\K[0-9.e+]+' || echo 0)
      v=$(echo "$m" | grep -oP 'verify_steps_total=\K[0-9.e+]+' || echo 0)
      ts=$(python3 -c "print(f'{$tot_tok/($tot_ms/1000):.2f}' if $tot_ms>0 else '0')")
      echo "$cls,$arm,$r,$tot_tok,$tot_ms,$ts,$d,$a,$v"
    done
  done
done
