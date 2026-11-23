#!/usr/bin/env bash
# VRAM touch audit of one model (docs/VRAM_AUDIT.md): boots imp-server with
# diagnostics.vram_touch_audit, drives a fixed workload, stops it and prints per pool the MiB held
# since warmup and the MiB written. Pools that hold REVIEW_MIB or more with under REVIEW_PCT % written
# are listed for review: capacity the workload did not use, or read-only data (weights).
# The caller holds the GPU (gpu-busy-check + lock). Results are not committed; curate into the doc.
set -euo pipefail

MODEL=${MODEL:?MODEL=/models/<dir> (as mounted from MODELS_DIR)}
IMG=${IMG:-imp:test}
PORT=${PORT:-8081}
MODELS_DIR=${MODELS_DIR:-$HOME/models}
EXTRA=${EXTRA:-}                 # extra imp-server flags
LONG_WORDS=${LONG_WORDS:-4000}   # words of the long prompt (about 3 tokens each)
SHORT_N=${SHORT_N:-10}           # short requests, one after the other
PAR_N=${PAR_N:-4}                # parallel requests at the end
MAX_TOKENS=${MAX_TOKENS:-300}
MIN_MIB=${MIN_MIB:-8}
REVIEW_MIB=${REVIEW_MIB:-64}
REVIEW_PCT=${REVIEW_PCT:-50}
# Accepted review rows of this model: "<pool> <max MiB held> <reason>" per line. A review row that is
# missing there, or holds over 5 % more, fails the run. UPDATE=1 rewrites the file (reasons kept).
BASELINE=${BASELINE:-tests/vram_touch_baseline/$(basename "$MODEL").txt}
UPDATE=${UPDATE:-0}
NAME=imp-vram-touch
TMP=$(mktemp -d)
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true; rm -rf "$TMP"' EXIT

docker rm -f "$NAME" >/dev/null 2>&1 || true
# shellcheck disable=SC2086
docker run -d --name "$NAME" --gpus all -p "127.0.0.1:$PORT:8080" -v "$MODELS_DIR:/models" "$IMG" \
  --model "$MODEL" --set diagnostics.vram_touch_audit=true $EXTRA >/dev/null
for _ in $(seq 1 300); do
  curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" = true ] || { docker logs "$NAME" 2>&1 | tail -20; exit 1; }
  sleep 2
done
ID=$(curl -fsS "http://127.0.0.1:$PORT/v1/models" | jq -r '.data[0].id')

ask() { # prompt file, max tokens
  jq -n --rawfile p "$1" --arg m "$ID" --argjson t "$2" \
    '{model:$m,messages:[{role:"user",content:$p}],max_tokens:$t,temperature:0}' >"$1.json"
  curl -sS "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d @"$1.json" |
    jq -r 'if .usage then "  prompt \(.usage.prompt_tokens) + completion \(.usage.completion_tokens) tokens"
           else "  refused: \(.error.message // "no usage in the reply")" end' | cut -c1-200
}

for i in $(seq 1 "$SHORT_N"); do
  echo "Explain item $i of a list of basic sorting algorithms in a few paragraphs." >"$TMP/s$i"
  ask "$TMP/s$i" "$MAX_TOKENS" >/dev/null
done
echo "short requests: $SHORT_N"
{ for i in $(seq 1 "$LONG_WORDS"); do printf 'word%d ' "$((i % 997))"; done; echo; echo "How many words was that, roughly?"; } >"$TMP/long"
echo "long request, twice:"
ask "$TMP/long" "$MAX_TOKENS"
ask "$TMP/long" "$MAX_TOKENS"
for i in $(seq 1 "$PAR_N"); do
  echo "Write a short story about the number $i." >"$TMP/p$i"
  ask "$TMP/p$i" "$MAX_TOKENS" >/dev/null &
done
wait
echo "parallel requests: $PAR_N"
echo "device used: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

docker stop -t 180 "$NAME" >/dev/null
docker logs "$NAME" >"$TMP/log" 2>&1
grep -a 'touch_audit.cu' "$TMP/log" | sed 's/^.*touch_audit.cu:[0-9]*: //' >"$TMP/rows"
if ! grep -q '^touch TOTAL' "$TMP/rows"; then
  echo "no touch audit table in the server log (image without diagnostics.vram_touch_audit?)" >&2
  exit 1
fi
echo
echo "pool                              MiB held  MiB written  (rows of $MIN_MIB MiB or more)"
awk -v min="$MIN_MIB" '$1 == "touch" && ($3 + 0 >= min || $2 == "TOTAL") { sub(/^touch /, ""); print }' "$TMP/rows"
echo
echo "review ($REVIEW_MIB MiB or more held, under $REVIEW_PCT % written; arena takes summed per file):"
# pool (an arena take without its line), MiB held, MiB written
awk -v mib="$REVIEW_MIB" -v pct="$REVIEW_PCT" \
  '$1 == "touch" && $2 != "TOTAL" && $3 + 0 >= mib && $5 + 0 < pct {
     p = $2; if (p ~ /^arena:/) sub(/:[0-9]+$/, "", p); held[p] += $3; wr[p] += $4 }
   END { for (p in held) printf "%s %.1f %.1f\n", p, held[p], wr[p] }' "$TMP/rows" | sort >"$TMP/review"
awk '{ printf "  %-44s %9.1f held %9.1f written\n", $1, $2, $3 }' "$TMP/review"

if [ "$UPDATE" = 1 ]; then
  mkdir -p "$(dirname "$BASELINE")"
  touch "$BASELINE"
  awk 'FILENAME == ARGV[1] { if ($1 !~ /^#/ && NF >= 3) { r = $3; for (i = 4; i <= NF; i++) r = r " " $i; why[$1] = r } next }
       { printf "%s %.0f %s\n", $1, $2 + 1, ($1 in why) ? why[$1] : "TODO" }' "$BASELINE" "$TMP/review" >"$TMP/base"
  { echo "# VRAM touch audit baseline: <pool> <max MiB held> <reason>. make vram-touch-audit MODEL=$MODEL"; cat "$TMP/base"; } >"$BASELINE"
  echo "baseline written: $BASELINE"
  exit 0
fi
if [ ! -f "$BASELINE" ]; then
  echo "no baseline at $BASELINE: nothing gated (UPDATE=1 writes one)"
  exit 0
fi
fail=0
while read -r pool held _; do
  line=$(grep -E "^$pool " "$BASELINE" || true)
  max=$(echo "$line" | awk '{ print $2 }')
  why=$(echo "$line" | cut -d' ' -f3-)
  if [ -z "$line" ]; then
    echo "FAIL $pool: $held MiB held and mostly unwritten, not in $BASELINE"; fail=1
  elif [ "$why" = TODO ] || [ -z "$why" ]; then
    echo "FAIL $pool: baseline row without a reason"; fail=1
  elif awk -v h="$held" -v m="$max" 'BEGIN { exit !(h > m * 1.05) }'; then
    echo "FAIL $pool: $held MiB held, baseline $max MiB"; fail=1
  fi
done <"$TMP/review"
[ "$fail" = 0 ] && echo "vram touch audit: OK against $BASELINE"
exit "$fail"
