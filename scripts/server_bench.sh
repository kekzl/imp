#!/usr/bin/env bash
# server_bench.sh <imp-server> <model> [clients] [max_tokens] [port]: boots imp-server, warms it with
# one request, then sends <clients> concurrent streamed chat requests (SSE writer included;
# ignore_eos, temperature 0) and prints the aggregate completion tok/s and the slowest request's
# wall ms: "<tok/s> <max_ms>".
# The verify.sh server gate and gen_perf_baseline.sh share it (roadmap row 44). Exit 1 = no number.
set -uo pipefail
SRV="$1" MODEL="$2" CLIENTS="${3:-8}" MAXTOK="${4:-128}" PORT="${5:-18180}"
LOG="$(mktemp)"
"$SRV" --model "$MODEL" --host 127.0.0.1 --port "$PORT" --set speculative.ngram=false >"$LOG" 2>&1 &
PID=$!
trap 'kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; rm -f "$LOG" "$LOG".r*' EXIT
for _ in $(seq 1 300); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 "$PID" 2>/dev/null || { tail -5 "$LOG" >&2; exit 1; }
    sleep 1
done
M=$(curl -sf "http://127.0.0.1:$PORT/v1/models" | jq -r '.data[0].id') || exit 1
body() {
    jq -nc --arg m "$M" --argjson n "$MAXTOK" --arg p "$1" \
        '{model:$m,messages:[{role:"user",content:$p}],max_tokens:$n,temperature:0,ignore_eos:true,
          stream:true,stream_options:{include_usage:true},chat_template_kwargs:{enable_thinking:false}}'
}
req() { curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$1"; }
req "$(body "warm up")" >/dev/null
t0=$(date +%s%N)
for i in $(seq 1 "$CLIENTS"); do
    (s=$(date +%s%N); r=$(req "$(body "Client $i: describe a city you like.")");
     echo "$(grep -o '"completion_tokens":[0-9]*' <<<"$r" | tail -1 | cut -d: -f2) $((($(date +%s%N) - s) / 1000000))" >"$LOG.r$i") &
done
wait $(jobs -p | grep -v "^$PID$") 2>/dev/null
wall_ms=$((($(date +%s%N) - t0) / 1000000))
cat "$LOG".r* | awk -v w="$wall_ms" '{t+=$1; if($2>m)m=$2} END{if(t==0)exit 1; printf "%.2f %d\n", t*1000/w, m}'
