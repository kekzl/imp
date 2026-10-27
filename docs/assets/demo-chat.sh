#!/usr/bin/env bash
# Demo client for docs/assets/demo.tape: one streaming chat request with curl, the answer printed
# as it arrives, then tokens, time to first token and decode tok/s. Needs only curl and bash >= 5.
# Usage: docs/assets/demo-chat.sh "prompt" [port 8090] [max_tokens 300]
PROMPT=${1:-"Explain in five short bullet points why a GPU needs high memory bandwidth for LLM decoding."}
PORT=${2:-8090}
MAX=${3:-300}
model=$(curl -s "http://127.0.0.1:$PORT/v1/models" | grep -o "\"id\":\"[^\"]*\"" | head -1 | cut -d\" -f4)
body=$(printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":%d,"temperature":0,"stream":true,"enable_thinking":false,"stream_options":{"include_usage":true}}' "$model" "$PROMPT" "$MAX")
t0=$EPOCHREALTIME; t1=""; n=0
while IFS= read -r line; do
    if [[ $line =~ \"content\":\"(([^\"\\]|\\.)*)\" ]]; then
        [ -n "$t1" ] || t1=$EPOCHREALTIME
        tok=${BASH_REMATCH[1]}; tok=${tok//\\\"/\"}
        printf '%b' "$tok"
    fi
    [[ $line =~ \"completion_tokens\":([0-9]+) ]] && n=${BASH_REMATCH[1]}
done < <(curl -sN "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$body")
t2=$EPOCHREALTIME
awk -v n="$n" -v a="$t0" -v b="$t1" -v c="$t2" 'BEGIN {
    printf "\n\n\033[1;32m%d tokens · first token after %.0f ms · %.1f tok/s\033[0m\n", n, (b - a) * 1000, (n - 1) / (c - b) }'
