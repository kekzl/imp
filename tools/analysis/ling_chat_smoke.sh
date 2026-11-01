#!/usr/bin/env bash
# Chat smoke on one imp-server (check-degeneration): capital, German story, 2-turn recall, code.
# Usage: ling_chat_smoke.sh <image> <out_dir> [model_dir=Ling-3.0-tiny] [port=18093]
# Writes <out_dir>/server.log and one <case>.json per request; prints content per case.
set -euo pipefail
IMG=$1
OUT=$2
MODEL=${3:-Ling-3.0-tiny}
PORT=${4:-18093}
NAME=ling-smoke-$$
mkdir -p "$OUT"
docker run -d --rm --name "$NAME" --gpus all -p "127.0.0.1:$PORT:8080" -v "$HOME/models:/models:ro" "$IMG" \
    imp-server --host 0.0.0.0 --port 8080 --model "/models/$MODEL" >/dev/null
trap 'docker logs "$NAME" >"$OUT/server.log" 2>&1 || true; docker stop "$NAME" >/dev/null 2>&1 || true' EXIT
for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 1
done
chat() {
    local name=$1 body=$2
    curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$body" \
        >"$OUT/$name.json"
    echo "=== $name"
    jq -r '.choices[0].message.content, "finish=\(.choices[0].finish_reason) tokens=\(.usage.completion_tokens)"' \
        "$OUT/$name.json"
}
chat capital '{"messages":[{"role":"user","content":"What is the capital of France? Answer in one sentence."}],"max_tokens":64,"temperature":0}'
chat story_de '{"messages":[{"role":"user","content":"Schreibe eine kurze Geschichte (etwa 150 Woerter) ueber einen Fuchs, der das Meer sucht."}],"max_tokens":400,"temperature":0.7}'
T1='{"role":"user","content":"My name is Ingrid and my favourite colour is teal. Please remember that."}'
chat recall_t1 "{\"messages\":[$T1],\"max_tokens\":64,\"temperature\":0}"
A1=$(jq -c '{role:"assistant",content:.choices[0].message.content}' "$OUT/recall_t1.json")
chat recall_t2 "{\"messages\":[$T1,$A1,{\"role\":\"user\",\"content\":\"What is my name and my favourite colour?\"}],\"max_tokens\":64,\"temperature\":0}"
chat code '{"messages":[{"role":"user","content":"Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring."}],"max_tokens":300,"temperature":0}'
