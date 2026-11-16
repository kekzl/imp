#!/usr/bin/env bash
# Greedy output of one image via imp-server JSON (temperature 0, runtime.deterministic): writes
# $OUT_DIR/<TAG>.json per prompt index; compare two tags with cmp. Usage: flint_greedy_parity.sh IMAGE TAG.
set -euo pipefail
IMG=$1; TAG=$2
MODEL=${MODEL:-/models/flint-nvfp4-official}
OUT_DIR=${OUT_DIR:-/mnt/data/scratch/flint_perf/parity}
PORT=${PORT:-8093}
MAX_TOKENS=${MAX_TOKENS:-400}
mkdir -p "$OUT_DIR"
name="parity_$TAG"
docker rm -f "$name" >/dev/null 2>&1 || true
docker run -d --name "$name" --gpus all -p "127.0.0.1:$PORT:8080" -v "$HOME/models:/models:ro" "$IMG" \
    imp-server --model "$MODEL" --host 0.0.0.0 --port 8080 --set runtime.deterministic=true >/dev/null
for _ in $(seq 1 120); do
    curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && break
    sleep 2
done
prompts=(
    "Write a Python function that merges two sorted lists, then explain its complexity."
    "Explain how a CPU cache hierarchy works, with concrete latency numbers."
    "List ten European capitals and one fact about each."
)
i=0
for p in "${prompts[@]}"; do
    body=$(printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":%d,"temperature":0}' \
        "$(basename "$MODEL")" "$p" "$MAX_TOKENS")
    curl -s "localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' -d "$body" \
        | sed -E 's/"(created|id|system_fingerprint)":[^,]*,//g' > "$OUT_DIR/${TAG}_$i.json"
    i=$((i + 1))
done
docker logs "$name" 2>&1 | grep -E 'shared_expert_overlap' | head -1 || true
docker rm -f "$name" >/dev/null
