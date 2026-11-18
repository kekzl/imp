#!/usr/bin/env bash
# Do the server caches (prefix hashes, SWA snapshots) cost decode? One imp-server per arm; a long prompt sent
# REPS times (first cold, then cached), streamed, ignore_eos. Client-side: ttft = first SSE delta, decode tok/s =
# (deltas - 1) / (last - first delta); the server log's ttft counts content only (0 when all is reasoning).
# Speculation off so tok/s does not depend on the text. kv_cache.swa_snapshot_mb=0 is no arm at 74k tokens:
# without snapshots prefix caching turns SWA sizing off and the context drops to 47216 (HTTP 400).
# Usage: tools/analysis/flint_cache_decode_ab.sh IMAGE TEXT_FILE TAG [extra --set args...]
set -euo pipefail
IMG=$1; TEXT=$2; TAG=$3; shift 3
MODEL=${MODEL:-/models/flint-nvfp4-official}
OUT_DIR=${OUT_DIR:-/mnt/data/scratch/flint_cache}
PORT=${PORT:-8091}
MAX_TOKENS=${MAX_TOKENS:-512}
REPS=${REPS:-3}
mkdir -p "$OUT_DIR"
body="$OUT_DIR/body.json"
jq -Rs --argjson n "$MAX_TOKENS" --arg m "$(basename "$MODEL")" '{model: $m, stream: true, max_tokens: $n,
    temperature: 0, ignore_eos: true,
    messages: [{role: "user", content: (. + "\n\nSummarize the document above section by section.")}]}' \
    "$TEXT" > "$body"
name="flint-cache-ab-$TAG"
docker rm -f "$name" > /dev/null 2>&1 || true
docker run -d --init --name "$name" --gpus all -p "127.0.0.1:$PORT:$PORT" -v "$HOME/models:/models:ro" \
    --entrypoint imp-server "$IMG" --model "$MODEL" --host 0.0.0.0 --port "$PORT" \
    --set speculative.ngram=false --set speculative.mtp_k=0 "$@" > /dev/null
until curl -sf "localhost:$PORT/health" > /dev/null; do
    docker inspect -f '{{.State.Running}}' "$name" | grep -q true || { docker logs "$name" 2>&1 | tail -5; exit 1; }
    sleep 2
done
for i in $(seq "$REPS"); do
    t0=$(date +%s.%N)
    curl -s -N "localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d @"$body" |
        while IFS= read -r line; do
            case $line in *'"delta"'*'"content"'* | *'"delta"'*'"reasoning'*) date +%s.%N ;; esac
        done > "$OUT_DIR/$TAG.req$i.ts"
    awk -v t="$TAG" -v i="$i" -v t0="$t0" 'NR == 1 { f = $1 } { l = $1; n++ }
        END { printf "%s\treq%d\tdeltas %d\tttft %.0f ms\tdecode %.2f tok/s\n", t, i, n, (f - t0) * 1000, (n - 1) / (l - f) }' \
        "$OUT_DIR/$TAG.req$i.ts"
done
docker logs "$name" > "$OUT_DIR/$TAG.log" 2>&1
grep -E 'completion tokens' "$OUT_DIR/$TAG.log" | sed -E 's/.*\] //' || true
docker rm -f "$name" > /dev/null
