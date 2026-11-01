#!/usr/bin/env bash
# KV eviction accuracy A/B (roadmap row 48, docs/plans/2026-11-01-kv-eviction.md): one server per
# arm (kv_cache.evict_keep_ratio 0 / 0.5 / 0.25, everything else pinned), the same NIAH +
# continuation-recall probes on each (tools/analysis/kv_evict_niah.py).
# Usage: bash tools/analysis/kv_evict_ab.sh <out_dir> [model] [lengths] [image] [build_dir]
# Runs imp-server from <build_dir> (default build-dev) inside <image> (default imp:toolchain).
# Port 8091: 8080 is a user service on this host. GPU lock and busy check are the caller's job.
set -u

OUT="${1:?out dir}"
MODEL="${2:-Qwen3-8B-NVFP4-cortecs}"
LENGTHS="${3:-8000,16000,32000}"
IMG="${4:-imp:toolchain}"
BUILD="${5:-build-dev}"
NAME=imp-kvevict
PORT=8091
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
mkdir -p "$OUT"

run_arm() {
    local arm="$1" ratio="$2"
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker run -d --name "$NAME" --gpus all --network host -v "$HOME/models:/models:ro" -v "$ROOT:/src:ro" \
        -w /src "$IMG" "/src/$BUILD/imp-server" --host 127.0.0.1 --port "$PORT" --model "/models/$MODEL" \
        --set runtime.max_seq_len=40960 \
        --set runtime.max_batch_size=4 \
        --set runtime.prefill_batch=false \
        --set runtime.prefill_graph=false \
        --set server.prefix_cache=false \
        --set speculative.ngram=false \
        --set speculative.mtp_k=0 \
        --set attention.sparse_topk_tokens=0 \
        --set attention.sparse_prefill_topk_tokens=0 \
        --set "kv_cache.evict_keep_ratio=$ratio" >/dev/null
    echo "--- arm $arm (kv_cache.evict_keep_ratio=$ratio)"
    local up=0
    for _ in $(seq 1 240); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            up=1
            break
        fi
        sleep 2
    done
    if [ "$up" -ne 1 ]; then
        echo "ABORT: server for arm $arm never became healthy"
        docker logs "$NAME" 2>&1 | tail -20
        docker rm -f "$NAME" >/dev/null 2>&1
        exit 2
    fi
    docker logs "$NAME" 2>&1 | grep -E "evict_keep_ratio|KV eviction harness" | head -3
    docker run --rm --network host -v "$ROOT/tools/analysis:/a:ro" -v "$ROOT/tools/eval/niah/data:/data:ro" \
        -v "$OUT:/out" python:3.13-slim python3 /a/kv_evict_niah.py --url "http://127.0.0.1:$PORT" \
        --model "$MODEL" --arm "$arm" --lengths "$LENGTHS" --out "/out/probes.jsonl" > "$OUT/client_$arm.log" 2>&1
    tail -6 "$OUT/client_$arm.log"
    docker logs "$NAME" > "$OUT/server_$arm.log" 2>&1
    # Positive control: an eviction arm whose eviction never ran is a dense run with a label.
    local evicted warned
    evicted=$(grep -c "kv evict: req" "$OUT/server_$arm.log")
    warned=$(grep -c "query statistics cover" "$OUT/server_$arm.log")
    echo "kv evict lines: $evicted, statistics mismatches: $warned"
    grep -E "CUDA error|capture failed|\bNaN\b|is NaN|is Inf" "$OUT/server_$arm.log" | head -3
    docker rm -f "$NAME" >/dev/null 2>&1
    if [ "$ratio" != "0" ] && [ "$evicted" -lt 1 ]; then
        echo "ABORT: arm $arm never evicted"
        exit 3
    fi
}

run_arm base 0
run_arm keep50 0.5
run_arm keep25 0.25
echo "=== probes in $OUT/probes.jsonl"
