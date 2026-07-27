#!/usr/bin/env bash
# Decode throughput vs CONFIGURED context capacity, at a fixed (short) live
# sequence — the measurement behind issue "decode pays for context capacity it
# never uses".
#
# WHY THIS EXISTS: the pinned perf baseline is measured with `imp-cli --bench`,
# which sizes the engine to the bench workload. imp-server defaults to the
# model's FULL context. On Qwen3-14B-Q6_K that difference is -38% decode on a
# ~280-token sequence — i.e. the number the gate protects is not the number a
# server delivers. This script makes that reproducible in one command.
#
# Reports decode tok/s and the captured decode-body graph size (the kernel-node
# count tracks the penalty exactly), per capacity.
#
# Usage: tools/analysis/ctx_capacity_decode_sweep.sh [MODEL] [PORT]
set -euo pipefail

MODEL="${1:-Qwen3-14B-Q6_K.gguf}"
PORT="${2:-8083}"
IMG="${DOCKER_IMG:-imp:test}"
NAME="imp-ctx-sweep"
CAPS="${CAPS:-1024 2048 4096 8192 16384 32768 40960}"
PROMPT='Explain step by step why a bandwidth-bound matrix-vector product does not get faster with more compute units.'

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "=== decode tok/s vs configured capacity — model=$MODEL ==="
echo "    (live sequence is ~280 tokens in EVERY row; only the capacity changes)"
printf "    %-14s %10s  %s\n" "max_seq_len" "tok/s" "decode-body graph"

for cap in $CAPS; do
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    # spec off + batch 1: isolate the capacity variable from the drafter and
    # from batched-decode machinery, both of which move the number too.
    docker run -d --name "$NAME" --gpus all -p "${PORT}:8080" -v "$HOME/models:/models" "$IMG" \
        imp-server --host 0.0.0.0 --model "/models/$MODEL" \
        --set speculative.ngram=false --set runtime.max_batch_size=1 \
        --set runtime.max_seq_len="$cap" >/dev/null 2>&1
    for _ in $(seq 1 90); do
        curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && break
        sleep 2
    done
    tps=$(MODEL="$MODEL" PORT="$PORT" PROMPT="$PROMPT" python3 - <<'PY'
import json, os, time, urllib.request
U = f"http://localhost:{os.environ['PORT']}"
def gen(mt=256):
    b = {"model": os.environ["MODEL"], "prompt": os.environ["PROMPT"],
         "max_tokens": mt, "temperature": 0, "stream": False}
    r = urllib.request.Request(U + "/v1/completions", json.dumps(b).encode(),
                               {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=900) as resp:
        d = json.load(resp)
    return d["usage"]["completion_tokens"] / (time.time() - t0)
gen(300)  # clock ramp: the first ~1s runs at reduced clocks and reads LOW
print(f"{max(gen() for _ in range(3)):.2f}")
PY
)
    edges=$(docker logs "$NAME" 2>&1 | grep -oE "[0-9]+ body graph edges" | tail -1 || true)
    printf "    %-14s %10s  %s\n" "$cap" "$tps" "${edges:-n/a}"
done
