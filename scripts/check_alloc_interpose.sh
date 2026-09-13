#!/usr/bin/env bash
# Invariant I2 (docs/internals/MEMORY.md): nothing allocates device memory while serving.
# steady_state_allocations() only sees Backend-routed calls; the --wrap interposer closes that
# gap for every direct allocation site. make check-alloc-interpose builds and runs this.
# Phase A: batch>1 + NVFP4 residual KV + MTP chain. Phase B: MTP off, default KV, ragged prefill
# + json_object constrained pipeline. Each phase has its own liveness check and pin.
set -uo pipefail

BIN=${BIN:-build-interpose/imp-server}
IMG=${DEV_IMG:-imp:toolchain}
MODEL=${INTERPOSE_MODEL:-/models/Qwen3.8-27B-NVFP4-vllm}
MODELS_DIR=${MODELS_DIR:-$HOME/models}
PORT=8099
M=$(basename "$MODEL")
URL="http://127.0.0.1:$PORT"

[ -x "$BIN" ] || { echo "FATAL: $BIN not built. Run: make check-alloc-interpose" >&2; exit 1; }
[ -d "$MODELS_DIR/$M" ] || { echo "FATAL: model not readable: $MODELS_DIR/$M" >&2; exit 1; }

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$used" -le 3000 ] || { echo "FATAL: $used MiB already held on the GPU" >&2; exit 1; }

trap 'docker rm -f interpose >/dev/null 2>&1' EXIT

# start_server <extra --set args...>: boots the interposer binary, waits on
# /ready (200 only once the model is loaded; /health is 200 before that).
start_server() {
    docker rm -f interpose >/dev/null 2>&1
    docker run -d --name interpose --gpus all -p $PORT:8080 \
        -v "$PWD":/src -w /src -v "$MODELS_DIR":/models "$IMG" \
        "$BIN" --host 0.0.0.0 --port 8080 --model "$MODEL" --think-budget 0 \
        --set runtime.max_batch_size=4 \
        --set diagnostics.log_level=debug "$@" >/dev/null
    for _ in $(seq 1 200); do
        curl -sf "$URL/ready" >/dev/null 2>&1 && break
        docker ps --format '{{.Names}}' | grep -q '^interpose$' || break
        sleep 3
    done
    curl -sf "$URL/ready" >/dev/null 2>&1 || {
        echo "FATAL: server never became ready" >&2; docker logs interpose 2>&1 | tail -20 >&2; exit 1; }
}

chat() {  # chat <json body>
    curl -s "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$1" >/dev/null
}

# stop_server <log>: clean shutdown, the report is a static destructor and
# SIGKILL loses it.
stop_server() {
    docker stop -t 60 interpose >/dev/null 2>&1
    docker logs interpose > "$1" 2>&1
}

# Matches "<class> <n> calls" anywhere in the line, not anchored at the start (an anchored
# read used to skip the first class, undercounting 2 vs 19 real allocations).
# awk, not bc: bc isn't installed on this host, and a missing binary would read as 0 allocations.
calls() {
    sed -n '/alloc-interpose\] I2 VIOLATIONS/,/pinned host/p' "$1" \
        | grep -oP '(cudaMalloc|cudaMallocAsync|pinned host)\s+\K[0-9]+(?=\s+calls)' \
        | awk '{s+=$1} END {print s+0}'
}

# verdict <phase> <log> <pinned>: exits 1 on any deviation from the pin.
verdict() {
    local phase=$1 log=$2 pinned=$3
    local clean viol n
    clean=$(grep -c 'alloc-interpose\] steady state clean' "$log")
    viol=$(grep -c 'alloc-interpose\] I2 VIOLATIONS' "$log")
    if [ "$clean" -eq 0 ] && [ "$viol" -eq 0 ]; then
        echo "FATAL($phase): neither report line appeared. The binary was not built with" >&2
        echo "       -DIMP_ALLOC_INTERPOSE=ON, or it did not shut down cleanly."          >&2
        tail -20 "$log" >&2
        exit 1
    fi
    if [ "$viol" -eq 0 ]; then
        n=0
    else
        n=$(calls "$log")
        if [ "${n:-0}" -eq 0 ]; then
            echo "FATAL($phase): the violation banner is present but no per-class call"  >&2
            echo "       counts parsed out of it. The report format changed; fix calls()." >&2
            sed -n '/alloc-interpose\] I2 VIOLATIONS/,/pinned host/p' "$log"              >&2
            exit 1
        fi
    fi
    if [ "$n" -gt "$pinned" ]; then
        echo "FAIL($phase): $n device allocations while serving, pin is $pinned (invariant I2)."
        echo "      A new serving-phase allocation. Find it with"
        echo "        addr2line -e $BIN -f -C <offset>   (offsets in $log)"
        sed -n '/alloc-interpose\] I2 VIOLATIONS/,/pinned host/p' "$log"
        exit 1
    fi
    if [ "$n" -lt "$pinned" ]; then
        echo "FAIL($phase): $n device allocations while serving, pin is $pinned."
        echo "      Fewer than pinned: someone fixed one. Lower the pin to $n and"
        echo "      remove the fixed site from the list above it."
        exit 1
    fi
    echo "PASS($phase): $n serving allocation(s), exactly the pinned residue (log: $log)"
}

# Per-phase call-count pin, named by site (docs/audit/DEBT_LEDGER_2026_08_21.md section g).
# Pin may only go DOWN: a rise fails (new serving allocation), a fall fails too (stale pin).
# Phase A: 0 (the MTP post-norm feed scratch is sized once at enable time since #1940).
PINNED_A=0
# Phase B: 3 calls, 0.7 MiB, JsonConstrainer::init (src/compute/constrain_device_buffers.h):
# per-json-request device tables built on the scheduler thread, not upload metadata.
PINNED_B=3

# ---------------------------------------------------------------- phase A
LOG_A=$(mktemp /tmp/interpose.A.XXXXXX.log)
start_server --set kv_cache.dtype=nvfp4 \
             --set kv_cache.bitdecoding_residual_tokens=128 \
             --set speculative.mtp_k=1
# 20 requests, four at a time: batch > 1 is what reaches the residual allocator.
for round in 1 2 3 4 5; do
    for slot in 1 2 3 4; do
        chat "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain topic $round.$slot in two paragraphs.\"}],\"max_tokens\":96,\"temperature\":0}" &
    done
    wait
done
stop_server "$LOG_A"
# Liveness, before any verdict: if the residual pool was declined (wrong kv
# dtype, non-uniform head_dim) the run cannot reach the allocator it is
# looking for, and a clean result means nothing.
if ! grep -q 'residual buffer enabled' "$LOG_A"; then
    echo "FATAL(A): the residual KV pool was never enabled, so this run did not"   >&2
    echo "       exercise the path it claims to. Check kv_cache.dtype=nvfp4 and" >&2
    echo "       kv_cache.bitdecoding_residual_tokens against this model."       >&2
    grep -i 'residual' "$LOG_A" | head -5                                       >&2
    exit 1
fi
verdict A "$LOG_A" "$PINNED_A"

# ---------------------------------------------------------------- phase B
LOG_B=$(mktemp /tmp/interpose.B.XXXXXX.log)
start_server --set speculative.mtp_k=0
# Ragged prefill needs two or more prompts in the SAME prefill step, which
# 15-token prompts never give (each is prefilled before the next curl lands).
long=$(seq 1 150 | sed 's/^/Fact number & about the system under test is recorded here. /' | tr -d '\n')
for slot in 1 2 3 4; do
    jq -cn --arg m "$M" --arg p "$long Summarise the above in one sentence ($slot)." \
        '{model:$m,messages:[{role:"user",content:$p}],max_tokens:48,temperature:0}' |
        curl -s "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d @- >/dev/null &
done
wait
for i in 1 2 3; do
    chat "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Return a JSON object with the keys name and age for person $i.\"}],\"max_tokens\":48,\"temperature\":0,\"response_format\":{\"type\":\"json_object\"}}"
done
stop_server "$LOG_B"
if ! grep -qE 'Ragged prefill: [2-9] seqs' "$LOG_B"; then
    echo "FATAL(B): no ragged prefill wave with 2+ members in the log, so the run did" >&2
    echo "       not exercise engine_prefill_ragged.cpp. Check runtime.prefill_batch"  >&2
    echo "       and that the four long prompts were admitted together."               >&2
    exit 1
fi
if ! grep -q 'ConstrainedPipeline: launched' "$LOG_B"; then
    echo "FATAL(B): the constrained pipeline never launched, so the run did not"    >&2
    echo "       exercise cpipe_ (json_object refused or routed to eager decode)." >&2
    grep -i 'constrained\|json' "$LOG_B" | head -5                                >&2
    exit 1
fi
verdict B "$LOG_B" "$PINNED_B"
exit 0
