#!/usr/bin/env bash
# serving_sparse_ab.sh - alternating serving A/B for attention.sparse_topk_tokens
# at long context: CONC streams x ~15.5k-token prompts, decode rate via the
# tg8/tg520 differential (prefill wall cancels). Fresh server per arm.
set -u
MODELS_DIR=$HOME/models
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL=/models/Qwen3-8B-Q8_0.gguf
PORT=8091
CONC=${CONC:-6}
TRIALS=${TRIALS:-3}
LOG="${TMPDIR:-/tmp}/serving_sparse_ab.log"
: > "$LOG"

start_server() {  # $1 = extra --set args
    docker rm -f imp-sab >/dev/null 2>&1
    # shellcheck disable=SC2086
    docker run -d --name imp-sab --gpus all -v $MODELS_DIR:/models \
        -p ${PORT}:${PORT} imp:test imp-server --model $MODEL --port $PORT \
        --host 0.0.0.0 --kv-fp8 --max-concurrent $CONC \
        --set runtime.max_batch_size=$CONC --set runtime.max_seq_len=${SEQLEN:-17408} \
        --set kv_cache.max_blocks=${KVBLOCKS:-6600} --set server.prefix_cache=false $1 >/dev/null
    for _ in $(seq 1 180); do
        sleep 2
        if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            return 0
        fi
        if [ -z "$(docker ps -q -f name=imp-sab)" ]; then
            echo "server died:" | tee -a "$LOG"
            docker logs imp-sab 2>&1 | tail -20 | tee -a "$LOG"
            return 1
        fi
    done
    echo "server never became healthy" | tee -a "$LOG"
    return 1
}

run_arm() {  # $1 = arm name, $2 = extra sets, $3 = trial
    ~/.claude/skills/gpu-stats/gpu-busy-check.sh >/dev/null || {
        echo "GPU BUSY before $1 - aborting" | tee -a "$LOG"; exit 2; }
    start_server "$2" || exit 3
    docker logs imp-sab 2>&1 | grep -E "Sparse decode attention|KV cache: .* blocks" | tee -a "$LOG"
    echo "== arm $1 trial $3 ==" | tee -a "$LOG"
    # warm wave (graph captures, clocks) - discarded
    python3 "$HERE/longctx_conc_client.py" $PORT $CONC 8 "$1$3-warm" >> "$LOG" 2>&1
    python3 "$HERE/longctx_conc_client.py" $PORT $CONC 8 "$1$3-tg8" 2>&1 | tee -a "$LOG"
    python3 "$HERE/longctx_conc_client.py" $PORT $CONC 520 "$1$3-tg520" 2>&1 | tee -a "$LOG"
    docker rm -f imp-sab >/dev/null 2>&1
    sleep 3
}

for t in $(seq 1 $TRIALS); do
    run_arm OFF "" "$t"
    run_arm ON "--set attention.sparse_topk_tokens=4096" "$t"
done
echo "=== raw ==="; grep -E "WAVE|arm" "$LOG"
