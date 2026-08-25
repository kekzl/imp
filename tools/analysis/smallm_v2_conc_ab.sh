#!/usr/bin/env bash
# smallm_v2_conc_ab.sh - alternating A/B for gemm.nvfp4_smallm (off vs on)
# 3 trials/arm, 3 waves/trial at CONC streams. mbs/seq pinned (free-VRAM swing).
set -u
MODELS_DIR=${MODELS_DIR:-$HOME/models}
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL=/models/Qwen3.8-27B-NVFP4
PORT=8090
CONC=${CONC:-32}
WAVES=3
TRIALS=${TRIALS:-3}
LOG="${TMPDIR:-/tmp}/ab_conc_${CONC}.log"
: > "$LOG"

start_server() {  # $1 = extra --set args
    docker rm -f imp-ab >/dev/null 2>&1
    # shellcheck disable=SC2086
    docker run -d --name imp-ab --gpus all -v ${MODELS_DIR:-$HOME/models}:/models \
        -p ${PORT}:${PORT} imp:test imp-server --model $MODEL --port $PORT \
        --host 0.0.0.0 --max-concurrent $CONC \
        --set runtime.max_batch_size=32 --set runtime.max_seq_len=4096 $1 >/dev/null
    for _ in $(seq 1 180); do
        sleep 2
        if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            return 0
        fi
        if [ -z "$(docker ps -q -f name=imp-ab)" ]; then
            echo "server died:" | tee -a "$LOG"
            docker logs imp-ab 2>&1 | tail -20 | tee -a "$LOG"
            return 1
        fi
    done
    echo "server never became healthy" | tee -a "$LOG"
    return 1
}

run_arm() {  # $1 = arm name, $2 = extra sets
    ~/.claude/skills/gpu-stats/gpu-busy-check.sh >/dev/null || {
        echo "GPU BUSY before $1 - aborting" | tee -a "$LOG"; exit 2; }
    start_server "$2" || exit 3
    echo "== arm $1 trial $3 ==" | tee -a "$LOG"
    python3 "$HERE/conc_client.py" $PORT $CONC $WAVES "$1$3" 2>&1 | tee -a "$LOG"
    docker rm -f imp-ab >/dev/null 2>&1
    sleep 3
}

for t in $(seq 1 $TRIALS); do
    run_arm A "" "$t"
    run_arm B "--set gemm.nvfp4_smallm=true" "$t"
done
echo "=== summary ===" | tee -a "$LOG"
grep -H "MEDIAN\|== arm" "$LOG" | tail -40
