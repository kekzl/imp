#!/usr/bin/env bash
# two_image_conc_ab.sh - alternating two-IMAGE aggregate-throughput A/B at
# CONC streams: arm A runs IMG_A, arm B runs IMG_B, same pinned config, fresh
# server per arm, TRIALS alternating pairs, WAVES waves each; the client is
# tools/analysis/conc_client.py (unique prompts, 300-token greedy gens).
# This is the "two-image A/B" of the benchmark-cuda skill for a CODE change
# (as opposed to smallm_v2_conc_ab.sh, which flips a config flag on one image).
#
# Usage: IMG_A=imp:ab-base IMG_B=imp:test bash tools/analysis/two_image_conc_ab.sh
#        CONC=32 TRIALS=3 WAVES=3 EXTRA="--set x=y" (EXTRA applies to both arms)
set -u
MODELS_DIR=${MODELS_DIR:-$HOME/models}
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL=${MODEL:-/models/Qwen3.8-27B-NVFP4-vllm}
PORT=${PORT:-8094}
CONC=${CONC:-32}
WAVES=${WAVES:-3}
TRIALS=${TRIALS:-3}
IMG_A=${IMG_A:-imp:ab-base}
IMG_B=${IMG_B:-imp:test}
EXTRA=${EXTRA:-}
LOG="${TMPDIR:-/tmp}/two_image_ab_${CONC}.log"
: > "$LOG"
echo "A=$IMG_A B=$IMG_B conc=$CONC waves=$WAVES trials=$TRIALS extra='$EXTRA'" | tee -a "$LOG"

start_server() {  # $1 = image
    docker rm -f imp-ab2 >/dev/null 2>&1
    # shellcheck disable=SC2086
    docker run -d --name imp-ab2 --gpus all -v "${MODELS_DIR}":/models \
        -p ${PORT}:${PORT} "$1" imp-server --model $MODEL --port $PORT \
        --host 0.0.0.0 --max-concurrent $CONC \
        --set runtime.max_batch_size=32 --set runtime.max_seq_len=4096 --set kv_cache.max_blocks=2387 \
        $EXTRA >/dev/null
    for _ in $(seq 1 180); do
        sleep 2
        curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && return 0
        if [ -z "$(docker ps -q -f name=imp-ab2)" ]; then
            echo "server died ($1):" | tee -a "$LOG"
            docker logs imp-ab2 2>&1 | tail -20 | tee -a "$LOG"
            return 1
        fi
    done
    echo "server never became healthy ($1)" | tee -a "$LOG"
    return 1
}

run_arm() {  # $1 = arm name, $2 = image, $3 = trial
    ~/.claude/skills/gpu-stats/gpu-busy-check.sh >/dev/null || {
        echo "GPU BUSY before $1 - aborting" | tee -a "$LOG"; exit 2; }
    start_server "$2" || exit 3
    echo "== arm $1 ($2) trial $3 ==" | tee -a "$LOG"
    python3 "$HERE/conc_client.py" $PORT $CONC $WAVES "$1$3" 2>&1 | tee -a "$LOG"
    docker rm -f imp-ab2 >/dev/null 2>&1
    sleep 3
}

for t in $(seq 1 $TRIALS); do
    if [ $((t % 2)) -eq 1 ]; then
        run_arm A "$IMG_A" "$t"; run_arm B "$IMG_B" "$t"
    else
        run_arm B "$IMG_B" "$t"; run_arm A "$IMG_A" "$t"
    fi
done
echo "=== summary ===" | tee -a "$LOG"
grep -H "MEDIAN\|== arm" "$LOG" | tail -40
