#!/usr/bin/env bash
# vllm_conc_ab.sh - alternating cross-ENGINE aggregate-throughput A/B at CONC
# streams on the same checkpoint: arm I runs imp (IMG_IMP, pinned 32/4096
# config of two_image_conc_ab.sh), arm V runs vLLM (IMG_VLLM, the flags of the
# 2026-08-25 BENCHMARKS.md row); fresh server per arm, TRIALS alternating
# pairs, WAVES waves each, client tools/analysis/conc_client.py (unique
# prompts, 300-token greedy gens, aggregate = completion tokens / wall).
#
# Usage: bash tools/analysis/vllm_conc_ab.sh
#        IMG_IMP=imp:test IMG_VLLM=vllm/vllm-openai:v0.27.1 CONC=32 TRIALS=3 WAVES=3
#        VLLM_EXTRA="--foo" IMP_EXTRA="--set x=y" PLEN=1000 (long-prompt shape)
set -u
MODELS_DIR=${MODELS_DIR:-$HOME/models}
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL_NAME=${MODEL_NAME:-Qwen3.8-27B-NVFP4-vllm}
PORT=${PORT:-8094}
CONC=${CONC:-32}
WAVES=${WAVES:-3}
TRIALS=${TRIALS:-3}
IMG_IMP=${IMG_IMP:-imp:test}
IMG_VLLM=${IMG_VLLM:-vllm/vllm-openai:v0.27.1}
IMP_EXTRA=${IMP_EXTRA:-}
VLLM_EXTRA=${VLLM_EXTRA:-}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-16384}
VLLM_UTIL=${VLLM_UTIL:-0.90}
PLEN=${PLEN:-0}
LOG="${TMPDIR:-/tmp}/vllm_conc_ab_${CONC}_p${PLEN}.log"
: > "$LOG"
echo "imp=$IMG_IMP vllm=$IMG_VLLM conc=$CONC waves=$WAVES trials=$TRIALS plen=$PLEN imp_extra='$IMP_EXTRA' vllm_extra='$VLLM_EXTRA'" | tee -a "$LOG"

wait_healthy() {  # $1 = name, $2 = seconds
    local n=$(( $2 / 2 ))
    for _ in $(seq 1 $n); do
        sleep 2
        curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && return 0
        if [ -z "$(docker ps -q -f name="$1")" ]; then
            echo "server died ($1):" | tee -a "$LOG"
            docker logs "$1" 2>&1 | tail -30 | tee -a "$LOG"
            return 1
        fi
    done
    echo "server never became healthy ($1)" | tee -a "$LOG"
    docker logs "$1" 2>&1 | tail -30 | tee -a "$LOG"
    return 1
}

start_imp() {
    docker rm -f engine-ab >/dev/null 2>&1
    # shellcheck disable=SC2086
    docker run -d --name engine-ab --gpus all -v "${MODELS_DIR}":/models \
        -p ${PORT}:${PORT} "$IMG_IMP" imp-server --model /models/${MODEL_NAME} --port $PORT \
        --host 0.0.0.0 --max-concurrent $CONC \
        --set runtime.max_batch_size=32 --set runtime.max_seq_len=4096 --set kv_cache.max_blocks=2387 \
        $IMP_EXTRA >/dev/null
    wait_healthy engine-ab 360
}

start_vllm() {
    docker rm -f engine-ab >/dev/null 2>&1
    # shellcheck disable=SC2086
    docker run -d --name engine-ab --gpus all --ipc=host -v "${MODELS_DIR}":/models \
        -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 -p ${PORT}:${PORT} "$IMG_VLLM" \
        --model /models/${MODEL_NAME} --served-model-name ${MODEL_NAME} \
        --port $PORT --host 0.0.0.0 \
        --gpu-memory-utilization ${VLLM_UTIL} --max-model-len ${VLLM_MAX_MODEL_LEN} \
        --max-num-seqs $CONC $VLLM_EXTRA >/dev/null
    wait_healthy engine-ab 1200
}

run_arm() {  # $1 = arm name (I|V), $2 = trial
    ~/.claude/skills/gpu-stats/gpu-busy-check.sh >/dev/null || {
        echo "GPU BUSY before $1 - aborting" | tee -a "$LOG"; exit 2; }
    if [ "$1" = I ]; then start_imp || exit 3; else start_vllm || exit 3; fi
    echo "== arm $1 trial $2 ==" | tee -a "$LOG"
    MODEL_NAME=$MODEL_NAME python3 "$HERE/conc_client.py" $PORT $CONC $WAVES "$1$2" $PLEN 2>&1 | tee -a "$LOG"
    docker rm -f engine-ab >/dev/null 2>&1
    sleep 5
}

for t in $(seq 1 $TRIALS); do
    if [ $((t % 2)) -eq 1 ]; then
        run_arm I "$t"; run_arm V "$t"
    else
        run_arm V "$t"; run_arm I "$t"
    fi
done
echo "=== summary ===" | tee -a "$LOG"
grep -H "MEDIAN\|== arm" "$LOG" | tail -40
