#!/usr/bin/env bash
# prefill_cap_conc_ab.sh - alternating config A/B/C at CONC streams on ONE
# image with the streaming burst client (tools/analysis/burst_stream_client.py:
# aggregate tok/s, TTFT p50/p90/max, ITL p50/p95/max, gaps > 100 ms per wave).
# Built for the long-prompt serving question (runtime.prefill_chunk_decode_cap,
# runtime.prefill_batch_decode_cap); any --set string works as an arm. Fresh
# server per arm, pinned mbs/seq/kv like two_image_conc_ab.sh, arm order
# rotates per trial.
#
# Usage: ARM_A="" ARM_B="--set runtime.prefill_chunk_decode_cap=0" \
#        [ARM_C="..."] CONC=32 PLEN=1000 TRIALS=3 WAVES=3 \
#        bash tools/analysis/prefill_cap_conc_ab.sh
# Two-IMAGE form (a code change): IMG_A=imp:base IMG_B=imp:test with the same
# ARM_* sets (default empty), IGNORE_EOS=1 forces equal token counts per arm.
set -u
MODELS_DIR=${MODELS_DIR:-$HOME/models}
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL_NAME=${MODEL_NAME:-Qwen3.8-27B-NVFP4-vllm}
IMG=${IMG:-imp:test}
IMG_A=${IMG_A:-$IMG}
IMG_B=${IMG_B:-$IMG}
IMG_C=${IMG_C:-$IMG}
IGNORE_EOS=${IGNORE_EOS:-0}
PORT=${PORT:-8095}
CONC=${CONC:-32}
WAVES=${WAVES:-3}
TRIALS=${TRIALS:-3}
PLEN=${PLEN:-1000}
ARM_A=${ARM_A-}
ARM_B=${ARM_B---set runtime.prefill_chunk_decode_cap=0}  # unset-only default: ARM_B="" is a real empty arm
ARM_C=${ARM_C-__none__}
LOG="${TMPDIR:-/tmp}/prefill_cap_ab_${CONC}_p${PLEN}.log"
: > "$LOG"
echo "img A=$IMG_A B=$IMG_B C=$IMG_C conc=$CONC waves=$WAVES trials=$TRIALS plen=$PLEN ignore_eos=$IGNORE_EOS A='$ARM_A' B='$ARM_B' C='$ARM_C'" | tee -a "$LOG"

start_server() {  # $1 = extra --set args, $2 = image
    docker rm -f imp-capab >/dev/null 2>&1
    # shellcheck disable=SC2086
    docker run -d --name imp-capab --gpus all -v "${MODELS_DIR}":/models \
        -p ${PORT}:${PORT} "$2" imp-server --model /models/${MODEL_NAME} --port $PORT \
        --host 0.0.0.0 --max-concurrent $CONC \
        --set runtime.max_batch_size=32 --set runtime.max_seq_len=4096 --set kv_cache.max_blocks=2387 \
        $1 >/dev/null
    for _ in $(seq 1 180); do
        sleep 2
        curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && return 0
        if [ -z "$(docker ps -q -f name=imp-capab)" ]; then
            echo "server died:" | tee -a "$LOG"
            docker logs imp-capab 2>&1 | tail -20 | tee -a "$LOG"
            return 1
        fi
    done
    echo "server never became healthy" | tee -a "$LOG"
    return 1
}

run_arm() {  # $1 = arm name, $2 = extra sets, $3 = trial, $4 = image
    ~/.claude/skills/gpu-stats/gpu-busy-check.sh >/dev/null || {
        echo "GPU BUSY before $1 - aborting" | tee -a "$LOG"; exit 2; }
    start_server "$2" "$4" || exit 3
    echo "== arm $1 trial $3 ($4 $2) ==" | tee -a "$LOG"
    MODEL_NAME=$MODEL_NAME IGNORE_EOS=$IGNORE_EOS python3 "$HERE/burst_stream_client.py" $PORT $CONC $WAVES "$1$3" $PLEN 2>&1 | tee -a "$LOG"
    docker rm -f imp-capab >/dev/null 2>&1
    sleep 3
}

arms=(A B)
sets=("$ARM_A" "$ARM_B")
imgs=("$IMG_A" "$IMG_B")
if [ "$ARM_C" != "__none__" ]; then arms+=(C); sets+=("$ARM_C"); imgs+=("$IMG_C"); fi
n=${#arms[@]}
for t in $(seq 1 $TRIALS); do
    for k in $(seq 0 $((n - 1))); do
        idx=$(( (t - 1 + k) % n ))
        run_arm "${arms[$idx]}" "${sets[$idx]}" "$t" "${imgs[$idx]}"
    done
done
echo "=== summary ===" | tee -a "$LOG"
grep -H "== arm\|^wave" "$LOG" | tail -60
