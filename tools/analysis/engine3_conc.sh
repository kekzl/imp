#!/usr/bin/env bash
# Cross-engine serving throughput, imp vs vLLM vs llama.cpp, each on its defaults, one fresh server
# per arm, client conc_client.py (300-token greedy gens, ignore_eos, aggregate = tokens / wall).
# Arms: imp + vLLM on the NVFP4 dir, imp + llama.cpp on the GGUF. Order alternates per trial.
# Usage: bash tools/analysis/engine3_conc.sh [CONCS="1 32"] [TRIALS=2] [WAVES=3]
# [NVFP4=Qwen3.8-27B-NVFP4-vllm] [GGUF=Qwen3.8-27B-UD-Q4_K_M.gguf] [IMG_IMP=imp:test]
# [IMG_VLLM=vllm/vllm-openai:v0.31.0] [IMG_LLAMA=ghcr.io/ggml-org/llama.cpp:server-cuda]
set -u
MODELS_DIR=${MODELS_DIR:-$HOME/models}
HERE="$(cd "$(dirname "$0")" && pwd)"
CONCS=${CONCS:-"1 32"}
TRIALS=${TRIALS:-2}
WAVES=${WAVES:-3}
NVFP4=${NVFP4:-Qwen3.8-27B-NVFP4-vllm}
GGUF=${GGUF:-Qwen3.8-27B-UD-Q4_K_M.gguf}
IMG_IMP=${IMG_IMP:-imp:test}
IMG_VLLM=${IMG_VLLM:-vllm/vllm-openai:v0.31.0}
IMG_LLAMA=${IMG_LLAMA:-ghcr.io/ggml-org/llama.cpp:server-cuda}
PORT=${PORT:-8095}
LOG="${LOG:-${TMPDIR:-/tmp}/engine3_conc.log}"
: > "$LOG"
for img in "$IMG_IMP" "$IMG_VLLM" "$IMG_LLAMA"; do
    echo "image $img $(docker image inspect --format '{{index .RepoDigests 0}}' "$img" 2>/dev/null)" | tee -a "$LOG"
done

wait_healthy() {  # $1 = seconds
    for _ in $(seq 1 $(( $1 / 2 ))); do
        sleep 2
        curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && return 0
        [ -n "$(docker ps -q -f name=engine3)" ] || break
    done
    echo "server not healthy:" | tee -a "$LOG"; docker logs engine3 2>&1 | tail -20 | tee -a "$LOG"
    return 1
}

start() {  # $1 = arm, $2 = conc
    docker rm -f engine3 >/dev/null 2>&1
    local common=(-d --name engine3 --gpus all -v "${MODELS_DIR}":/models -p "${PORT}:${PORT}")
    case "$1" in
        imp-nvfp4|imp-gguf)
            local m=$NVFP4; [ "$1" = imp-gguf ] && m=$GGUF
            docker run "${common[@]}" "$IMG_IMP" imp-server --model "/models/$m" --host 0.0.0.0 \
                --port "$PORT" --max-concurrent "$2" --set "runtime.max_batch_size=$2" >/dev/null
            wait_healthy 600 ;;
        vllm-nvfp4)
            docker run "${common[@]}" --ipc=host -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 "$IMG_VLLM" \
                --model "/models/$NVFP4" --served-model-name "$NVFP4" --host 0.0.0.0 --port "$PORT" \
                --max-model-len 16384 --max-num-seqs "$2" >/dev/null
            wait_healthy 1200 ;;
        llama-gguf)
            docker run "${common[@]}" "$IMG_LLAMA" -m "/models/$GGUF" --host 0.0.0.0 --port "$PORT" \
                -ngl 999 -fa on -np "$2" -c $(( $2 * 4096 )) >/dev/null
            wait_healthy 600 ;;
    esac
}

run_arm() {  # $1 = arm, $2 = conc, $3 = trial
    # The previous arm's VRAM can take seconds to return on WSL2: 6 checks, 10 s apart, then abort.
    local free=1
    for _ in 1 2 3 4 5 6; do
        ~/.claude/skills/gpu-stats/gpu-busy-check.sh >/dev/null && { free=0; break; }
        sleep 10
    done
    [ $free -eq 0 ] || { echo "GPU BUSY before $1" | tee -a "$LOG"; exit 2; }
    start "$1" "$2" || exit 3
    local name=$NVFP4; case "$1" in *gguf) name=$GGUF ;; esac
    echo "== arm $1 conc $2 trial $3 ==" | tee -a "$LOG"
    docker run --rm --network host -e IGNORE_EOS=1 -e MODEL_NAME="$name" -v "$HERE":/t:ro imp:toolchain \
        python3 /t/conc_client.py "$PORT" "$2" "$WAVES" "$1-c$2-t$3" 0 2>&1 | grep -v -e CUDA -e NVIDIA \
        | tee -a "$LOG"
    docker rm -f engine3 >/dev/null 2>&1
    sleep 5
}

ARMS=(imp-nvfp4 vllm-nvfp4 imp-gguf llama-gguf)
for c in $CONCS; do
    for t in $(seq 1 "$TRIALS"); do
        if [ $((t % 2)) -eq 1 ]; then order=("${ARMS[@]}"); else order=(llama-gguf imp-gguf vllm-nvfp4 imp-nvfp4); fi
        for a in "${order[@]}"; do run_arm "$a" "$c" "$t"; done
    done
done
echo "=== summary ===" | tee -a "$LOG"
grep -E "MEDIAN|== arm" "$LOG"
