#!/usr/bin/env bash
# Serving VRAM per phase on a default imp-server: boot, warm, c1 chats, c4 bursts, then c32 bursts one wave
# at a time; after each phase /metrics (own bytes, device live, KV blocks) + nvidia-smi used, then a clean
# stop so the shutdown VRAM audit (per-pool table) lands in the log. Fresh server per arm.
# Usage: ARMS="label|image|k=v,k=v ..." (image or dev:<binary path in this tree>) OUT=/mnt/data/scratch/dir [PORT=8300] [C32_WAVES=4] \
#        tools/analysis/serving_vram_phases.sh
set -uo pipefail
MODELS_DIR=${MODELS_DIR:-$HOME/models}
HERE="$(cd "$(dirname "$0")" && pwd)"
ARMS=${ARMS:?ARMS="label|image|k=v,... ..."}
OUT=${OUT:?OUT=dir for logs}
PORT=${PORT:-8300}
C32_WAVES=${C32_WAVES:-4}
NVFP4=${NVFP4:-Qwen3.8-27B-NVFP4-vllm}
CTR=${CTR:-vramphase-$PORT}
mkdir -p "$OUT"
mib() { awk -v n="$1" 'BEGIN {printf "%.0f", n / 1048576}'; }
sample() {  # $1 = tag, $2 = phase label
    local m smi
    m=$(curl -s "http://127.0.0.1:$PORT/metrics")
    smi=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    own=$(grep -E '^imp_vram_own_bytes ' <<< "$m" | awk '{print $2}')
    peak=$(grep -E '^imp_vram_own_peak_bytes ' <<< "$m" | awk '{print $2}')
    dev=$(grep -E '^imp_memory_live_bytes\{tier="device"\}' <<< "$m" | awk '{print $2}')
    pool=$(grep -E '^imp_memory_reserved_bytes\{tier="async_pool"\}' <<< "$m" | awk '{print $2}')
    untr=$(grep -E '^imp_vram_untracked_bytes ' <<< "$m" | awk '{print $2}')
    kv=$(grep -E '^imp_kv_blocks_total ' <<< "$m" | awk '{print $2}')
    echo "$1 $2 own_mib $(mib "${own:-0}") own_peak_mib $(mib "${peak:-0}") device_live_mib $(mib "${dev:-0}")" \
         "async_pool_mib $(mib "${pool:-0}") untracked_mib $(mib "${untr:-0}") kv_blocks ${kv:-0} smi_used_mib $smi" |
        tee -a "$OUT/summary.txt"
    echo "$m" > "$OUT/$1.$2.metrics.txt"
}
for arm in $ARMS; do
    IFS='|' read -r label img sets <<< "$arm"
    tag="$label"
    docker rm -f "$CTR" >/dev/null 2>&1; sleep 5
    # image "dev:<path>" = a binary from this tree in imp:toolchain (e.g. dev:build-interpose/imp-server)
    run=("$img" imp-server)
    [[ $img == dev:* ]] && run=(-v "$HERE/../..":/src -w /src imp:toolchain "/src/${img#dev:}")
    docker run -d --name "$CTR" --gpus all -v "$MODELS_DIR":/models -p "$PORT:$PORT" "${run[@]}" \
        --model "/models/$NVFP4" --host 0.0.0.0 --port "$PORT" \
        $(for s in ${sets//,/ }; do printf -- "--set %s " "$s"; done) >/dev/null
    for _ in $(seq 1 600); do sleep 2; curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
        [ -n "$(docker ps -q -f name="$CTR")" ] || { echo "$tag: server died"; docker logs "$CTR" 2>&1 | tail -5; exit 3; }; done
    sample "$tag" 0_boot
    docker run --rm --network host -e IGNORE_EOS=1 -e MODEL_NAME="$NVFP4" -v "$HERE":/t:ro imp:toolchain \
        python3 /t/conc_client.py "$PORT" 1 1 warm >/dev/null 2>&1
    sample "$tag" 1_warm
    docker run --rm --network host -e ROUNDS=1 -e MODEL_NAME="$NVFP4" -v "$HERE":/t:ro imp:toolchain \
        python3 /t/seq_stream_client.py "$PORT" "$tag" > "$OUT/$tag.c1.txt" 2>&1
    sample "$tag" 2_c1
    docker run --rm --network host -e MODEL_NAME="$NVFP4" -v "$HERE":/t:ro imp:toolchain \
        python3 /t/conc_client.py "$PORT" 4 4 "$tag" > "$OUT/$tag.c4.txt" 2>&1
    sample "$tag" 3_c4
    for w in $(seq 1 "$C32_WAVES"); do
        docker run --rm --network host -e MODEL_NAME="$NVFP4" -v "$HERE":/t:ro imp:toolchain \
            python3 /t/conc_client.py "$PORT" 32 1 "$tag" > "$OUT/$tag.c32w$w.txt" 2>&1
        sample "$tag" "4_c32w$w"
    done
    docker stop -t 60 "$CTR" >/dev/null 2>&1
    docker logs "$CTR" > "$OUT/$tag.server.log" 2>&1
    docker rm -f "$CTR" >/dev/null 2>&1
done
