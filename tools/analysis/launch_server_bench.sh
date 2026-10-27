#!/usr/bin/env bash
# README-shaped server measurement: TRIALS fresh containers of IMAGE serving MODEL (the quickstart
# `docker run`), per trial: seconds to /health ok, host MemAvailable before / after load, card peak
# VRAM while serving, seq_stream_client.py (6 real prompts x GEN tokens), the server log.
# Usage: tools/analysis/launch_server_bench.sh OUTDIR /models/<model> [extra imp-server args]
# Env: IMAGE (ghcr.io/kekzl/imp:latest), TRIALS (3), PORT (8090), GEN (400), MODELS_DIR ($HOME/models),
#      CACHE_VOL (docker volume at /home/imp/.cache/imp, as in the README; unset = none).
set -uo pipefail
OUT="$1"; MODEL="$2"; shift 2
IMAGE=${IMAGE:-ghcr.io/kekzl/imp:latest}
TRIALS=${TRIALS:-3}
PORT=${PORT:-8090}
MODELS_DIR=${MODELS_DIR:-$HOME/models}
HERE="$(cd "$(dirname "$0")" && pwd)"
SMI=$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)
mkdir -p "$OUT"
avail_gib() { awk '/MemAvailable/ {printf "%.1f", $2 / 1048576}' /proc/meminfo; }
vram_mib() { "$SMI" --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
echo "image $IMAGE $(docker image inspect --format '{{index .RepoDigests 0}} v{{index .Config.Labels "org.opencontainers.image.version"}}' "$IMAGE")" | tee "$OUT/summary.txt"
for t in $(seq 1 "$TRIALS"); do
    docker rm -f launch-bench >/dev/null 2>&1; sleep 5
    bash "$HERE/../../scripts/require_free_gpu.sh" "launch_server_bench trial $t" >/dev/null || { echo "GPU BUSY before trial $t" | tee -a "$OUT/summary.txt"; exit 2; }
    ram0=$(avail_gib); vram0=$(vram_mib); t0=$(date +%s)
    docker run -d --name launch-bench --gpus all -v "$MODELS_DIR":/models ${CACHE_VOL:+-v "$CACHE_VOL":/home/imp/.cache/imp} -p "127.0.0.1:$PORT:$PORT" "$IMAGE" \
        --model "$MODEL" --host 0.0.0.0 --port "$PORT" "$@" >/dev/null
    ok=0
    for _ in $(seq 1 450); do
        sleep 2
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
        [ -n "$(docker ps -q -f name=launch-bench)" ] || break
    done
    load_s=$(( $(date +%s) - t0 ))
    docker logs launch-bench > "$OUT/server_t$t.log" 2>&1
    [ $ok -eq 1 ] || { echo "trial $t: server not healthy" | tee -a "$OUT/summary.txt"; tail -20 "$OUT/server_t$t.log"; exit 3; }
    ram1=$(avail_gib); health=$(curl -s "http://127.0.0.1:$PORT/health" | jq -c '{status,kv_blocks_total,kv_capacity_tokens}')
    ( peak=0; while [ ! -f "$OUT/.stop" ]; do v=$(vram_mib); [ "$v" -gt "$peak" ] && peak=$v; echo "$peak" > "$OUT/.peak"; sleep 0.5; done ) &
    sampler=$!
    docker run --rm --network host -e GEN="${GEN:-400}" -v "$HERE":/t:ro imp:toolchain \
        python3 /t/seq_stream_client.py "$PORT" "t$t" 2>&1 | grep -v -e CUDA -e NVIDIA | tee "$OUT/client_t$t.txt"
    touch "$OUT/.stop"; wait "$sampler"; rm -f "$OUT/.stop"
    docker logs launch-bench > "$OUT/server_t$t.log" 2>&1
    echo "trial $t: load ${load_s} s, host MemAvailable ${ram0} -> ${ram1} GiB, card VRAM ${vram0} MiB idle -> $(cat "$OUT/.peak") MiB peak, health $health" | tee -a "$OUT/summary.txt"
    grep MEDIAN "$OUT/client_t$t.txt" >> "$OUT/summary.txt"
    docker rm -f launch-bench >/dev/null 2>&1
done
rm -f "$OUT/.peak"
