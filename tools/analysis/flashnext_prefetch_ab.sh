#!/usr/bin/env bash
# Flash-Next server decode A/B for moe.expert_prefetch: fresh imp-server per arm, one warm request, then
# seq_stream_client.py (6 real prompts x GEN greedy tokens, decode tok/s per request + MEDIAN), graceful stop so
# the device expert cache stats line (hits, misses, prefetched, routed) reaches the log. Arms alternate PAIRS times.
# Usage: ARMS="label|image|k=v,k=v ..." OUT=/mnt/data/scratch/dir [PAIRS=2] [PORT=8260] [GEN=400] \
#        tools/analysis/flashnext_prefetch_ab.sh
set -uo pipefail
MODELS_DIR=${MODELS_DIR:-$HOME/models}
HERE="$(cd "$(dirname "$0")" && pwd)"
ARMS=${ARMS:?ARMS="label|image|k=v,... ..."}
PAIRS=${PAIRS:-2}
PORT=${PORT:-8260}
GEN=${GEN:-400}
MODEL=${MODEL:-Qwen3.8-Flash-Next-NVFP4}
CTR=${CTR:-flashnextpf-$PORT}
OUT=${OUT:?OUT=dir for logs}
mkdir -p "$OUT" && chmod 777 "$OUT"
echo "harness: md5=$(md5sum "$0" | cut -c1-12)" | tee -a "$OUT/summary.txt"
for pair in $(seq 1 "$PAIRS"); do
    for arm in $ARMS; do
        IFS='|' read -r label img sets <<< "$arm"
        tag="p${pair}_${label}"
        docker rm -f "$CTR" >/dev/null 2>&1
        "$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh" > "$OUT/$tag.busy.txt" 2>&1 || { echo "GPU BUSY before $tag"; exit 2; }
        # shellcheck disable=SC2046
        docker run -d --name "$CTR" --gpus all -v "$MODELS_DIR":/models:ro -v "$OUT":/out -p "$PORT:$PORT" "$img" \
            imp-server --model "/models/$MODEL" --host 0.0.0.0 --port "$PORT" \
            $(for s in ${sets//,/ }; do printf -- "--set %s " "$s"; done) >/dev/null
        for _ in $(seq 1 900); do sleep 2; curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
            [ -n "$(docker ps -q -f name="$CTR")" ] || { echo "$tag: server died"; docker logs "$CTR" 2>&1 | tail -5; exit 3; }; done
        docker run --rm --network host -e GEN=32 -v "$HERE":/t:ro imp:toolchain \
            python3 /t/seq_stream_client.py "$PORT" warm > "$OUT/$tag.warm.txt" 2>&1
        docker run --rm --network host -e GEN="$GEN" -v "$HERE":/t:ro imp:toolchain \
            python3 /t/seq_stream_client.py "$PORT" "$tag" > "$OUT/$tag.c1.txt" 2>&1
        docker stop -t 120 "$CTR" >/dev/null 2>&1
        docker logs "$CTR" > "$OUT/$tag.server.log" 2>&1
        docker rm -f "$CTR" >/dev/null 2>&1
        med=$(awk '/MEDIAN/ {print $4}' "$OUT/$tag.c1.txt")
        rates=$(awk '/ decode / && !/MEDIAN/ {printf "%s ", $(NF-1)}' "$OUT/$tag.c1.txt")
        stats=$(sed -nE 's/.*Device expert cache stats: (.*)/\1/p' "$OUT/$tag.server.log" | tail -1)
        echo "$tag median ${med:-FAIL} [ $rates] | $stats" | tee -a "$OUT/summary.txt"
    done
done
