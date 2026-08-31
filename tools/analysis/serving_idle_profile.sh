#!/usr/bin/env bash
# serving_idle_profile.sh - nsys profile of imp-server under a 32-stream wave
# burst, then the idle attribution (tools/analysis/nsys_gap_attribution.py).
#
# The server runs from the dev build (build-dev/imp-server) inside
# imp:toolchain (the only image that ships nsys); the .qdstrm -> .nsys-rep
# conversion fails silently inside that image (missing libcap/libdw), so the
# export runs on the host's nsys. Same pinned 32-stream shape as the
# BENCHMARKS.md attribution (mbs 32, seq 4096, kv blocks 2387) so numbers
# compare to the 2026-08-24/26 rows.
#
# Usage: bash tools/analysis/serving_idle_profile.sh [OUTDIR] [CONC] [WAVES]
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT=${1:-/tmp/serving_idle}
CONC=${2:-32}
WAVES=${3:-3}
PORT=${PORT:-8093}
MODEL=/models/Qwen3.8-27B-NVFP4
MODELS_DIR=${MODELS_DIR:-$HOME/models}
EXTRA=${EXTRA:-}
mkdir -p "$OUT"
chmod 777 "$OUT"
docker rm -f imp-idle >/dev/null 2>&1

~/.claude/skills/gpu-stats/gpu-busy-check.sh >/dev/null || { echo "GPU BUSY - aborting"; exit 2; }

# shellcheck disable=SC2086
docker run -d --name imp-idle --gpus all -v "$MODELS_DIR":/models -v "$ROOT":/src -v "$OUT":/out \
    -w /tmp -p ${PORT}:${PORT} imp:toolchain \
    nsys profile --sample=none --cpuctxsw=none --backtrace=none -t cuda --cuda-graph-trace=node \
    -o /out/serving --force-overwrite=true \
    /src/build-dev/imp-server --model $MODEL --port $PORT --host 0.0.0.0 --max-concurrent $CONC \
    --set runtime.max_batch_size=32 --set runtime.max_seq_len=4096 --set kv_cache.max_blocks=2387 \
    --set diagnostics.step_timing=true $EXTRA >/dev/null
for _ in $(seq 1 240); do
    sleep 2
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
    if [ -z "$(docker ps -q -f name=imp-idle)" ]; then
        echo "server died:"; docker logs imp-idle 2>&1 | tail -20; exit 3
    fi
done
python3 "$ROOT/tools/analysis/conc_client.py" $PORT $CONC $WAVES idle 2>&1 | tee "$OUT/client.log"
# Graceful stop: nsys must see the process exit to flush the trace.
docker kill --signal=SIGINT imp-idle >/dev/null 2>&1
for _ in $(seq 1 60); do
    sleep 2
    [ -z "$(docker ps -q -f name=imp-idle)" ] && break
done
docker logs imp-idle > "$OUT/server.log" 2>&1
docker rm -f imp-idle >/dev/null 2>&1
ls -la "$OUT"
grep -E 'step-timing|outside-timing' "$OUT/server.log" | tail -6
if [ ! -f "$OUT/serving.nsys-rep" ] && ls "$OUT"/*.qdstrm >/dev/null 2>&1; then
    IMP=$(ls /opt/nvidia/nsight-systems/*/host-linux-x64/QdstrmImporter 2>/dev/null | head -1)
    [ -n "$IMP" ] && "$IMP" -i "$OUT"/serving.qdstrm -o "$OUT/serving.nsys-rep"
fi
nsys export --type sqlite --force-overwrite=true -o "$OUT/serving.sqlite" "$OUT/serving.nsys-rep" >/dev/null 2>&1
python3 "$ROOT/tools/analysis/nsys_gap_attribution.py" "$OUT/serving.sqlite" | tee "$OUT/attribution.txt"
