#!/usr/bin/env bash
# degen_suite.py against a fresh imp-server of IMAGE + MODEL (python in a container, never the host).
# Usage: degen_suite_image.sh IMAGE MODEL_DIR TAG; writes $OUT_DIR/TAG.{json,log,server.log}.
set -euo pipefail
IMG=$1; MODEL=$2; TAG=$3
OUT_DIR=${OUT_DIR:-/mnt/data/scratch/flint_perf/degen}
PORT=${PORT:-8094}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$OUT_DIR"; chmod 777 "$OUT_DIR"
name="degen_$TAG"
docker rm -f "$name" >/dev/null 2>&1 || true
docker run -d --name "$name" --gpus all -p "127.0.0.1:$PORT:8080" -v "$HOME/models:/models:ro" "$IMG" \
    imp-server --model "$MODEL" --host 0.0.0.0 --port 8080 >/dev/null
for _ in $(seq 1 150); do
    curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && break
    sleep 2
done
rc=0
docker run --rm --network host -v "$REPO/tools/analysis:/t:ro" -v "$OUT_DIR:/out" python:3.13-slim \
    python3 -I /t/degen_suite.py --url "http://localhost:$PORT" --json "/out/$TAG.json" > "$OUT_DIR/$TAG.log" 2>&1 || rc=$?
docker logs "$name" > "$OUT_DIR/$TAG.server.log" 2>&1
docker rm -f "$name" >/dev/null
echo "$TAG suite_rc=$rc $(grep -cE 'FAIL' "$OUT_DIR/$TAG.log") FAIL lines; stderr hits: $(grep -cE 'CUDA error|capture failed|falling back|warmup.*invalid|\bNaN\b|is NaN|is Inf' "$OUT_DIR/$TAG.server.log")"
