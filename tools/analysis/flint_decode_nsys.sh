#!/usr/bin/env bash
# nsys kernel sums of Flint decode (graphs on, node trace): imp-cli --bench pp N tg 256, 1 rep.
# Usage: tools/analysis/flint_decode_nsys.sh IMAGE TAG PP [extra args]; writes $OUT_DIR/TAG.{nsys-rep,sqlite,log}.
set -euo pipefail
IMG=$1; TAG=$2; PP=$3; shift 3
MODEL=${MODEL:-/models/flint-nvfp4-official}
OUT_DIR=${OUT_DIR:-/mnt/data/scratch/flint_perf}
NSYS=${NSYS:-/opt/nvidia/nsight-systems/2026.1.3}
mkdir -p "$OUT_DIR"; chmod 777 "$OUT_DIR"
docker run --rm --gpus all --user root -v "$HOME/models:/models:ro" -v "$NSYS:/nsys" -v "$OUT_DIR:/out" \
    --entrypoint /nsys/target-linux-x64/nsys "$IMG" profile --sample=none --cpuctxsw=none --backtrace=none \
    -t cuda --cuda-graph-trace=node -o "/out/$TAG" --force-overwrite=true \
    imp-cli --model "$MODEL" --bench --bench-pp "$PP" --max-tokens 256 --bench-reps 1 "$@" > "$OUT_DIR/$TAG.log" 2>&1
docker run --rm --user root -v "$NSYS:/nsys" -v "$OUT_DIR:/out" --entrypoint /nsys/target-linux-x64/nsys "$IMG" \
    export --type sqlite -o "/out/$TAG.sqlite" --force-overwrite=true "/out/$TAG.nsys-rep" >/dev/null
grep -E '^(pp|tg) .*tok/s' "$OUT_DIR/$TAG.log"
