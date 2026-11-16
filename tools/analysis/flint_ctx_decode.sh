#!/usr/bin/env bash
# Flint decode tok/s vs context: imp-cli --bench at --bench-pp {512,16384,65536}, tg128, one process per point.
# Usage: tools/analysis/flint_ctx_decode.sh IMAGE TAG [extra --set args...]; writes $OUT_DIR/<TAG>_pp<N>.log.
set -euo pipefail
IMG=$1; TAG=$2; shift 2
MODEL=${MODEL:-/models/flint-nvfp4-official}
OUT_DIR=${OUT_DIR:-/mnt/data/scratch/flint_perf}
PPS=${PPS:-"512 16384 65536"}
REPS=${REPS:-3}
mkdir -p "$OUT_DIR"
for pp in $PPS; do
    log="$OUT_DIR/${TAG}_pp${pp}.log"
    docker run --rm --gpus all -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -v "$HOME/models:/models:ro" \
        --entrypoint imp-cli "$IMG" --model "$MODEL" --bench --bench-pp "$pp" --max-tokens 128 \
        --bench-reps "$REPS" "$@" > "$log" 2>&1 || echo "rc=$? $log"
    printf '%s\tpp%s\t%s\n' "$TAG" "$pp" "$(grep -E 'tg[0-9]+.*tok/s|decode.*tok/s' "$log" | tail -1)"
    sleep 10
done
