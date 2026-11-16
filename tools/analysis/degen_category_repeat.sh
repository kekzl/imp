#!/usr/bin/env bash
# One imp-server of IMAGE + MODEL, degen_suite.py --only CATEGORY run REPS times: pass/fail per run.
# Usage: degen_category_repeat.sh IMAGE MODEL_DIR CATEGORY REPS
set -euo pipefail
IMG=$1; MODEL=$2; CAT=$3; REPS=$4
PORT=${PORT:-8097}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
name="degenrep_$$"
docker run -d --name "$name" --gpus all -p "127.0.0.1:$PORT:8080" -v "$HOME/models:/models:ro" "$IMG" \
    imp-server --model "$MODEL" --host 0.0.0.0 --port 8080 >/dev/null
for _ in $(seq 1 150); do
    curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && break
    sleep 2
done
for r in $(seq 1 "$REPS"); do
    out=$(docker run --rm --network host -v "$REPO/tools/analysis:/t:ro" python:3.13-slim \
        python3 -I /t/degen_suite.py --url "http://localhost:$PORT" --only "$CAT" 2>&1 || true)
    echo "run $r: $(echo "$out" | grep -E '^degen_suite: [0-9]+ checks' | head -1)"
    echo "$out" | grep -E '^  FAIL' | cut -c1-200 || true
done
docker rm -f "$name" >/dev/null
