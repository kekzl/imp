#!/bin/bash
# Guarded baseline re-pin: refuses to run on top-range or volatile host days (issue #526/#697
# lesson: re-pinning on a top day bakes a threshold that fails on normal days).
# Gate: two gate-model tg128@pp512 probes must BOTH land inside the healthy band AND agree
# within 2%. On pass: gen_perf_baseline.sh for the gate model -> perf_baseline.json, for the
# north-star model -> perf_baseline_north_star.candidate.json (merge deliberately, then
# make verify-north-star). Run from repo root after make build: bash scripts/repin_baselines_if_median.sh.
set -euo pipefail

# Band for the spec-OFF gate metric on GATE_MODEL (tight, ~0.5% spread); the spec-ON metric's
# verify-path volatility (~11% across restarts) does not belong in this band.
# Pin 2026-10-31: tg128 281.35 (two cold-median passes 281.27-281.35); band = pin -4 % / +1.5 %.
# An intentional speedup lands above BAND_HI: pass BAND_LO/BAND_HI for that re-pin, then move the pin here.
BAND_LO="${BAND_LO:-270}"
BAND_HI="${BAND_HI:-286}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
IMG="${IMG:-imp:test}"
GATE_MODEL="${GATE_MODEL:-/models/Qwen3-8B-NVFP4-cortecs}"
NORTH_STAR_MODEL="${NORTH_STAR_MODEL:-/models/Qwen3-14B-NVFP4}"

# Load criterion (utilization + VRAM), not `docker ps`: idle non-GPU containers are no tenant,
# a Windows-side tenant is invisible there.
bash scripts/require_free_gpu.sh "guarded re-pin" || exit 2

probe() {
    docker run --rm --gpus all -u "$(id -u):$(id -g)" -w /tmp \
        -v "$MODELS_DIR":/models -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        --entrypoint bash "$IMG" -c "imp-cli --model $GATE_MODEL --bench --bench-pp 512 \
            --bench-reps 10 --max-tokens 128 --temperature 0 --json \
            --set speculative.ngram=false 2>/dev/null | jq -er .decode_tps"
}

P1=$(probe); P2=$(probe)
CLK=$(nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw --format=csv,noheader)
echo "probes: $P1 / $P2 tok/s   clocks: $CLK   band: [$BAND_LO, $BAND_HI]"

awk -v p1="$P1" -v p2="$P2" -v lo="$BAND_LO" -v hi="$BAND_HI" 'BEGIN {
    spread = (p1 > p2 ? p1 - p2 : p2 - p1) / (p1 > p2 ? p1 : p2)
    ok = p1 >= lo && p1 <= hi && p2 >= lo && p2 <= hi && spread <= 0.02
    printf "spread: %.1f%%  -> %s\n", 100 * spread, ok ? "MEDIAN-DAY: proceeding" : "NOT a stable median day: refusing to re-pin"
    exit !ok
}'

run_gen() {
    # Host facts for the pin: the runtime image has no git or nvidia-smi (see Makefile gen-perf-baseline).
    docker run --rm --gpus all -v "$MODELS_DIR":/models -v "$PWD":/src -w /src \
        -u "$(id -u):$(id -g)" -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        -e IMP_BASELINE_COMMIT="$(git rev-parse --short=8 HEAD)$(git diff --quiet && git diff --cached --quiet || echo -dirty)" \
        -e IMP_BASELINE_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" \
        -e IMP_BASELINE_CUDA="$(nvidia-smi | grep -oP 'CUDA (UMD )?Version:\s*\K[0-9.]+' | head -1)" \
        -e IMP_BASELINE_VRAM_TOTAL="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)" \
        --entrypoint bash "$IMG" scripts/gen_perf_baseline.sh "$1"
}

echo "== re-pinning gate baseline ($GATE_MODEL) =="
run_gen "$GATE_MODEL"

echo "== measuring north-star candidate ($NORTH_STAR_MODEL) =="
# gen_perf_baseline.sh carries thresholds and notes over from the file it overwrites:
# stage the north-star pin there so the candidate inherits its own policy.
gate_pin=$(mktemp)
trap 'cp "$gate_pin" tests/perf_baseline.json; rm -f "$gate_pin"' EXIT
cp tests/perf_baseline.json "$gate_pin"
cp tests/perf_baseline_north_star.json tests/perf_baseline.json
run_gen "$NORTH_STAR_MODEL"
mv tests/perf_baseline.json tests/perf_baseline_north_star.candidate.json

echo
echo "DONE. Next steps:"
echo "  - merge tests/perf_baseline_north_star.candidate.json into"
echo "    tests/perf_baseline_north_star.json (keep its schema/_note fields)"
echo "  - make verify-fast && make verify-north-star"
echo "  - one PR with both baselines; state the intended deltas"
