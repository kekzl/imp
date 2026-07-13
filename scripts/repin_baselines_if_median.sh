#!/bin/bash
# Guarded baseline re-pin: refuses to run on top-range or volatile host days.
#
# Both open re-pins (tests/perf_baseline.json after the 07-12 top-range pin,
# tests/perf_baseline_north_star.json after the #982 lm_head net rule, −1.9%
# intended) must be sampled on an ORDINARY median host day — re-pinning on a
# top day bakes a gate threshold that fails on normal days (issue #526 /
# #697 lesson). 2026-07-12 and 2026-07-13 both probed top-range (285.5 and
# 271→295 volatile, mem-clock 14001 vs nominal 13801), so the refresh waits.
#
# Run from the repo root on the host (invokes docker itself):
#   bash scripts/repin_baselines_if_median.sh
#
# Gate: two Qwen3-8B Q8_0 tg128@pp512 probes must BOTH land inside the
# healthy band AND agree within 2%. On pass:
#   1. scripts/gen_perf_baseline.sh (Q8)  -> tests/perf_baseline.json
#   2. scripts/gen_perf_baseline.sh (14B) -> tests/perf_baseline_north_star.candidate.json
#      (candidate only — the north-star file carries extra schema fields and
#      an explanatory _note; merge deliberately, then `make verify-north-star`.)
# Then: make verify-fast, review diffs, and ship both files in one PR that
# states the intended deltas (north star −~1.9% from the #982 net rule).
set -euo pipefail

BAND_LO="${BAND_LO:-266}"
BAND_HI="${BAND_HI:-278}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
IMG="${IMG:-imp:test}"

if [ "$(docker ps -q | wc -l)" != "0" ]; then
    echo "ABORT: containers running — GPU must be exclusive"; exit 2
fi

probe() {
    docker run --rm --gpus all -u "$(id -u):$(id -g)" -w /tmp \
        -v "$MODELS_DIR":/models -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        --entrypoint /usr/local/bin/imp-cli "$IMG" \
        --model /models/Qwen3-8B-Q8_0.gguf --bench --bench-pp 512 \
        --bench-reps 10 --max-tokens 128 --temperature 0 2>&1 |
        sed -n 's/^tg .*( *\([0-9.]*\) tok\/s.*/\1/p'
}

P1=$(probe); P2=$(probe)
CLK=$(nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw --format=csv,noheader)
echo "probes: $P1 / $P2 tok/s   clocks: $CLK   band: [$BAND_LO, $BAND_HI]"

python3 - "$P1" "$P2" "$BAND_LO" "$BAND_HI" <<'EOF'
import sys
p1, p2, lo, hi = map(float, sys.argv[1:5])
spread = abs(p1 - p2) / max(p1, p2)
ok = lo <= p1 <= hi and lo <= p2 <= hi and spread <= 0.02
print(f"spread: {100*spread:.1f}%  -> {'MEDIAN-DAY: proceeding' if ok else 'NOT a stable median day: refusing to re-pin'}")
sys.exit(0 if ok else 1)
EOF

run_gen() {
    docker run --rm --gpus all -v "$MODELS_DIR":/models -v "$PWD":/src -w /src \
        -u "$(id -u):$(id -g)" -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        --entrypoint bash "$IMG" scripts/gen_perf_baseline.sh "$1"
}

echo "== re-pinning Q8 gate baseline =="
run_gen "/models/Qwen3-8B-Q8_0.gguf"

echo "== measuring north-star candidate (14B Q6_K) =="
cp tests/perf_baseline.json /tmp/repin_q8_baseline.json
run_gen "/models/Qwen3-14B-Q6_K.gguf"
mv tests/perf_baseline.json tests/perf_baseline_north_star.candidate.json
mv /tmp/repin_q8_baseline.json tests/perf_baseline.json

echo
echo "DONE. Next steps:"
echo "  - merge tests/perf_baseline_north_star.candidate.json into"
echo "    tests/perf_baseline_north_star.json (keep its schema/_note fields)"
echo "  - make verify-fast && make verify-north-star"
echo "  - one PR with both baselines; state the intended deltas"
