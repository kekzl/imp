#!/usr/bin/env bash
# #2267 GPU run: bit-exact tall-tile test, per-shape GEMM microbench, e2e prefill sweep, then the
# PASS/FAIL verdict (scripts/accept_2267.sh). One GPU lock for all three.
# Usage: make build && bash scripts/bench_2267.sh [out_dir]. Exit = accept_2267.sh exit (0 = PASS).
# Env: IMP_BENCH_2267_OUT (default ~/.cache/imp/bench_2267/<utc stamp>), IMP_DEV_IMG (imp:toolchain),
#      IMP_GPU_BUSY_CHECK; sweep env (IMP_SWEEP_*) passes through to scripts/sweep_2267.sh.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
OUT="${1:-${IMP_BENCH_2267_OUT:-$HOME/.cache/imp/bench_2267/$(date -u +%Y%m%dT%H%M%SZ)}}"
DEV_IMG="${IMP_DEV_IMG:-imp:toolchain}"

if [ -z "${IMP_BENCH_2267_LOCKED:-}" ]; then
    # CPU part before the lock: the dev build holds test-quant and imp-bench (the image has no imp-bench).
    make dev >"${TMPDIR:-/tmp}/bench_2267_make_dev.log" 2>&1 ||
        { echo "ERROR setup: make dev failed (${TMPDIR:-/tmp}/bench_2267_make_dev.log)"; exit 1; }
    BUSY="${IMP_GPU_BUSY_CHECK:-$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh}"
    if [ -x "$BUSY" ]; then "$BUSY" || { echo "ERROR setup: GPU busy ($BUSY)"; exit 1; }; fi
    bash scripts/require_free_gpu.sh "bench_2267" || exit 1
    IMP_BENCH_2267_LOCKED=1 exec bash scripts/gpu_lock.sh run "bench_2267" -- bash "$0" "$OUT"
fi

mkdir -p "$OUT" || { echo "ERROR setup: cannot create $OUT"; exit 1; }
for b in build-dev/test-quant build-dev/imp-bench; do
    [ -x "$b" ] || { echo "ERROR setup: $b missing (make dev)"; exit 1; }
done
devrun() { docker run --rm --gpus all -v "$ROOT":/src -w /src "$DEV_IMG" "$@"; }

echo "== 1/3 bit-exact test -> $OUT/unit.log"
devrun ./build-dev/test-quant --gtest_filter='MmqQ8Imma.TallTilesBitIdenticalToBm128' >"$OUT/unit.log" 2>&1
echo "unit exit=$?" >>"$OUT/unit.log"
tail -3 "$OUT/unit.log"

echo "== 2/3 per-shape microbench -> $OUT/micro.log"
devrun ./build-dev/imp-bench q8imma >"$OUT/micro.log" 2>&1
echo "micro exit=$?" >>"$OUT/micro.log"
grep '^layer ' "$OUT/micro.log"

echo "== 3/3 e2e prefill sweep -> $OUT/e2e.log, $OUT/e2e.tsv"
IMP_SWEEP_2267_LOCKED=1 IMP_SWEEP_TSV="$OUT/e2e.tsv" bash scripts/sweep_2267.sh >"$OUT/e2e.log" 2>&1
echo "e2e exit=$?" >>"$OUT/e2e.log"
grep -v ' done$' "$OUT/e2e.log"

bash scripts/accept_2267.sh "$OUT"
