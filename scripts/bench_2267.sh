#!/usr/bin/env bash
# #2267 GPU run: bit-exact tests, legacy-vs-pipeline kernel microbench, IMMA-vs-off route microbench,
# e2e prefill sweep (arms new / off / old), then the PASS/FAIL verdict (scripts/accept_2267.sh).
# One GPU lock for all four. Old arm = image built from IMP_2267_BASE (default: merge-base with
# origin/main) in a detached worktree at $TMPDIR/imp-2267-base, tagged imp:2267-base-<sha12>.
# Usage: bash scripts/bench_2267.sh [out_dir]. Exit = accept_2267.sh exit (0 = PASS).
# Env: IMP_BENCH_2267_OUT (default ~/.cache/imp/bench_2267/<utc stamp>), IMP_2267_BASE, IMP_DEV_IMG
#      (imp:toolchain), IMP_GPU_BUSY_CHECK; sweep env (IMP_SWEEP_*) passes through to scripts/sweep_2267.sh.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
OUT="${1:-${IMP_BENCH_2267_OUT:-$HOME/.cache/imp/bench_2267/$(date -u +%Y%m%dT%H%M%SZ)}}"
DEV_IMG="${IMP_DEV_IMG:-imp:toolchain}"
LOGDIR="${TMPDIR:-/tmp}"

if [ -z "${IMP_BENCH_2267_LOCKED:-}" ]; then
    # CPU part before the lock: dev build (test-quant, imp-bench), this tree's image, the base image.
    make dev >"$LOGDIR/bench_2267_make_dev.log" 2>&1 ||
        { echo "ERROR setup: make dev failed ($LOGDIR/bench_2267_make_dev.log)"; exit 1; }
    make build >"$LOGDIR/bench_2267_make_build.log" 2>&1 ||
        { echo "ERROR setup: make build failed ($LOGDIR/bench_2267_make_build.log)"; exit 1; }
    BASE="${IMP_2267_BASE:-$(git merge-base HEAD origin/main)}"
    BASE_SHA="$(git rev-parse --verify "$BASE^{commit}")" || { echo "ERROR setup: base $BASE unknown"; exit 1; }
    [ "$BASE_SHA" != "$(git rev-parse HEAD)" ] ||
        { echo "ERROR setup: base $BASE_SHA is HEAD (set IMP_2267_BASE to the pre-#2267 commit)"; exit 1; }
    OLD_IMG="imp:2267-base-${BASE_SHA:0:12}"
    BASE_WT="$LOGDIR/imp-2267-base"
    git worktree remove --force "$BASE_WT" >/dev/null 2>&1 || rm -rf "$BASE_WT"
    git worktree add --detach "$BASE_WT" "$BASE_SHA" >"$LOGDIR/bench_2267_base_build.log" 2>&1 &&
        make -C "$BASE_WT" build DOCKER_IMG="$OLD_IMG" >>"$LOGDIR/bench_2267_base_build.log" 2>&1
    rc=$?
    git worktree remove --force "$BASE_WT" >/dev/null 2>&1
    [ "$rc" = 0 ] || { echo "ERROR setup: base image $OLD_IMG failed ($LOGDIR/bench_2267_base_build.log)"; exit 1; }
    BUSY="${IMP_GPU_BUSY_CHECK:-$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh}"
    if [ -x "$BUSY" ]; then "$BUSY" || { echo "ERROR setup: GPU busy ($BUSY)"; exit 1; }; fi
    bash scripts/require_free_gpu.sh "bench_2267" || exit 1
    IMP_BENCH_2267_LOCKED=1 IMP_SWEEP_OLD_IMG="$OLD_IMG" \
        exec bash scripts/gpu_lock.sh run "bench_2267" -- bash "$0" "$OUT"
fi

mkdir -p "$OUT" || { echo "ERROR setup: cannot create $OUT"; exit 1; }
for b in build-dev/test-quant build-dev/imp-bench; do
    [ -x "$b" ] || { echo "ERROR setup: $b missing (make dev)"; exit 1; }
done
[ -n "${IMP_SWEEP_OLD_IMG:-}" ] || { echo "ERROR setup: IMP_SWEEP_OLD_IMG unset"; exit 1; }
devrun() { docker run --rm --gpus all -v "$ROOT":/src -w /src "$DEV_IMG" "$@"; }

echo "== 1/4 bit-exact tests -> $OUT/unit.log"
devrun ./build-dev/test-quant \
    --gtest_filter='MmqQ8Imma.PipelineBitIdenticalToLegacy:MmqQ8Imma.RouteBitIdenticalToLegacy' >"$OUT/unit.log" 2>&1
echo "unit exit=$?" >>"$OUT/unit.log"
tail -3 "$OUT/unit.log"

echo "== 2/4 kernel microbench, legacy vs pipeline -> $OUT/micro.log"
devrun ./build-dev/test-quant --gtest_also_run_disabled_tests \
    --gtest_filter='MmqQ8ImmaBench.DISABLED_PipelineVsLegacy' >"$OUT/micro.log" 2>&1
echo "micro exit=$?" >>"$OUT/micro.log"
grep '^q8pipe ' "$OUT/micro.log"

echo "== 3/4 route microbench, IMMA vs off -> $OUT/route.log"
devrun ./build-dev/imp-bench q8imma >"$OUT/route.log" 2>&1
echo "route exit=$?" >>"$OUT/route.log"
grep '^layer ' "$OUT/route.log"

echo "== 4/4 e2e prefill sweep (new, off, old@$IMP_SWEEP_OLD_IMG) -> $OUT/e2e.log, $OUT/e2e.tsv"
IMP_SWEEP_2267_LOCKED=1 IMP_SWEEP_TSV="$OUT/e2e.tsv" bash scripts/sweep_2267.sh >"$OUT/e2e.log" 2>&1
echo "e2e exit=$?" >>"$OUT/e2e.log"
grep -v ' done$' "$OUT/e2e.log"

bash scripts/accept_2267.sh "$OUT"
