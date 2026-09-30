#!/usr/bin/env bash
# #2267e GPU run + verdict: 2-CTA/SM mbarrier Q8 IMMA kernel against the legacy kernel. One GPU lock.
# Gates (PASS iff all hold):
#   G1 unit    test-quant MmqQ8Imma.* exit 0 (new vs legacy <= 1e-3 rel, deterministic)
#   G2 ncu     median gpu__time_duration of the new BM=128 kernel, Qwen3-8B q_o, M=2048, base clock: <= 260 us
#   G3 e2e     scripts/accept_2267.sh arm on: off/on >= 1.00 at 2048, 4096, 8192 tokens, both models
#   G4 micro   imp-bench q8imma layer sums, new/legacy <= 1.02 at M = 256, 512, 1024, both models
#   G5 KL      scripts/measure_2256.sh arm A (IMMA on): KL mean <= 0.005, p99 <= 0.05
# Usage: make build && bash scripts/bench_2267e.sh [out_dir]. Exit 0 = PASS, 1 = FAIL or setup error.
# Env: IMP_BENCH_2267E_OUT (default ~/.cache/imp/bench_2267e/<utc stamp>), IMP_DEV_IMG (imp:toolchain),
#      IMP_GPU_BUSY_CHECK; IMP_SWEEP_* and IMP_MEASURE_* pass through to the sweep / KL scripts.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
OUT="${1:-${IMP_BENCH_2267E_OUT:-$HOME/.cache/imp/bench_2267e/$(date -u +%Y%m%dT%H%M%SZ)}}"
DEV_IMG="${IMP_DEV_IMG:-imp:toolchain}"
NCU_MAX_US=260
KL_MEAN_MAX=0.005
KL_P99_MAX=0.05

if [ -z "${IMP_BENCH_2267E_LOCKED:-}" ]; then
    mkdir -p "$OUT" || { echo "ERROR setup: cannot create $OUT"; exit 1; }
    # CPU parts before the lock: dev build (test-quant, imp-bench), image check, KL reference.
    make dev >"$OUT/make_dev.log" 2>&1 || { echo "ERROR setup: make dev failed ($OUT/make_dev.log)"; exit 1; }
    IMG="${IMP_TEST_IMG:-$(bash scripts/image_tag.sh)}"
    docker image inspect "$IMG" >/dev/null 2>&1 || { echo "ERROR setup: image $IMG missing (make build)"; exit 1; }
    bash scripts/image_tag.sh check "$IMG" || { echo "ERROR setup: $IMG is not this tree (make build)"; exit 1; }
    IMP_MEASURE_REF_ONLY=1 timeout 10800 bash scripts/measure_2256.sh >"$OUT/kl_ref.log" 2>&1 ||
        { echo "ERROR setup: KL reference failed ($OUT/kl_ref.log)"; exit 1; }
    BUSY="${IMP_GPU_BUSY_CHECK:-$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh}"
    if [ -x "$BUSY" ]; then "$BUSY" || { echo "ERROR setup: GPU busy ($BUSY)"; exit 1; }; fi
    bash scripts/require_free_gpu.sh "bench_2267e" || exit 1
    IMP_BENCH_2267E_LOCKED=1 exec bash scripts/gpu_lock.sh run "bench_2267e" -- bash "$0" "$OUT"
fi

mkdir -p "$OUT" || { echo "ERROR setup: cannot create $OUT"; exit 1; }
for b in build-dev/imp-bench build-dev/test-quant; do
    [ -x "$b" ] || { echo "ERROR setup: $b missing (make dev)"; exit 1; }
done
CTR=imp_bench_2267e
# docker run under timeout: the client proxies SIGTERM; the EXIT trap removes a container that outlives it.
# shellcheck disable=SC2329  # invoked by the EXIT trap
cleanup() { docker rm -f "$CTR" >/dev/null 2>&1 || true; }
trap cleanup EXIT
DEVRUN=(docker run --rm --name "$CTR" --gpus all -v "$ROOT":/src -w /src "$DEV_IMG")
FAIL=0
gate() {  # gate <PASS|FAIL> <text>
    echo "$1 $2" | tee -a "$OUT/verdict.log"
    [ "$1" = PASS ] || FAIL=1
}
: >"$OUT/verdict.log"

echo "== 1/5 unit tests -> $OUT/unit.log"
cleanup
timeout 1800 "${DEVRUN[@]}" ./build-dev/test-quant --gtest_filter='MmqQ8Imma.*' >"$OUT/unit.log" 2>&1
rc=$?
grep -E '^new_vs_legacy|^\[  (PASSED|FAILED) ' "$OUT/unit.log"
if [ "$rc" = 0 ]; then gate PASS "G1 unit: test-quant MmqQ8Imma.* exit 0"; else gate FAIL "G1 unit: exit $rc"; fi

echo "== 2/5 ncu Qwen3-8B q_o M=2048 -> $OUT/ncu_{new,legacy}.csv"
NCU_METRICS=gpu__time_duration.sum,launch__registers_per_thread,launch__occupancy_limit_registers,\
launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps,sm__warps_active.avg.pct_of_peak_sustained_active
ncu_run() {  # ncu_run <tag> <demangled-name regex>
    cleanup
    timeout 1800 docker run --rm --name "$CTR" --gpus all --cap-add SYS_ADMIN -v "$ROOT":/src -w /src \
        -e IMP_Q8IMMA_MODEL=Qwen3-8B -e IMP_Q8IMMA_SHAPE=q_o -e IMP_Q8IMMA_M=2048 "$DEV_IMG" \
        ncu --csv --print-units base --kernel-name-base demangled --kernel-name "regex:$2" \
        --launch-skip 3 --launch-count 5 --metrics "$NCU_METRICS" \
        ./build-dev/imp-bench q8imma >"$OUT/ncu_$1.csv" 2>&1
}
ncu_metric() {  # ncu_metric <tag> <metric> -> median over the profiled launches ("" if none)
    grep "^\"" "$OUT/ncu_$1.csv" | awk -F'","' -v m="$2" '$(NF-2) == m { v = $NF; gsub(/[",]/, "", v); print v }' |
        sort -g | awk '{ a[NR] = $1 } END { if (NR) print a[int((NR + 1) / 2)] }'
}
ncu_run new '::mmq_imma_kernel<128, false, false, false>'
rc_new=$?
ncu_run legacy '::mmq_imma_kernel_legacy<128, false, false, false>'
rc_legacy=$?
for t in new legacy; do
    ns=$(ncu_metric "$t" gpu__time_duration.sum)
    us=""; [ -n "$ns" ] && us=$(awk -v n="$ns" 'BEGIN { printf "%.2f", n / 1000 }')
    echo "ncu $t: duration_us=${us:-n/a} regs=$(ncu_metric "$t" launch__registers_per_thread)" \
        "ctas_per_sm_limit_regs=$(ncu_metric "$t" launch__occupancy_limit_registers)" \
        "ctas_per_sm_limit_smem=$(ncu_metric "$t" launch__occupancy_limit_shared_mem)" \
        "achieved_occupancy_pct=$(ncu_metric "$t" sm__warps_active.avg.pct_of_peak_sustained_active)" |
        tee -a "$OUT/ncu_summary.log"
    [ "$t" = new ] && NEW_US="$us"
done
if [ "$rc_new" != 0 ] || [ "$rc_legacy" != 0 ] || [ -z "${NEW_US:-}" ]; then
    gate FAIL "G2 ncu: exit new=$rc_new legacy=$rc_legacy, new duration '${NEW_US:-}' (see $OUT/ncu_new.csv)"
elif awk -v u="$NEW_US" -v m="$NCU_MAX_US" 'BEGIN { exit !(u <= m) }'; then
    gate PASS "G2 ncu: new kernel ${NEW_US} us <= ${NCU_MAX_US} us"
else
    gate FAIL "G2 ncu: new kernel ${NEW_US} us > ${NCU_MAX_US} us"
fi

echo "== 3/5 per-shape microbench -> $OUT/micro.log"
cleanup
timeout 3600 "${DEVRUN[@]}" ./build-dev/imp-bench q8imma >"$OUT/micro.log" 2>&1
rc=$?
grep -E '^(q8imma|layer) ' "$OUT/micro.log" |
    awk '{ for (i = 2; i <= NF; i++) { split($i, kv, "="); v[kv[1]] = kv[2] }
           r = (v["legacy"] > 0) ? v["imma"] / v["legacy"] : -1
           printf "%s new/legacy %.3f off/new %.3f\n", $0, r, (v["imma"] > 0) ? v["off"] / v["imma"] : -1 }' |
    tee "$OUT/micro_ratios.log"
g4=$(grep '^layer ' "$OUT/micro.log" | awk '
    { for (i = 2; i <= NF; i++) { split($i, kv, "="); v[kv[1]] = kv[2] } }
    v["M"] == 256 || v["M"] == 512 || v["M"] == 1024 {
        n++; r = (v["legacy"] > 0 && v["imma"] > 0) ? v["imma"] / v["legacy"] : 99
        if (r > 1.02) { bad++; printf "%s M=%s new/legacy %.3f; ", v["model"], v["M"], r } }
    END { if (n != 6) printf "only %d of 6 layer rows; ", n; exit (bad > 0 || n != 6) }')
g4rc=$?
if [ "$rc" = 0 ] && [ "$g4rc" = 0 ]; then
    gate PASS "G4 micro: layer new/legacy <= 1.02 at M 256/512/1024, both models"
else
    gate FAIL "G4 micro: exit $rc; ${g4:-ok}"
fi

echo "== 4/5 e2e prefill sweep on/off -> $OUT/e2e.log, $OUT/e2e.tsv"
IMP_SWEEP_2267_LOCKED=1 IMP_SWEEP_TSV="$OUT/e2e.tsv" timeout 14400 bash scripts/sweep_2267.sh >"$OUT/e2e.log" 2>&1
rc=$?
grep -v ' done$' "$OUT/e2e.log" | tail -40
bash scripts/accept_2267.sh "$OUT" on >"$OUT/accept_2267.log" 2>&1
arc=$?
cat "$OUT/accept_2267.log"
if [ "$rc" = 0 ] && [ "$arc" = 0 ]; then
    gate PASS "G3 e2e: accept_2267.sh on PASS (off/on >= 1.00 at 2048/4096/8192, both models)"
else
    gate FAIL "G3 e2e: sweep exit $rc, accept_2267.sh exit $arc"
fi

echo "== 5/5 KL vs HF fp32 -> $OUT/kl.log"
IMP_MEASURE_2256_LOCKED=1 timeout 7200 bash scripts/measure_2256.sh >"$OUT/kl.log" 2>&1
rc=$?
grep -E '^(metric|KL mean|\|dlogp|top-1|PPL|prefill) ' "$OUT/kl.log"
# row(): printf '%-34s %-42s %s' -> arm A = "mean / p50 / p99 / max"
kl_a=$(grep '^KL mean / p50 / p99 / max' "$OUT/kl.log" | head -1 | cut -c36-77)
kl_mean=$(awk -F' / ' '{ print $1 + 0 }' <<<"$kl_a")
kl_p99=$(awk -F' / ' '{ print $3 + 0 }' <<<"$kl_a")
if [ "$rc" = 0 ] && [ -n "$kl_a" ] &&
    awk -v m="$kl_mean" -v p="$kl_p99" -v mm="$KL_MEAN_MAX" -v pm="$KL_P99_MAX" 'BEGIN { exit !(m <= mm && p <= pm) }'; then
    gate PASS "G5 KL: arm A mean $kl_mean <= $KL_MEAN_MAX, p99 $kl_p99 <= $KL_P99_MAX"
else
    gate FAIL "G5 KL: measure exit $rc, arm A mean '${kl_mean:-}' p99 '${kl_p99:-}' (limits $KL_MEAN_MAX / $KL_P99_MAX)"
fi

echo "== verdict ($OUT/verdict.log)"
cat "$OUT/verdict.log"
[ "$FAIL" = 0 ] && { echo "BENCH_2267E PASS"; exit 0; }
echo "BENCH_2267E FAIL"
exit 1
