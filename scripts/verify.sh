#!/bin/bash
# verify.sh — pre-commit/pre-push verification for imp.
#
# Four-step gate:
#   1. Build (incremental)
#   2. Tests (gtest filter or full suite)
#   3. Perf vs. tests/perf_baseline.json (decode regression > 3% = fail)
#   4. Smoke prompts on real models (degeneration detector)
#
# Modes:
#   verify.sh fast    Unit tests + perf baseline + 1 smoke prompt   (~90s)
#   verify.sh full    Full ctest + perf baseline + 2 smoke prompts  (~5min)
#
# Env overrides:
#   IMP_VERIFY_BIN=build/imp-cli
#   IMP_VERIFY_TESTS=build/imp-tests
#   IMP_VERIFY_MODELS=models
#   IMP_VERIFY_BASELINE=tests/perf_baseline.json
#   IMP_VERIFY_CHUNK_SIZE=0   prefill chunk size for perf bench (0 = single-chunk,
#                             default: 0 for legacy baseline, per-json for v1)
#   IMP_VERIFY_SKIP_BUILD=1   skip cmake build step
#   IMP_VERIFY_SKIP_PERF=1    skip perf-baseline regression check (use when the
#                             baseline is known-stale; refresh with
#                             scripts/gen_perf_baseline.sh)
#   IMP_VERIFY_TRIALS=3       independent PROCESSES the perf gate medians over.
#                             1 restores the old single-shot behaviour (fast,
#                             but see the note at the gate itself).
#   IMP_VERIFY_IN_DOCKER=1    sentinel set by the auto-re-exec block; do not set manually
#
# Auto-Docker fallback: if cmake is not on PATH (Clean-Host workflow), the
# script re-execs itself inside the imp:test container, mounting the repo at
# /src and using the prebuilt /usr/local/bin/imp-cli + imp-tests. Requires
# 'make build' to have produced the imp:test image first.
#
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Auto re-exec in the imp:test container when cmake is unavailable on the host
# (Clean-Host workflow: no language toolchains installed). The runtime image
# has prebuilt binaries at /usr/local/bin/, so we skip the build step and point
# IMP_VERIFY_BIN/TESTS at them. IMP_VERIFY_IN_DOCKER guards against infinite
# re-exec if the runtime image somehow also lacks cmake.
if ! command -v cmake >/dev/null 2>&1 && [ "${IMP_VERIFY_IN_DOCKER:-0}" != "1" ]; then
    if ! docker image inspect imp:test >/dev/null 2>&1; then
        echo "verify: cmake not found on host and imp:test image not built." >&2
        echo "        Run 'make build' first, then re-run." >&2
        exit 1
    fi
    echo "verify: host cmake unavailable — re-executing in imp:test container"
    exec docker run --rm --gpus all \
        -v "$ROOT":/src -w /src \
        -v "$HOME/models":"$HOME/models":ro \
        -e IMP_VERIFY_IN_DOCKER=1 \
        -e IMP_VERIFY_BIN=/usr/local/bin/imp-cli \
        -e IMP_VERIFY_TESTS=/usr/local/bin/imp-tests \
        -e IMP_VERIFY_SKIP_BUILD=1 \
        -e IMP_VERIFY_SKIP_PERF="${IMP_VERIFY_SKIP_PERF:-0}" \
        -e IMP_VERIFY_SKIP_VRAM="${IMP_VERIFY_SKIP_VRAM:-0}" \
        -e IMP_VERIFY_SKIP_GRAPHS="${IMP_VERIFY_SKIP_GRAPHS:-0}" \
        -e IMP_VERIFY_BASELINE="${IMP_VERIFY_BASELINE:-tests/perf_baseline.json}" \
        -e IMP_VERIFY_CHUNK_SIZE="${IMP_VERIFY_CHUNK_SIZE:-0}" \
        -e IMP_VERIFY_TRIALS="${IMP_VERIFY_TRIALS:-3}" \
        --entrypoint bash imp:test scripts/verify.sh "$@"
fi

MODE="${1:-fast}"

BIN="${IMP_VERIFY_BIN:-build/imp-cli}"
TESTS_BIN="${IMP_VERIFY_TESTS:-build/imp-tests}"
MODELS="${IMP_VERIFY_MODELS:-models}"
BASELINE="${IMP_VERIFY_BASELINE:-tests/perf_baseline.json}"
# IMP_VERIFY_CHUNK_SIZE: prefill chunk size to use for the perf bench.
# Empty/unset = use whatever the baseline JSON specifies, or 0 (single-chunk) by default.
CHUNK_SIZE="${IMP_VERIFY_CHUNK_SIZE:-0}"

RED=$'\033[0;31m'; GRN=$'\033[0;32m'; YLW=$'\033[0;33m'; RST=$'\033[0m'
FAIL=0
section() { echo; echo "${YLW}== $* ==${RST}"; }
pass()    { echo "${GRN}PASS${RST} $*"; }
fail()    { echo "${RED}FAIL${RST} $*"; FAIL=$((FAIL+1)); }
skip()    { echo "${YLW}SKIP${RST} $*"; }
warn()    { echo "${YLW}WARN${RST} $*"; }

# --- Host-drift guard (#526) -------------------------------------------------
# This WSL2 box has day-level "depressed host" states where decode reads 8-15%
# low DESPITE full methodology (memory: q8_drift_host_artifact_2026_06_05). The
# culprit is host/driver state, not code — but the perf gate cannot tell the two
# apart from a single bench number. So we sample GPU clocks/power DURING the
# decode bench and, if a regression coincides with the depressed-host signature,
# we degrade FAIL→WARN (clearly labeled) instead of crying false-positive.
#
# Healthy under-load signature: ~2850 MHz SM / 13801 MHz mem / ~500 W.
# Depressed         => mem-clock median < 13801 MHz, OR power max < 400 W,
#                       OR SM-clock median < 2000 MHz (healthy decode load is
#                       ~2850 MHz sustained; the 2026-07-11 episode pinned SM
#                       at 285-367 MHz with mem partly still at 13801).
#   - mem-clock is the cleanest tell (GDDR7 either P0-clocks or it doesn't).
#   - power uses MAX over the run (a robust upper bound) with a conservative
#     400 W floor so a healthy ~500 W run never trips it.
#   - The SM clock ramps ~1s at bench start (cold-start artifact, audit §5), so
#     we drop the first 2 samples before aggregating to avoid a false depressed
#     classification from the ramp. A run with only 1-2 samples can't drop the
#     ramp, so it's treated as no-data (gate fails open: plain FAIL).
# Fail-open: if nvidia-smi is missing or errors, the sampler no-ops and the gate
# behaves exactly as before (no degradation logic kicks in).
GPU_DRIFT_MEM_FLOOR=13801   # mem clock (MHz) at/above which the host is healthy
GPU_DRIFT_POWER_FLOOR=400   # power (W) max below which the host is depressed
GPU_DRIFT_SM_FLOOR=2000     # SM clock (MHz) median below which the host is depressed
_SAMPLE_FILE=""
_SAMPLE_PID=""

# Start a background sampler writing "sm,mem,power" CSV rows every 1s.
# No-op (leaves _SAMPLE_PID empty) if nvidia-smi can't be queried.
gpu_sample_start() {
    _SAMPLE_FILE=""; _SAMPLE_PID=""
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw \
        --format=csv,noheader,nounits >/dev/null 2>&1 || return 0
    _SAMPLE_FILE=$(mktemp)
    ( while true; do
        nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw \
            --format=csv,noheader,nounits 2>/dev/null | head -1
        sleep 1
      done ) >>"$_SAMPLE_FILE" 2>/dev/null &
    _SAMPLE_PID=$!
}

# Stop the sampler and classify the run. Sets globals:
#   GPU_DRIFT_DEPRESSED = 0|1
#   GPU_DRIFT_DESC      = human-readable "mem=… power=…" summary ("" if no data)
gpu_sample_stop() {
    GPU_DRIFT_DEPRESSED=0
    GPU_DRIFT_DESC=""
    [ -n "$_SAMPLE_PID" ] && kill "$_SAMPLE_PID" >/dev/null 2>&1
    [ -n "$_SAMPLE_PID" ] && wait "$_SAMPLE_PID" 2>/dev/null
    [ -n "$_SAMPLE_FILE" ] && [ -s "$_SAMPLE_FILE" ] || { _cleanup_sample; return 0; }
    # Drop the first 2 samples (clock-ramp cold-start), then aggregate:
    # median mem clock + max power over the steady portion of the run.
    read -r MEM_MED PWR_MAX SM_MED N < <(awk -F, '
        { gsub(/ /,""); n++; sm[n]=$1+0; mem[n]=$2+0; pwr[n]=$3+0 }
        END {
            # n<=2 is too short to drop the ~1s cold-ramp samples: classify as
            # no-data (N=0) so the gate fails open (plain FAIL on regression),
            # never letting cold-ramp low-mem samples force a depressed WARN.
            if (n <= 2) { print 0, 0, 0, 0; exit }
            start = 3; cnt = 0; pmax = 0;
            for (i = start; i <= n; i++) {
                m[++cnt] = mem[i]; sm2[cnt] = sm[i];
                if (pwr[i] > pmax) pmax = pwr[i];
            }
            if (cnt == 0) { print 0, 0, 0, 0; exit }
            # medians of m[1..cnt] and sm2[1..cnt]
            for (a = 1; a <= cnt; a++) for (b = a+1; b <= cnt; b++) {
                if (m[b] < m[a]) { t=m[a]; m[a]=m[b]; m[b]=t }
                if (sm2[b] < sm2[a]) { t=sm2[a]; sm2[a]=sm2[b]; sm2[b]=t }
            }
            med = (cnt % 2) ? m[(cnt+1)/2] : (m[cnt/2] + m[cnt/2+1]) / 2;
            smmed = (cnt % 2) ? sm2[(cnt+1)/2] : (sm2[cnt/2] + sm2[cnt/2+1]) / 2;
            printf "%.0f %.0f %.0f %d\n", med, pmax, smmed, cnt;
        }' "$_SAMPLE_FILE")
    _cleanup_sample
    [ "${N:-0}" -gt 0 ] || return 0
    GPU_DRIFT_DESC="sm=${SM_MED}MHz(med) mem=${MEM_MED}MHz(med) power=${PWR_MAX}W(max) n=${N}"
    if [ "$MEM_MED" -lt "$GPU_DRIFT_MEM_FLOOR" ] || [ "$PWR_MAX" -lt "$GPU_DRIFT_POWER_FLOOR" ] ||
       [ "$SM_MED" -lt "$GPU_DRIFT_SM_FLOOR" ]; then
        GPU_DRIFT_DEPRESSED=1
    fi
}

_cleanup_sample() {
    [ -n "$_SAMPLE_FILE" ] && rm -f "$_SAMPLE_FILE"
    _SAMPLE_FILE=""; _SAMPLE_PID=""
}

# Report a decode regression, degrading FAIL→WARN under the depressed-host
# signature (#526). $1 = context label for the message.
decode_regression() {
    local ctx="$1"
    if [ "${GPU_DRIFT_DEPRESSED:-0}" = "1" ]; then
        warn "$ctx: depressed-host signature ($GPU_DRIFT_DESC) — decode delta not attributable to code, FAIL degraded to WARN (#526)"
    else
        fail "$ctx"
    fi
}

# Docker-only host (host has neither cmake nor a build/ directory): run
# the canonical Docker build, then exit early — the test / perf / smoke
# gates below need the host build artefacts and there's no clean way to
# reach them from here. Override with IMP_VERIFY_SKIP_BUILD=1 if
# cmake-on-host is preferred.
if [ "${IMP_VERIFY_SKIP_BUILD:-0}" != "1" ] && ! command -v cmake >/dev/null 2>&1; then
    section "build (docker — host has no cmake)"
    if make build >/tmp/imp_verify_build.log 2>&1; then
        pass "docker build"
    else
        fail "docker build (see /tmp/imp_verify_build.log)"
        tail -20 /tmp/imp_verify_build.log
        exit 1
    fi
    section "remaining gates (skipped — Docker-only host)"
    skip "tests / perf / smoke require host build artefacts; run 'make test-gpu'"
    skip "and 'make verify' inside Docker manually for the full pre-merge gate"
    exit 0
fi

# -------------------------------------------------------------------- 1. build
section "build"
if [ "${IMP_VERIFY_SKIP_BUILD:-0}" = "1" ]; then
    skip "build (IMP_VERIFY_SKIP_BUILD=1)"
elif [ ! -d build ]; then
    fail "no build/ directory — run 'cmake -B build -DCMAKE_BUILD_TYPE=Release' first"
else
    if cmake --build build -j"$(nproc)" >/tmp/imp_verify_build.log 2>&1; then
        pass "incremental build"
    else
        fail "build (see /tmp/imp_verify_build.log)"
        tail -20 /tmp/imp_verify_build.log
        exit 1
    fi
fi

# -------------------------------------------------------------------- 2. tests
section "tests"
if [ ! -x "$TESTS_BIN" ]; then
    fail "$TESTS_BIN not found"
else
    if [ "$MODE" = "fast" ]; then
        # VramBudgetReserve is in here because CI structurally cannot run it:
        # compute_vram_budget queries the device for total VRAM, so the suite is
        # SKIP_IF_NO_CUDA and the GPU-less CI lane skips it. Three of its tests
        # were red for three PRs before anyone looked (AUDIT B63). The pre-push
        # gate is the only place that can catch that class.
        # ForwardPassTest is here for the same reason: it is SKIP_IF_NO_CUDA, so
        # CI cannot run it, and DecodeLogitsInvariantToBatchComposition (#1314)
        # is the only assert in the tree that a sequence's logits do not depend
        # on its batch neighbours — the class #1044/#1045 came from.
        FILTER="TensorTest.*:GgufLoaderTest.*:Tokenizer*:ChatTemplate*:KVCache*:GemmTest.*:FP8GemmTest.*:SamplingTest.*:SoftmaxTest.*:AttentionTest.*:VramBudget*:ForwardPassTest.*"
        if "$TESTS_BIN" --gtest_filter="$FILTER" >/tmp/imp_verify_tests.log 2>&1; then
            pass "fast gtest filter"
        else
            fail "gtest (see /tmp/imp_verify_tests.log)"
            tail -30 /tmp/imp_verify_tests.log
        fi
    else
        if "$TESTS_BIN" >/tmp/imp_verify_tests.log 2>&1; then
            pass "full gtest suite"
        else
            fail "gtest (see /tmp/imp_verify_tests.log)"
            grep -E "FAIL|fatal" /tmp/imp_verify_tests.log | head -20
        fi

        # test-e2e unit/gpu lane-split guard (R5/#580): the unit lane is a
        # gtest_filter, so a rename could silently move a CPU test into the GPU
        # lane. This asserts the filter still resolves to the frozen CPU set.
        # Fail-open: skips cleanly if the binary or script isn't locatable.
        _E2E_BIN="$(dirname "$TESTS_BIN")/test-e2e"
        # DUPLICATE of _unit_e2e_filter in CMakeLists.txt (~line 607) — keep in
        # sync. Can't source it here: the container path runs without a
        # configured build/ dir, so ctest/CMake variables are unreachable.
        # Going stale is caught by the primary ctest guard (guard_e2e_lane_split)
        # plus the frozen EXPECTED list inside check_e2e_lane_split.sh itself.
        _LANE_FILTER="BatchBuilderTest.*:SchedulerTest.*:RequestTest.*:EndToEndTest.*:StubModelTest.LoadStubModel:StubModelTest.TokenizeStub"
        if [ -x "$_E2E_BIN" ] && [ -x scripts/check_e2e_lane_split.sh ]; then
            if scripts/check_e2e_lane_split.sh "$_E2E_BIN" "$_LANE_FILTER" >/tmp/imp_verify_lane.log 2>&1; then
                pass "e2e unit/gpu lane split"
            else
                fail "e2e lane split (see /tmp/imp_verify_lane.log)"
                cat /tmp/imp_verify_lane.log
            fi
        else
            skip "e2e lane-split guard (test-e2e or guard script not found)"
        fi
    fi
fi

# --------------------------------------------------------------------- 3. perf
section "perf vs baseline"
if [ "${IMP_VERIFY_SKIP_PERF:-0}" = "1" ]; then
    skip "perf gate (IMP_VERIFY_SKIP_PERF=1)"
elif [ ! -f "$BASELINE" ]; then
    skip "no $BASELINE — run scripts/gen_perf_baseline.sh to create one"
elif [ ! -x "$BIN" ]; then
    fail "$BIN not found"
elif ! command -v jq >/dev/null 2>&1; then
    skip "jq not installed (needed to parse $BASELINE)"
else
    # Detect baseline schema version (R4/#579: all three baselines now carry a
    # common "schema_version" string — "legacy-v1" | "multi-model-v1"). Fall back
    # to the historical numeric ".version" (multi-model == 1) for forward-tolerance
    # if an older/regenerated file still lacks schema_version.
    BL_SCHEMA=$(jq -r '.schema_version // empty' "$BASELINE")
    BL_VERSION=$(jq -r '.version // 0' "$BASELINE")

    if [ "$BL_SCHEMA" = "multi-model-v1" ] || [ "$BL_VERSION" = "1" ]; then
        # ---- Multi-model v1 schema ----
        DEC_THR=$(jq -r '.regression_thresholds.decode_pct' "$BASELINE")
        PRE_THR=$(jq -r '.regression_thresholds.prefill_pct' "$BASELINE")
        BL_CHUNK=$(jq -r '.prefill_chunk_size // 0' "$BASELINE")
        REPS=3
        BENCH_CHUNK="${CHUNK_SIZE:-$BL_CHUNK}"
        ANY_MEASURED=0
        # Iterate over every model entry in the JSON
        while IFS= read -r BL_MODEL; do
            BL_TG=$(jq -r ".models[\"$BL_MODEL\"].tg256" "$BASELINE")
            BL_PP=$(jq -r ".models[\"$BL_MODEL\"].pp512" "$BASELINE")
            MODEL_PATH="$MODELS/$BL_MODEL"
            if [ ! -f "$MODEL_PATH" ]; then
                skip "  skip $BL_MODEL (not present)"
                continue
            fi
            ANY_MEASURED=1
            ERR=$(mktemp)
            gpu_sample_start
            "$BIN" --model "$MODEL_PATH" --bench --bench-pp 512 --bench-reps $REPS \
                  --prefill-chunk-size "$BENCH_CHUNK" --max-tokens 256 --temperature 0 \
                  --set speculative.ngram=false >/dev/null 2>"$ERR"
            gpu_sample_stop
            PP=$(grep -oP '^pp\s+512\s.*\(\s*\K[0-9.]+(?=\s+tok/s)' "$ERR" | head -1)
            TG=$(grep -oP '^tg\s+256\s.*\(\s*\K[0-9.]+(?=\s+tok/s)' "$ERR" | head -1)
            if [ -z "$PP" ] || [ -z "$TG" ]; then
                fail "$BL_MODEL: could not parse bench output (see $ERR)"
                tail -10 "$ERR"
            else
                rm -f "$ERR"
                DEC_DELTA=$(awk -v cur="$TG" -v base="$BL_TG" 'BEGIN{printf "%.2f", (cur-base)/base*100}')
                PRE_DELTA=$(awk -v cur="$PP" -v base="$BL_PP" 'BEGIN{printf "%.2f", (cur-base)/base*100}')
                DEC_REG=$(awk -v d="$DEC_DELTA" -v t="$DEC_THR" 'BEGIN{print (-d > t) ? 1 : 0}')
                PRE_REG=$(awk -v d="$PRE_DELTA" -v t="$PRE_THR" 'BEGIN{print (-d > t) ? 1 : 0}')
                printf "  %-42s  tg256=%7.2f (base %7.2f, %+.1f%%)  pp512=%7.1f (base %7.1f, %+.1f%%)\n" \
                    "$BL_MODEL" "$TG" "$BL_TG" "$DEC_DELTA" "$PP" "$BL_PP" "$PRE_DELTA"
                [ -n "${GPU_DRIFT_DESC:-}" ] && echo "    GPU during bench: $GPU_DRIFT_DESC"
                if [ "$DEC_REG" = "1" ]; then
                    decode_regression "$BL_MODEL: decode regression > ${DEC_THR}%"
                else
                    pass "$BL_MODEL: decode within ${DEC_THR}% threshold"
                fi
                if [ "$PRE_REG" = "1" ]; then
                    echo "${YLW}WARN${RST} $BL_MODEL: prefill regression > ${PRE_THR}% (cuBLAS variance is common)"
                fi
            fi
        done < <(jq -r '.models | keys[]' "$BASELINE")
        if [ "$ANY_MEASURED" = "0" ]; then
            skip "no baseline models found in $MODELS/"
        fi
    else
        # ---- Legacy schema (perf_baseline.json) ----
        BL_MODEL=$(jq -r '.model' "$BASELINE")
        BL_TG=$(jq -r '.metrics.decode_tps.tg128' "$BASELINE")
        BL_PP=$(jq -r '.metrics.prefill_tps.pp512' "$BASELINE")
        DEC_THR=$(jq -r '.thresholds.decode_regression_pct' "$BASELINE")
        PRE_THR=$(jq -r '.thresholds.prefill_regression_pct' "$BASELINE")
        MODEL_PATH="$MODELS/$BL_MODEL"

        if [ ! -f "$MODEL_PATH" ]; then
            skip "baseline model $MODEL_PATH not present"
        else
            REPS=3
            ERR=$(mktemp)
            # --prefill-chunk-size 0 forces single-chunk prefill so the baseline
            # remains apples-to-apples with the pre-chunked-prefill measurements.
            # speculative.ngram=false: the bench prompt is self-repetitive
            # (~99.9% draft accept), so with speculation ON tg measures the
            # batched spec-verify GEMMs — restart-volatile like cuBLAS prefill
            # (observed 11% swing across healthy-clock restarts, 2026-07-15).
            #
            # MEDIAN OVER INDEPENDENT PROCESSES, not one shot. --bench-reps
            # averages WITHIN a process; it cannot see the variance that lives
            # BETWEEN them (cuBLASLt heuristic re-selection, allocator layout,
            # clock/host state at load). Measured 2026-08-03 on an idle host,
            # six runs of near-identical code: 278.59 … 289.77 tok/s — a 4.01 %
            # spread against a 3.00 % threshold. A single-shot gate that tight
            # reports its own noise: the same tree failed at -3.25 % and passed
            # at +0.90 % on the same day.
            #
            # The line this replaces claimed "the decode GEMV hot path ... is
            # stable (<1%)". That is true within a process and false across
            # them, which is the number the gate actually compares.
            #
            # What the median does and does not buy: it drops single-run
            # outliers, which is the failure that produced the -3.25 % scare.
            # It does NOT flatten the hours-scale host drift — three runs taken
            # back to back spread 0.09 %, the same binary hours apart spread
            # 4.01 %. That is why the spread is PRINTED: a tight spread means
            # the median is worth believing, a wide one means the host moved
            # mid-gate and the verdict is about the box, not the diff.
            TRIALS="${IMP_VERIFY_TRIALS:-3}"
            TG_ALL=""; PP_ALL=""
            gpu_sample_start
            for _t in $(seq 1 "$TRIALS"); do
                "$BIN" --model "$MODEL_PATH" --bench --bench-pp 512 --bench-reps $REPS \
                      --prefill-chunk-size "${CHUNK_SIZE}" --max-tokens 128 --temperature 0 \
                      --set speculative.ngram=false >/dev/null 2>"$ERR"
                _tg=$(grep -oP '^tg\s+128\s.*\(\s*\K[0-9.]+(?=\s+tok/s)' "$ERR" | head -1)
                _pp=$(grep -oP '^pp\s+512\s.*\(\s*\K[0-9.]+(?=\s+tok/s)' "$ERR" | head -1)
                [ -n "$_tg" ] && TG_ALL="$TG_ALL$_tg\n"
                [ -n "$_pp" ] && PP_ALL="$PP_ALL$_pp\n"
            done
            gpu_sample_stop
            median() { printf "$1" | grep -v '^$' | sort -n | awk '{a[NR]=$1} END{if(NR==0)exit 1; print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }
            spread() { printf "$1" | grep -v '^$' | sort -n | awk '{a[NR]=$1} END{if(NR<2)exit 0; printf "%.2f", (a[NR]-a[1])/a[1]*100}'; }
            # Bench lines (stderr) have variable spacing inside parens for short numbers:
            #   "pp   512 tokens  avg    38.47 ms  (13310.12 tok/s)  [3 reps]"
            #   "tg   128 tokens  avg   861.50 ms  ( 148.58 tok/s)  [3 reps]"
            PP=$(median "$PP_ALL" || true)
            TG=$(median "$TG_ALL" || true)
            TG_SPREAD=$(spread "$TG_ALL")
            if [ -z "$PP" ] || [ -z "$TG" ]; then
                fail "could not parse bench output (see $ERR)"
                tail -15 "$ERR"
            else
                rm -f "$ERR"
                DEC_DELTA=$(awk -v cur="$TG" -v base="$BL_TG" 'BEGIN{printf "%.2f", (cur-base)/base*100}')
                PRE_DELTA=$(awk -v cur="$PP" -v base="$BL_PP" 'BEGIN{printf "%.2f", (cur-base)/base*100}')
                DEC_REG=$(awk -v d="$DEC_DELTA" -v t="$DEC_THR" 'BEGIN{print (-d > t) ? 1 : 0}')
                PRE_REG=$(awk -v d="$PRE_DELTA" -v t="$PRE_THR" 'BEGIN{print (-d > t) ? 1 : 0}')

                if [ "$TRIALS" -gt 1 ]; then
                    printf "  perf gate: median of %s independent runs (spread %s%%)\n" "$TRIALS" \
                           "${TG_SPREAD:-n/a}"
                fi
                printf "  decode  tg128 = %7.2f tok/s  (baseline %7.2f, delta %+s%%)\n" "$TG" "$BL_TG" "$DEC_DELTA"
                printf "  prefill pp512 = %7.2f tok/s  (baseline %7.2f, delta %+s%%)\n" "$PP" "$BL_PP" "$PRE_DELTA"
                [ -n "${GPU_DRIFT_DESC:-}" ] && echo "  GPU during bench: $GPU_DRIFT_DESC"

                if [ "$DEC_REG" = "1" ]; then
                    decode_regression "decode regression > ${DEC_THR}%  (if expected: ./scripts/gen_perf_baseline.sh $MODEL_PATH)"
                else
                    pass "decode within ${DEC_THR}% threshold"
                fi
                if [ "$PRE_REG" = "1" ]; then
                    # prefill is noisy (cuBLAS autotuning), warn only
                    echo "${YLW}WARN${RST} prefill regression > ${PRE_THR}% (often cuBLAS variance, not real)"
                else
                    pass "prefill within ${PRE_THR}% threshold"
                fi

                # ---- long-context prefill (single-chunk FA2): pp4096..pp65536 ----
                # Crosses the cuBLAS→FMHA threshold so it exercises the
                # register-resident FA2 kernel (attention.fmha_fa2=on default).
                # Guards the FA2 prefill win AND the #1022 32K-64K TTFT band.
                # Warn-only (prefill-noise convention) — though large-context
                # prefill is actually stable (<0.5% cross-restart at 32K, measured
                # 2026-07-16, unlike pp512's cuBLAS-algo jitter). Lengths past the
                # model ctx or the 70K bench ceiling simply have no baseline key
                # and are skipped. TTFT ms ≈ pp_len / pp_tps.
                for PPLEN in 4096 8192 16384 32768 65536; do
                    BL_PP=$(jq -r ".metrics.prefill_tps.pp${PPLEN} // empty" "$BASELINE")
                    [ -z "$BL_PP" ] && continue
                    ERR2=$(mktemp)
                    "$BIN" --model "$MODEL_PATH" --bench --bench-pp "$PPLEN" --bench-reps $REPS \
                          --prefill-chunk-size 0 --max-tokens 1 --temperature 0 --max-seq-len 70000 \
                          >/dev/null 2>"$ERR2"
                    PPTPS=$(grep -oP "^pp\s+${PPLEN}\s.*\(\s*\K[0-9.]+(?=\s+tok/s)" "$ERR2" | head -1)
                    rm -f "$ERR2"
                    [ -z "$PPTPS" ] && continue
                    PP_DELTA=$(awk -v cur="$PPTPS" -v base="$BL_PP" 'BEGIN{printf "%.2f", (cur-base)/base*100}')
                    PP_REG=$(awk -v d="$PP_DELTA" -v t="$PRE_THR" 'BEGIN{print (-d > t) ? 1 : 0}')
                    printf "  prefill pp%-5s= %7.2f tok/s  (baseline %7.2f, delta %+s%%, FA2)\n" "$PPLEN" "$PPTPS" "$BL_PP" "$PP_DELTA"
                    if [ "$PP_REG" = "1" ]; then
                        echo "${YLW}WARN${RST} long-ctx prefill (pp${PPLEN}/FA2) regression > ${PRE_THR}% — check attention.fmha_fa2"
                    else
                        pass "long-ctx prefill (pp${PPLEN}/FA2) within ${PRE_THR}% threshold"
                    fi
                done
            fi
        fi
    fi
fi

# ------------------------ 3.4. peak-VRAM gate (memory acceptance criterion 8)
# The baseline has carried a `memory_mb` block and a `vram_increase_pct`
# threshold since it was written, and nothing ever read them — a declared gate
# that never existed. Peak VRAM is exactly the quantity that creeps silently:
# every "just cache one more thing" lands here, and it is invisible in the perf
# numbers until a model stops fitting (#1103 is that failure, and it cost 7x
# decode).
#
# Gated on own_peak, NOT device peak_used: device-used also carries the CUDA
# primary context (~1.7 GiB here) and anything a neighbour process holds, so it
# moves for reasons that have nothing to do with the change under test.
# own_peak is this process's allocations since engine init and nothing else.
#
# Its own short run rather than piggy-backing on the perf bench: --mem-report
# starts a 2 ms sampler thread, and perturbing the number the perf gate reads
# to save ten seconds is a bad trade.
section "peak VRAM vs baseline"
if [ "${IMP_VERIFY_SKIP_VRAM:-0}" = "1" ]; then
    skip "peak-VRAM gate (IMP_VERIFY_SKIP_VRAM=1)"
elif [ ! -f "$BASELINE" ] || ! command -v jq >/dev/null 2>&1; then
    skip "peak-VRAM gate (no baseline or no jq)"
else
    # MODEL_PATH is set inside the perf gate above, which IMP_VERIFY_SKIP_PERF=1
    # skips entirely — referencing it unguarded killed the whole script under
    # `set -u`. Resolve it here instead of inheriting it, so this gate stands on
    # its own whether or not the perf gate ran.
    VRAM_MODEL=$(jq -r '.model // empty' "$BASELINE")
    VRAM_MODEL_PATH="$MODELS/$VRAM_MODEL"
    if [ -z "$VRAM_MODEL" ] || [ ! -x "$BIN" ] || [ ! -f "$VRAM_MODEL_PATH" ]; then
        skip "peak-VRAM gate (binary or model missing)"
        BL_VRAM=""
    else
        BL_VRAM=$(jq -r '.metrics.memory_mb.own_peak_mb // empty' "$BASELINE")
    fi
    VRAM_THR=$(jq -r '.thresholds.vram_increase_pct // 10' "$BASELINE")
    if [ -z "$BL_VRAM" ] && [ -n "$VRAM_MODEL" ] && [ -x "$BIN" ] && [ -f "$VRAM_MODEL_PATH" ]; then
        skip "no .metrics.memory_mb.own_peak_mb in $BASELINE — run scripts/gen_perf_baseline.sh"
    elif [ -z "$BL_VRAM" ]; then
        :  # already reported above
    else
        ERR_V=$(mktemp)
        # BOTH streams: imp's INFO logs (which carry the VRAM audit table) go to
        # STDOUT, only the bench result lines go to stderr. Capturing stderr
        # alone — as the perf gate above correctly does for its own numbers —
        # silently yields nothing here.
        "$BIN" --model "$VRAM_MODEL_PATH" --bench --bench-pp 128 --bench-reps 1 --max-tokens 8 \
              --temperature 0 --set speculative.ngram=false --mem-report \
              >"$ERR_V" 2>&1
        OWN_PEAK=$(grep -oP 'own_peak=\K[0-9]+' "$ERR_V" | tail -1)
        if [ -z "$OWN_PEAK" ]; then
            warn "could not read own_peak from --mem-report — gate not evaluated"
        else
            DELTA=$(awk -v a="$OWN_PEAK" -v b="$BL_VRAM" 'BEGIN{printf "%.2f", (a-b)/b*100}')
            echo "  own peak VRAM = $OWN_PEAK MiB  (baseline $BL_VRAM, delta ${DELTA}%)"
            OVER=$(awk -v d="$DELTA" -v t="$VRAM_THR" 'BEGIN{print (d>t)?1:0}')
            if [ "$OVER" = "1" ]; then
                fail "peak VRAM grew more than ${VRAM_THR}% over the baseline"
            else
                pass "peak VRAM within ${VRAM_THR}% of baseline"
            fi
        fi
        rm -f "$ERR_V"
    fi
fi

# ------------------------ 3.5. graphs-ON vs graphs-OFF decode regression gate
# Catches future PRs that silently break CUDA Graph capture in the decode loop.
# Without graphs, decode is launch-overhead-bound (875-1170 launches/step on
# dense Q8). Graphs-ON wins +95-376% (memo: cuda_graphs_moe_works_2026_05_07).
# A graphs-ON improvement that drops below kMinSpeedupX = 1.5× signals a path
# that fell out of capture (host sync inside captured region, malloc on hot
# path, etc.). Skipped by IMP_VERIFY_SKIP_GRAPHS=1.
section "graphs ON vs OFF decode gate"
if [ "${IMP_VERIFY_SKIP_GRAPHS:-0}" = "1" ]; then
    skip "graphs gate (IMP_VERIFY_SKIP_GRAPHS=1)"
elif [ ! -f "$BIN" ]; then
    skip "graphs gate requires host build artefacts"
else
    GRAPHS_MODEL="${IMP_VERIFY_GRAPHS_MODEL:-Qwen3-4B-Instruct-2507-Q8_0.gguf}"
    GRAPHS_MODEL_PATH="$MODELS/$GRAPHS_MODEL"
    if [ ! -f "$GRAPHS_MODEL_PATH" ]; then
        skip "graphs gate model $GRAPHS_MODEL_PATH not present"
    else
        # Threshold lowered from 1.5 → 1.3 after F1's warmup-pre-pass made
        # graphs-OFF significantly faster on dense Q8 (compresses the ratio).
        # Cross-model A/B (post-patches, reps=2, pp=256 tg=256):
        #   Qwen3-4B Q8       1.90x
        #   Qwen3.5-GDN Q8    2.23x
        #   Llama-3.2-3B Q8   2.38x
        #   Qwen3-8B Q8       1.20x   ← bigger model = larger kernel time =
        #                                less launch-overhead share = lower ratio.
        # 1.3 catches catastrophic graph failures (≈ 1.0x = full fallback to
        # per-step decode) without rejecting healthy big-model decodes.
        MIN_SPEEDUP_X="${IMP_VERIFY_MIN_GRAPH_SPEEDUP:-1.3}"
        ERR_NG=$(mktemp); ERR_G=$(mktemp)
        # Sample clocks/power across BOTH runs: a depressed host (#526 — SM
        # pinned low / mem at half / power capped) makes kernels dominate and
        # compresses the ON/OFF ratio toward 1.0x on ANY build (measured
        # 0.85-1.0x on main and branches alike, 2 consecutive nights) — the
        # ratio stops being attributable to code.
        # NOTE: the power floor (400 W) is calibrated on 8B-class decode; the
        # 4B gate model can draw less even when healthy, so this gate leans
        # WARN. That is deliberate: a false WARN costs a re-run on a healthy
        # host, a false FAIL blocks a push on host noise. The hard FAIL still
        # fires whenever the signature reads healthy.
        gpu_sample_start
        "$BIN" --model "$GRAPHS_MODEL_PATH" --bench --bench-pp 256 --bench-reps 2 \
              --max-tokens 256 --temperature 0 --no-cuda-graphs \
              --set speculative.ngram=false >/dev/null 2>"$ERR_NG"
        "$BIN" --model "$GRAPHS_MODEL_PATH" --bench --bench-pp 256 --bench-reps 2 \
              --max-tokens 256 --temperature 0 \
              --set speculative.ngram=false >/dev/null 2>"$ERR_G"
        gpu_sample_stop
        TG_NG=$(grep -oP '^tg\s+256\s.*\(\s*\K[0-9.]+(?=\s+tok/s)' "$ERR_NG" | head -1)
        TG_G=$(grep -oP '^tg\s+256\s.*\(\s*\K[0-9.]+(?=\s+tok/s)' "$ERR_G" | head -1)
        if [ -z "$TG_NG" ] || [ -z "$TG_G" ]; then
            fail "could not parse graphs bench output"
            echo "  no-graphs stderr tail:"; tail -8 "$ERR_NG"
            echo "  graphs stderr tail:";    tail -8 "$ERR_G"
        else
            SPEEDUP=$(awk -v g="$TG_G" -v ng="$TG_NG" 'BEGIN{printf "%.3f", g/ng}')
            BELOW=$(awk -v s="$SPEEDUP" -v t="$MIN_SPEEDUP_X" 'BEGIN{print (s < t) ? 1 : 0}')
            printf "  graphs OFF tg256 = %7.2f tok/s\n" "$TG_NG"
            printf "  graphs ON  tg256 = %7.2f tok/s   (%.2fx, threshold %sx)\n" \
                   "$TG_G" "$SPEEDUP" "$MIN_SPEEDUP_X"
            if [ "$BELOW" = "1" ]; then
                if [ "${GPU_DRIFT_DEPRESSED:-0}" = "1" ]; then
                    warn "graphs speedup ${SPEEDUP}x < ${MIN_SPEEDUP_X}x under depressed-host signature ($GPU_DRIFT_DESC) — ratio not attributable to code, FAIL degraded to WARN (#526). Re-run on a healthy host; if it still fails there, look for new host syncs / cudaMalloc on the decode hot path"
                else
                    fail "graph capture broken: speedup ${SPEEDUP}x < ${MIN_SPEEDUP_X}x — \
look for new host syncs / cudaMalloc on the decode hot path"
                fi
            else
                pass "graphs ON delivers ≥${MIN_SPEEDUP_X}x decode speedup"
            fi
        fi
        rm -f "$ERR_NG" "$ERR_G"
    fi
fi

# ------------------------------------------------------------- 4. smoke prompts
section "smoke prompts (degeneration check)"

# Greedy decode on a known-deterministic prompt.
# Quality gate: output must contain expected substring AND last 32 tokens must
# have at least 8 distinct tokens (catches "own own own" stuck-token failures).
smoke_prompt() {
    local label="$1" model="$2" prompt="$3" expect="$4"
    if [ ! -f "$MODELS/$model" ]; then
        skip "$label ($model not present)"
        return
    fi
    local ERR; ERR=$(mktemp)
    OUT=$("$BIN" --model "$MODELS/$model" --prompt "$prompt" \
          --max-tokens 64 --temperature 0 --chat-template none 2>"$ERR")
    # Token markers '[tok=NNNN ' word']' land on stderr.
    TOKS=$(grep -oP '\[tok=\K[0-9]+' "$ERR" | tail -32)
    DISTINCT=$(echo "$TOKS" | sort -u | wc -l)
    # Generated word stream (stripped of token id prefix) for NaN/Inf scan
    WORDS=$(grep -oP "\[tok=[0-9]+ '\K[^']*" "$ERR" | tr '\n' ' ')
    rm -f "$ERR"

    if echo " $WORDS " | grep -qiE ' (nan|inf|-inf|-nan) '; then
        fail "$label — NaN/Inf token in output"
        echo "  words: $WORDS"
        return
    fi
    if [ "$DISTINCT" -lt 8 ]; then
        fail "$label — degenerate (only $DISTINCT distinct tokens in last 32)"
        echo "  tokens: $(echo "$TOKS" | tr '\n' ' ')"
        return
    fi
    # Generated text appears interleaved with logs on stdout — substring match works
    if ! echo "$OUT$WORDS" | grep -q "$expect"; then
        fail "$label — expected '$expect' in output"
        echo "  words: $WORDS"
        return
    fi
    pass "$label (distinct=$DISTINCT, contains '$expect')"
}

smoke_prompt "Qwen3-4B Q8_0 (dense)" \
    "Qwen3-4B-Instruct-2507-Q8_0.gguf" \
    "The capital of France is" \
    "Paris"

if [ "$MODE" = "full" ]; then
    smoke_prompt "Qwen3.5-4B Q8_0 (GDN)" \
        "Qwen3.5-4B-Q8_0.gguf" \
        "The capital of France is" \
        "Paris"
fi

# ------------------------------------------------------------------- summary
echo
if [ "$FAIL" -eq 0 ]; then
    echo "${GRN}=== verify $MODE: OK ===${RST}"
    exit 0
else
    echo "${RED}=== verify $MODE: $FAIL failure(s) ===${RST}"
    exit 1
fi
