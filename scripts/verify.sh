#!/bin/bash
# Pre-commit/pre-push verification for imp. Four-step gate: build (incremental), tests (gtest
# filter or full suite), perf vs tests/perf_baseline.json (decode regression >3% fails), smoke
# prompts on real models (degeneration detector). Steps 3/4 plus peak-VRAM/CUDA-graph gates need
# real checkpoints under $MODELS and skip individually when absent; a run where no model-backed
# gate measured anything fails rather than reporting OK.
# Modes: verify.sh fast (unit tests + perf + 1 smoke, ~90s), verify.sh full (full ctest + perf +
# 2 smoke, ~5min).
# Env: IMP_VERIFY_BIN, IMP_VERIFY_TESTS, IMP_VERIFY_MODELS, IMP_VERIFY_BASELINE,
# IMP_VERIFY_CHUNK_SIZE (0=single-chunk), IMP_VERIFY_SKIP_BUILD=1, IMP_VERIFY_SKIP_PERF=1
# (stale baseline; refresh via gen_perf_baseline.sh), IMP_VERIFY_TRIALS=3 (processes the perf
# gate medians over), IMP_VERIFY_IN_DOCKER=1 (internal re-exec sentinel, do not set manually).
# Auto-Docker fallback: re-execs inside imp:test when cmake is not on PATH (Clean-Host), using
# the prebuilt /usr/local/bin binaries; requires make build first.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Auto re-exec into imp:test when cmake is unavailable on the host (Clean-Host: no toolchains
# installed); points IMP_VERIFY_BIN/TESTS at the runtime image's prebuilt /usr/local/bin binaries.
# IMP_VERIFY_IN_DOCKER guards against infinite re-exec if the runtime image also lacks cmake.
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
        -e IMP_VERIFY_MODELS="${IMP_VERIFY_MODELS:-models}" \
        -e IMP_VERIFY_SKIP_PERF="${IMP_VERIFY_SKIP_PERF:-0}" \
        -e IMP_VERIFY_SKIP_VRAM="${IMP_VERIFY_SKIP_VRAM:-0}" \
        -e IMP_VERIFY_SKIP_GRAPHS="${IMP_VERIFY_SKIP_GRAPHS:-0}" \
        -e IMP_VERIFY_BASELINE="${IMP_VERIFY_BASELINE:-tests/perf_baseline.json}" \
        -e IMP_VERIFY_CHUNK_SIZE="${IMP_VERIFY_CHUNK_SIZE:-0}" \
        -e IMP_VERIFY_TRIALS="${IMP_VERIFY_TRIALS:-3}" \
        --entrypoint bash imp:test scripts/verify.sh "$@"
fi

MODE="${1:-fast}"

# Wall clock at start and summary: every measurement below is a throughput number on a card
# shared with other sessions, and without a timestamp an overlapping neighbour job can't be
# ruled in or out after the fact.
# UTC with the Z: this script re-execs into the imp:test container, whose clock is UTC while
# the host runs local time; a bare local time from one side is worse than none.
VERIFY_T_START="$(date -u +%H:%M:%SZ)"
echo "verify: started $VERIFY_T_START ($MODE)"

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

# Model-backed gates that actually measured something. Every gate that loads a checkpoint
# degrades to SKIP when the file is absent, so a run can reach the summary having put nothing
# in front of the GPU; the summary refuses to report OK while this is still zero.
MODEL_GATES_RUN=0
model_gate_ran() { MODEL_GATES_RUN=$((MODEL_GATES_RUN+1)); }

# Host-drift guard (#526): this WSL2 box has day-level "depressed host" states where decode
# reads 8-15% low despite full methodology (host/driver state, not code). Samples GPU
# clocks/power during the decode bench; a regression coinciding with the depressed-host
# signature degrades FAIL->WARN instead of a false positive.
# Depressed: mem-clock median < 13801 MHz, OR power max < 400W, OR SM-clock median < 2000MHz
# (healthy decode is ~2850MHz sustained). First 2 samples dropped (SM ramps ~1s at bench start);
# a run with only 1-2 samples can't drop the ramp and is treated as no-data (fails open: plain
# FAIL). Fail-open if nvidia-smi is missing: sampler no-ops, no degradation logic kicks in.
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

# Docker-only host (no cmake, no build/ dir): run the canonical Docker build, then exit early;
# the test/perf/smoke gates below need host build artifacts with no clean path from here.
# Override with IMP_VERIFY_SKIP_BUILD=1 if cmake-on-host is preferred.
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

# Every GPU-touching gate below resolves its checkpoint under $MODELS and degrades to SKIP
# when absent; with the directory missing outright, everything skips and the summary used to
# print OK anyway. models/ is a gitignored symlink farm present only in the main checkout, so a
# fresh worktree produced a green pre-push gate that tested nothing.
case "$MODELS" in
    /*) MODELS_ABS="$MODELS";        MODELS_IS_RELATIVE=0 ;;
    *)  MODELS_ABS="$ROOT/$MODELS";  MODELS_IS_RELATIVE=1 ;;
esac
if [ ! -d "$MODELS_ABS" ]; then
    section "models"
    fail "model directory not found: $MODELS_ABS"
    if [ "${IMP_VERIFY_IN_DOCKER:-0}" = "1" ] && [ "$MODELS_IS_RELATIVE" = "1" ]; then
        echo "  ($ROOT is where this container has the repo mounted, so the"
        echo "   missing directory is '$MODELS/' in the checkout you ran this from)"
    fi
    echo "  perf, peak VRAM, graphs ON/OFF and the smoke prompts all read their"
    echo "  checkpoints from there, so this run would have measured nothing."
    echo "  Fix on the host, from the repo root, with either of:"
    echo "    ln -s \"\$HOME/models\" models"
    echo "    IMP_VERIFY_MODELS=\"\$HOME/models\" make verify-fast"
    echo
    echo "${RED}=== verify $MODE: $FAIL failure(s) ===${RST} (started $VERIFY_T_START, ended $(date -u +%H:%M:%SZ))"
    exit 1
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
        # VramBudgetReserve and ForwardPassTest are SKIP_IF_NO_CUDA, so CI structurally cannot run
        # them; this pre-push gate is the only place that can (three VramBudgetReserve tests sat red
        # for three PRs unnoticed, AUDIT B63). DecodeLogitsInvariantToBatchComposition (#1314) is the
        # only assert that a sequence's logits don't depend on its batch neighbours.
        # `*Attention*` matched a renamed-away suite for four months and gtest reported success (zero
        # attention tests ran, #1586); check_verify_filter.sh now fails on any pattern matching nothing.
        # `*Attention*` covers paged decode only, not FMHA prefill; default-path FA2 fixtures
        # (FmhaFA2Test etc, ~40s total) are added by name. Left out on purpose (run via make test-gpu):
        # FmhaSm120Test (legacy tier), FmhaFP8Test (opt-in), FmhaFA2Dense2CtaTest,
        # FmhaFA2Hd256Bkv32Test (opt-in bkv=32).
        FILTER="TensorTest.*:GgufLoaderTest.*:Tokenizer*:ChatTemplate*:KVCache*:GemmTest.*:FP8GemmTest.*:SamplingTest.*:SoftmaxTest.*:*Attention*:VramBudget*:ForwardPassTest.*:FmhaFA2Test.*:FmhaFA2Hd256Test.*:FmhaFA2PvF16Test.*:FmhaHd512Test.*:FmhaFA2HeavyFirstTest.*:FmhaFA2Fp8ScaledTest.*:FhmaMxFP4Test.*"
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

        # test-e2e unit/gpu lane-split guard (R5/#580): the unit lane is a gtest_filter, so a rename
        # could silently move a CPU test into the GPU lane. Asserts the filter resolves to the frozen
        # CPU set. Fail-open: skips cleanly if the binary or script isn't locatable.
        _E2E_BIN="$(dirname "$TESTS_BIN")/test-e2e"
        # DUPLICATE of _unit_e2e_filter in CMakeLists.txt, kept in sync: can't source CMake variables
        # here (container path has no configured build/ dir). Went stale for nine days after #1795
        # before anything diffed it (AUDIT_arch_2026 I-3); check_lane_filter_copy.sh now diffs the two
        # literals in the CPU lane.
        _LANE_FILTER="BatchBuilderTest.*:SchedulerTest.*:RequestTest.*:EndToEndTest.*:StoragePlanner.*:WeightRegistryPreservation.*:StubModelTest.LoadStubModel:StubModelTest.TokenizeStub"
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

# Age/provenance of the perf pin (#1624): AGENTS.md/BENCHMARKING.md both say a comparison is
# only meaningful within one session on one host, so a pin measured weeks earlier is comparing
# across a distance the gate can't see. Warning, not a failure: a stale pin still catches a
# 30% regression, and failing on a calendar date would train people to regenerate it just to
# silence the warning.
# 90 days, not 30 (AUDIT_arch_2026 H-9): 30 fired on every run for a month and carried nothing.
# The paired arm (make verify-ab) now measures cross-session drift directly; this single arm is
# the coarse net.
if [ -f "$BASELINE" ]; then
    _pin_ts=$(grep -oE '"timestamp"[[:space:]]*:[[:space:]]*"[^"]*"' "$BASELINE" |
              head -1 | sed 's/.*"\([^"]*\)"$/\1/')
    if [ -n "$_pin_ts" ]; then
        _pin_epoch=$(date -u -d "$_pin_ts" +%s 2>/dev/null || echo "")
        _now_epoch=$(date -u +%s)
        if [ -n "$_pin_epoch" ]; then
            _age_days=$(( (_now_epoch - _pin_epoch) / 86400 ))
            _pin_model=$(grep -oE '"model"[[:space:]]*:[[:space:]]*"[^"]*"' "$BASELINE" |
                         head -1 | sed 's/.*"\([^"]*\)"$/\1/')
            echo "  pin: $_pin_ts (${_age_days} days old), model $_pin_model"
            if [ "$_age_days" -gt 90 ]; then
                echo "  WARNING: the baseline is ${_age_days} days old. The measurement contract"
                echo "           (AGENTS.md) is single-session; host drift over that span is larger"
                echo "           than this gate can distinguish from a code change. The paired arm"
                echo "           (make verify-ab) is the instrument for that; re-pin with"
                echo "           scripts/gen_perf_baseline.sh on a day whose numbers are trusted."
            fi
        fi
    fi
fi

if [ "${IMP_VERIFY_SKIP_PERF:-0}" = "1" ]; then
    skip "perf gate (IMP_VERIFY_SKIP_PERF=1)"
elif [ ! -f "$BASELINE" ]; then
    skip "no $BASELINE — run scripts/gen_perf_baseline.sh to create one"
elif [ ! -x "$BIN" ]; then
    fail "$BIN not found"
elif ! command -v jq >/dev/null 2>&1; then
    skip "jq not installed (needed to parse $BASELINE)"
else
    # Detects baseline schema version (R4/#579): all three baselines carry schema_version
    # ("legacy-v1"|"multi-model-v1"). Falls back to the historical numeric .version (multi-model==1)
    # for an older/regenerated file that still lacks schema_version.
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
            model_gate_ran
            ERR=$(mktemp)
            gpu_sample_start
            OUT=$(mktemp)
            "$BIN" --model "$MODEL_PATH" --bench --bench-pp 512 --bench-reps $REPS \
                  --prefill-chunk-size "$BENCH_CHUNK" --max-tokens 256 --temperature 0 --json \
                  --set speculative.ngram=false >"$OUT" 2>"$ERR"
            gpu_sample_stop
            PP=$(jq -er '.prefill_tps // empty' "$OUT" 2>/dev/null || true)
            TG=$(jq -er '.decode_tps // empty' "$OUT" 2>/dev/null || true)
            rm -f "$OUT"
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
            model_gate_ran
            REPS=3
            ERR=$(mktemp)
            # --prefill-chunk-size 0 forces single-chunk prefill so the baseline stays apples-to-apples.
            # speculative.ngram=false: the bench prompt is self-repetitive (~99.9% draft accept), so
            # spec-ON would measure the batched verify GEMMs instead (restart-volatile, ~11% swing).
            # MEDIAN OVER INDEPENDENT PROCESSES, not one shot: --bench-reps averages WITHIN a process and
            # can't see between-process variance (cuBLASLt heuristic reselection, allocator layout, clock
            # state). Between-process spread measured up to ~4% against a 3% threshold, so a single-shot
            # gate reports its own noise. The median drops single-run outliers but does NOT flatten
            # hours-scale host drift; the spread is printed so a wide one reads as "host moved", not "diff
            # regressed".
            TRIALS="${IMP_VERIFY_TRIALS:-3}"
            TG_ALL=""; PP_ALL=""
            gpu_sample_start
            for _t in $(seq 1 "$TRIALS"); do
                OUT=$(mktemp)
                "$BIN" --model "$MODEL_PATH" --bench --bench-pp 512 --bench-reps $REPS \
                      --prefill-chunk-size "${CHUNK_SIZE}" --max-tokens 128 --temperature 0 --json \
                      --set speculative.ngram=false >"$OUT" 2>"$ERR"
                _tg=$(jq -er '.decode_tps // empty' "$OUT" 2>/dev/null || true)
                _pp=$(jq -er '.prefill_tps // empty' "$OUT" 2>/dev/null || true)
                rm -f "$OUT"
                [ -n "$_tg" ] && TG_ALL="$TG_ALL$_tg\n"
                [ -n "$_pp" ] && PP_ALL="$PP_ALL$_pp\n"
            done
            gpu_sample_stop
            median() { printf "$1" | grep -v '^$' | sort -n | awk '{a[NR]=$1} END{if(NR==0)exit 1; print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }
            spread() { printf "$1" | grep -v '^$' | sort -n | awk '{a[NR]=$1} END{if(NR<2)exit 0; printf "%.2f", (a[NR]-a[1])/a[1]*100}'; }
            # Numbers come from --bench --json (#1583), not regexed out of the stderr table (whose paren
            # spacing varies with magnitude and was undocumented, load-bearing formatting). An empty
            # capture there used to produce a median over fewer samples than the header claimed.
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

                # Long-context prefill (single-chunk FA2), pp4096..pp65536: crosses the cuBLAS->FMHA threshold
                # to exercise the register-resident FA2 kernel (attention.fmha_fa2=on default). Guards the FA2
                # prefill win and the #1022 32K-64K TTFT band. Warn-only: large-context prefill is stable
                # (<0.5% cross-restart at 32K) unlike pp512's cuBLAS-algo jitter. Lengths past model ctx or
                # the
                # 70K bench ceiling have no baseline key and are skipped.
                for PPLEN in 4096 8192 16384 32768 65536; do
                    BL_PP=$(jq -r ".metrics.prefill_tps.pp${PPLEN} // empty" "$BASELINE")
                    [ -z "$BL_PP" ] && continue
                    ERR2=$(mktemp)
                    OUT2=$(mktemp)
                    "$BIN" --model "$MODEL_PATH" --bench --bench-pp "$PPLEN" --bench-reps $REPS \
                          --prefill-chunk-size 0 --max-tokens 1 --temperature 0 --max-seq-len 70000 \
                          --json >"$OUT2" 2>"$ERR2"
                    PPTPS=$(jq -er '.prefill_tps // empty' "$OUT2" 2>/dev/null || true)
                    rm -f "$ERR2" "$OUT2"
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

# Peak-VRAM gate (memory acceptance criterion 8): the baseline carried a memory_mb block and a
# vram_increase_pct threshold that nothing ever read; peak VRAM creeps silently until a model
# stops fitting (#1103, 7x decode cost).
# Gated on own_peak, NOT device peak_used: device-used also carries the CUDA primary context
# (~1.7 GiB) and any neighbour process. Runs its own short bench rather than piggy-backing on
# the perf bench: --mem-report's 2ms sampler thread would perturb the number the perf gate reads.
section "peak VRAM vs baseline"
if [ "${IMP_VERIFY_SKIP_VRAM:-0}" = "1" ]; then
    skip "peak-VRAM gate (IMP_VERIFY_SKIP_VRAM=1)"
elif [ ! -f "$BASELINE" ] || ! command -v jq >/dev/null 2>&1; then
    skip "peak-VRAM gate (no baseline or no jq)"
else
    # MODEL_PATH is set inside the perf gate above, which IMP_VERIFY_SKIP_PERF=1 skips entirely;
    # referencing it unguarded killed the script under `set -u`. Resolved here instead so this gate
    # stands on its own regardless of whether the perf gate ran.
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
        model_gate_ran
        ERR_V=$(mktemp)
        # BOTH streams: imp's INFO logs (VRAM audit table) go to STDOUT, only bench result lines go to
        # stderr. Capturing stderr alone, correct for the perf gate's own numbers, yields nothing here.
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

# Graphs-ON vs graphs-OFF decode regression gate: catches a PR that silently breaks CUDA Graph
# capture. Without graphs, decode is launch-overhead-bound (875-1170 launches/step on dense Q8);
# graphs-ON wins +95-376%. A ratio dropping below kMinSpeedupX = 1.5x signals a path that fell
# out of capture (host sync inside the region, malloc on the hot path). Skipped by
# IMP_VERIFY_SKIP_GRAPHS=1.
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
        # Threshold lowered 1.5 -> 1.3 after F1's warmup-pre-pass made graphs-OFF faster on dense Q8
        # (compresses the ratio). Cross-model A/B: Qwen3-4B 1.90x, Qwen3.5-GDN 2.23x, Llama-3.2-3B 2.38x,
        # Qwen3-8B 1.20x (bigger model = larger kernel time = less launch-overhead share = lower ratio).
        # 1.3 catches catastrophic graph failures (~1.0x = full fallback) without rejecting healthy
        # big-model decodes.
        model_gate_ran
        MIN_SPEEDUP_X="${IMP_VERIFY_MIN_GRAPH_SPEEDUP:-1.3}"
        ERR_NG=$(mktemp); ERR_G=$(mktemp)
        # Samples clocks/power across BOTH runs: a depressed host (#526) makes kernels dominate and
        # compresses the ON/OFF ratio toward 1.0x on ANY build, so the ratio stops being attributable
        # to code. Power floor (400W) is calibrated on 8B-class decode; the 4B gate model can draw less
        # even when healthy, so this gate leans WARN (a false WARN costs a re-run, a false FAIL blocks
        # a push on host noise). Hard FAIL still fires whenever the signature reads healthy.
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

# Greedy decode on a known-deterministic prompt. Quality gate: output must contain the
# expected substring AND the last 32 tokens must have >=8 distinct tokens (catches "own own own"
# stuck-token failures). $5 (optional): minimum generated tokens, default 10 with an exception
# for single-word-factual prompts (this gate's own prompt answers in 11 tokens on the 4B model).
smoke_prompt() {
    local label="$1" model="$2" prompt="$3" expect="$4" min_toks="${5:-10}"
    if [ ! -f "$MODELS/$model" ]; then
        skip "$label ($model not present)"
        return
    fi
    model_gate_ran
    local ERR; ERR=$(mktemp)
    # --token-trace is load-bearing, not cosmetic: without it imp-cli caps markers at the first
    # ten decode steps (mode_oneshot.cpp), so this gate would read eleven tokens of a 64-token run
    # and call them "the last 32", missing any repetition loop starting at step 11. Also failed the
    # distinct-token count on a correct short answer.
    OUT=$("$BIN" --model "$MODELS/$model" --prompt "$prompt" --token-trace \
          --max-tokens 64 --temperature 0 --chat-template none 2>"$ERR")
    # Token markers '[tok=NNNN ' word']' land on stderr.
    ALL_TOKS=$(grep -oP '\[tok=\K[0-9]+' "$ERR")
    WORDS=$(grep -oP "\[tok=[0-9]+ '\K[^']*" "$ERR" | tr '\n' ' ')
    rm -f "$ERR"

    # NaN/Inf checked first: a different failure from degeneration, and the verdict script only
    # sees ids, not pieces.
    # Herestrings, not `echo ... | grep -q`: grep -q exits at the first match, echo dies of EPIPE,
    # and set -o pipefail turns that into a non-zero pipeline even though grep MATCHED - which
    # would swallow a NaN report here, or fail a correct smoke run where the test is negated.
    if grep -qiE ' (nan|inf|-inf|-nan) ' <<< " $WORDS "; then
        fail "$label — NaN/Inf token in output"
        echo "  words: $WORDS"
        return
    fi

    # Thresholds live in scripts/degen_verdict.sh so the CPU lane can exercise them without a GPU
    # (guard_degen_thresholds). This gate used to judge the eleven markers imp-cli printed and call
    # them "the last 32 tokens".
    local VERDICT
    VERDICT=$(printf '%s\n' "$ALL_TOKS" | bash "$ROOT/scripts/degen_verdict.sh" "$min_toks" 2>&1)
    if [ "${VERDICT#OK}" = "$VERDICT" ]; then
        fail "$label — ${VERDICT#FAIL }"
        echo "  tokens: $(printf '%s ' $ALL_TOKS)"
        return
    fi

    # Generated text appears interleaved with logs on stdout — substring match works
    if ! grep -q "$expect" <<< "$OUT$WORDS"; then
        fail "$label — expected '$expect' in output"
        echo "  words: $WORDS"
        return
    fi
    pass "$label (${VERDICT#OK }, contains '$expect')"
}

smoke_prompt "Qwen3-4B Q8_0 (dense)" \
    "Qwen3-4B-Instruct-2507-Q8_0.gguf" \
    "The capital of France is" \
    "Paris" \
    5

if [ "$MODE" = "full" ]; then
    smoke_prompt "Qwen3.5-4B MXFP4 (GDN)" \
        "Qwen3.5-4B-mxfp4.gguf" \
        "The capital of France is" \
        "Paris" \
        5
fi

# A run in which every model-backed gate skipped measured nothing on the GPU, so it is not a
# pass. A single absent checkpoint still only skips: the gate stays green as long as at least
# one model-backed gate ran.
if [ "$MODEL_GATES_RUN" -eq 0 ]; then
    fail "no model-backed gate ran: perf, peak VRAM, graphs and smoke all skipped"
    echo "  Model directory in use: $MODELS_ABS"
    echo "  None of the checkpoints those gates need are in it, so nothing was measured."
    echo "  Point the gate at the model store: IMP_VERIFY_MODELS=\"\$HOME/models\""
fi

echo
if [ "$FAIL" -eq 0 ]; then
    echo "${GRN}=== verify $MODE: OK ===${RST} (started $VERIFY_T_START, ended $(date -u +%H:%M:%SZ))"
    exit 0
else
    echo "${RED}=== verify $MODE: $FAIL failure(s) ===${RST} (started $VERIFY_T_START, ended $(date -u +%H:%M:%SZ))"
    exit 1
fi
