#!/bin/bash
# verify_ab.sh - paired, alternating A/B of the perf-gate bench between two images.
#
# WHY THIS EXISTS
# scripts/verify.sh compares one arm against tests/perf_baseline.json, a pin
# measured weeks earlier. This host's same-tree movement between sessions is
# 4-6 % (release day 2026-09-04: 294.53 vs 277.73 tg128 on one tree), so the
# single-arm 8 % gate cannot resolve anything smaller: its one demonstrated
# catch is -36 % (M29), and -7.3 % shipped at +0.33 % (#1270). A paired A/B in
# one session cancels the host: arm A = origin/main built into imp:ab-<sha>
# (scripts/ab_base_image.sh), arm B = imp:test (this tree). PAIRS alternating
# pairs (A B, B A, A B), the bench line verify.sh uses, one process per arm
# run, the delta per pair and the mean over pairs (AUDIT_arch_2026 H-3).
#
# Usage: scripts/verify_ab.sh [IMG_A] [IMG_B]
#   AB_PAIRS=3                 pairs (each pair = one run of each arm)
#   AB_THRESHOLD_PCT=<n>       overrides thresholds.paired_decode_regression_pct
#   MODELS_DIR=$HOME/models    where the pin's model lives
#   IMP_VERIFY_BASELINE        pin to read model + threshold from
#   IMP_VERIFY_CHUNK_SIZE=0    prefill chunk, as in verify.sh
#   AB_MODEL=<file>            bench this model (under MODELS_DIR) instead of the pin's
#   AB_EXTRA="--set k=v ..."   extra imp-cli args, applied to BOTH arms (an ad-hoc
#                              A/B of a path the pin's default config does not take)
# Exit 1 when the mean paired decode delta is below -threshold; prefill warns.
set -uo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
IMG_A="${1:-${IMG_A:-imp:ab-base}}"
IMG_B="${2:-${IMG_B:-imp:test}}"
PAIRS="${AB_PAIRS:-3}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
BASELINE="${IMP_VERIFY_BASELINE:-tests/perf_baseline.json}"
CHUNK="${IMP_VERIFY_CHUNK_SIZE:-0}"
REPS=3

RED=$'\033[0;31m'; GRN=$'\033[0;32m'; YLW=$'\033[0;33m'; RST=$'\033[0m'

command -v jq >/dev/null 2>&1 || { echo "verify-ab: jq not installed" >&2; exit 2; }
[ -f "$BASELINE" ] || { echo "verify-ab: no $BASELINE" >&2; exit 2; }
MODEL="${AB_MODEL:-$(jq -r '.model' "$BASELINE")}"
EXTRA="${AB_EXTRA:-}"
THR="${AB_THRESHOLD_PCT:-$(jq -r '.thresholds.paired_decode_regression_pct // empty' "$BASELINE")}"
[ -n "$THR" ] || { echo "verify-ab: no thresholds.paired_decode_regression_pct in $BASELINE and no AB_THRESHOLD_PCT" >&2; exit 2; }
[ -e "$MODELS_DIR/$MODEL" ] || { echo "verify-ab: model $MODELS_DIR/$MODEL not present" >&2; exit 2; }
for img in "$IMG_A" "$IMG_B"; do
    docker image inspect "$img" >/dev/null 2>&1 || { echo "verify-ab: image $img missing (make build / scripts/ab_base_image.sh)" >&2; exit 2; }
done
if [ -x scripts/require_free_gpu.sh ]; then
    scripts/require_free_gpu.sh "verify-ab" || exit 2
fi

TS="$(date -u +%Y%m%d_%H%M%S)"
LOG="${AB_LOG_DIR:-${TMPDIR:-/tmp}}/verify_ab_${TS}.log"
ERR="$(mktemp)"
echo
echo "${YLW}== paired A/B: A=$IMG_A  B=$IMG_B  pairs=$PAIRS  model=$MODEL  threshold=-${THR}%${EXTRA:+  extra=$EXTRA} ==${RST}"
echo "  log: $LOG"

# One bench process of one arm. Prints "<tg> <pp>"; the line is verify.sh's
# (single-chunk prefill, speculation off, --json so nothing is regexed).
bench_arm() {  # $1 = image
    local out
    out="$(docker run --rm --gpus all -v "$MODELS_DIR":/models "$1" \
        imp-cli --model "/models/$MODEL" --bench --bench-pp 512 --bench-reps $REPS \
        --prefill-chunk-size "$CHUNK" --max-tokens 128 --temperature 0 --json \
        --set speculative.ngram=false $EXTRA 2>>"$ERR")" || return 1
    local tg pp
    tg="$(jq -er '.decode_tps' <<<"$out" 2>/dev/null)" || return 1
    pp="$(jq -er '.prefill_tps' <<<"$out" 2>/dev/null)" || return 1
    echo "$tg $pp"
}

run_arm() {  # $1 = arm letter, $2 = image, $3 = pair -> sets R_TG R_PP
    local r
    if ! r="$(bench_arm "$2")"; then
        echo "${RED}FAIL${RST} arm $1 ($2) pair $3: bench produced no numbers" | tee -a "$LOG"
        tail -15 "$ERR"
        exit 1
    fi
    R_TG="${r% *}"; R_PP="${r#* }"
    printf "  pair %d  arm %s  tg128 %8.2f  pp512 %9.2f  (%s)\n" "$3" "$1" "$R_TG" "$R_PP" "$2" | tee -a "$LOG"
}

D_TG=""; D_PP=""
for p in $(seq 1 "$PAIRS"); do
    if [ $((p % 2)) -eq 1 ]; then
        run_arm A "$IMG_A" "$p"; A_TG=$R_TG; A_PP=$R_PP
        run_arm B "$IMG_B" "$p"; B_TG=$R_TG; B_PP=$R_PP
    else
        run_arm B "$IMG_B" "$p"; B_TG=$R_TG; B_PP=$R_PP
        run_arm A "$IMG_A" "$p"; A_TG=$R_TG; A_PP=$R_PP
    fi
    d_tg="$(awk -v a="$A_TG" -v b="$B_TG" 'BEGIN{printf "%.2f", (b-a)/a*100}')"
    d_pp="$(awk -v a="$A_PP" -v b="$B_PP" 'BEGIN{printf "%.2f", (b-a)/a*100}')"
    D_TG="$D_TG $d_tg"; D_PP="$D_PP $d_pp"
    printf "  pair %d  delta B/A: decode %+s%%  prefill %+s%%\n" "$p" "$d_tg" "$d_pp" | tee -a "$LOG"
done
rm -f "$ERR"

stats() {  # mean min max n_negative over a space-separated list
    echo "$1" | tr ' ' '\n' | grep -v '^$' | awk '{s+=$1; if(NR==1||$1<mn)mn=$1; if(NR==1||$1>mx)mx=$1; if($1<0)neg++} END{printf "%.2f %.2f %.2f %d", s/NR, mn, mx, neg+0}'
}
read -r TG_MEAN TG_MIN TG_MAX TG_NEG <<<"$(stats "$D_TG")"
read -r PP_MEAN PP_MIN PP_MAX PP_NEG <<<"$(stats "$D_PP")"

echo
printf "  paired decode  delta: mean %+s%%  (pairs %s; min %+s%% max %+s%%; %d of %d negative)\n" \
       "$TG_MEAN" "$(echo "$D_TG" | sed 's/^ //')" "$TG_MIN" "$TG_MAX" "$TG_NEG" "$PAIRS" | tee -a "$LOG"
printf "  paired prefill delta: mean %+s%%  (pairs %s)\n" "$PP_MEAN" "$(echo "$D_PP" | sed 's/^ //')" | tee -a "$LOG"
echo "AB_RESULT a=$IMG_A b=$IMG_B pairs=$PAIRS tg_mean=$TG_MEAN tg_min=$TG_MIN tg_max=$TG_MAX tg_neg=$TG_NEG pp_mean=$PP_MEAN thr=$THR" >>"$LOG"

FAIL=0
if awk -v m="$TG_MEAN" -v t="$THR" 'BEGIN{exit !(m < -t)}'; then
    if [ "$TG_NEG" -eq "$PAIRS" ]; then
        echo "${RED}FAIL${RST} paired decode regression: mean ${TG_MEAN}% < -${THR}%, every pair negative"
    else
        echo "${RED}FAIL${RST} paired decode regression: mean ${TG_MEAN}% < -${THR}% (sign not unanimous: $TG_NEG of $PAIRS pairs negative - re-run before trusting either verdict)"
    fi
    FAIL=1
else
    echo "${GRN}PASS${RST} paired decode within ${THR}% (mean ${TG_MEAN}%)"
fi
if awk -v m="$PP_MEAN" -v t="$THR" 'BEGIN{exit !(m < -t)}'; then
    echo "${YLW}WARN${RST} paired prefill regression: mean ${PP_MEAN}% < -${THR}% (cuBLAS autotune variance is common; a unanimous sign over the pairs is the tell)"
else
    echo "${GRN}PASS${RST} paired prefill within ${THR}% (mean ${PP_MEAN}%)"
fi
exit $FAIL
