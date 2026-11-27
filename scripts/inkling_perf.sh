#!/usr/bin/env bash
# inkling_perf.sh <imp-cli> <model-dir> [baseline]: the Inkling-2-Mini row of tests/perf_baseline.json (B-111).
# Same bench as the verify.sh perf gate (pp512 + tg128 in one process, chunk 0, speculative.ngram=false) plus
# the row's `set` flags, IMP_INKLING_PERF_TRIALS (3) independent processes, median. Decode worse than
# thresholds.decode_regression_pct fails, prefill warns. Exit 0 = within; `make verify-inkling` runs it.
set -uo pipefail
CLI="$1" MODEL="$2" BASELINE="${3:-tests/perf_baseline.json}"
KEY="Inkling-2-Mini"
TRIALS="${IMP_INKLING_PERF_TRIALS:-3}"
row() { jq -er --arg k "$KEY" ".extra_models[\$k]$1" "$BASELINE"; }
BL_TG=$(row '.decode_tps.tg128') || { echo "FAIL no .extra_models[\"$KEY\"].decode_tps.tg128 in $BASELINE"; exit 1; }
BL_PP=$(row '.prefill_tps.pp512') || { echo "FAIL no .extra_models[\"$KEY\"].prefill_tps.pp512 in $BASELINE"; exit 1; }
DEC_THR=$(jq -er '.thresholds.decode_regression_pct' "$BASELINE") || exit 1
PRE_THR=$(jq -er '.thresholds.prefill_regression_pct' "$BASELINE") || exit 1
SETS=()
while IFS= read -r s; do SETS+=(--set "$s"); done < <(row '.set[]')
TG_ALL="" PP_ALL=""
for t in $(seq 1 "$TRIALS"); do
    out=$("$CLI" --model "$MODEL" --bench --bench-pp 512 --bench-reps 3 --prefill-chunk-size 0 --max-tokens 128 \
        --temperature 0 --json --set speculative.ngram=false "${SETS[@]}" 2>/dev/null)
    tg=$(jq -er '.decode_tps' <<< "$out" 2>/dev/null) || { echo "FAIL trial $t: no decode_tps in the bench output"; exit 1; }
    pp=$(jq -er '.prefill_tps' <<< "$out" 2>/dev/null) || { echo "FAIL trial $t: no prefill_tps in the bench output"; exit 1; }
    echo "  trial $t: tg128 $tg, pp512 $pp tok/s"
    TG_ALL+="$tg"$'\n' PP_ALL+="$pp"$'\n'
done
median() { grep -v '^$' | sort -g | awk '{a[NR]=$1} END {print (NR % 2) ? a[(NR + 1) / 2] : (a[NR / 2] + a[NR / 2 + 1]) / 2}'; }
spread() { grep -v '^$' | sort -g | awk '{a[NR]=$1} END {printf "%.2f", (a[NR] - a[1]) / a[1] * 100}'; }
TG=$(median <<< "$TG_ALL") PP=$(median <<< "$PP_ALL")
DEC=$(awk -v c="$TG" -v b="$BL_TG" 'BEGIN {printf "%.2f", (c - b) / b * 100}')
PRE=$(awk -v c="$PP" -v b="$BL_PP" 'BEGIN {printf "%.2f", (c - b) / b * 100}')
echo "  $KEY ${SETS[*]}: median of $TRIALS, tg128 spread $(spread <<< "$TG_ALL") %"
echo "  decode  tg128 = $TG tok/s (baseline $BL_TG, delta $DEC %)"
echo "  prefill pp512 = $PP tok/s (baseline $BL_PP, delta $PRE %)"
awk -v d="$PRE" -v t="$PRE_THR" 'BEGIN {exit !(-d > t)}' && echo "WARN prefill regression > $PRE_THR %"
if awk -v d="$DEC" -v t="$DEC_THR" 'BEGIN {exit !(-d > t)}'; then
    echo "FAIL decode regression > $DEC_THR % (if intended: re-pin .extra_models[\"$KEY\"] from these medians)"
    exit 1
fi
echo "ok   decode within $DEC_THR %"
