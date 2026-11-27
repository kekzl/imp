#!/usr/bin/env bash
# inkling_perf.sh <imp-cli> <model-dir> [baseline] | --selftest: the Inkling-2-Mini row of tests/perf_baseline.json (B-111).
# Same bench as the verify.sh perf gate (pp512 + tg128 in one process, chunk 0, speculative.ngram=false) plus
# the row's `set` flags, IMP_INKLING_PERF_TRIALS (3) independent processes, median. Thresholds: the row's
# own `.thresholds` (5 %, run medians spread 2.7 %), else the file's top-level ones (8 %). Decode past it
# fails, prefill warns. Exit 0 = within; `make verify-inkling` runs it.
set -uo pipefail

selftest() {  # a stub imp-cli against a temp baseline: the row threshold wins, the top-level one is the fallback
    local d pass=0 bad=0
    d="$(mktemp -d)" || return 1
    printf '#!/usr/bin/env bash\necho "{\\"decode_tps\\": $STUB_TG, \\"prefill_tps\\": $STUB_PP}"\n' > "$d/cli"
    chmod +x "$d/cli"
    printf '%s\n' '{"thresholds": {"decode_regression_pct": 8, "prefill_regression_pct": 8},' \
        ' "extra_models": {"Inkling-2-Mini": {"set": [], "decode_tps": {"tg128": 100}, "prefill_tps": {"pp512": 100},' \
        '  "thresholds": {"decode_regression_pct": 5, "prefill_regression_pct": 5}}}}' > "$d/row.json"
    jq 'del(.extra_models[].thresholds)' "$d/row.json" > "$d/top.json" || { rm -rf "$d"; return 1; }
    expect() {  # expect <rc> <grep or -> <label> <baseline> <tg> <pp>
        local want="$1" pat="$2" label="$3" out rc
        out="$(STUB_TG="$5" STUB_PP="$6" IMP_INKLING_PERF_TRIALS=1 bash "$0" "$d/cli" "$d" "$d/$4" 2>&1)"
        rc=$?
        if [ "$rc" = "$want" ] && { [ "$pat" = "-" ] || grep -qF -- "$pat" <<< "$out"; }; then
            pass=$((pass + 1))
        else
            bad=$((bad + 1)); echo "selftest FAIL: $label: want rc $want + '$pat', got rc $rc: $out"
        fi
    }
    expect 1 "FAIL decode regression > 5 %" "row 5 %: decode -6 % fails"        row.json 94 100
    expect 0 "ok   decode within 5 %"        "row 5 %: decode -4 % passes"       row.json 96 100
    expect 1 "-9.09 %"                      "row 5 %: stored tg128 x 1.10 fails" row.json 90.909 100
    expect 0 "WARN prefill regression > 5 %" "row 5 %: prefill -6 % warns"       row.json 100 94
    expect 0 "ok   decode within 8 %"        "no row thresholds: top-level 8 %"  top.json 94 100
    rm -rf "$d"
    echo "selftest: $pass/$((pass + bad))"
    [ "$bad" -eq 0 ]
}
[ "${1:-}" = "--selftest" ] && { selftest; exit; }

CLI="$1" MODEL="$2" BASELINE="${3:-tests/perf_baseline.json}"
KEY="Inkling-2-Mini"
TRIALS="${IMP_INKLING_PERF_TRIALS:-3}"
row() { jq -er --arg k "$KEY" ".extra_models[\$k]$1" "$BASELINE"; }
thr() { jq -er --arg k "$KEY" "(.extra_models[\$k].thresholds.$1 // .thresholds.$1)" "$BASELINE"; }
BL_TG=$(row '.decode_tps.tg128') || { echo "FAIL no .extra_models[\"$KEY\"].decode_tps.tg128 in $BASELINE"; exit 1; }
BL_PP=$(row '.prefill_tps.pp512') || { echo "FAIL no .extra_models[\"$KEY\"].prefill_tps.pp512 in $BASELINE"; exit 1; }
DEC_THR=$(thr decode_regression_pct) || exit 1
PRE_THR=$(thr prefill_regression_pct) || exit 1
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
echo "  decode  tg128 = $TG tok/s (baseline $BL_TG, delta $DEC %, threshold $DEC_THR %)"
echo "  prefill pp512 = $PP tok/s (baseline $BL_PP, delta $PRE %, threshold $PRE_THR %)"
awk -v d="$PRE" -v t="$PRE_THR" 'BEGIN {exit !(-d > t)}' && echo "WARN prefill regression > $PRE_THR %"
if awk -v d="$DEC" -v t="$DEC_THR" 'BEGIN {exit !(-d > t)}'; then
    echo "FAIL decode regression > $DEC_THR % (if intended: re-pin .extra_models[\"$KEY\"] from these medians)"
    exit 1
fi
echo "ok   decode within $DEC_THR %"
