#!/usr/bin/env bash
# GPU acceptance for #2218: int64 device offsets. PASS/FAIL per criterion, exit 0 only if all pass.
# C1 make test-gpu exit 0, 0 FAILED. C2/C3 paired A/B vs origin/main per model: decode and
# prefill mean delta >= -thr (tests/perf_baseline.json paired_decode_regression_pct, 2).
# Usage: bash scripts/accept_2218.sh
# Env: IMP_ACCEPT_2218_MODELS (space-separated, default: the perf baseline model), AB_PAIRS.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

# Busy check first, then hold the card for the whole run (scripts/gpu_lock.sh).
if [ -z "${IMP_ACCEPT_2218_LOCKED:-}" ]; then
    HOST_CHECK="$HOME/.claude/skills/gpu-stats/gpu-busy-check.sh"
    if [ -x "$HOST_CHECK" ]; then
        "$HOST_CHECK" || { echo "FAIL setup: GPU busy ($HOST_CHECK)"; exit 1; }
    fi
    bash scripts/require_free_gpu.sh "accept_2218" || exit 1
    IMP_ACCEPT_2218_LOCKED=1 exec bash scripts/gpu_lock.sh run "accept_2218" -- bash "$ROOT/scripts/accept_2218.sh" "$@"
fi

for tool in docker jq make awk; do
    command -v "$tool" >/dev/null || { echo "FAIL setup: $tool not on PATH"; exit 1; }
done
BASELINE=tests/perf_baseline.json
THR="$(jq -r '.thresholds.paired_decode_regression_pct' "$BASELINE")"
MODELS="${IMP_ACCEPT_2218_MODELS:-$(jq -r '.model' "$BASELINE")}"
ADIR="$(mktemp -d "${TMPDIR:-/tmp}/accept_2218.XXXXXX")"
echo "logs: $ADIR"

declare -a RESULTS=()
verdict() {  # verdict <id> <PASS|FAIL> <detail>
    RESULTS+=("$2 $1: $3")
    echo "$2 $1: $3"
}

# C1: GPU test suite on this tree.
make test-gpu >"$ADIR/test_gpu.log" 2>&1
rc=$?
nfail="$(grep -c 'FAILED' "$ADIR/test_gpu.log")"
if [ "$rc" -eq 0 ] && [ "$nfail" -eq 0 ]; then
    verdict C1-test-gpu PASS "exit 0, 0 FAILED lines"
else
    verdict C1-test-gpu FAIL "exit $rc, $nfail FAILED lines ($ADIR/test_gpu.log)"
fi

# C2/C3: paired A/B per model; verify_ab.sh gates decode only, prefill is gated here.
for model in $MODELS; do
    tag="$(basename "$model")"
    mkdir -p "$ADIR/ab_$tag"
    AB_MODEL="$model" AB_LOG_DIR="$ADIR/ab_$tag" make verify-ab >"$ADIR/ab_$tag.out" 2>&1
    rc=$?
    line="$(cat "$ADIR/ab_$tag"/verify_ab_*.log 2>/dev/null | grep '^AB_RESULT' | tail -1)"
    if [ -z "$line" ]; then
        verdict "C2-decode[$tag]" FAIL "no AB_RESULT (exit $rc, $ADIR/ab_$tag.out)"
        verdict "C3-prefill[$tag]" FAIL "no AB_RESULT (exit $rc, $ADIR/ab_$tag.out)"
        continue
    fi
    tg="$(sed -n 's/.* tg_mean=\([^ ]*\).*/\1/p' <<<"$line")"
    pp="$(sed -n 's/.* pp_mean=\([^ ]*\).*/\1/p' <<<"$line")"
    for pair in "C2-decode:$tg" "C3-prefill:$pp"; do
        id="${pair%%:*}"; val="${pair#*:}"
        if awk -v m="$val" -v t="$THR" 'BEGIN{exit !(m != "" && m >= -t)}'; then
            verdict "${id}[$tag]" PASS "mean ${val}% >= -${THR}%"
        else
            verdict "${id}[$tag]" FAIL "mean ${val}% < -${THR}%"
        fi
    done
done

echo
echo "== summary =="
printf '%s\n' "${RESULTS[@]}"
for r in "${RESULTS[@]}"; do
    case "$r" in FAIL*) exit 1 ;; esac
done
exit 0
