#!/usr/bin/env bash
# Pin the degeneration verdicts against known token streams.
#
# The thresholds in degen_verdict.sh had never been exercised on a real
# generation: verify.sh read imp-cli's `[tok=...]` markers and imp-cli capped
# them at the first ten decode steps, so the gate judged eleven tokens of every
# run and called them "the last 32". The first run that could see a whole
# generation failed CORRECT output on two of the three thresholds. They are
# calibrated here, against streams whose verdict is not in question, so a future
# edit cannot quietly re-break them.
#
# The healthy streams are real: the 64 tokens Qwen3-4B-Instruct-2507-Q8_0 and
# Qwen3.5-4B-mxfp4 produced for "The capital of France is" on 2026-09-07.
#
# Usage: check_degen_thresholds.sh <repo-root>
# Exit 0 = every case matches; 1 = a verdict moved.

set -uo pipefail
ROOT="${1:?usage: check_degen_thresholds.sh <repo-root>}"
VERDICT="$ROOT/scripts/degen_verdict.sh"

if [ ! -x "$VERDICT" ] && [ ! -r "$VERDICT" ]; then
    echo "check_degen_thresholds: $VERDICT not found" >&2
    exit 1
fi

FAILURES=0

# want: OK | FAIL
check() {
    local name="$1" want="$2" min_toks="$3" toks="$4"
    local out got
    out=$(printf '%s\n' $toks | bash "$VERDICT" "$min_toks" 2>&1)
    case "$out" in
        OK*)   got=OK ;;
        FAIL*) got=FAIL ;;
        *)     got="?($out)" ;;
    esac
    if [ "$got" = "$want" ]; then
        printf '  ok    %-22s %s\n' "$name" "$out"
    else
        printf '  FAIL  %-22s want %s, got: %s\n' "$name" "$want" "$out"
        FAILURES=$((FAILURES + 1))
    fi
}

# Some cases must fail for a specific reason, not merely fail.
check_reason() {
    local name="$1" want="$2" min_toks="$3" toks="$4" out
    out=$(printf '%s\n' $toks | bash "$VERDICT" "$min_toks" 2>&1)
    case "$out" in
        *"$want"*) printf '  ok    %-22s reason: %s\n' "$name" "$want" ;;
        *) printf '  FAIL  %-22s want reason %s, got: %s\n' "$name" "$want" "$out"
           FAILURES=$((FAILURES + 1)) ;;
    esac
}

rep() { python3 -c "import sys; print(' '.join(sys.argv[1].split()*int(sys.argv[2])))" "$1" "$2"; }

echo "degeneration verdicts:"

# --- healthy, must pass -----------------------------------------------------
# Qwen3-4B dense: " Paris, and the capital of Germany is Berlin. The capital of
# Spain is Madrid, ..." - "the capital of" recurs 4 times because the CONTENT is
# a list of capitals. The old flat "3-gram more than 3 times" limit failed this.
check "dense-capitals" OK 5 \
  "12095 11 323 279 6722 315 9856 374 19846 13 576 6722 315 17689 374 24081 11 323 279 6722 315 15344 374 21718 13 576 6722 315 279 3639 15072 374 7148 11 323 279 6722 315 279 3639 4180 374 6515 11 422 727 13 576 6722 315 6323 374 26194 11 323 279 6722 315 5616 374 26549 13 576 6722"

# Qwen3.5-4B MXFP4 (GDN), same prompt, an A/B/C/D list. This is the stream that
# had `make verify` red: eleven of these tokens carry 7 distinct values, under
# the old flat threshold of 8.
check "gdn-list" OK 5 \
  "11751 13 32 13 2503 33 13 3557 271 785 6722 315 9856 374 25 4230 13 19846 425 13 21718 356 13 24081 422 13 7148 271 785 6722 315 17689 374 25 4230 13 21718 425 13 24081 356 13 19846 422 13 7148 271 785 6722"

# A short but correct answer: the window is 11, so the distinct threshold has to
# scale or this reads as degenerate.
check "short-correct" OK 5 "11751 13 198 32 13 2912 198 33 13 3439 198"

# --- degenerate, must fail --------------------------------------------------
# The classic loop. runmax is 1, so the run-length check cannot see it.
check "abc-loop" FAIL 5 "$(rep '101 102 103' 21)"
# A stuck single token.
check "stuck-token" FAIL 5 "$(rep '999' 64)"
# Two-token alternation: runmax 1 again.
check "alternation" FAIL 5 "$(rep '77 88' 32)"
# A long phrase on a loop. This is the case ONLY the coverage rule catches: 10
# distinct tokens is well above the distinct threshold and runmax is 1, so both
# other checks pass while the output is a loop.
check "phrase-loop" FAIL 5 "$(rep '11 12 13 14 15 16 17 18 19 20' 6)"
check_reason "phrase-loop" "repeated 3-grams" 5 "$(rep '11 12 13 14 15 16 17 18 19 20' 6)"
# Low variety WITHOUT literal repetition: a de Bruijn sequence over 4 token ids
# contains every 3-gram exactly once, so the repeated-3-gram share is 0 and the
# run length is 1. Only the distinct-token check sees it. A model emitting four
# token types in a shuffled order is broken even though nothing repeats.
check "low-variety" FAIL 5 "101 101 101 102 101 101 103 101 101 104 101 102 102 101 102 103 101 102 104 101 103 102 101 103 103 101 103 104 101 104 102 101 104 103 101 104 104 102 102 102 103 102 102 104 102 103 103 102 103 104 102 104 103 102 104 104 103 103 103 104 103 104 104 104"
check_reason "low-variety" "distinct tokens" 5 "101 101 101 102 101 101 103 101 101 104 101 102 102 101 102 103 101 102 104 101 103 102 101 103 103 101 103 104 101 104 102 101 104 103 101 104 104 102 102 102 103 102 102 104 102 103 103 102 103 104 102 104 103 102 104 104 103 103 103 104 103 104 104 104"
# ` own own own own own` - the shape the gate is named for.
check "own-run" FAIL 5 "1 2 3 4 5 6 7 7 7 7 7 7 7 7 8 9 10 11 12 13 14 15 16"
# Early abort: three tokens against a min of 5. Must fail on LENGTH, not on a
# coverage figure computed from three tokens, so the reason is checked too.
check "early-abort" FAIL 5 "100 200 300"
check_reason "early-abort" "stopped after 3 tokens" 5 "100 200 300"
# A short but varied output above the minimum must not trip the coverage rule
# just because the sample is small.
check "short-varied" OK 5 "11 22 33 44 55 66 77 88 99 10 20 21"
# Nothing at all.
check "empty" FAIL 5 ""

if [ "$FAILURES" -ne 0 ]; then
    echo "check_degen_thresholds: $FAILURES verdict(s) moved" >&2
    exit 1
fi
echo "check_degen_thresholds: all verdicts hold"
