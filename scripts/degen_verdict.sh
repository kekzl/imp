#!/usr/bin/env bash
# The degeneration verdict, as a function of a token stream.
#
# Extracted from scripts/verify.sh so the thresholds can be exercised without a
# GPU. They had never been exercised at all: verify.sh read imp-cli's `[tok=...]`
# markers, imp-cli capped those at the first ten decode steps, and the gate
# treated the eleven it got as "the last 32 tokens". Every threshold below was
# therefore being applied to the opening of a generation, and a repetition loop
# starting at step 11 was invisible to the detector built to catch it.
#
# Usage:  degen_verdict.sh <min_toks> < token-ids-one-per-line
# Prints  OK <stats>            and exits 0
#     or  FAIL <reason>         and exits 1
#
# scripts/check_degen_thresholds.sh pins the verdicts against known streams.

set -uo pipefail

MIN_TOKS="${1:?usage: degen_verdict.sh <min_toks> < token-ids}"

ALL_TOKS=$(cat)
N_TOKS=$(printf '%s\n' "$ALL_TOKS" | grep -c .)

if [ "$N_TOKS" -eq 0 ]; then
    echo "FAIL no tokens were generated"
    exit 1
fi

# The trailing window the distinct-token check looks at.
TOKS=$(printf '%s\n' "$ALL_TOKS" | tail -32)
WINDOW=$(printf '%s\n' "$TOKS" | grep -c .)
DISTINCT=$(printf '%s\n' "$TOKS" | sort -u | grep -c .)

# Longest run of one token, and the most frequent 3-gram.
RUNMAX=$(printf '%s\n' "$ALL_TOKS" | uniq -c | awk '{if($1>m)m=$1}END{print m+0}')
GRAMMAX=$(printf '%s\n' "$ALL_TOKS" | awk '{a[NR]=$0} END{
    for(i=1;i+2<=NR;i++){k=a[i]" "a[i+1]" "a[i+2]; c[k]++; if(c[k]>m)m=c[k]}
    print m+0}')

# 1. Distinct tokens, scaled to the window that actually exists. The flat "8 in
#    the last 32" was written for a full window; applied to a short run it fails
#    correct output, because a legitimate short answer repeats its separators.
#    A run that stops early is the min_toks check's job, not this one's.
DISTINCT_MIN=$(( WINDOW / 4 ))
[ "$DISTINCT_MIN" -gt 8 ] && DISTINCT_MIN=8
[ "$DISTINCT_MIN" -lt 2 ] && DISTINCT_MIN=2

# 2. Repetition is degeneration when the output is MADE OF repeats, not when a
#    phrase merely recurs. The share of 3-gram positions whose 3-gram occurs at
#    least twice separates the two with a wide margin, measured over the streams
#    in check_degen_thresholds.sh:
#
#      healthy dense prose  37 %      "a b c" loop            100 %
#      healthy GDN list     31 %      stuck single token      100 %
#      short correct answer  0 %      two-token alternation   100 %
#                                     10-token phrase on loop 100 %
#
#    The top-3-gram share, which is the obvious metric, does NOT work: a
#    10-token phrase repeated six times spreads over ten different 3-grams, so
#    its top one covers only 30 % while the output is entirely a loop. The
#    run-length check misses every one of these four (runmax is 1 for three of
#    them), so this is the check carrying them.
REPEAT_PCT=$(printf '%s\n' "$ALL_TOKS" | awk '{a[NR]=$0} END{
    n=0; for(i=1;i+2<=NR;i++){k=a[i]" "a[i+1]" "a[i+2]; c[k]++; n++}
    if(n==0){print 0; exit}
    r=0; for(k in c) if(c[k]>=2) r+=c[k]
    printf "%d", r*100/n}')

STATS="distinct=$DISTINCT/$WINDOW need $DISTINCT_MIN, tokens=$N_TOKS, max-run=$RUNMAX, 3gram=$GRAMMAX repeat=$REPEAT_PCT%"

# Length first, so a run that stopped early is reported as that and not as a
# statistic derived from three tokens.
if [ "$N_TOKS" -lt "$MIN_TOKS" ]; then
    echo "FAIL stopped after $N_TOKS tokens (limit: $MIN_TOKS)"
    exit 1
fi
if [ "$DISTINCT" -lt "$DISTINCT_MIN" ]; then
    echo "FAIL degenerate (only $DISTINCT distinct tokens in a $WINDOW-token window, need $DISTINCT_MIN)"
    exit 1
fi
if [ "$RUNMAX" -gt 4 ]; then
    echo "FAIL a token repeats $RUNMAX times in a row (limit: 4)"
    exit 1
fi
# Needs a sample to be a statistic; below 12 tokens the length and distinct
# checks above carry the range. Threshold sits in the gap between the 37 % of
# the worst healthy stream and the 100 % of every looped one.
if [ "$N_TOKS" -ge 12 ] && [ "$REPEAT_PCT" -gt 70 ]; then
    echo "FAIL $REPEAT_PCT% of the output is repeated 3-grams (limit: 70%), top 3-gram x$GRAMMAX"
    exit 1
fi

echo "OK $STATS"
