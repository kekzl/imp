#!/usr/bin/env bash
# Degeneration verdict as a function of a token stream, extracted from verify.sh so
# thresholds can be exercised without a GPU.
# Usage: degen_verdict.sh <min_toks> < token-ids-one-per-line. Prints OK <stats> (exit 0) or
# FAIL <reason> (exit 1); check_degen_thresholds.sh pins the verdicts against known streams.

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

# Distinct-token threshold scales to the window that actually exists: a flat threshold sized
# for a full window fails a short-but-correct run (legitimate answers repeat separators).
DISTINCT_MIN=$(( WINDOW / 4 ))
[ "$DISTINCT_MIN" -gt 8 ] && DISTINCT_MIN=8
[ "$DISTINCT_MIN" -lt 2 ] && DISTINCT_MIN=2

# 3-gram-repeat share (percent of 3-gram positions whose 3-gram recurs >=2x) separates real
# repetition-loop degeneration from a phrase merely recurring: worst healthy observation is
# 45%, loops sit at 100%. Top-3-gram share doesn't work: a 10-token phrase looped 6x spreads
# across 10 different 3-grams, covering only 30% of positions while the output is a pure loop.
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
