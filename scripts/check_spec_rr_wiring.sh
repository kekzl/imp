#!/bin/sh
# Guard: batch>1 speculation must gate on "is there a draft source", not speculative.ngram
# (the key the measured MTP recipe sets false, docs/MODELS.md). Source-reading script, not a
# GTest body: shipped test binaries run without the tree (imp:test has no /src).

set -eu

ROOT="${1:-.}"
FAIL=0

# block <file> <first-anchor> <end-marker>: the text from the first line
# containing the anchor up to (not including) the next line with the marker.
block() {
    awk -v a="$2" -v e="$3" '
        !on && index($0, a) { on = 1 }
        on && index($0, e) { exit }
        on { print }
    ' "$1"
}

check() {
    f="$ROOT/$1"; a="$2"; e="$3"; what="$4"
    if [ ! -f "$f" ]; then
        echo "check_spec_rr_wiring: $f not found" >&2
        exit 2
    fi
    blk=$(block "$f" "$a" "$e")
    if [ -z "$blk" ]; then
        echo "check_spec_rr_wiring: anchor '$a' not found in $1" >&2
        exit 2
    fi
    if ! printf '%s\n' "$blk" | grep -q 'spec_any_drafter_enabled_'; then
        echo "FAIL: $what must select by 'can anyone draft' (spec_any_drafter_enabled_)" >&2
        FAIL=1
    fi
    if printf '%s\n' "$blk" | grep -q 'spec_ngram_enabled_'; then
        echo "FAIL: spec_ngram_enabled_ is back in $what: with the documented MTP pair" \
             "(mtp_k=2, ngram=false) it selects no row and batch>1 speculation never fires" >&2
        FAIL=1
    fi
    if printf '%s\n' "$blk" | grep -q 'speculative\.ngram'; then
        echo "FAIL: $what must not read speculative.ngram" >&2
        FAIL=1
    fi
}

check src/runtime/engine_scheduler.cpp "SpecBatchRrState rr_state" "spec_rr_yield_interval_ =" \
      "the scheduler round-robin branch"
check src/runtime/engine_decode_pipeline.cpp "speculative.batch_rr" "const int next_parity" \
      "the #1003 pipeline spec yield"

if [ "$FAIL" -ne 0 ]; then
    exit 1
fi
echo "check_spec_rr_wiring: both batch_rr sites ask the drafter question"
