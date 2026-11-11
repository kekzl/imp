#!/usr/bin/env bash
# Engine-level check of speculative.mtp_batch_verify_factor_row (bound verify). With an f32 GDN state the fold
# is the scan's own update, so greedy texts must come out byte-identical (runtime.deterministic turns drafting
# off, so the arms run without it). Exit 0 = every text md5 equal and drafts were verified; 1 = a text differs.
#   pair:  2 drafts, factor row vs a second spare per row, CONC chats STAGGER_MS apart (the texts follow the
#          arrival order of the streams, 2 outcomes at STAGGER_MS=0). Quiet host only: beside `make preflight`
#          1 run differed.
#   depth: 4 drafts verified, 1 chat, speculative.mtp_k = 4 / 3 / 2 / 1 (spare adopted, 3 / 2 / 1 rows
#          folded). 1 chat only: the rows per step change the text, so a second stream's timing would too.
#          THINK_BUDGET=1: a forced closer lands up to mtp_k tokens past the budget (engine_spec_ngram.cpp).
# Usage: OUT=/mnt/data/scratch/dir [IMAGE=imp:test | dev] [CTX=20000] [CONC=2] [CORPUS=dir] [STAGGER_MS=300]
#        [CHECKS="pair depth"] tools/analysis/mtp_factor_row_equiv.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=${OUT:?OUT=dir for logs}
I=${IMAGE:-imp:test}
# check <name> <reference arm> <conc> <stagger ms> <arms>: every arm's texts equal the reference arm's.
check() {
    local name=$1 ref=$2 conc=$3 stagger=$4 arms=$5
    ARMS="$arms" PAIRS=1 CONC="$conc" CTX="${CTX:-20000}" STAGGER_MS="$stagger" OUT="$OUT/$name" \
        bash "$HERE/mtp_longctx_ab.sh" >"$OUT/$name.log" 2>&1 ||
        { echo "$name: harness failed, see $OUT/$name.log"; return 2; }
    # REQ <tag> ctx <n> rep <r> s<i> ... drafted <d> accepted <a> md5 <h>
    awk -v ref="$ref" -v name="$name" '$1 == "REQ" {
             key = $4 " " $7; arm = $2; sub(/^p[0-9]+_/, "", arm)
             md5[arm, key] = $NF; keys[key] = 1; arms[arm] = 1
             for (i = 1; i < NF; i++) if ($i == "drafted") drafted[arm] += $(i + 1)
         }
         END {
             bad = 0; n = 0
             for (a in arms) {
                 printf "%s %s: drafted %d\n", name, a, drafted[a]
                 if (drafted[a] == 0) { print "no drafts verified: the check measured nothing"; exit 2 }
                 for (k in keys) {
                     n++
                     same = md5[a, k] == md5[ref, k] && md5[ref, k] != ""
                     printf "%s ctx/stream %s: %s %s %s %s %s\n", name, k, ref, md5[ref, k], a, md5[a, k], same ? "equal" : "DIFFER"
                     bad += same ? 0 : 1
                 }
             }
             if (n == 0) { print "no REQ line: the check measured nothing"; exit 2 }
             exit bad ? 1 : 0
         }' "$OUT/$name/summary.txt"
}
mkdir -p "$OUT"
rc=0
for c in ${CHECKS:-pair depth}; do
    case $c in
    pair)
        B="speculative.mtp_batch_verify_k=2,gdn.state_bf16=false"
        check pair spare "${CONC:-2}" "${STAGGER_MS:-300}" \
            "spare|$I|$B,speculative.mtp_batch_verify_factor_row=false row|$I|$B,speculative.mtp_batch_verify_factor_row=true"
        ;;
    depth)
        B="speculative.ngram=false,speculative.mtp_batch_verify_k=4,gdn.state_bf16=false"
        THINK_BUDGET=1 check depth k1 1 0 \
            "k1|$I|$B,speculative.mtp_k=1 k2|$I|$B,speculative.mtp_k=2 k3|$I|$B,speculative.mtp_k=3 k4|$I|$B,speculative.mtp_k=4"
        ;;
    *)
        echo "unknown check '$c'"
        false
        ;;
    esac
    r=$?
    [ "$r" -gt "$rc" ] && rc=$r
done
exit "$rc"
