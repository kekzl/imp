#!/usr/bin/env bash
# Engine-level check of speculative.mtp_batch_verify_factor_row (bound verify, 2 drafts). With an f32 GDN state
# the fold is the scan's own update, so CONC greedy chats must come out byte-identical with the factor row and
# with a second spare per row. Exit 0 = every text md5 equal and drafts were verified; 1 = a text differs.
# The texts follow the arrival order of the streams (2 outcomes at STAGGER_MS=0), so the streams start
# STAGGER_MS apart; runtime.deterministic turns drafting off, so the arms run without it.
# Usage: OUT=/mnt/data/scratch/dir [IMAGE=imp:test | dev] [CTX=20000] [CONC=2] [CORPUS=dir] [STAGGER_MS=300]
#        tools/analysis/mtp_factor_row_equiv.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=${OUT:?OUT=dir for logs}
I=${IMAGE:-imp:test}
BASE="speculative.mtp_batch_verify_k=2,gdn.state_bf16=false"
ARMS="spare|$I|$BASE,speculative.mtp_batch_verify_factor_row=false row|$I|$BASE,speculative.mtp_batch_verify_factor_row=true"
ARMS="$ARMS" PAIRS=1 CONC="${CONC:-2}" CTX="${CTX:-20000}" STAGGER_MS="${STAGGER_MS:-300}" OUT="$OUT" \
    bash "$HERE/mtp_longctx_ab.sh" >"$OUT.log" 2>&1 ||
    { echo "harness failed, see $OUT.log"; exit 2; }
# REQ <tag> ctx <n> rep <r> s<i> ... drafted <d> accepted <a> md5 <h>
awk '$1 == "REQ" {
         key = $4 " " $7; arm = $2; sub(/^p[0-9]+_/, "", arm)
         md5[arm, key] = $NF; keys[key] = 1
         for (i = 1; i < NF; i++) if ($i == "drafted") drafted[arm] += $(i + 1)
     }
     END {
         bad = 0; n = 0
         for (k in keys) {
             n++
             same = md5["spare", k] == md5["row", k] && md5["row", k] != ""
             printf "ctx/stream %s: spare %s row %s %s\n", k, md5["spare", k], md5["row", k], same ? "equal" : "DIFFER"
             bad += same ? 0 : 1
         }
         printf "drafted: spare %d row %d\n", drafted["spare"], drafted["row"]
         if (n == 0 || drafted["spare"] == 0 || drafted["row"] == 0) { print "no drafts verified: the check measured nothing"; exit 2 }
         exit bad ? 1 : 0
     }' "$OUT/summary.txt"
