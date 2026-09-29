#!/usr/bin/env bash
# #2267 acceptance over a scripts/bench_2267.sh result dir (CPU only, reads files). PASS iff all of:
#   bit-exact: MmqQ8Imma.{PipelineBitIdenticalToLegacy,RouteBitIdenticalToLegacy} OK, unit exit 0;
#   micro: new_ms / old_ms <= 0.88 at Qwen3-8B q_o (N=K=4096) M=2048 (micro.log `q8pipe` line);
#   e2e, every model: off / new >= 1.00 at 2048, 4096, 8192 tokens; new / old <= 1.02 at 256, 512, 1024.
# Usage: bash scripts/accept_2267.sh <out_dir>
set -uo pipefail

OUT="${1:?usage: accept_2267.sh <out_dir>}"
TSV="$OUT/e2e.tsv"
UNIT="$OUT/unit.log"
MICRO="$OUT/micro.log"
FAIL=0

for t in MmqQ8Imma.PipelineBitIdenticalToLegacy MmqQ8Imma.RouteBitIdenticalToLegacy; do
    if [ -s "$UNIT" ] && grep -q "^\[       OK \] $t " "$UNIT" && grep -q '^unit exit=0$' "$UNIT"; then
        echo "PASS bit-exact: $t"
    else
        echo "FAIL bit-exact: $t not OK in $UNIT"
        FAIL=1
    fi
done

micro=$(awk '/^q8pipe / && / model=Qwen3-8B / && / shape=q_o / && / M=2048 / {
        for (i = 1; i <= NF; i++) { split($i, kv, "="); v[kv[1]] = kv[2] }
        if (v["old_ms"] > 0 && v["new_ms"] > 0) printf "%.3f %s %s\n", v["new_ms"] / v["old_ms"], v["new_ms"], v["old_ms"]
    }' "$MICRO" 2>/dev/null | head -1)
if [ -z "$micro" ]; then
    echo "FAIL micro: no Qwen3-8B q_o M=2048 q8pipe line in $MICRO"
    FAIL=1
else
    read -r r n o <<<"$micro"
    if awk -v r="$r" 'BEGIN { exit !(r <= 0.88) }'; then v=PASS; else v=FAIL; FAIL=1; fi
    echo "$v micro: Qwen3-8B q_o M=2048 new/old $r (new $n ms, old $o ms; need <= 0.880)"
fi

[ -s "$TSV" ] || { echo "FAIL e2e: $TSV missing or empty"; echo "ACCEPT_2267 FAIL"; exit 1; }

# TSV rows: model tokens arm median_s
e2e=$(awk -F'\t' '
    { t[$1 SUBSEP $2 SUBSEP $3] = $4; models[$1] = 1 }
    function chk(m, n, num, den, lim, ge,    a, b, r, pass) {
        a = t[m SUBSEP n SUBSEP num]; b = t[m SUBSEP n SUBSEP den]
        if (a == "" || b == "" || a == "ERR" || b == "ERR") { printf "FAIL e2e %s %s: %s or %s missing\n", m, n, num, den; return 0 }
        r = a / b; pass = ge ? (r >= lim) : (r <= lim)
        printf "%s e2e %s %s tokens: %s/%s %.3f (need %s %.3f)\n", pass ? "PASS" : "FAIL", m, n, num, den, r, ge ? ">=" : "<=", lim
        return pass
    }
    END {
        ok = 1
        for (m in models) {
            split("2048 4096 8192", lng, " ")
            for (i = 1; i <= 3; i++) ok = chk(m, lng[i], "off", "new", 1.00, 1) && ok
            split("256 512 1024", sht, " ")
            for (i = 1; i <= 3; i++) ok = chk(m, sht[i], "new", "old", 1.02, 0) && ok
        }
        printf "E2E %s\n", ok ? "PASS" : "FAIL"
    }' "$TSV")
echo "$e2e" | sort
echo "$e2e" | grep -q '^E2E PASS$' || FAIL=1

[ "$FAIL" = 0 ] && { echo "ACCEPT_2267 PASS"; exit 0; }
echo "ACCEPT_2267 FAIL"
exit 1
