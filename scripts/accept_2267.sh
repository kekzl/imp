#!/usr/bin/env bash
# #2267 acceptance over a scripts/bench_2267.sh result dir (CPU only, reads files).
# PASS iff at least one IMMA arm meets, on every model:
#   off / arm >= 1.00 at 2048, 4096, 8192 tokens;  arm <= 1.02 x on (today's default) at 256, 512, 1024.
# Usage: bash scripts/accept_2267.sh <out_dir> [arm ...]   (arms default: on)
set -uo pipefail

OUT="${1:?usage: accept_2267.sh <out_dir> [arm ...]}"
shift
ARMS=("$@")
[ ${#ARMS[@]} -gt 0 ] || ARMS=(on)
TSV="$OUT/e2e.tsv"
FAIL=0

[ -s "$TSV" ] || { echo "FAIL e2e: $TSV missing or empty"; exit 1; }

# TSV rows: model tokens arm median_s
verdict() {  # verdict <arm> -> per-check lines, last line "ARM <arm> PASS|FAIL"
    awk -F'\t' -v arm="$1" '
        { t[$1 SUBSEP $2 SUBSEP $3] = $4; models[$1] = 1 }
        END {
            ok = 1
            for (m in models) {
                split("2048 4096 8192", lng, " ")
                for (i = 1; i <= 3; i++) {
                    o = t[m SUBSEP lng[i] SUBSEP "off"]; x = t[m SUBSEP lng[i] SUBSEP arm]
                    if (o == "" || x == "" || o == "ERR" || x == "ERR") { printf "FAIL %s %s %s: missing\n", arm, m, lng[i]; ok = 0; continue }
                    r = o / x; pass = (r >= 1.00)
                    printf "%s %s %s %s tokens: off/arm %.3f (need >= 1.000)\n", pass ? "PASS" : "FAIL", arm, m, lng[i], r
                    ok = ok && pass
                }
                split("256 512 1024", sht, " ")
                for (i = 1; i <= 3; i++) {
                    b = t[m SUBSEP sht[i] SUBSEP "on"]; x = t[m SUBSEP sht[i] SUBSEP arm]
                    if (b == "" || x == "" || b == "ERR" || x == "ERR") { printf "FAIL %s %s %s: missing\n", arm, m, sht[i]; ok = 0; continue }
                    r = x / b; pass = (r <= 1.02)
                    printf "%s %s %s %s tokens: arm/on %.3f (need <= 1.020)\n", pass ? "PASS" : "FAIL", arm, m, sht[i], r
                    ok = ok && pass
                }
            }
            printf "ARM %s %s\n", arm, ok ? "PASS" : "FAIL"
        }' "$TSV"
}

PASSING=()
for a in "${ARMS[@]}"; do
    res=$(verdict "$a")
    echo "$res" | sort
    echo "$res" | grep -q "^ARM $a PASS$" && PASSING+=("$a")
done
if [ ${#PASSING[@]} -eq 0 ]; then
    echo "FAIL no arm meets the #2267 goal"
    FAIL=1
fi
[ "$FAIL" = 0 ] && { echo "ACCEPT_2267 PASS: ${PASSING[*]}"; exit 0; }
echo "ACCEPT_2267 FAIL"
exit 1
