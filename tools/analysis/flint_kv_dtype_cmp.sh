#!/usr/bin/env bash
# Paired per-token NLL comparison of two ppl_dump files, binned by position: mean(B - A), its standard error
# (independent-token approximation), and the share of tokens where B is worse by > 0.1 nats.
# Usage: tools/analysis/flint_kv_dtype_cmp.sh A.txt B.txt [BIN_EDGES]   (default 0,4096,16384,32768,49152,65536,131072)
set -euo pipefail
A=$1; B=$2; EDGES=${3:-0,4096,16384,32768,49152,65536,131072}
join <(sort -k1,1 "$A") <(sort -k1,1 "$B") | awk -v edges="$EDGES" '
BEGIN { n = split(edges, e, ","); printf "%-14s %8s %10s %10s %10s %10s %8s\n", "positions", "tokens", "nll_A", "nll_B", "B-A", "se", ">0.1" }
{
    for (i = 1; i < n; ++i) if ($1 >= e[i] && $1 < e[i + 1]) { k = i; break }
    d = $3 - $2; c[k]++; sa[k] += $2; sb[k] += $3; sd[k] += d; sq[k] += d * d; w[k] += (d > 0.1)
    C++; SA += $2; SB += $3; SD += d; SQ += d * d; W += (d > 0.1)
}
function row(lbl, cc, a, b, dd, qq, ww,   m, v) {
    m = dd / cc; v = (qq / cc - m * m) / cc
    printf "%-14s %8d %10.5f %10.5f %+10.5f %10.5f %7.2f%%\n", lbl, cc, a / cc, b / cc, m, sqrt(v > 0 ? v : 0), 100 * ww / cc
}
END {
    for (i = 1; i < n; ++i) if (c[i]) row(e[i] "-" e[i + 1], c[i], sa[i], sb[i], sd[i], sq[i], w[i])
    row("all", C, SA, SB, SD, SQ, W)
    printf "ppl_A %.4f  ppl_B %.4f\n", exp(SA / C), exp(SB / C)
}'
