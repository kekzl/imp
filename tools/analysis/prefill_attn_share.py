#!/usr/bin/env python3
"""Attention share of prefill kernel time from an nsys sqlite (prefill_attn_share.sh).

Window: kernels starting inside the NVTX range "bench:pp" (imp-cli --bench). Attention class =
name matches ATTN_RE; the top kernels are printed so the classification can be checked by eye.
Bound = speedup if attention got 4x cheaper: 1 / (1 - share * 0.75).
Usage: prefill_attn_share.py <file.sqlite> [--range bench:pp]
"""
import argparse
import re
import sqlite3
from collections import defaultdict

ATTN_RE = re.compile(r"fmha|flash|attention|attn_|paged_attn", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--range", default="bench:pp")
    a = ap.parse_args()
    db = sqlite3.connect(a.sqlite)
    nv = db.execute(
        "SELECT n.start, n.end FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id = n.textId "
        "WHERE COALESCE(n.text, s.value) = ? AND n.end IS NOT NULL ORDER BY n.start",
        (a.range,),
    ).fetchall()
    if not nv:
        print(f"no NVTX range {a.range}")
        return 1
    lo, hi = nv[0]
    rows = db.execute(
        "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id = COALESCE(k.demangledName, k.shortName) WHERE k.start >= ? AND k.start < ?",
        (lo, hi),
    ).fetchall()
    per = defaultdict(lambda: [0.0, 0])
    for s, e, name in rows:
        per[name][0] += (e - s) / 1e6
        per[name][1] += 1
    total = sum(v[0] for v in per.values())
    if total <= 0:
        print("no kernels in the window")
        return 1
    attn = sum(v[0] for n, v in per.items() if ATTN_RE.search(n))
    attn_n = sum(v[1] for n, v in per.items() if ATTN_RE.search(n))
    share = attn / total
    print(f"range={a.range} ranges_found={len(nv)} window_ms={(hi - lo) / 1e6:.1f}")
    print(f"kernels_ms={total:.1f} launches={sum(v[1] for v in per.values())}")
    print(f"attention_ms={attn:.1f} launches={attn_n} share={100.0 * share:.1f}%")
    print(f"bound=1/(1-share*0.75)={1.0 / (1.0 - 0.75 * share):.3f}x")
    print("top kernels (ms, launches, A=attention):")
    for n, v in sorted(per.items(), key=lambda kv: -kv[1][0])[:15]:
        print(f"  {v[0]:9.1f} {v[1]:7d} {'A' if ATTN_RE.search(n) else '-'} {n[:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
