#!/usr/bin/env python3
"""nsys_kernel_windows.py - per-load-window kernel time share of named kernel families.

Splits the kernels of an nsys SQLite export (imp-server under `--cuda-graph-trace=node`)
into windows at device idle gaps > --gap-s seconds (the harness sleeps between client
levels), then prints per window the summed kernel duration and, per substring pattern,
launches, summed duration and share of the window's kernel time.

Usage:
  nsys_kernel_windows.py profile.sqlite [--gap-s 4] [--pat kpar_mb_fp16 --pat smallm ...]
"""
import argparse
import sqlite3

DEFAULT_PATS = [
    "gemv_nvfp4_kpar_mb_fp16",
    "gemv_nvfp4_multirow_mb",
    "gemv_nvfp4_kpar_mb_fp32",
    "smallm",
    "gemv_nvfp4",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--gap-s", type=float, default=4.0)
    ap.add_argument("--pat", action="append")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    pats = a.pat or DEFAULT_PATS
    rows = sqlite3.connect(a.sqlite).execute(
        "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id = k.demangledName ORDER BY k.start").fetchall()
    if not rows:
        print("no kernels")
        return
    wins, cur, last_end = [], [rows[0]], rows[0][1]
    for r in rows[1:]:
        if r[0] - last_end > a.gap_s * 1e9:
            wins.append(cur)
            cur = []
        cur.append(r)
        last_end = max(last_end, r[1])
    wins.append(cur)
    t0 = rows[0][0]
    for i, w in enumerate(wins):
        tot = sum(e - s for s, e, _ in w)
        wall = w[-1][1] - w[0][0]
        print(f"window {i}: t={(w[0][0] - t0) / 1e9:.1f}s wall {wall / 1e6:.1f} ms, "
              f"{len(w)} kernels, kernel sum {tot / 1e6:.2f} ms")
        for p in pats:
            sel = [(s, e) for s, e, n in w if p in n]
            d = sum(e - s for s, e in sel)
            print(f"  {p:28s} n={len(sel):7d} sum {d / 1e6:9.2f} ms share {100.0 * d / max(tot, 1):6.2f} %")
        agg = {}
        for s, e, n in w:
            k = n.split("(")[0][:110]
            agg[k] = agg.get(k, 0) + (e - s)
        for k, d in sorted(agg.items(), key=lambda kv: -kv[1])[:a.top]:
            print(f"    top {100.0 * d / max(tot, 1):6.2f} % {d / 1e6:9.2f} ms {k}")
        # Span of the first pattern's launches (the eager-verify decode phase, prefill excluded).
        hits = [(s, e) for s, e, n in w if pats[0] in n]
        if hits:
            lo, hi = hits[0][0], hits[-1][1]
            span = [(s, e) for s, e, _ in w if s >= lo and e <= hi]
            st = sum(e - s for s, e in span)
            d = sum(e - s for s, e in hits)
            print(f"  span of {pats[0]}: wall {(hi - lo) / 1e6:.1f} ms, kernel sum {st / 1e6:.2f} ms, "
                  f"{pats[0]} share {100.0 * d / max(st, 1):.2f} %")


if __name__ == "__main__":
    main()
