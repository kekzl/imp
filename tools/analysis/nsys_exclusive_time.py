#!/usr/bin/env python3
"""nsys_exclusive_time.py - per-kernel critical-path time in an nsys SQLite export.

A PDL kernel (griddepcontrol.wait) starts before its producer ends, so its nsys
duration includes the wait: on Qwen3.8 MTP verify the smallm v2 GEMMs read 1.20 TB/s
by duration and 1.34-1.67 TB/s by exclusive time. Exclusive time charges each kernel
end - max(start, frontier), frontier = the latest end seen so far.

Steps are delimited by an anchor kernel (one launch per step, e.g. fp8_head_mma for
an MTP verify). Kernels group by (short name, gridX, gridY).

Usage:
  nsys_exclusive_time.py profile.sqlite ANCHOR_SUBSTRING [--skip 5] [--top 30]
"""
import argparse
import sqlite3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("anchor", help="substring of the kernel that runs once per step")
    ap.add_argument("--skip", type=int, default=5, help="warmup steps to drop")
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()
    rows = sqlite3.connect(a.db).execute(
        "SELECT s.value, k.start, k.end, k.gridX, k.gridY FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id = k.shortName ORDER BY k.start").fetchall()
    idx = [i for i, r in enumerate(rows) if a.anchor in r[0]]
    steps = list(zip(idx, idx[1:]))[a.skip:]
    if not steps:
        raise SystemExit(f"fewer than {a.skip + 2} '{a.anchor}' launches")
    agg, wall, idle = {}, 0, 0
    for i0, i1 in steps:
        frontier = rows[i0][1]
        for name, s, e, gx, gy in rows[i0:i1]:
            idle += max(0, s - frontier)
            excl = max(0, e - max(s, frontier))
            frontier = max(frontier, e)
            c, t, d = agg.get((name[:55], gx, gy), (0, 0, 0))
            agg[(name[:55], gx, gy)] = (c + 1, t + excl, d + (e - s))
        wall += rows[i1][1] - rows[i0][1]
    n = len(steps)
    print(f"steps {n}, wall/step {wall / n / 1e3:.1f} us, idle/step {idle / n / 1e3:.1f} us")
    print("excl us/step | dur us/launch | excl us/launch | launches/step | grid | kernel")
    for (name, gx, gy), (c, t, d) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:a.top]:
        print(f"{t / n / 1e3:9.1f} | {d / c / 1e3:8.2f} | {t / c / 1e3:8.2f} | {c / n:6.1f} | "
              f"({gx},{gy}) | {name}")


if __name__ == "__main__":
    main()
