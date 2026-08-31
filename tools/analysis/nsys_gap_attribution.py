#!/usr/bin/env python3
"""nsys_gap_attribution.py - where the GPU sits idle in a serving profile.

Reads an nsys SQLite export (`nsys export --type sqlite`) of an imp-server
run captured with `--cuda-graph-trace=node`, merges every kernel and memcpy
/ memset interval on the device into one busy timeline, and attributes the
idle time between them:

  - busy / idle share of the analysed window
  - gap histogram: <10 us (launch/replay density), 10-100 us, 100 us-1 ms,
    >1 ms (host moments)
  - the largest gaps with the kernel before and after each (what the host
    was between)
  - per-step view when the graph replays are visible: steps are delimited
    by the sampler/argmax kernel that ends a decode step

Usage:
  nsys_gap_attribution.py profile.sqlite [--window START_S END_S] [--top 20]

The window is in seconds from the first kernel; pick the steady-state part
(after the warmup waves, before the drain). All numbers are printed with
their definitions so the output can be pasted into a doc as-is.
"""
import argparse
import sqlite3
import sys
from collections import Counter, defaultdict


def load_intervals(db, kinds):
    cur = db.cursor()
    strings = {}
    try:
        for sid, val in cur.execute("SELECT id, value FROM StringIds"):
            strings[sid] = val
    except sqlite3.OperationalError:
        pass
    rows = []
    if "kernel" in kinds:
        for start, end, name_id in cur.execute(
                "SELECT start, end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"):
            rows.append((start, end, strings.get(name_id, str(name_id))))
    if "memcpy" in kinds:
        try:
            for start, end in cur.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
                rows.append((start, end, "[memcpy]"))
        except sqlite3.OperationalError:
            pass
    if "memset" in kinds:
        try:
            for start, end in cur.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMSET"):
                rows.append((start, end, "[memset]"))
        except sqlite3.OperationalError:
            pass
    rows.sort()
    return rows


def short(name, n=60):
    name = name.split("(")[0]
    return name if len(name) <= n else name[:n - 3] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--window", nargs=2, type=float, metavar=("START_S", "END_S"),
                    help="analysed window in seconds from the first kernel (default: all)")
    ap.add_argument("--top", type=int, default=20, help="largest gaps to list")
    ap.add_argument("--kinds", default="kernel,memcpy,memset")
    args = ap.parse_args()

    db = sqlite3.connect(args.sqlite)
    rows = load_intervals(db, set(args.kinds.split(",")))
    if not rows:
        sys.exit("no device intervals found")
    t0 = rows[0][0]
    if args.window:
        lo = t0 + int(args.window[0] * 1e9)
        hi = t0 + int(args.window[1] * 1e9)
        rows = [r for r in rows if r[0] >= lo and r[1] <= hi]
        if not rows:
            sys.exit("window contains no intervals")
    w_start, w_end = rows[0][0], max(r[1] for r in rows)
    window_ns = w_end - w_start

    # Merge overlapping intervals (streams overlap) into one busy timeline.
    busy_ns = 0
    gaps = []  # (gap_ns, before_name, after_name, at_s)
    cur_s, cur_e, cur_name = rows[0][0], rows[0][1], rows[0][2]
    last_name = cur_name
    for s, e, name in rows[1:]:
        if s <= cur_e:
            if e > cur_e:
                cur_e = e
                last_name = name
            continue
        busy_ns += cur_e - cur_s
        gaps.append((s - cur_e, last_name, name, (cur_e - t0) / 1e9))
        cur_s, cur_e, last_name = s, e, name
    busy_ns += cur_e - cur_s
    idle_ns = window_ns - busy_ns

    print(f"window: {window_ns / 1e9:.3f} s, device intervals: {len(rows)}, merged gaps: {len(gaps)}")
    print(f"busy: {busy_ns / 1e9:.3f} s ({100.0 * busy_ns / window_ns:.1f}%)  "
          f"idle: {idle_ns / 1e9:.3f} s ({100.0 * idle_ns / window_ns:.1f}%)")

    bins = [(0, 10e3, "<10 us"), (10e3, 100e3, "10-100 us"), (100e3, 1e6, "100 us-1 ms"),
            (1e6, float("inf"), ">1 ms")]
    print("\ngap histogram (idle between merged device intervals):")
    print("  bin          count      total ms   share of idle")
    for lo, hi, label in bins:
        sel = [g for g in gaps if lo <= g[0] < hi]
        tot = sum(g[0] for g in sel)
        print(f"  {label:<11} {len(sel):>7}   {tot / 1e6:>10.1f}   {100.0 * tot / max(idle_ns, 1):>6.1f}%")

    print(f"\nlargest {args.top} gaps (ms, at s from first kernel, kernel before -> after):")
    for g, before, after, at in sorted(gaps, reverse=True)[:args.top]:
        print(f"  {g / 1e6:8.2f}  @{at:9.3f}  {short(before)} -> {short(after)}")

    # Which kernel pairs the sub-100-us gaps sit between (launch density).
    pair_tot = Counter()
    pair_n = Counter()
    for g, before, after, _ in gaps:
        if g < 100e3:
            key = (short(before, 40), short(after, 40))
            pair_tot[key] += g
            pair_n[key] += 1
    print("\nsub-100-us gaps by (before -> after) pair, top 15 by total idle:")
    for key, tot in pair_tot.most_common(15):
        print(f"  {tot / 1e6:8.1f} ms  n={pair_n[key]:>7}  avg {tot / pair_n[key] / 1e3:6.1f} us  "
              f"{key[0]} -> {key[1]}")

    # Kernel time by name (device-side cost, for the launch-count view).
    by_name = defaultdict(lambda: [0, 0])
    for s, e, name in rows:
        by_name[short(name, 50)][0] += e - s
        by_name[short(name, 50)][1] += 1
    print("\ntop 12 device intervals by total time:")
    for name, (tot, n) in sorted(by_name.items(), key=lambda kv: -kv[1][0])[:12]:
        print(f"  {tot / 1e6:9.1f} ms  n={n:>7}  avg {tot / n / 1e3:7.1f} us  {name}")


if __name__ == "__main__":
    main()
