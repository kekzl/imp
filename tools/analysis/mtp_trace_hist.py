#!/usr/bin/env python3
"""Per (n, T) of "spec-batch-trace n=N T=T fwd=MS head=MS | id:nd:m:e ..." lines: steps, median step cost,
matched histogram of full-draft rows, tokens per step ms.

Usage: mtp_trace_hist.py SERVER_LOG...
"""
import re
import statistics
import sys
from collections import Counter, defaultdict

LINE = re.compile(r"spec-batch-trace n=(\d+) T=(\d+) fwd=([\d.]+) head=([\d.]+) \|(.*)")
for path in sys.argv[1:]:
    cost = defaultdict(list)
    hist = defaultdict(Counter)
    emitted = Counter()
    short = Counter()
    for line in open(path, errors="replace"):
        m = LINE.search(line)
        if not m:
            continue
        n, t = int(m.group(1)), int(m.group(2))
        key = (n, t)
        cost[key].append((float(m.group(3)), float(m.group(4))))
        for tok in m.group(5).split():
            _, nd, mt, em = (int(x) for x in tok.split(":"))
            emitted[key] += em
            if nd == t - 1:
                hist[key][mt] += 1
            else:
                short[key] += 1
    print(path)
    tot_e = tot_ms = 0.0
    for key in sorted(cost):
        n, t = key
        c = cost[key]
        fwd = statistics.median(x[0] for x in c)
        head = statistics.median(x[1] for x in c)
        ms = sum(x[0] + x[1] for x in c)
        rows = sum(hist[key].values())
        h = " ".join(f"m{k}={hist[key][k] / rows:.3f}" for k in range(t)) if rows else "-"
        tot_e += emitted[key]
        tot_ms += ms
        print(f"  n={n} T={t} steps={len(c)} fwd_med={fwd:.3f} head_med={head:.3f} emitted={emitted[key]} "
              f"tok_per_s={1000.0 * emitted[key] / ms:.2f} rows={rows} short={short[key]} {h}")
    if tot_ms:
        print(f"  all: emitted={tot_e:.0f} step_ms={tot_ms:.0f} tok_per_s={1000.0 * tot_e / tot_ms:.2f}")
