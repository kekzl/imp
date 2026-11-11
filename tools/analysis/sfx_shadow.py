#!/usr/bin/env python3
"""Suffix-draft shadow of a bound-verify run: what the suffix index would have drafted at every MTP step.

Usage: sfx_shadow.py SERVER_LOG [...] [--row-ms MS]
Lines read (diagnostics.spec_trace):
  spec-batch-trace n=N T=T fwd=MS head=MS | id:nd:m:e ...          step cost per (requests, rows)
  spec-sfx-shadow id=I pos=P n=N T=T mtp=nd:m:e sfx=len:match:agree  one per request step
`agree` = leading suffix-draft tokens equal to the tokens the request emitted from pos on; `match` = the
backward match length behind the draft (capped at speculative.max_match), 0 = no draft.

Tables:
  1. by match bucket: MTP tokens per step against the suffix draft's agreement at the same positions.
  2. policy replay per request-count n: a step whose match >= M runs the suffix draft at depth
     D = min(len, K) and emits min(agree, D) + 1 tokens; every other step is the recorded MTP step.
     A suffix step that ends inside a recorded step pays that step's remainder pro rata.
     Step cost: median fwd + head of (n, T), a request's share = cost / n. Rows past the measured T cost
     n * row_ms per row (uniform T: the other requests pad), all charged to the request that deepens.
     row_ms: least-squares slope of cost over n * T when the log holds 2+ row counts, else --row-ms.
A replay is a calculation: it sizes the lever, the server A/B decides it.
"""
import argparse
import re
import statistics
from collections import defaultdict

STEP = re.compile(r"spec-batch-trace n=(\d+) T=(\d+) fwd=([\d.]+) head=([\d.]+) \|")
SHADOW = re.compile(r"spec-sfx-shadow id=(\d+) pos=(\d+) n=(\d+) T=(\d+) mtp=(\d+):(\d+):(\d+) sfx=(\d+):(\d+):(\d+)")
BUCKETS = ((0, 0), (1, 7), (8, 11), (12, 15), (16, 23), (24, 31), (32, 1 << 30))
THRESHOLDS = (6, 8, 12, 16, 24, 32)
DEPTHS = (4, 8, 16, 32)


def parse(paths):
    cost = defaultdict(list)
    recs = defaultdict(list)  # (log, id) -> [(pos, n, T, nd, m, e, len, match, agree)]
    for li, path in enumerate(paths):
        for line in open(path, errors="replace"):
            m = SHADOW.search(line)
            if m:
                v = [int(x) for x in m.groups()]
                recs[(li, v[0])].append(tuple(v[1:]))
                continue
            m = STEP.search(line)
            if m:
                cost[(int(m.group(1)), int(m.group(2)))].append(float(m.group(3)) + float(m.group(4)))
    return cost, recs


def row_slope(cost, fallback):
    pts = [(n * T, statistics.median(v), len(v)) for (n, T), v in cost.items() if len(v) >= 20]
    if len({p[0] for p in pts}) < 2:
        return fallback, "--row-ms"
    w = sum(p[2] for p in pts)
    mx = sum(p[0] * p[2] for p in pts) / w
    my = sum(p[1] * p[2] for p in pts) / w
    den = sum(p[2] * (p[0] - mx) ** 2 for p in pts)
    return sum(p[2] * (p[0] - mx) * (p[1] - my) for p in pts) / den, "fit"


def replay(rs, thr, depth, med, row_ms):
    """(base ms, policy ms, tokens, suffix steps) of one request under suffix precedence at match >= thr."""
    base = pol = 0.0
    toks = taken = 0
    p = -1
    for pos, n, T, _nd, _m, e, ln, match, agree in rs:
        if e <= 0 or (n, T) not in med:
            continue
        share = med[(n, T)] / n
        base += share
        toks += e
        end = pos + e
        if p >= end:
            continue  # covered by an earlier suffix step
        if p > pos:
            pol += share * (end - p) / e
            p = end
            continue
        if ln > 0 and match >= thr:
            d = min(ln, depth)
            pol += share + max(0, d + 1 - T) * n * row_ms
            taken += 1
            p = pos + min(agree, d) + 1
            if p < end:
                pol += share * (end - p) / e
                p = end
        else:
            pol += share
            p = end
    return base, pol, toks, taken


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--row-ms", type=float, default=0.27, help="cost of one more row when the log cannot fit it")
    a = ap.parse_args()
    cost, recs = parse(a.logs)
    med = {k: statistics.median(v) for k, v in cost.items() if len(v) >= 20}
    row_ms, src = row_slope(cost, a.row_ms)
    print("# step cost ms (median fwd + head)")
    for k in sorted(med):
        print(f"cost n={k[0]} T={k[1]} {med[k]:.3f} ms steps={len(cost[k])}")
    print(f"row_ms {row_ms:.3f} ({src})")
    flat = [r for rs in recs.values() for r in rs]
    print(f"# requests {len(recs)} shadow steps {len(flat)} tokens {sum(r[5] for r in flat)}")
    if not flat:
        raise SystemExit("no spec-sfx-shadow line: diagnostics.spec_trace off, or no bound verify ran")

    print("match | steps | share % | mtp tok/step | sfx len | sfx agree | agree>=4 % | >=8 % | >=16 % | >=32 % | "
          "sfx tok/step at mtp depth | sfx beats mtp %")
    for lo, hi in BUCKETS:
        b = [r for r in flat if lo <= r[7] <= hi]
        if not b:
            continue
        k = len(b)
        ge = [100.0 * sum(1 for r in b if r[8] >= d) / k for d in DEPTHS]
        at_depth = sum(min(r[8], r[2] - 1) + 1 for r in b) / k
        beats = 100.0 * sum(1 for r in b if min(r[8], r[2] - 1) + 1 > r[5]) / k
        name = "none" if hi == 0 else (f"{lo}+" if hi > 1000 else f"{lo}-{hi}")
        print(f"{name:>5} | {k:7d} | {100.0 * k / len(flat):5.1f} | {sum(r[5] for r in b) / k:5.2f} | "
              f"{sum(r[6] for r in b) / k:5.1f} | {sum(r[8] for r in b) / k:5.2f} | {ge[0]:5.1f} | {ge[1]:5.1f} | "
              f"{ge[2]:5.1f} | {ge[3]:5.1f} | {at_depth:5.2f} | {beats:5.1f}")

    by_n = defaultdict(list)
    for rs in recs.values():
        rs.sort()
        by_n[statistics.median_low([r[1] for r in rs])].append(rs)
    print("n | M | K | suffix steps | base tok/s | policy tok/s | gain %")
    for n in sorted(by_n):
        for thr in THRESHOLDS:
            for depth in DEPTHS:
                tb = tp = 0.0
                tk = st = 0
                for rs in by_n[n]:
                    b, p, t, s = replay(rs, thr, depth, med, row_ms)
                    tb += b
                    tp += p
                    tk += t
                    st += s
                if tb <= 0 or tp <= 0:
                    continue
                print(f"{n} | {thr:2d} | {depth:2d} | {st:6d} | {1000.0 * tk / tb:7.2f} | {1000.0 * tk / tp:7.2f} | "
                      f"{100.0 * (tb / tp - 1.0):+6.2f}")


if __name__ == "__main__":
    main()
