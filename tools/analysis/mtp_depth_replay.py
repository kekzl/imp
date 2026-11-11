#!/usr/bin/env python3
"""Replay of a bound-verify trace under per-step depth policies.

Usage: mtp_depth_replay.py D3_SERVER_LOG [D2_SERVER_LOG ...]
Trace line (diagnostics.spec_trace): spec-batch-trace n=N T=T fwd=MS head=MS | id:nd:m:e ...
Step cost per (n, depth) = median fwd + head over every given log. The first log (depth 3) supplies the
outcomes: a step replayed at depth 2 emits min(e, 3) tokens per request.
"""
import re
import statistics
import sys
from collections import defaultdict

LINE = re.compile(r"spec-batch-trace n=(\d+) T=(\d+) fwd=([\d.]+) head=([\d.]+) \|(.*)")


def parse(path, idx):
    steps = []
    for line in open(path, errors="replace"):
        m = LINE.search(line)
        if not m:
            continue
        reqs = [tuple(int(x) for x in r.split(":")) for r in m.group(5).split()]
        reqs = [(idx * 100000 + r[0],) + r[1:] for r in reqs]
        steps.append((int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4)), reqs))
    return steps


def main():
    logs = [parse(p, i) for i, p in enumerate(sys.argv[1:])]
    cost = defaultdict(list)
    for steps in logs:
        for n, T, fwd, head, _ in steps:
            cost[(n, T - 1)].append(fwd + head)
    c = {k: statistics.median(v) for k, v in cost.items() if len(v) >= 20}
    print("# step cost ms (median fwd + head) by (requests, depth), n steps")
    for k in sorted(c):
        print(f"cost n={k[0]} D={k[1]} {c[k]:.3f} ms steps={len(cost[k])}")
    steps = [s for lg in logs for s in lg if s[1] == 4]
    print(f"# depth-3 steps replayed: {len(steps)}")
    hist = defaultdict(lambda: [0, 0, 0, 0])
    for n, _, _, _, reqs in steps:
        for _, nd, m, _ in reqs:
            if nd == 3:
                hist[n][m] += 1
    for n in sorted(hist):
        h = hist[n]
        tot = sum(h)
        print(f"matched n={n} rows={tot} m0={h[0] / tot:.3f} m1={h[1] / tot:.3f} m2={h[2] / tot:.3f} "
              f"m3={h[3] / tot:.3f}")
    # run lengths of full / non-full depth-3 steps per request (burstiness)
    last = {}
    flips = same = 0
    for _, _, _, _, reqs in steps:
        for rid, nd, m, _ in reqs:
            if nd != 3:
                continue
            full = m == 3
            if rid in last:
                if last[rid] == full:
                    same += 1
                else:
                    flips += 1
            last[rid] = full
    print(f"# full-accept persistence: same {same} flips {flips} P(same)={same / max(1, same + flips):.3f}")

    def replay(policy_init, want, update):
        st = {}
        t_ms = 0.0
        toks = 0
        at3 = 0
        by_n = defaultdict(lambda: [0.0, 0])
        for n, _, _, _, reqs in steps:
            if (n, 2) not in c or (n, 3) not in c:
                continue
            for rid, *_ in reqs:
                if rid not in st:
                    st[rid] = policy_init()
            d = max(want(st[rid]) for rid, *_ in reqs)
            at3 += d == 3
            step_t = c[(n, d)]
            step_k = 0
            for rid, nd, m, e in reqs:
                e_d = e if d == 3 else min(e, 3)
                m_d = m if d == 3 else min(m, 2)
                step_k += e_d
                if nd >= d:
                    update(st[rid], d, m_d)
            t_ms += step_t
            toks += step_k
            by_n[n][0] += step_t
            by_n[n][1] += step_k
        return t_ms, toks, at3, by_n

    def report(name, res, base=None):
        t_ms, toks, at3, by_n = res
        rate = 1000.0 * toks / t_ms
        parts = " ".join(f"n{n}={1000.0 * v[1] / v[0]:.1f}" for n, v in sorted(by_n.items()))
        rel = f" vs_fixed3 {100.0 * (rate / base - 1):+.2f}%" if base else ""
        print(f"{name:34s} tok/s {rate:8.2f}{rel}  steps_at_3 {at3:6d}  {parts}")
        return rate

    fixed = lambda d: (lambda: {}, lambda s: d, lambda s, dd, m: None)
    base = report("fixed3", replay(*fixed(3)))
    report("fixed2", replay(*fixed(2)), base)

    # AIMD: full accept at the step's depth -> 3, anything else -> 2
    def aimd():
        def upd(s, d, m):
            s["w"] = 3 if m >= d else 2
        return (lambda: {"w": 3}, lambda s: s["w"], upd)

    report("aimd", replay(*aimd()), base)

    # streak: drop after K non-full steps at 3 in a row, climb after J full steps at 2 in a row
    def streak(K, J):
        def upd(s, d, m):
            if d == 3:
                s["miss"] = 0 if m == 3 else s["miss"] + 1
                if s["miss"] >= K:
                    s["w"], s["hit"] = 2, 0
            else:
                s["hit"] = s["hit"] + 1 if m >= 2 else 0
                if s["hit"] >= J:
                    s["w"], s["miss"] = 3, 0
        return (lambda: {"w": 3, "miss": 0, "hit": 0}, lambda s: s["w"], upd)

    for K in (2, 3, 4, 6, 8):
        for J in (1, 2, 3, 4):
            report(f"streak drop{K} climb{J}", replay(*streak(K, J)), base)

    # EMA: e3 = EMA of [m == 3] at depth 3, e2 = EMA of [m >= 2]; drop below lo, climb above hi
    def ema(a, lo, hi):
        def upd(s, d, m):
            s["e2"] += a * ((1.0 if m >= 2 else 0.0) - s["e2"])
            if d == 3:
                s["e3"] += a * ((1.0 if m == 3 else 0.0) - s["e3"])
                if s["e3"] < lo:
                    s["w"], s["e2"] = 2, min(s["e2"], hi - 0.1)
            elif s["e2"] > hi:
                s["w"], s["e3"] = 3, 0.5
        return (lambda: {"w": 3, "e3": 0.5, "e2": 0.5}, lambda s: s["w"], upd)

    for a in (0.05, 0.1, 0.2):
        for lo in (0.2, 0.3, 0.4):
            for hi in (0.6, 0.75, 0.9):
                report(f"ema a{a} lo{lo} hi{hi}", replay(*ema(a, lo, hi)), base)

    # oracle with the outcome known: depth 3 iff the step's extra tokens pay its extra cost
    t3, k3, _, _ = replay(*fixed(3))
    ms_per_tok = t3 / k3
    t_ms = 0.0
    toks = 0
    at3 = 0
    for n, _, _, _, reqs in steps:
        if (n, 2) not in c or (n, 3) not in c:
            continue
        extra = sum(1 for _, _, _, e in reqs if e == 4)
        d = 3 if extra * ms_per_tok > c[(n, 3)] - c[(n, 2)] else 2
        at3 += d == 3
        t_ms += c[(n, d)]
        toks += sum(e if d == 3 else min(e, 3) for _, _, _, e in reqs)
    rate = 1000.0 * toks / t_ms
    print(f"{'oracle (outcome known)':34s} tok/s {rate:8.2f} vs_fixed3 {100.0 * (rate / base - 1):+.2f}%  "
          f"steps_at_3 {at3:6d}")


main()
