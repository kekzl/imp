#!/usr/bin/env python3
"""Price the MTP draft depth of the lone batched verify from diagnostics.spec_trace server logs.

Reads the [bv-verify] (per request) and [bv-step] (per step) lines that
docs/audit/qwen38_mtp_bv_adaptive_depth_2026_11_04/adaptive_depth.diff.txt adds to the batched verify (main has
none), lone steps only (n=1). Per request: the accept distribution at each draft depth, the same conditioned on
the previous step's outcome, step prices split by the depth that ran (forward) and by the chain the head pass
ran (= the next step's depth), and an AIMD estimate (depth 1 after a rejection, 2 after a full accept) priced
from a fixed-depth-2 trace over a grid of per-step savings (aimd_estimate).

Usage: mtp_bv_depth_price.py SERVER_LOG [SERVER_LOG ...]
"""
import re
import sys
from collections import defaultdict

VER = re.compile(r"\[bv-verify\] req=(\d+) n=(\d+) p0=(\d+) nd=(\d+) matched=(\d+) emitted=(\d+) mtp=(\d)")
STEP = re.compile(r"\[bv-step\] n=(\d+) ctx=(\d+) drafted=(\d+) emitted=(\d+) fwd_ms=([\d.]+) head_ms=([\d.]+)")


def parse(path):
    """Per request: list of (nd, matched, emitted, fwd_ms, head_ms, ctx) of its lone MTP steps."""
    reqs = defaultdict(list)
    pending = None
    with open(path, errors="replace") as f:
        for line in f:
            m = VER.search(line)
            if m:
                req, n, p0, nd, mt, em, mtp = (int(x) for x in m.groups())
                pending = (req, n, p0, nd, mt, em, mtp)
                continue
            s = STEP.search(line)
            if s and pending is not None:
                req, n, p0, nd, mt, em, mtp = pending
                pending = None
                if n == 1 and mtp == 1 and nd > 0:
                    reqs[req].append((nd, mt, em, float(s.group(5)), float(s.group(6)), p0))
    return reqs


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def outcome(nd, mt):
    return "full" if mt >= nd else ("zero" if mt == 0 else "part")


def report(label, steps):
    n = len(steps)
    if n < 20:
        return
    ctx = steps[0][5]
    emitted = sum(s[2] for s in steps)
    wall = sum(s[3] + s[4] for s in steps)
    print(f"{label} ctx {ctx} steps {n} emitted {emitted} tok/step {emitted / n:.3f} "
          f"step_ms {wall / n:.3f} tok_per_s_in_steps {1000.0 * emitted / wall:.2f}")
    for d in (1, 2):
        ms = [s[1] for s in steps if s[0] == d]
        if ms:
            dist = " ".join(f"m{k} {ms.count(k) / len(ms):.3f}" for k in range(d + 1))
            print(f"  nd={d} steps {len(ms)} {dist} fwd_ms {mean([s[3] for s in steps if s[0] == d]):.3f}")
    # Head pass of step i ran the chain iff step i + 1 drafted 2.
    for d in (1, 2):
        hs = [steps[i][4] for i in range(n - 1) if steps[i + 1][0] == d]
        if hs:
            print(f"  head_ms before nd={d}: {mean(hs):.3f} (n={len(hs)})")
    # Accept at depth 2 conditioned on the previous step's outcome.
    cond = defaultdict(list)
    for i in range(1, n):
        if steps[i][0] == 2:
            cond[outcome(steps[i - 1][0], steps[i - 1][1])].append(steps[i][1])
    for k in ("full", "part", "zero"):
        ms = cond[k]
        if ms:
            dist = " ".join(f"m{j} {ms.count(j) / len(ms):.3f}" for j in range(3))
            print(f"  nd=2 after {k}: n={len(ms)} {dist}")
    return cond


def aimd_estimate(cond, t2, t1):
    """Fixed depth 2 vs AIMD, tok/s. t2 / t1: step ms at depth 2 / 1.

    Hidden class of a step = what a depth-2 verify accepts there (m 0 / 1 / 2), drawn from p[class of the
    previous step]. Depth 2 emits 1 + m and sees full iff m = 2; depth 1 emits 1 + min(m, 1) and sees full iff
    m >= 1. AIMD: full -> depth 2 next, else depth 1. States (depth, previous hidden class).
    """
    cls = ("zero", "part", "full")
    p = {k: [ms.count(j) / len(ms) for j in range(3)] for k, ms in cond.items() if ms}
    if not all(k in p for k in cls):
        return None
    pi = {(d, c): 1.0 / 6 for d in (1, 2) for c in cls}
    for _ in range(1000):
        nxt = dict.fromkeys(pi, 0.0)
        for (d, prev), w in pi.items():
            for m, q in enumerate(p[prev]):
                full = m == 2 if d == 2 else m >= 1
                nxt[(2 if full else 1, cls[m])] += w * q
        pi = nxt
    tok = sum(w * sum(q * (1 + (m if d == 2 else min(m, 1))) for m, q in enumerate(p[prev]))
              for (d, prev), w in pi.items())
    time = sum(w * (t2 if d == 2 else t1) for (d, _), w in pi.items())
    fx = dict.fromkeys(cls, 1.0 / 3)
    for _ in range(1000):
        n2 = dict.fromkeys(cls, 0.0)
        for prev, w in fx.items():
            for m, q in enumerate(p[prev]):
                n2[cls[m]] += w * q
        fx = n2
    tok_fx = sum(w * (1 + p[k][1] + 2 * p[k][2]) for k, w in fx.items())
    share1 = sum(w for (d, _), w in pi.items() if d == 1)
    return 1000.0 * tok_fx / t2, 1000.0 * tok / time, share1


def main():
    for path in sys.argv[1:]:
        reqs = parse(path)
        print(f"== {path}: {len(reqs)} requests with lone MTP steps")
        for req, steps in sorted(reqs.items()):
            cond = report(f"req {req}", steps)
            if not cond:
                continue
            t2 = mean([s[3] + s[4] for s in steps if s[0] == 2])
            # Depth 1 saves the chain head pass and the second draft row: a grid, the trace of a k=1 or
            # adaptive arm prices the real saving.
            for saved in (0.5, 1.0, 1.5, 2.0, 3.0):
                est = aimd_estimate(cond, t2, t2 - saved)
                if est:
                    print(f"  depth-1 step saves {saved:.1f} ms: fixed-2 {est[0]:.2f} tok/s, AIMD {est[1]:.2f} tok/s "
                          f"(depth-1 share {est[2]:.3f})")


if __name__ == "__main__":
    main()
