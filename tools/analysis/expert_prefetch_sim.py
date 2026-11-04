#!/usr/bin/env python3
"""Price moe.expert_prefetch offline from a diagnostics.moe_expert_trace with prediction records.

Usage: expert_prefetch_sim.py trace.json SLOTS [budget ...] [--window K] [--tag B[,B...]] [--drop] [--margin M]

The trace holds the routed records [layer, e0..e_{k-1}] and, when the engine ran with the
trace on, prediction records [1000 + layer, p0..p_{k-1}] (layer L's MoE input through layer
L + 1's router, written just before layer L + 1 routes). The replay mirrors the device cache:
one LRU per layer of SLOTS keys (projection, expert), victims never taken from keys stamped in
the same step; a prediction restamps its resident keys and inserts up to `budget` absent ones
(rank-major: expert k's gate, up, down before expert k + 1), which the next routed record of
that layer counts as prefetch hits. `--window K`: keys the gather stream moves between the
prefetch issue and that layer's resolve; prefetched keys past it delay the layer's own misses.
`--tag B`: replay the prediction records tagged B + layer (default 1000); other sources a trace
holds under other multiples of 1000 are ignored, one table per listed B.
`--drop`: unrouted prefetched keys leave the cache at the next resolve (the abort bound).
`--margin M`: only predicted experts whose logit is >= M above the k-th predicted logit get prefetched.
"""
import json
from collections import OrderedDict
import sys

TAG = 1000


def load(path, margin=None):
    """margin: a predicted expert is a prefetch candidate only when its logit is >= margin above the
    last (k-th) predicted logit; the others still restamp their resident keys (ids stored as -1 - e)."""
    with open(path) as f:
        d = json.load(f)
    recs = d["trace"]
    if margin is not None:
        for rec, w in zip(recs, d["weights"]):
            if rec[0] >= TAG:
                rec[1:] = [e if lg - w[-1] >= margin else -1 - e for e, lg in zip(rec[1:], w)]
    return recs, int(d.get("top_k", 10))


class Layer:
    """Recency order = stamp order; the oldest key is never from this step while slots > 3 * top_k."""

    def __init__(self, slots):
        self.slots = slots
        self.lru = OrderedDict()
        self.pending = set()

    def __contains__(self, key):
        return key in self.lru

    def touch(self, key):
        self.lru.move_to_end(key)

    def insert(self, key):
        if len(self.lru) >= self.slots:
            self.lru.popitem(last=False)
        self.lru[key] = True


def keys_of(experts):
    return [(p, e) for e in experts for p in range(3)]  # rank-major, gate / up / down


def replay(recs, slots, budget, window, base=TAG, drop=False):
    layers = {}
    hits = misses = pf_issued = pf_used = critical = 0
    pred_hit = pred_tot = 0
    last_pred = {}
    issued_for = {}
    for rec in recs:
        tag, experts = rec[0], rec[1:]
        if tag >= TAG:
            if tag // TAG * TAG != base:
                continue
            tag -= base - TAG
            lay = layers.setdefault(tag - TAG, Layer(slots))
            cand = [e for e in experts if e >= 0]  # --margin marks the others -1 - e
            experts = [e if e >= 0 else -1 - e for e in experts]
            last_pred[tag - TAG] = experts
            # Resident keys restamp before any victim is picked (both engine kernels do).
            absent = [k for k in keys_of(cand) if k not in lay]
            for key in keys_of(experts):
                if key in lay:
                    lay.touch(key)
            for key in absent[:budget]:
                lay.insert(key)
                lay.pending.add(key)
            n = min(budget, len(absent))
            pf_issued += n
            issued_for[tag - TAG] = n
            continue
        lay = layers.setdefault(tag, Layer(slots))
        if tag in last_pred:
            pred_hit += len(set(last_pred.pop(tag)) & set(experts))
            pred_tot += len(experts)
        absent = [k for k in keys_of(experts) if k not in lay]
        for key in keys_of(experts):
            if key in lay:
                hits += 1
                lay.touch(key)
                if key in lay.pending:
                    pf_used += 1
        for key in absent:
            lay.insert(key)
        m = len(absent)
        misses += m
        if drop:  # moe.expert_prefetch_abort upper bound: every unrouted prefetch frees its slot
            for key in lay.pending - set(keys_of(experts)):
                lay.lru.pop(key, None)
        lay.pending.clear()
        critical += m + max(0, issued_for.pop(tag, 0) - window)
    return hits, misses, pf_issued, pf_used, critical, pred_hit, pred_tot


def main():
    args = sys.argv[1:]
    window = 3
    if "--window" in args:
        i = args.index("--window")
        window = int(args[i + 1])
        del args[i:i + 2]
    bases = [TAG]
    if "--tag" in args:
        i = args.index("--tag")
        bases = [int(x) for x in args[i + 1].split(",")]
        del args[i:i + 2]
    margin = None
    if "--margin" in args:
        i = args.index("--margin")
        margin = float(args[i + 1])
        del args[i:i + 2]
    drop = "--drop" in args
    if drop:
        args.remove("--drop")
    path, slots = args[0], int(args[1])
    budgets = [int(x) for x in args[2:]] or [0, 4, 8, 16]
    recs, top_k = load(path, margin)
    routed = [r for r in recs if r[0] < TAG]
    layers = {r[0] for r in routed}
    tokens = len(routed) / max(1, len(layers))
    print(f"records {len(recs)} (routed {len(routed)}, predictions {len(recs) - len(routed)}), "
          f"{len(layers)} layers, ~{tokens:.0f} tokens, top_k {top_k}, {slots} slots/layer, window {window} keys, margin {margin}, drop {drop}")
    for base in bases:
        print(f"{'tag':>5} {'budget':>6} {'hit%':>6} {'miss/tok':>9} {'pf/tok':>7} {'used%':>6} "
              f"{'moved/tok':>10} {'critical/tok':>13} {'pred recall':>12}")
        for b in budgets:
            h, m, pi, pu, crit, ph, pt = replay(recs, slots, b, window, base, drop)
            print(f"{base:>5} {b:>6} {100.0 * h / (h + m):>5.1f}% {m / tokens:>9.1f} {pi / tokens:>7.1f} "
                  f"{100.0 * pu / max(1, pi):>5.1f}% {(m + pi) / tokens:>10.1f} {crit / tokens:>13.1f} "
                  f"{100.0 * ph / max(1, pt):>11.1f}%")


if __name__ == "__main__":
    main()
