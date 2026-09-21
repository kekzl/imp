#!/usr/bin/env python3
"""Replay a diagnostics.moe_expert_trace against expert-cache policies.

Usage: cachesim.py trace.json [slots_per_layer ...]

The trace is decode-only, one record per (token, layer): [layer, e0..e_{top_k-1}].
A key in the device cache is (projection, expert), three per selected expert, and all
three share a slot budget, so the simulation counts 3 * top_k keys per record.
"""
import json
import sys
from collections import Counter, OrderedDict


def load(path):
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, dict):
        for k in ("trace", "entries", "data"):
            if k in d and isinstance(d[k], list):
                return d[k], d
    return d, {}


def records(raw, top_k):
    """Yield (layer, [experts]) from either list-of-lists or a flat int stream."""
    if raw and isinstance(raw[0], (list, tuple)):
        for r in raw:
            yield int(r[0]), [int(x) for x in r[1:]]
        return
    stride = top_k + 1
    for i in range(0, len(raw) - stride + 1, stride):
        yield int(raw[i]), [int(x) for x in raw[i + 1:i + stride]]


def sim(recs, slots, policy, hot=None):
    """Per-layer cache of `slots` entries. Returns (hits, misses)."""
    lru = {}      # layer -> OrderedDict(key -> True)
    pinned = {}   # layer -> set(key)
    hits = misses = 0
    for layer, experts in recs:
        c = lru.setdefault(layer, OrderedDict())
        if policy == "pin_hot" and layer not in pinned:
            p = set()
            for e in (hot.get(layer) or [])[: slots // 2]:
                for proj in range(3):
                    p.add((proj, e))
            # keep at most half the budget pinned
            pinned[layer] = set(list(p)[: slots // 2])
        pin = pinned.get(layer, set())
        for e in experts:
            for proj in range(3):
                key = (proj, e)
                if key in pin or key in c:
                    hits += 1
                    if key in c:
                        c.move_to_end(key)
                    continue
                misses += 1
                c[key] = True
                while len(c) + len(pin) > slots:
                    c.popitem(last=False)
    return hits, misses


def main():
    path = sys.argv[1]
    budgets = [int(x) for x in sys.argv[2:]] or [186]
    raw, meta = load(path)
    top_k = int(meta.get("top_k", 10))
    recs = list(records(raw, top_k))
    if not recs:
        print("no records")
        return
    layers = sorted({l for l, _ in recs})
    tokens = len(recs) // max(1, len(layers))
    print(f"records {len(recs)}  layers {len(layers)}  ~tokens {tokens}  top_k {top_k}")

    # per-layer expert frequency, hottest first
    freq = {}
    for layer, experts in recs:
        c = freq.setdefault(layer, Counter())
        c.update(experts)
    hot = {l: [e for e, _ in c.most_common()] for l, c in freq.items()}

    l0 = layers[0]
    c0 = freq[l0]
    tot = sum(c0.values())
    run = 0
    marks = []
    for i, (_, n) in enumerate(c0.most_common(), 1):
        run += n
        if i in (16, 32, 64, 128, 256):
            marks.append(f"top{i} {100.0 * run / tot:.0f}%")
    print(f"layer {l0}: {len(c0)} distinct experts, skew: " + "  ".join(marks))

    print(f"{'slots':>6}  {'LRU':>8}  {'pin_hot':>8}")
    for s in budgets:
        out = []
        for pol in ("lru", "pin_hot"):
            h, m = sim(recs, s, pol, hot)
            out.append(100.0 * h / (h + m))
        print(f"{s:>6}  {out[0]:>7.1f}%  {out[1]:>7.1f}%")


if __name__ == "__main__":
    main()
