#!/usr/bin/env python3
"""Expert-pruning coverage from diagnostics.moe_expert_trace files (#2613 step 1).

Usage: expert_prune_sim.py trace.json [trace.json ...] [--keep 448,384,256,128] [--slots 355]
                           [--experts 512] [--rank count|weight]

Keep set per layer = the k experts with the most picks (count) or the most routing weight
(weight, needs a trace with "weights"). Held-out = leave-one-trace-out: rank on the other
traces, score on the left-out one.
cov: share of top_k picks inside the keep set. mass: share of routing weight inside it.
full: share of (token, layer) records whose top_k picks are all kept. hit: LRU hit rate of
`slots` keys/layer (3 keys per expert) on kept picks only; dropped picks would reroute,
which the trace cannot replay.
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expert_cache_sim import records, sim  # noqa: E402


def read(path):
    """[(layer, experts, weights)]; weights are 1.0 when the trace predates them."""
    with open(path) as f:
        d = json.load(f)
    top_k = int(d.get("top_k", 10))
    recs = list(records(d["trace"], top_k))
    ws = d.get("weights") or [[1.0] * len(e) for _, e in recs]
    return [(l, e, [float(x) for x in w]) for (l, e), w in zip(recs, ws)], "weights" in d


def rank(recs_list, n_experts, by):
    """Hottest first; experts never routed in the calibration traces pad the tail by id."""
    freq = {}
    for recs in recs_list:
        for layer, experts, weights in recs:
            c = freq.setdefault(layer, Counter())
            for e, w in zip(experts, weights):
                c[e] += w if by == "weight" else 1
    out = {}
    for l, c in freq.items():
        seen = [e for e, _ in c.most_common()]
        out[l] = seen + sorted(set(range(n_experts)) - set(seen))
    return out


def score(recs, ranked, k, slots):
    keep = {l: set(r[:k]) for l, r in ranked.items()}
    picks = kept = full = 0
    mass = mass_kept = 0.0
    filtered = []
    for layer, experts, weights in recs:
        ks = keep.get(layer, set())
        inside = [e for e in experts if e in ks]
        picks += len(experts)
        kept += len(inside)
        full += len(inside) == len(experts)
        mass += sum(weights)
        mass_kept += sum(w for e, w in zip(experts, weights) if e in ks)
        filtered.append((layer, inside))
    h, m = sim(filtered, slots, "lru")
    return kept / picks, mass_kept / mass, full / len(recs), h / max(1, h + m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--keep", default="448,384,288,256,128")
    ap.add_argument("--slots", type=int, default=355)
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--rank", choices=("count", "weight"), default="count")
    a = ap.parse_args()
    keeps = [int(x) for x in a.keep.split(",")]
    data, weighted = {}, True
    for p in a.traces:
        data[os.path.basename(p)], w = read(p)
        weighted &= w
    if a.rank == "weight" and not weighted:
        sys.exit("--rank weight needs traces with a \"weights\" array")
    n_layers = len({l for recs in data.values() for l, _, _ in recs})
    print(f"traces {len(data)}  records {sum(len(r) for r in data.values())}  layers {n_layers}  "
          f"slots {a.slots}  rank {a.rank}  weights {'yes' if weighted else 'no'}")

    touched = {}
    for recs in data.values():
        for layer, experts, _ in recs:
            touched.setdefault(layer, set()).update(experts)
    distinct = sorted(len(s) for s in touched.values())
    print(f"distinct experts touched per layer: min {distinct[0]}  median {distinct[len(distinct) // 2]}  max {distinct[-1]}")

    ranked_all = rank(data.values(), a.experts, a.rank)
    print(f"{'keep':>5}  {'cov in':>7}  {'cov out':>7}  {'mass in':>7}  {'mass out':>8}  {'full out':>8}  "
          f"{'hit out':>7}  worst trace (mass out)")
    for k in keeps:
        ins = [score(r, ranked_all, k, a.slots) for r in data.values()]
        outs = {}
        for name, recs in data.items():
            ranked = rank([r for n, r in data.items() if n != name], a.experts, a.rank)
            outs[name] = score(recs, ranked, k, a.slots)

        def mean(rows, i):
            return 100 * sum(r[i] for r in rows) / len(rows)

        o = list(outs.values())
        worst = min(outs, key=lambda n: outs[n][1])
        print(f"{k:>5}  {mean(ins, 0):>6.1f}%  {mean(o, 0):>6.1f}%  {mean(ins, 1):>6.1f}%  {mean(o, 1):>7.1f}%  "
              f"{mean(o, 2):>7.1f}%  {mean(o, 3):>6.1f}%  {worst} {100 * outs[worst][1]:.1f}%")


if __name__ == "__main__":
    main()
