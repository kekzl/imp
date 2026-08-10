#!/usr/bin/env python3
"""Turn a MoE expert-activation histogram into the number the roadmap needs.

docs/roadmap.md ("CPU-resident cold experts") and docs/GOAL.md both gate that
idea on a measured budget rather than on how attractive it sounds. The
bandwidth half is measurable directly; the half that was missing is ROUTING
SKEW: if VRAM holds fraction f of each layer's experts, what share of the
token's expert activations actually land on resident ones?

A flat router means coverage(f) = f and the host has to stream (1-f) of the
active expert weights every token. A skewed router means a small resident set
absorbs most activations and the host streams far less. The difference decides
whether an 80B-120B MoE lands at a usable tok/s or not, and it is a property of
the model, not of imp.

Record the histogram (needs a GPU and a MoE model):

    imp-cli --model /models/<moe>.gguf \\
        --set diagnostics.moe_expert_hist=/tmp/moe_hist.json \\
        --prompt "..." --max-tokens 512 --temperature 0

Then:

    python3 tools/analysis/moe_routing_skew.py /tmp/moe_hist.json

Self-test (no GPU, no histogram needed):

    python3 tools/analysis/moe_routing_skew.py --self-test
"""

import argparse
import json
import math
import sys

# Measured on this host 2026-08-10: streaming read, 16 threads, 24 GiB buffer,
# 512-bit non-temporal loads, three runs within 0.2%. See docs/roadmap.md.
DEFAULT_HOST_BW_GBS = 62.5

# docs/roadmap.md: each H2D transfer blocks the host ~165 us on this WSL2/WDDM
# box, regardless of size. A host-side expert needs the token's activations over
# and its output back, so two per MoE layer per token.
DEFAULT_TRANSFER_US = 165.0


def coverage_curve(counts):
    """Share of activations covered when the top-k experts of EACH layer are resident.

    Residency is decided per layer — expert 7 of layer 3 and expert 7 of layer 4
    are different tensors — so the top-k is taken per layer and then summed.
    Returns a list `cov` of length n_experts+1 where cov[k] is the covered share
    for "k experts per layer resident"; cov[0] == 0.0 and cov[n_experts] == 1.0.
    """
    if not counts:
        return [0.0]
    n_experts = len(counts[0])
    total = sum(sum(layer) for layer in counts)
    if total == 0:
        return None  # caller reports this; a zero histogram is a failed run
    covered = [0] * (n_experts + 1)
    for layer in counts:
        ordered = sorted(layer, reverse=True)
        run = 0
        for k in range(1, n_experts + 1):
            run += ordered[k - 1]
            covered[k] += run
    return [c / total for c in covered]


def coverage_at(cov, frac):
    """Coverage when a `frac` fraction of each layer's experts is resident.

    ceil() rather than round(): holding 40% of 128 experts means 52 of them, and
    a resident set is a count, not a ratio.
    """
    n_experts = len(cov) - 1
    k = min(n_experts, math.ceil(frac * n_experts))
    return cov[k], k


def budget(cold_share, active_routed_b, bytes_per_param, n_moe_layers,
           bw_gbs=DEFAULT_HOST_BW_GBS, transfer_us=DEFAULT_TRANSFER_US):
    """Per-token host cost, split into its two terms.

    Both matter: on the numbers in the roadmap they are the same order, so an
    optimisation at one end alone changes little.
    """
    cold_bytes = cold_share * active_routed_b * 1e9 * bytes_per_param
    bw_ms = cold_bytes / (bw_gbs * 1e9) * 1e3
    lat_ms = n_moe_layers * 2 * transfer_us / 1e3
    total_ms = bw_ms + lat_ms
    return {
        "cold_gb": cold_bytes / 1e9,
        "bandwidth_ms": bw_ms,
        "latency_ms": lat_ms,
        "total_ms": total_ms,
        "ceiling_tok_s": (1e3 / total_ms) if total_ms > 0 else float("inf"),
    }


def load_and_sum(paths):
    """Sum several runs into one workload.

    One prompt is one trajectory's expert taste; the question is about a
    workload, so the runs are added. Shapes must agree — summing histograms of
    different models would produce a plausible-looking average of nothing.
    """
    total = None
    for path in paths:
        with open(path) as f:
            h = json.load(f)
        if total is None:
            total = h
            continue
        if (h["n_layers"], h["n_experts"]) != (total["n_layers"], total["n_experts"]):
            raise SystemExit(
                f"refusing to sum {path}: {h['n_layers']}x{h['n_experts']} vs "
                f"{total['n_layers']}x{total['n_experts']} — different models")
        for l in range(total["n_layers"]):
            row, add = total["counts"][l], h["counts"][l]
            for e in range(total["n_experts"]):
                row[e] += add[e]
        total["total_activations"] = total.get("total_activations", 0) + h.get("total_activations", 0)
    return total


def report(hist, args):
    counts = hist["counts"]
    n_layers = hist["n_layers"]
    n_experts = hist["n_experts"]
    total = hist.get("total_activations", sum(sum(l) for l in counts))

    print(f"histogram: {n_layers} layers x {n_experts} experts, "
          f"top_k={hist.get('top_k', '?')}, {total} activations")

    if total == 0:
        print("\nERROR: the histogram is empty — nothing was recorded.")
        print("  Either the model has no MoE layers, or the run produced no tokens.")
        print("  A zero histogram is NOT a flat distribution; refusing to report skew.")
        return 1

    active_layers = sum(1 for l in counts if sum(l) > 0)
    if active_layers != n_layers:
        print(f"  ({active_layers} of {n_layers} layers routed — the rest are dense/attention-only)")

    cov = coverage_curve(counts)

    print("\nrouting skew — activations covered by the top experts of each layer")
    print(f"  {'resident':>9}  {'experts/layer':>13}  {'coverage':>9}  {'vs flat':>8}")
    for frac in (0.05, 0.10, 0.25, 0.40, 0.50, 0.75):
        c, k = coverage_at(cov, frac)
        print(f"  {frac*100:>8.0f}%  {k:>13}  {c*100:>8.1f}%  {c/frac:>7.2f}x")

    # The roadmap's own budget assumed proportional hits (coverage == residency).
    # Anything above 1.00x is what skew buys; at 1.00x the entry's arithmetic stands.
    resident = args.resident_frac
    c, k = coverage_at(cov, resident)
    cold_share = 1.0 - c

    print(f"\nbudget at {resident*100:.0f}% of experts resident "
          f"({k}/{n_experts} per layer), {args.active_routed_b}B active routed params, "
          f"{args.bytes_per_param} B/param")
    b_skew = budget(cold_share, args.active_routed_b, args.bytes_per_param,
                    active_layers, args.bandwidth, args.transfer_us)
    b_flat = budget(1.0 - resident, args.active_routed_b, args.bytes_per_param,
                    active_layers, args.bandwidth, args.transfer_us)
    for name, b, share in (("measured skew", b_skew, cold_share),
                           ("flat assumption", b_flat, 1.0 - resident)):
        print(f"  {name:>16}: cold {share*100:>5.1f}% = {b['cold_gb']:.2f} GB/token"
              f" -> {b['bandwidth_ms']:.1f} ms bandwidth + {b['latency_ms']:.1f} ms transfer"
              f" = {b['total_ms']:.1f} ms  ->  {b['ceiling_tok_s']:.0f} tok/s ceiling")

    print(f"\n  host bandwidth {args.bandwidth} GB/s, {args.transfer_us} us per blocking transfer,"
          f" 2 per MoE layer")
    print("  This is a CEILING: it assumes the host half is perfectly overlapped, ignores")
    print("  dequant and FMA cost, and assumes residency can actually be chosen by frequency.")
    return 0


def self_test():
    """The two distributions whose answers are known, plus the empty case."""
    ok = True

    # Flat router: coverage must equal residency, so skew buys exactly nothing.
    flat = [[10] * 8 for _ in range(4)]
    cov = coverage_curve(flat)
    for frac in (0.25, 0.5, 0.75):
        c, _ = coverage_at(cov, frac)
        if abs(c - frac) > 1e-9:
            print(f"FAIL flat: coverage({frac}) = {c}, expected {frac}")
            ok = False

    # Fully concentrated: one expert per layer takes everything, so ANY nonzero
    # resident set covers 100%.
    spike = [[100] + [0] * 7 for _ in range(4)]
    cov = coverage_curve(spike)
    c, k = coverage_at(cov, 0.125)  # 1 of 8
    if k != 1 or abs(c - 1.0) > 1e-9:
        print(f"FAIL spike: coverage(1/8) = {c} with k={k}, expected 1.0 with k=1")
        ok = False

    # Per-layer residency, not global: layer A concentrates on expert 0 and
    # layer B on expert 7. A global top-1 would cover only half; the per-layer
    # top-1 covers everything. This is the bug the curve must not have.
    split = [[100, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 100]]
    cov = coverage_curve(split)
    c, _ = coverage_at(cov, 0.125)
    if abs(c - 1.0) > 1e-9:
        print(f"FAIL per-layer: coverage(1/8) = {c}, expected 1.0 "
              f"(a global top-k would give 0.5)")
        ok = False

    # ceil(), not round(): 0.4 of 128 experts is 52, never 51.
    cov128 = coverage_curve([[1] * 128])
    _, k = coverage_at(cov128, 0.40)
    if k != 52:
        print(f"FAIL ceil: 40% of 128 experts = {k}, expected 52")
        ok = False

    # Summing must add counts, not average or overwrite them.
    import tempfile, os
    a = {"n_layers": 1, "n_experts": 4, "top_k": 1, "total_activations": 4, "counts": [[4, 0, 0, 0]]}
    b = {"n_layers": 1, "n_experts": 4, "top_k": 1, "total_activations": 4, "counts": [[0, 0, 0, 4]]}
    paths = []
    for d in (a, b):
        fd, pth = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as fh:
            json.dump(d, fh)
        paths.append(pth)
    summed = load_and_sum(paths)
    for pth in paths:
        os.unlink(pth)
    if summed["counts"][0] != [4, 0, 0, 4] or summed["total_activations"] != 8:
        print(f"FAIL sum: {summed['counts'][0]}, total {summed['total_activations']}")
        ok = False
    # ... and the summed histogram must NOT read as concentrated: two runs that
    # each spike a different expert are a flat pair together.
    cov_sum = coverage_curve(summed["counts"])
    c, _ = coverage_at(cov_sum, 0.25)
    if abs(c - 0.5) > 1e-9:
        print(f"FAIL sum-coverage: {c}, expected 0.5")
        ok = False

    # An empty histogram must be refused, not reported as flat.
    if coverage_curve([[0] * 8 for _ in range(4)]) is not None:
        print("FAIL empty: a zero histogram must return None, not a curve")
        ok = False

    # The budget's two terms must both be present and both scale as stated.
    b = budget(0.5, 4.3, 0.53, 36, bw_gbs=62.5, transfer_us=165.0)
    if not (b["bandwidth_ms"] > 0 and abs(b["latency_ms"] - 11.88) < 0.01):
        print(f"FAIL budget: {b}")
        ok = False
    b2 = budget(0.25, 4.3, 0.53, 36, bw_gbs=62.5, transfer_us=165.0)
    if abs(b2["bandwidth_ms"] - b["bandwidth_ms"] / 2) > 1e-6:
        print("FAIL budget: bandwidth term must be linear in the cold share")
        ok = False

    print("self-test: PASS" if ok else "self-test: FAIL")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("histogram", nargs="*",
                   help="one or more JSONs written by diagnostics.moe_expert_hist; "
                        "several are summed into one workload")
    p.add_argument("--self-test", action="store_true", help="check the analysis on known inputs")
    p.add_argument("--resident-frac", type=float, default=0.40,
                   help="fraction of each layer's experts held in VRAM (default 0.40)")
    p.add_argument("--active-routed-b", type=float, default=4.3,
                   help="billions of routed-expert params active per token (default 4.3, "
                        "a 120B-A5B class model)")
    p.add_argument("--bytes-per-param", type=float, default=0.53,
                   help="stored bytes per param (default 0.53 = MXFP4 + scales)")
    p.add_argument("--bandwidth", type=float, default=DEFAULT_HOST_BW_GBS,
                   help=f"host streaming read GB/s (default {DEFAULT_HOST_BW_GBS}, measured)")
    p.add_argument("--transfer-us", type=float, default=DEFAULT_TRANSFER_US,
                   help=f"blocking cost per H2D transfer (default {DEFAULT_TRANSFER_US})")
    args = p.parse_args()

    if args.self_test:
        return self_test()
    if not args.histogram:
        p.error("give a histogram path, or --self-test")
    hist = load_and_sum(args.histogram)
    return report(hist, args)


if __name__ == "__main__":
    sys.exit(main())
