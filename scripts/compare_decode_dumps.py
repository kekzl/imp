#!/usr/bin/env python3
"""Compare imp vs llama.cpp per-layer hidden states across decode steps.

Usage:
  python3 scripts/compare_decode_dumps.py /tmp/imp_decode_drift.log /tmp/llama_decode_drift.log
"""
import sys
import re
from collections import defaultdict

def parse_imp(path):
    """Parse imp [step=N] l_out-L[0] L2=X lines."""
    data = {}
    with open(path) as f:
        for line in f:
            m = re.search(r'\[step=(\d+)\]\s+l_out-(\d+)\[(\d+)\]\s+L2=([\d.]+)', line)
            if m:
                step, layer, row, l2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))
                # For prefill (step=0), take last row. For decode (step>0), row=0.
                key = (step, layer)
                data[key] = l2  # last row wins for prefill
    return data

def parse_llama(path):
    """Parse llama [DUMP] l_out-L ... L2=X lines. Sequential evals."""
    data = {}
    eval_counts = defaultdict(int)  # count how many times each layer appears
    with open(path) as f:
        for line in f:
            m = re.search(r'\[DUMP\]\s+l_out-(\d+)\s+.*?L2=([\d.]+)', line)
            if m:
                layer = int(m.group(1))
                l2 = float(m.group(2))
                eval_idx = eval_counts[layer]
                eval_counts[layer] += 1
                data[(eval_idx, layer)] = l2
    return data

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <imp_log> <llama_log>")
        sys.exit(1)

    imp = parse_imp(sys.argv[1])
    llama = parse_llama(sys.argv[2])

    # Auto-align: find which llama eval index corresponds to imp step 0
    # by matching L0 L2 values
    imp_l0 = imp.get((0, 0), 0)
    best_offset = 0
    best_diff = float('inf')
    for (e, l), l2 in llama.items():
        if l == 0:
            diff = abs(imp_l0 - l2) / max(imp_l0, 1e-6)
            if diff < best_diff:
                best_diff = diff
                best_offset = e

    print(f"imp L0 step=0: L2={imp_l0:.2f}")
    print(f"llama best match: eval {best_offset} (L0 L2 diff: {best_diff*100:.1f}%)")
    print()

    # Determine available steps
    imp_steps = sorted(set(s for s, l in imp.keys()))
    max_step = min(20, max(imp_steps) + 1) if imp_steps else 0

    layers = list(range(0, 30, 5)) + [29]  # [0, 5, 10, 15, 20, 25, 29]

    # Header
    print(f"{'step':>5}", end="")
    for l in layers:
        print(f"  L{l:02d}%", end="")
    print("  | worst")
    print("-" * (6 + 7 * len(layers) + 12))

    for s in range(max_step):
        print(f"{s:5d}", end="")
        worst_err = 0
        worst_layer = -1
        for l in layers:
            imp_l2 = imp.get((s, l), 0)
            llama_l2 = llama.get((s + best_offset, l), 0)
            if llama_l2 > 0.1:
                rel_err = abs(imp_l2 - llama_l2) / llama_l2
            elif imp_l2 > 0.1:
                rel_err = 1.0
            else:
                rel_err = 0
            marker = "*" if rel_err > 0.10 else " "
            print(f"  {rel_err*100:4.1f}{marker}", end="")
            if rel_err > worst_err:
                worst_err = rel_err
                worst_layer = l
        status = f"L{worst_layer}={worst_err*100:.0f}%" if worst_err > 0.05 else "ok"
        print(f"  | {status}")

    print()
    print("Legend: % = relative L2 error | * = >10%")

if __name__ == "__main__":
    main()
