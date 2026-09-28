#!/usr/bin/env python3
"""First bit-level difference between two imp hidden-state dump dirs (same model, same request).

Dumps come from `diagnostics.dump_hidden_dir` (`imp_step<step>_L<layer>_<tag>.npy`). Tensors are
compared in execution order (step, layer, stage); the first one that is not bit-identical owns the
divergence, since its inputs (the tensors before it) were identical. Built for #2168 (two server
processes, same request, different result).

usage: dump_first_diff.py DIR_A DIR_B [--all]   (--all lists every differing tensor, not only the first)
"""
import argparse
import os
import re
import sys

import numpy as np

# Stage order within a layer; unknown tags sort after the known ones, by name.
STAGES = ["A_pre_attn", "gdn_ssm_in_out", "gdn_conv_f32", "gdn_alpha", "gdn_beta", "gdn_y_post_scan",
          "gdn_y_post_norm", "gdn_linear_attn_out", "B_post_attn", "C_post_layer", "C_fp32_shadow"]
PAT = re.compile(r"imp_step(\d+)_L(\d+)_(\w+)\.npy$")


def index(d):
    out = {}
    for f in os.listdir(d):
        m = PAT.match(f)
        if m:
            out[(int(m.group(1)), int(m.group(2)), m.group(3))] = os.path.join(d, f)
    return out


def order(key):
    step, layer, tag = key
    return (step, layer, STAGES.index(tag) if tag in STAGES else len(STAGES), tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    a, b = index(args.dir_a), index(args.dir_b)
    common = sorted(set(a) & set(b), key=order)
    only = len(set(a) ^ set(b))
    if not common:
        print("no common dump files")
        return 2
    print(f"{len(common)} common tensors, {only} present in one dir only")
    found = 0
    for key in common:
        x, y = np.load(a[key]), np.load(b[key])
        if x.shape != y.shape:
            print(f"step{key[0]:02d} L{key[1]:02d} {key[2]}: shape {x.shape} vs {y.shape}")
            found += 1
        elif not np.array_equal(x.view(np.uint32), y.view(np.uint32)):
            d = np.abs(x.astype(np.float64) - y.astype(np.float64))
            rows = np.nonzero(d.reshape(d.shape[0], -1).max(axis=1) > 0)[0] if d.ndim > 1 else np.array([0])
            scale = max(float(np.abs(x).max()), 1e-12)
            print(f"step{key[0]:02d} L{key[1]:02d} {key[2]}: {int((d > 0).sum())}/{d.size} elements differ, "
                  f"rel max {d.max() / scale:.3e}, rows {rows[:8].tolist()}{'...' if rows.size > 8 else ''} "
                  f"({rows.size} of {d.shape[0]})")
            found += 1
        if found and not args.all:
            break
    if not found:
        print("all common tensors bit-identical")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
