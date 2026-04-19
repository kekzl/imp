#!/usr/bin/env python3
"""Compare imp's FP32 shadow vs FP16 view vs llama.cpp l_out-N per layer.

If fp32_shadow sums are closer to llama.cpp than the FP16 h sums, we prove
the drift happens during FP16 downcast. If both drift identically, the
problem is deeper (inside FP32 accumulator math).
"""
import argparse
import os
import re
import sys
import numpy as np

TENSOR_HEADER_RE = re.compile(r"common_debug_cb_eval:\s+(\S+)\s*=\s*\(f32\)")
SUM_RE = re.compile(r"^\s+sum\s*=\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*$", re.IGNORECASE)


def parse_llamacpp_l_out(path):
    """forward_idx -> layer -> sum(l_out-N)"""
    results = {}
    current_name = None
    forward_idx = -1
    seen_0 = False
    with open(path) as f:
        for line in f:
            m = TENSOR_HEADER_RE.search(line)
            if m:
                current_name = m.group(1)
                if current_name == "attn_out-0":
                    if not seen_0:
                        forward_idx += 1
                        seen_0 = True
                continue
            m = SUM_RE.match(line)
            if m and current_name:
                nm = re.match(r"^l_out-(\d+)$", current_name)
                if nm:
                    layer = int(nm.group(1))
                    results.setdefault(forward_idx, {}).setdefault(layer, float(m.group(1)))
                if current_name.endswith("-29"):
                    seen_0 = False
                current_name = None
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imp-dir", required=True)
    ap.add_argument("--llamacpp-log", required=True)
    ap.add_argument("--last-row-only-layers", default="29",
                    help="Layers where imp must sum its last token only "
                         "(llama.cpp GET_ROWS truncates to 1 token).")
    args = ap.parse_args()

    last_row_layers = {
        int(s) for s in args.last_row_only_layers.split(",") if s.strip()
    }

    lc = parse_llamacpp_l_out(args.llamacpp_log)
    print(f"llama.cpp: {sum(len(v) for v in lc.values())} l_out sums parsed", file=sys.stderr)

    print(f"{'step':>4} {'layer':>5} {'imp_h_sum':>14} {'imp_fp32_sum':>14} "
          f"{'lc_sum':>14} {'h_vs_lc':>9} {'fp32_vs_lc':>10}")

    for step in (0, 1):
        for layer in range(30):
            h_path = os.path.join(args.imp_dir,
                                   f"imp_step{step:02d}_L{layer:02d}_C_post_layer.npy")
            fp32_path = os.path.join(args.imp_dir,
                                      f"imp_step{step:02d}_L{layer:02d}_C_fp32_shadow.npy")
            if not os.path.exists(h_path) or not os.path.exists(fp32_path):
                continue
            if step not in lc or layer not in lc[step]:
                continue
            h = np.load(h_path)
            fp32 = np.load(fp32_path)
            if layer in last_row_layers and h.ndim == 2 and h.shape[0] > 1:
                h_sum = float(h[-1].sum())
                fp32_sum = float(fp32[-1].sum())
            else:
                h_sum = float(h.sum())
                fp32_sum = float(fp32.sum())
            lc_sum = lc[step][layer]
            denom = max(abs(lc_sum), 1e-6)
            h_rel = (h_sum - lc_sum) / denom * 100
            f_rel = (fp32_sum - lc_sum) / denom * 100
            print(f"{step:>4} {layer:>5} {h_sum:>+14.4f} {fp32_sum:>+14.4f} "
                  f"{lc_sum:>+14.4f} {h_rel:>+8.2f}% {f_rel:>+9.2f}%")


if __name__ == "__main__":
    main()
