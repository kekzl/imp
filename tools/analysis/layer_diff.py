#!/usr/bin/env python3
"""
Diff per-layer hidden-state tensors between imp and llama.cpp.

imp side: .npy dumps from IMP_DUMP_HIDDEN=<dir> runs.
llama.cpp side: log output from llama-eval-callback (per-tensor sum + 3+3 slice).

Computes relative sum divergence per (layer, snapshot) and identifies the
first layer where imp and llama.cpp diverge beyond threshold.

The llama.cpp sum-per-tensor is enough to spot a magnitude blowup;
full cosine-sim would need a patched eval-callback (not available in the
ghcr.io public image).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


TENSOR_HEADER_RE = re.compile(
    r"common_debug_cb_eval:\s+(\S+)\s*=\s*\(f32\)"
)
SUM_RE = re.compile(r"^\s+sum\s*=\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*$", re.IGNORECASE)


@dataclass
class LlamaTensor:
    name: str
    sum: float
    forward_idx: int   # 0 = prefill, 1 = first decode, ...


def parse_llamacpp_log(path: str) -> List[LlamaTensor]:
    """Parse llama-eval-callback stderr log into per-forward tensor sums.

    Each forward pass is identified by re-appearance of attn_out-0.
    """
    tensors: List[LlamaTensor] = []
    current_name: Optional[str] = None
    forward_idx = -1
    seen_attn0_in_current_forward = False

    with open(path, "r", errors="replace") as f:
        for line in f:
            m = TENSOR_HEADER_RE.search(line)
            if m:
                current_name = m.group(1)
                if current_name == "attn_out-0":
                    if not seen_attn0_in_current_forward:
                        forward_idx += 1
                        seen_attn0_in_current_forward = True
                continue
            m = SUM_RE.match(line)
            if m and current_name is not None:
                # sum = belongs to current_name (printed AFTER data values)
                try:
                    val = float(m.group(1))
                except ValueError:
                    continue
                tensors.append(LlamaTensor(
                    name=current_name, sum=val, forward_idx=max(forward_idx, 0)
                ))
                # reset current_name so lone "sum =" doesn't re-attribute
                current_name = None
            # new forward pass indicator: reset flag on any non-debug-cb line
            # (actually: just reset per eval based on attn_out-0 reappearance,
            # which we handle via seen_attn0_in_current_forward above).
            # To re-detect next forward, clear the flag when we pass attn_out-29
            # or any sensible boundary.
            if current_name is not None and current_name.endswith("-29"):
                seen_attn0_in_current_forward = False
    return tensors


def load_imp_dumps(dump_dir: str) -> Dict[tuple, np.ndarray]:
    """Load all .npy dumps keyed by (step, layer, snapshot)."""
    out: Dict[tuple, np.ndarray] = {}
    pattern = re.compile(r"imp_step(\d+)_L(\d+)_([ABC])_\w+\.npy")
    for fname in sorted(os.listdir(dump_dir)):
        m = pattern.match(fname)
        if not m:
            continue
        step = int(m.group(1))
        layer = int(m.group(2))
        snap = m.group(3)
        arr = np.load(os.path.join(dump_dir, fname))
        out[(step, layer, snap)] = arr
    return out


def build_imp_sums(imp: Dict[tuple, np.ndarray],
                   last_row_only_layers: set = frozenset()
                   ) -> Dict[tuple, float]:
    """Sum per (step, layer, snapshot). imp step == llama forward_idx.

    For layers in `last_row_only_layers`, sum only the last token's row.
    llama.cpp's Gemma-4 graph inserts GET_ROWS(l_out-prev, last_idx) before
    the final transformer block so L29 operates on a single token; imp still
    dumps the full prefill batch. Comparing imp's full-batch sum to llama's
    last-token sum inflates the "drift" by roughly the prompt length.
    """
    sums: Dict[tuple, float] = {}
    for key, arr in imp.items():
        layer = key[1]
        if arr.ndim == 2 and arr.shape[0] > 1 and layer in last_row_only_layers:
            sums[key] = float(arr[-1].astype(np.float64).sum())
        else:
            sums[key] = float(arr.astype(np.float64).sum())
    return sums


def build_llama_sums(
    llama: List[LlamaTensor], n_layers: int
) -> Dict[tuple, float]:
    """Sum per (forward_idx, layer, snapshot) for B (attn_out) and C (l_out)."""
    sums: Dict[tuple, float] = {}
    for t in llama:
        m = re.match(r"^(attn_out|l_out)-(\d+)$", t.name)
        if not m:
            continue
        layer = int(m.group(2))
        if layer >= n_layers:
            continue
        snap = "B" if m.group(1) == "attn_out" else "C"
        key = (t.forward_idx, layer, snap)
        # keep first occurrence per forward (there can be duplicates in log)
        if key not in sums:
            sums[key] = t.sum
    return sums


def compute_diff_table(
    imp_sums: Dict[tuple, float],
    llama_sums: Dict[tuple, float],
    n_layers: int,
    n_steps: int,
) -> List[dict]:
    rows = []
    for step in range(n_steps):
        for layer in range(n_layers):
            for snap in ("B", "C"):
                k_imp = (step, layer, snap)
                k_lc = (step, layer, snap)
                imp_s = imp_sums.get(k_imp)
                lc_s = llama_sums.get(k_lc)
                if imp_s is None or lc_s is None:
                    continue
                denom = max(abs(lc_s), 1e-6)
                rel = (imp_s - lc_s) / denom
                rows.append(dict(
                    step=step, layer=layer, snap=snap,
                    imp_sum=imp_s, lc_sum=lc_s,
                    abs_diff=imp_s - lc_s,
                    rel_diff=rel,
                ))
    return rows


def write_markdown(rows: List[dict], out_path: str, threshold: float):
    from collections import defaultdict
    by_step = defaultdict(list)
    for r in rows:
        by_step[r["step"]].append(r)

    with open(out_path, "w") as f:
        f.write("# Gemma-4 Layer-Diff: imp vs llama.cpp\n\n")
        f.write(
            "Per-tensor sum comparison. Snapshot B = `attn_out-N` / imp post-attn. "
            "Snapshot C = `l_out-N` / imp post-layer (incl. layer_out_scale).\n\n"
        )
        f.write(f"Divergence threshold: |rel_diff| > {threshold}\n\n")

        first_div = None
        for r in rows:
            if abs(r["rel_diff"]) > threshold and first_div is None:
                first_div = r

        if first_div:
            f.write(
                f"## First significant divergence\n\n"
                f"**Step {first_div['step']}, Layer {first_div['layer']}, "
                f"Snapshot {first_div['snap']}**: "
                f"imp_sum={first_div['imp_sum']:.4f}, "
                f"lc_sum={first_div['lc_sum']:.4f}, "
                f"rel_diff={first_div['rel_diff']*100:+.2f}%\n\n"
            )
        else:
            f.write("## No divergence above threshold\n\n")

        for step, step_rows in sorted(by_step.items()):
            f.write(f"## Step {step} ({'prefill' if step == 0 else f'decode {step}'})\n\n")
            f.write("| Layer | Snap | imp_sum | lc_sum | abs_diff | rel_diff |\n")
            f.write("|---|---|---|---|---|---|\n")
            for r in step_rows:
                marker = " ⚠️" if abs(r["rel_diff"]) > threshold else ""
                f.write(
                    f"| {r['layer']:2d} | {r['snap']} | "
                    f"{r['imp_sum']:+.4f} | {r['lc_sum']:+.4f} | "
                    f"{r['abs_diff']:+.4f} | {r['rel_diff']*100:+.2f}%{marker} |\n"
                )
            f.write("\n")
    print(f"[layer_diff] wrote {out_path}", file=sys.stderr)


def write_plot(rows: List[dict], out_path: str, threshold: float):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[layer_diff] matplotlib not available, skipping plot", file=sys.stderr)
        return

    from collections import defaultdict
    steps = sorted({r["step"] for r in rows})
    fig, axes = plt.subplots(len(steps), 1, figsize=(12, 3.5 * len(steps)), squeeze=False)
    for i, step in enumerate(steps):
        ax = axes[i][0]
        for snap, color in [("B", "tab:blue"), ("C", "tab:orange")]:
            xs, ys = [], []
            for r in rows:
                if r["step"] == step and r["snap"] == snap:
                    xs.append(r["layer"])
                    ys.append(abs(r["rel_diff"]) * 100)
            ax.plot(xs, ys, marker="o", label=f"Snap {snap}", color=color)
        ax.axhline(threshold * 100, linestyle="--", color="red", alpha=0.5,
                   label=f"threshold {threshold*100:.0f}%")
        ax.set_xlabel("Layer")
        ax.set_ylabel("|rel_diff| sum  (%)")
        ax.set_title(f"Step {step}: {'prefill' if step == 0 else f'decode {step}'}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"[layer_diff] wrote {out_path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imp-dir", required=True)
    ap.add_argument("--llamacpp-log", required=True)
    ap.add_argument("--out-md", default="docs/gemma4_layer_diff.md")
    ap.add_argument("--out-plot", default="docs/gemma4_layer_diff.png")
    ap.add_argument("--n-layers", type=int, default=30)
    ap.add_argument("--n-steps", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="Relative divergence threshold (fraction)")
    ap.add_argument("--last-row-only-layers", default="29",
                    help="Comma-separated layers where imp must sum its last "
                         "token only (llama.cpp GET_ROWS truncation). "
                         "Default '29' matches Gemma-4's single-token L29.")
    args = ap.parse_args()

    llama = parse_llamacpp_log(args.llamacpp_log)
    print(f"[layer_diff] parsed {len(llama)} llama.cpp tensor sums", file=sys.stderr)

    imp_raw = load_imp_dumps(args.imp_dir)
    print(f"[layer_diff] loaded {len(imp_raw)} imp dumps", file=sys.stderr)

    last_row_layers = frozenset(
        int(s) for s in args.last_row_only_layers.split(",") if s.strip()
    )
    imp_sums = build_imp_sums(imp_raw, last_row_only_layers=last_row_layers)
    llama_sums = build_llama_sums(llama, args.n_layers)

    rows = compute_diff_table(imp_sums, llama_sums, args.n_layers, args.n_steps)

    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    write_markdown(rows, args.out_md, args.threshold)
    write_plot(rows, args.out_plot, args.threshold)

    if rows:
        print(f"[layer_diff] {len(rows)} comparable (layer,snap,step) pairs")
        n_div = sum(1 for r in rows if abs(r["rel_diff"]) > args.threshold)
        print(f"[layer_diff] {n_div} above threshold ({args.threshold*100:.1f}%)")


if __name__ == "__main__":
    main()
