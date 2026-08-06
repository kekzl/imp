#!/usr/bin/env python3
"""
Per-layer, per-snapshot divergence between two runs of the SAME architecture.

Built for #1273 (hybrid GDN + NVFP4 degrades), where the question was not "is
this model worse" — perplexity answers that — but "WHICH BLOCK makes it worse".

imp writes three snapshots per layer when `diagnostics.dump_hidden_dir` is set:

    A_pre_attn    hidden state entering the attention/GDN block
    B_post_attn   after that block, residual added (input to FFN)
    C_post_layer  after the FFN, i.e. the layer output

Diffing A against B isolates what the attention block ITSELF contributes,
independently of the error it inherited. That distinction is what settled #1273:
the attention blocks add divergence (median +0.0156 per block) while the GDN
blocks are slightly corrective (-0.0017), which rules out "softmax amplifies an
error created elsewhere" — an amplifier cannot show a clean input and a corrupt
output.

Usage:
    # 1. dump both runs (PREFILL ONLY — see the warning below)
    imp-cli --model <A> --prompt "Hello world" --max-tokens 1 --temperature 0 \
            --set diagnostics.dump_hidden_dir=all      # writes to /tmp
    imp-cli --model <B> ... same, different output dir

    # 2. compare
    tools/analysis/layer_ab_diff.py <dirA> <dirB> [--config <model>/config.json]

WARNING — decode dumps are useless here. Decode is captured in a CUDA graph, so
the host-side copy sees only the final buffer state and EVERY layer's file is
identical (this cost an hour: RMS matching to four decimals across 40 layers).
Use step00, the prefill pass. The tool selects it by default.

No numpy: the containers that run imp do not ship it, and needing a second image
to read imp's own diagnostics defeats the point.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import struct
import sys


def load_npy(path: str, cap: int = 20000) -> list[float]:
    """Minimal .npy v1.0 reader for the 2-D FP32 arrays imp writes."""
    with open(path, "rb") as f:
        if f.read(6) != b"\x93NUMPY":
            raise ValueError(f"not a .npy file: {path}")
        f.read(2)  # version
        header_len = struct.unpack("<H", f.read(2))[0]
        header = f.read(header_len).decode()
        m = re.search(r"'shape':\s*\(([^)]*)\)", header)
        dims = [int(x) for x in m.group(1).replace(" ", "").rstrip(",").split(",") if x]
        total = 1
        for d in dims:
            total *= d
        raw = f.read()
    # Subsample rather than load whole tensors: we want norms, not exactness,
    # and a 40-layer x 3-snapshot x 2-model sweep is 240 files.
    stride = max(1, total // cap)
    return [struct.unpack_from("<f", raw, i * 4)[0] for i in range(0, total, stride)]


def rel_err(a: list[float], b: list[float]) -> float:
    """||a-b|| / ||a||, i.e. divergence of a from b relative to a's magnitude."""
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))) / na


def cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def layer_types(config_path: str | None, n_layers: int) -> list[str]:
    if not config_path or not os.path.exists(config_path):
        return ["?"] * n_layers
    with open(config_path) as f:
        cfg = json.load(f)
    cfg = cfg.get("text_config", cfg)
    lt = cfg.get("layer_types")
    return lt if lt else ["?"] * n_layers


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir_a", help="dump dir of the run under test (e.g. the NVFP4 checkpoint)")
    p.add_argument("dir_b", help="dump dir of the reference run (e.g. its GGUF twin)")
    p.add_argument("--config", help="config.json of either model, for layer_types labels")
    p.add_argument("--step", default="00", help="which forward step (default 00 = prefill)")
    p.add_argument("--cap", type=int, default=20000, help="max elements sampled per tensor")
    args = p.parse_args()

    pattern = os.path.join(args.dir_a, f"imp_step{args.step}_L*_A_pre_attn.npy")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"no snapshots matching {pattern}", file=sys.stderr)
        print("did the run use --set diagnostics.dump_hidden_dir=all ?", file=sys.stderr)
        return 2

    types = layer_types(args.config, len(files))
    rows = []
    for fa in files:
        layer = int(re.search(r"_L(\d+)_", fa).group(1))
        try:
            a_in = load_npy(fa, args.cap)
            b_in = load_npy(fa.replace(args.dir_a, args.dir_b), args.cap)
            fb = fa.replace("_A_pre_attn", "_B_post_attn")
            a_out = load_npy(fb, args.cap)
            b_out = load_npy(fb.replace(args.dir_a, args.dir_b), args.cap)
        except (FileNotFoundError, ValueError) as e:
            print(f"  L{layer}: skipped ({e})", file=sys.stderr)
            continue
        if not (len(a_in) == len(b_in) and len(a_out) == len(b_out)):
            print(f"  L{layer}: skipped (shape mismatch — same architecture?)", file=sys.stderr)
            continue
        ri, ro = rel_err(a_in, b_in), rel_err(a_out, b_out)
        rows.append((layer, types[layer] if layer < len(types) else "?", ri, ro, ro - ri,
                     cosine(a_out, b_out)))

    if not rows:
        print("no comparable layers — are both dirs from the same architecture?", file=sys.stderr)
        return 2

    print(f"{'L':>3} {'block type':>18} {'rel@in':>9} {'rel@out':>9} {'added':>9} {'cos@out':>9}")
    for layer, t, ri, ro, d, c in rows:
        mark = "   <== block injects" if d > 0.02 else ""
        print(f"{layer:3d} {t:>18} {ri:9.4f} {ro:9.4f} {d:+9.4f} {c:9.5f}{mark}")

    by_type: dict[str, list[float]] = {}
    for _, t, _, _, d, _ in rows:
        by_type.setdefault(t, []).append(d)
    print("\nDivergence ADDED by the block itself (rel@out - rel@in):")
    for t, vals in sorted(by_type.items()):
        vals.sort()
        # True median: average the middle pair on even counts. Using the upper
        # element instead reported +0.0285 where the real figure is +0.0156 —
        # same conclusion, wrong number, and the number is what gets quoted.
        mid = len(vals) // 2
        median = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2
        print(f"  {t:>18} (n={len(vals):3d}): median {median:+.4f}   max {max(vals):+.4f}")
    print("\nA block type with a positive median CREATES divergence; near-zero or")
    print("negative means it carries the inherited error without adding to it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
