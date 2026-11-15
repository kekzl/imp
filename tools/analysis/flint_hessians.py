#!/usr/bin/env python3
"""Calibration statistics for a checkpoint whose BF16 weights fit no card: input second moments
H = E[x^T x] of the Linear sites, from the BF16 CPU reference, one .pt per site in --out and, with --imphes,
imp's IMPHES01 file for `imp-quantize --gptq` (src/quant/calibration_hessian.h: "IMPHES01", u32 count, per
entry i32 layer, u32 len + kind, i64 K, u64 rows, K*K float32). Kinds: WQ (q/k/v input), WO (o_proj input),
W_GATE (gate/up input of the dense MLP, the shared expert and every routed expert), W_DOWN (dense MLP or
shared-expert down input; routed-expert down projections stay round-to-nearest).

Usage: flint_hessians.py --model BF16_DIR --text FILE --max-tokens N --out DIR [--imphes FILE]"""
import argparse
import contextlib
import io
import json
import pathlib
import re
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import flint_reference as fr  # noqa: E402

torch.set_grad_enabled(False)
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--text", required=True)
ap.add_argument("--max-tokens", type=int, default=16384)
ap.add_argument("--chunk", type=int, default=2048)
ap.add_argument("--out", required=True)
ap.add_argument("--imphes", default="")
ap.add_argument("--sites", default=r"attn_in$|o_in$|mlp_in$|\.mlp\.down_proj$|shared_expert\.down_proj$")
a = ap.parse_args()
from tokenizers import Tokenizer  # noqa: E402

cfg = json.loads((pathlib.Path(a.model) / "config.json").read_text())
ids = Tokenizer.from_file(str(pathlib.Path(a.model) / "tokenizer.json")).encode(
    pathlib.Path(a.text).read_text(), add_special_tokens=False).ids[: a.max_tokens]
W = fr.Weights(a.model)
fr.HESS = {}
fr.HESS_RX = re.compile(a.sites)
for i in range(0, len(ids), a.chunk):
    part = ids[i:i + a.chunk]
    if len(part) < 16:
        break
    with contextlib.redirect_stdout(io.StringIO()):
        fr.forward(cfg, W, torch.tensor(part))
    print(f"chunk at {i}: {len(part)} tokens, {len(fr.HESS)} sites", flush=True)
out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for k, (h, n) in fr.HESS.items():
    torch.save({"h": (h / n).to(torch.float32), "rows": n}, out / f"{k}.pt")
print(f"saved {len(fr.HESS)} Hessians, tokens {len(ids)}")

if a.imphes:
    import struct

    kinds = {"attn_in": "WQ", "o_in": "WO", "mlp_in": "W_GATE", "mlp.down_proj": "W_DOWN",
             "mlp.shared_expert.down_proj": "W_DOWN"}
    entries = []
    for k, (h, n) in fr.HESS.items():
        layer, _, site = k.partition(".")
        if site in kinds:
            entries.append((int(layer), kinds[site], (h / n).to(torch.float32).contiguous(), n))
    entries.sort(key=lambda e: (e[0], e[1]))
    with open(a.imphes, "wb") as o:
        o.write(b"IMPHES01")
        o.write(struct.pack("<I", len(entries)))
        for layer, kind, h, n in entries:
            o.write(struct.pack("<iI", layer, len(kind)))
            o.write(kind.encode())
            o.write(struct.pack("<qQ", h.shape[0], n))
            o.write(h.numpy().tobytes())
    print(f"wrote {len(entries)} IMPHES01 entries to {a.imphes}")
