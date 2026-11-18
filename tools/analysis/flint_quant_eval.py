#!/usr/bin/env python3
"""Quantization experiments on the Flint CPU reference: quantize selected BF16 tensors on the fly (NVFP4:
tensor scale absmax/6, E4M3 block scale per 16, E2M1 values; FP8; GPTQ onto the NVFP4 grid) or score a
ModelOpt NVFP4 checkpoint (--ckpt NAME:DIR), each against the BF16 forward: KL, NLL difference with
standard errors, PPL, argmax agreement. Record: docs/audit/PERF_LOG.md 2026-11-15 (Flint).

Usage: quant_exp.py --model BF16_DIR --ids FILE --max-tokens N --arm NAME:REGEX:MODE[:ARG] [--arm ...]
MODE: minmax (block scale = block absmax / 6, what imp-quantize writes without calibration)
      mse    (block scale searched over shrink factors, plain weight MSE)
      wmse   (as mse, error weighted per input column by ARG = stats file from --collect)
REGEX selects tensor names to quantize ('.' = every 2-D .weight except embed / lm_head / mlp.gate).
"""
import argparse
import json
import math
import pathlib
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import flint_reference as fr  # noqa: E402
from flint_raw_weights import E4M3  # noqa: E402

torch.set_grad_enabled(False)
E4 = torch.tensor(sorted({float(v) for v in E4M3 if v > 0}))
BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
GRID = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6.0])
FACTORS = torch.tensor([1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5])


def e4m3_nearest(x):
    i = torch.searchsorted(E4, x).clamp(1, len(E4) - 1)
    lo, hi = E4[i - 1], E4[i]
    return torch.where((x - lo) <= (hi - x), lo, hi)


def e2m1(y):
    return GRID[torch.bucketize(y.abs().clamp(max=6.0), BOUNDS)] * y.sign()


def quantize(w, mode, colw=None):
    n, k = w.shape
    ts = float(w.abs().max()) / 6.0
    if ts == 0:
        return w
    b = w.view(n, k // 16, 16)
    s0 = b.abs().amax(-1) / 6.0 / ts  # ideal block scale in tensor-scale units
    if mode == "minmax":
        s = (e4m3_nearest(s0.clamp(min=float(E4[0]))) * ts)[..., None]
        return (e2m1(b / s) * s).view(n, k)
    cw = None if colw is None else colw.view(1, k // 16, 16)
    best_err = torch.full(s0.shape, float("inf"))
    best = torch.zeros_like(b)
    for f in FACTORS:
        s = (e4m3_nearest((s0 * f).clamp(min=float(E4[0]))) * ts)[..., None]
        q = e2m1(b / s) * s
        d = (q - b).pow(2)
        err = (d * cw).sum(-1) if cw is not None else d.sum(-1)
        take = err < best_err
        best_err = torch.where(take, err, best_err)
        best = torch.where(take[..., None], q, best)
    return best.view(n, k)


def fp8(w):
    """E4M3 with one per-tensor scale (absmax -> 448)."""
    sc = float(w.abs().max()) / 448.0
    if sc == 0:
        return w
    y = (w / sc).abs()
    i = torch.searchsorted(E4, y).clamp(1, len(E4) - 1)
    lo, hi = E4[i - 1], E4[i]
    q = torch.where((y - lo) <= (hi - y), lo, hi)
    q = torch.where(y < E4[0] / 2, torch.zeros_like(q), q)
    return q * w.sign() * sc


class Combo:
    """Several (regex, mode, arg) rules, first match wins: 'rx1=mode1@arg;rx2=mode2'."""

    def __init__(self, base, spec):
        self.base, self.rules, self.n = base, [], 0
        for part in spec.split(";"):
            rx, rest = part.split("=", 1)
            mode, _, arg = rest.partition("@")
            self.rules.append((re.compile(rx), mode, arg))

    def has(self, name):
        return self.base.has(name)

    def get(self, name, dtype=torch.float32):
        w = self.base.get(name, dtype)
        if w.ndim != 2 or SKIP.search(name) or not name.endswith(".weight"):
            return w
        for rx, mode, arg in self.rules:
            if rx.search(name):
                self.n += 1
                if mode == "fp8":
                    return fp8(w)
                if mode.startswith("gptq"):
                    f = pathlib.Path(arg) / (stat_site(name) + ".pt")
                    return gptq(w, torch.load(f)["h"], float(mode[4:] or 0.1)) if f.exists() else quantize(w, "minmax")
                return quantize(w, mode)
        return w


def gptq(w, h, damp, blocksize=128):
    """GPTQ onto the NVFP4 grid: the 16-wide block scale is fixed from the updated weights when the block
    starts, columns are rounded one by one, the error is fed forward through the Cholesky factor of H^-1."""
    n, k = w.shape
    ts = float(w.abs().max()) / 6.0
    if ts == 0:
        return w
    W = w.to(torch.float64).clone()
    H = h.to(torch.float64).clone()
    d = torch.diagonal(H)
    dead = d <= 0
    H[dead, dead] = 1.0
    W[:, dead] = 0
    H += torch.eye(k, dtype=torch.float64) * damp * float(d.mean())
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    lo = float(E4[0])
    E4d = E4.to(torch.float64)
    for i1 in range(0, k, blocksize):
        i2 = min(i1 + blocksize, k)
        W1 = W[:, i1:i2].clone()
        Err = torch.zeros_like(W1)
        U1 = U[i1:i2, i1:i2]
        sc = None
        for j in range(i2 - i1):
            if (i1 + j) % 16 == 0:
                s0 = (W1[:, j:j + 16].abs().amax(1) / 6.0 / ts).clamp(min=lo)
                idx = torch.searchsorted(E4d, s0).clamp(1, len(E4d) - 1)
                a, b = E4d[idx - 1], E4d[idx]
                sc = torch.where((s0 - a) <= (b - s0), a, b) * ts
            y = W1[:, j] / sc
            q = (GRID[torch.bucketize(y.abs().clamp(max=6.0).to(torch.float32), BOUNDS)].to(torch.float64) * y.sign()) * sc
            Q[:, i1 + j] = q
            err = (W1[:, j] - q) / U1[j, j]
            W1[:, j:] -= err[:, None] * U1[j, j:][None, :]
            Err[:, j] = err
        W[:, i2:] -= Err @ U[i1:i2, i2:]
    return Q.to(torch.float32)


SKIP = re.compile(r"embed_tokens|lm_head|\.mlp\.gate\.weight$|norm")


class QWeights:
    def __init__(self, base, regex, mode, stats=None):
        self.base, self.rx, self.mode, self.stats = base, re.compile(regex), mode, stats
        self.n = 0

    def has(self, name):
        return self.base.has(name)

    def get(self, name, dtype=torch.float32):
        w = self.base.get(name, dtype)
        if w.ndim != 2 or SKIP.search(name) or not name.endswith(".weight") or not self.rx.search(name):
            return w
        self.n += 1
        if self.mode.startswith("gptq"):
            f = pathlib.Path(self.stats) / (stat_site(name) + ".pt")
            if not f.exists():
                return quantize(w, "minmax")
            self.n_gptq = getattr(self, "n_gptq", 0) + 1
            return gptq(w, torch.load(f)["h"], float(self.mode[4:] or 0.01))
        colw = None
        if self.mode == "wmse":
            colw = self.stats.get(stat_site(name))
        return quantize(w, "mse" if self.mode == "wmse" else self.mode, colw)


def stat_site(name):
    """Input-activation site of a Linear: tensors sharing an input share the statistics."""
    m = re.match(r"model\.layers\.(\d+)\.(.*)\.weight$", name)
    l, rest = m.group(1), m.group(2)
    if rest in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
        return f"{l}.attn_in"
    if rest == "self_attn.o_proj":
        return f"{l}.o_in"
    if rest.endswith("down_proj"):
        return f"{l}.{rest}"  # each down_proj has its own input (the SwiGLU product)
    return f"{l}.mlp_in"  # dense / shared / every expert gate_proj, up_proj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--arm", action="append", default=[])
    ap.add_argument("--ckpt", action="append", default=[],
                    help="NAME:DIR[:A4REGEX[:fp8]] Modelopt NVFP4 checkpoint to score, activations at A4REGEX sites as NVFP4 (fp8: per-token E4M3)")
    ap.add_argument("--ref-cache", default="", help="file for the BF16 logits: loaded if present, else written")
    a = ap.parse_args()
    cfg = json.loads((pathlib.Path(a.model) / "config.json").read_text())
    ids = [int(x) for x in pathlib.Path(a.ids).read_text().split()][: a.max_tokens]
    base = fr.Weights(a.model)
    tgt = torch.tensor(ids[1:])
    import contextlib
    import io
    cache = pathlib.Path(a.ref_cache) if a.ref_cache else None
    if cache and cache.exists():
        ref_logits = torch.load(cache)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            ref_logits, _ = fr.forward(cfg, base, torch.tensor(ids))
        if cache:
            torch.save(ref_logits.to(torch.float32), cache)
    ref_lp = torch.log_softmax(ref_logits[:-1].to(torch.float64), -1)
    ref_nll = -ref_lp.gather(1, tgt[:, None])[:, 0]
    print(f'BF16 rows {len(tgt)} mean NLL {ref_nll.mean():.6f} PPL {math.exp(ref_nll.mean()):.4f}', flush=True)

    def score(name, W, extra, t0):
        with contextlib.redirect_stdout(io.StringIO()):
            logits, _ = fr.forward(cfg, W, torch.tensor(ids))
        lp = torch.log_softmax(logits[:-1].to(torch.float64), -1)
        nll = -lp.gather(1, tgt[:, None])[:, 0]
        kl = (ref_lp.exp() * (ref_lp - lp)).sum(-1)
        d = nll - ref_nll
        print(f'ARM {name:20s} {extra:44s} KL {kl.mean():.5f} (se {kl.std() / len(kl) ** 0.5:.5f})  dNLL {d.mean():+.5f} '
              f'(se {d.std() / len(d) ** 0.5:.5f})  PPL {math.exp(nll.mean()):.4f}  top1 agree '
              f'{(logits[:-1].argmax(-1) == ref_logits[:-1].argmax(-1)).float().mean():.4f}  {time.time() - t0:.0f}s', flush=True)

    from flint_raw_weights import RawWeights
    for c in a.ckpt:
        cname, cdir, *a4 = c.split(':', 3)
        fr.A4 = re.compile(a4[0]) if a4 else None
        fr.A_FMT = a4[1] if len(a4) > 1 else "nvfp4"
        score(cname, RawWeights(cdir), f'checkpoint {cdir} A4 {a4[0] if a4 else "-"}', time.time())
        fr.A4 = None
    for arm in a.arm:
        parts = arm.split(":")
        name, regex, mode = parts[0], parts[1], parts[2]
        stats = None
        if len(parts) > 3:
            stats = parts[3] if mode.startswith("gptq") else torch.load(parts[3])["stats"]
        t0 = time.time()
        W = Combo(base, regex) if mode == "combo" else QWeights(base, regex, mode, stats)
        score(name, W, f'{regex} {mode}', t0)


if __name__ == "__main__":
    main()
