#!/usr/bin/env python3
"""CPU reference forward of Flint (RecursiveSI/flint), written from config.json and the vendor's SGLang model
code (flint_sglang/model.py 0.1.4): Qwen3-MoE block + learned attention sinks, sliding/full layers,
no RoPE on no_rope_layer_indices, dense SwiGLU on mlp_only_layers, sigmoid top-k router renormalized to 1,
ungated shared expert. FP32 math, weights read lazily: BF16 SafeTensors, or a ModelOpt NVFP4 checkpoint
(hf_quant_config.json present) dequantized per tensor (flint_raw_weights.py). The 74 GiB BF16 checkpoint fits no
32 GB card, so this is the reference imp's Flint port is checked against (1024 tokens in about 30 s on 16 cores).

Usage: flint_reference.py --model DIR (--text FILE | --ids FILE) [--max-tokens N] [--out DIR] [--dump-layers]
                          [--a4 | --a4-sites REGEX]   (--a4 rounds the input of every quantized Linear to
                          NVFP4: W4A4 emulation; --a4-sites only at sites matching REGEX, names in aq())
Prints PPL over rows [0, n-2] (row i predicts token i+1) and writes nll.txt, ids.txt, top1.txt to --out.
Runs in python:3.13-slim + torch (CPU), safetensors, tokenizers, numpy.
"""
import argparse
import json
import math
import pathlib
import re
import time

import torch
from safetensors import safe_open
from flint_raw_weights import RawWeights

torch.set_grad_enabled(False)


class Weights:
    def __init__(self, model_dir):
        self.dir = pathlib.Path(model_dir)
        idx = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.map = idx
        self.handles = {}

    def has(self, name):
        return name in self.map

    def get(self, name, dtype=torch.float32):
        f = self.map[name]
        if f not in self.handles:
            self.handles[f] = safe_open(str(self.dir / f), framework="pt")
        return self.handles[f].get_tensor(name).to(dtype)


def rmsnorm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope(x, pos, theta):
    # x [T, H, D]; HF rotate_half convention: pairs (i, i + D/2).
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    ang = pos.to(torch.float64)[:, None] * inv[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], -1).to(torch.float32)[:, None, :]
    sin = torch.cat([ang.sin(), ang.sin()], -1).to(torch.float32)[:, None, :]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


A4 = None  # --a4 / --a4-sites: regex over activation sites rounded to NVFP4 (W4A4 emulation); None = W4A16
_E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6.0])
_E4M3_POS = None


def aq(x, site):
    """Activation as NVFP4 if A4 matches site (L.attn.qkv, L.attn.o, L.{dense,shared,routed}.{gu,down}): 16-wide
    blocks, block scale absmax / 6 rounded up to E4M3, E2M1 values."""
    global _E4M3_POS
    if A4 is None or not A4.search(site):
        return x
    if _E4M3_POS is None:
        from flint_raw_weights import E4M3
        _E4M3_POS = torch.tensor(sorted({float(v) for v in E4M3 if v > 0}))
    b = x.reshape(-1, 16)
    s = (b.abs().amax(-1) / 6.0).clamp(min=float(_E4M3_POS[0]))
    s = _E4M3_POS[torch.searchsorted(_E4M3_POS, s).clamp(max=len(_E4M3_POS) - 1)][:, None]
    y = (b / s).clamp(-6, 6)
    q = _E2M1[(y.abs()[..., None] - _E2M1).abs().argmin(-1)] * y.sign()
    return (q * s).reshape(x.shape)


COLLECT = None  # site -> [sum of x^2 per column, rows]
HESS = None  # site -> [sum of x^T x, rows] for the sites HESS_RX matches
HESS_RX = None


def rec(site, x):
    if HESS is not None and site is not None and HESS_RX.search(site):
        xx = x.reshape(-1, x.shape[-1]).to(torch.float32)
        acc = HESS.setdefault(site, [torch.zeros(xx.shape[-1], xx.shape[-1], dtype=torch.float64), 0])
        acc[0] += (xx.T @ xx).to(torch.float64)
        acc[1] += xx.shape[0]
    if COLLECT is None or site is None:
        return
    x = x.reshape(-1, x.shape[-1]).to(torch.float64)
    acc = COLLECT.setdefault(site, [torch.zeros(x.shape[-1], dtype=torch.float64), 0])
    acc[0] += x.pow(2).sum(0)
    acc[1] += x.shape[0]


def swiglu(x, wg, wu, wd, site=None, a4_site=""):
    x = aq(x, a4_site + ".gu")
    mid = torch.nn.functional.silu(x @ wg.T) * (x @ wu.T)
    rec(site, mid)
    return aq(mid, a4_site + ".down") @ wd.T


def attention(cfg, W, l, x, pos):
    p = f"model.layers.{l}.self_attn."
    nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    t = x.shape[0]
    rec(f"{l}.attn_in", x)
    x = aq(x, f"{l}.attn.qkv")
    q = (x @ W.get(p + "q_proj.weight").T).view(t, nh, hd)
    k = (x @ W.get(p + "k_proj.weight").T).view(t, nkv, hd)
    v = (x @ W.get(p + "v_proj.weight").T).view(t, nkv, hd)
    eps = cfg["rms_norm_eps"]
    q = rmsnorm(q, W.get(p + "q_norm.weight"), eps)
    k = rmsnorm(k, W.get(p + "k_norm.weight"), eps)
    if l not in cfg["no_rope_layer_indices"]:
        q = rope(q, pos, cfg["rope_theta"])
        k = rope(k, pos, cfg["rope_theta"])
    rep = nh // nkv
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    logits = torch.einsum("qhd,khd->hqk", q, k) * (hd ** -0.5)
    qi = pos[:, None]
    ki = pos[None, :]
    mask = ki > qi
    if cfg["layer_types"][l] == "sliding_attention":
        mask = mask | ((qi - ki) >= cfg["sliding_window"])
    logits = logits.masked_fill(mask[None], float("-inf"))
    sink = W.get(p + "softmax_offset")  # [nh], a logit in the denominator only
    full = torch.cat([logits, sink[:, None, None].expand(nh, t, 1)], -1)
    probs = torch.softmax(full, -1)[..., :-1]
    out = torch.einsum("hqk,khd->qhd", probs, v).reshape(t, nh * hd)
    rec(f"{l}.o_in", out)
    return aq(out, f"{l}.attn.o") @ W.get(p + "o_proj.weight").T


def moe(cfg, W, l, x, stats):
    p = f"model.layers.{l}.mlp."
    scores = torch.sigmoid(x @ W.get(p + "gate.weight").T)
    choice = scores
    if W.has(p + "gate.expert_bias"):
        choice = scores + W.get(p + "gate.expert_bias")[None]
    top = torch.topk(choice, cfg["num_experts_per_tok"], dim=-1).indices
    w = scores.gather(1, top)
    if cfg.get("norm_topk_prob", True):
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    out = torch.zeros_like(x)
    for e in top.unique().tolist():
        rows, slot = (top == e).nonzero(as_tuple=True)
        y = swiglu(x[rows], W.get(f"{p}experts.{e}.gate_proj.weight"), W.get(f"{p}experts.{e}.up_proj.weight"),
                   W.get(f"{p}experts.{e}.down_proj.weight"), site=f"{l}.mlp.experts.{e}.down_proj",
                   a4_site=f"{l}.routed")
        out.index_add_(0, rows, y * w[rows, slot][:, None])
    stats.append(top)
    return out + swiglu(x, W.get(p + "shared_expert.gate_proj.weight"), W.get(p + "shared_expert.up_proj.weight"),
                        W.get(p + "shared_expert.down_proj.weight"), site=f"{l}.mlp.shared_expert.down_proj",
                        a4_site=f"{l}.shared")


def forward(cfg, W, ids, dump=None):
    h = W.get("model.embed_tokens.weight")[ids]
    pos = torch.arange(len(ids))
    eps = cfg["rms_norm_eps"]
    routes = []
    for l in range(cfg["num_hidden_layers"]):
        t0 = time.time()
        h = h + attention(cfg, W, l, rmsnorm(h, W.get(f"model.layers.{l}.input_layernorm.weight"), eps), pos)
        x = rmsnorm(h, W.get(f"model.layers.{l}.post_attention_layernorm.weight"), eps)
        rec(f"{l}.mlp_in", x)
        if l in cfg["mlp_only_layers"]:
            p = f"model.layers.{l}.mlp."
            h = h + swiglu(x, W.get(p + "gate_proj.weight"), W.get(p + "up_proj.weight"), W.get(p + "down_proj.weight"),
                           site=f"{l}.mlp.down_proj", a4_site=f"{l}.dense")
        else:
            h = h + moe(cfg, W, l, x, routes)
        if dump is not None:
            torch.save(h.to(torch.float16), dump / f"layer_{l:02d}.pt")
        print(f"layer {l:2d} {cfg['layer_types'][l][:7]} rms {h.pow(2).mean().sqrt():.4f} {time.time() - t0:.1f}s", flush=True)
    h = rmsnorm(h, W.get("model.norm.weight"), eps)
    return h @ W.get("lm_head.weight").T, routes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text")
    ap.add_argument("--ids")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", default="")
    ap.add_argument("--dump-layers", action="store_true")
    ap.add_argument("--bos", action="store_true")
    ap.add_argument("--a4", action="store_true")
    ap.add_argument("--a4-sites", default=None)
    a = ap.parse_args()
    global A4
    A4 = re.compile(a.a4_sites if a.a4_sites is not None else ".") if (a.a4 or a.a4_sites is not None) else None
    cfg = json.loads((pathlib.Path(a.model) / "config.json").read_text())
    if a.ids:
        ids = [int(x) for x in pathlib.Path(a.ids).read_text().split()]
    else:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(pathlib.Path(a.model) / "tokenizer.json"))
        ids = tok.encode(pathlib.Path(a.text).read_text(), add_special_tokens=False).ids
    if a.bos:
        ids = [cfg["bos_token_id"]] + ids
    ids = ids[: a.max_tokens]
    out = pathlib.Path(a.out) if a.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "ids.txt").write_text(" ".join(map(str, ids)))
    W = RawWeights(a.model) if (pathlib.Path(a.model) / "hf_quant_config.json").exists() else Weights(a.model)
    logits, _ = forward(cfg, W, torch.tensor(ids), out if a.dump_layers else None)
    logp = torch.log_softmax(logits[:-1].to(torch.float64), -1)
    tgt = torch.tensor(ids[1:])
    nll = -logp.gather(1, tgt[:, None])[:, 0]
    top1 = logits[:-1].argmax(-1)
    print(f"tokens {len(ids)} rows {len(nll)} mean NLL {nll.mean():.6f} PPL {math.exp(nll.mean()):.4f} "
          f"top1 acc {(top1 == tgt).float().mean():.4f}")
    if out:
        (out / "nll.txt").write_text("\n".join(f"{x:.6f}" for x in nll.tolist()))
        (out / "top1.txt").write_text(" ".join(map(str, top1.tolist())))
        torch.save(logits[-1].to(torch.float32), out / "last_logits.pt")


if __name__ == "__main__":
    main()
