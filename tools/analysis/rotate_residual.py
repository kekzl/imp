"""Offline residual-stream rotation of a dense LLaMA/Qwen3 BF16 checkpoint (roadmap row 101).

Folds every RMSNorm gamma into the weights that read it (norm weights become ones), then rotates
the residual basis h -> h Q: embeddings E Q, readers (q/k/v, gate/up, lm_head) W Q, writers
(o_proj, down_proj) Q^T W. Optional per-head Hadamard on v_proj rows / o_proj columns (--vo).
The BF16 forward is unchanged up to BF16 rounding; tied embeddings are untied (lm_head carries
the final-norm gamma). CPU only: python -I in a torch + safetensors image (Dockerfile.gptqref).
Usage: rotate_residual.py --in SRC --out DST --rotation {none,had16,full} [--vo] [--seed N]
       rotate_residual.py --selftest
"""
import argparse
import glob
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

AUX = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
       "merges.txt", "special_tokens_map.json", "added_tokens.json")


def hadamard(n):
    """Sylvester Hadamard / sqrt(n), n a power of two, float64."""
    assert n > 0 and n & (n - 1) == 0, f"{n} is not a power of two"
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / (n ** 0.5)


def rotation(kind, d, seed):
    """Orthogonal [d, d]: diag(random signs) times Hadamard (full) or block-diagonal Had16."""
    if kind == "none":
        return torch.eye(d, dtype=torch.float64)
    signs = torch.randint(0, 2, (d,), generator=torch.Generator().manual_seed(seed)).double() * 2 - 1
    if kind == "full":
        h = hadamard(d)
    elif kind == "had16":
        assert d % 16 == 0
        h = torch.block_diag(*([hadamard(16)] * (d // 16)))
    else:
        raise ValueError(kind)
    return signs[:, None] * h


def transform(t, cfg, q, vo):
    """t: name -> float64 tensor (modified in place). q: [d, d] orthogonal. vo: per-head Had or None."""
    n_layers = cfg["num_hidden_layers"]
    hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    emb = t["model.embed_tokens.weight"]
    lm = t.get("lm_head.weight", emb)
    gf = t["model.norm.weight"]
    t["lm_head.weight"] = (lm * gf[None, :]) @ q
    t["model.norm.weight"] = torch.ones_like(gf)
    t["model.embed_tokens.weight"] = emb @ q
    for i in range(n_layers):
        p = f"model.layers.{i}."
        for norm, readers in (("input_layernorm", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
                              ("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj"))):
            g = t[p + norm + ".weight"]
            for r in readers:
                t[p + r + ".weight"] = (t[p + r + ".weight"] * g[None, :]) @ q
            t[p + norm + ".weight"] = torch.ones_like(g)
        for w in ("self_attn.o_proj", "mlp.down_proj"):
            t[p + w + ".weight"] = q.T @ t[p + w + ".weight"]
            if p + w + ".bias" in t:
                t[p + w + ".bias"] = q.T @ t[p + w + ".bias"]
        if vo is not None:  # v_h' = v_h H: rows of v_proj per kv head, columns of o_proj per q head
            wv = t[p + "self_attn.v_proj.weight"]
            t[p + "self_attn.v_proj.weight"] = (vo.T @ wv.reshape(-1, hd, wv.shape[1])).reshape(wv.shape)
            if p + "self_attn.v_proj.bias" in t:
                bv = t[p + "self_attn.v_proj.bias"]
                t[p + "self_attn.v_proj.bias"] = (bv.reshape(-1, hd) @ vo).reshape(-1)
            wo = t[p + "self_attn.o_proj.weight"]
            t[p + "self_attn.o_proj.weight"] = (wo.reshape(wo.shape[0], -1, hd) @ vo).reshape(wo.shape)


def rms(x, g, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * g


def tiny_forward(t, cfg, ids):
    """Qwen3-shaped decoder in float64 (q/k norm, GQA, causal softmax, SwiGLU; no RoPE: it acts after q/k)."""
    hd, nh, nkv = cfg["head_dim"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    x = t["model.embed_tokens.weight"][ids]
    s = x.shape[0]
    mask = torch.full((s, s), float("-inf"), dtype=torch.float64).triu(1)
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        h = rms(x, t[p + "input_layernorm.weight"])
        qh = rms((h @ t[p + "self_attn.q_proj.weight"].T).view(s, nh, hd), t[p + "self_attn.q_norm.weight"])
        kh = rms((h @ t[p + "self_attn.k_proj.weight"].T).view(s, nkv, hd), t[p + "self_attn.k_norm.weight"])
        vh = (h @ t[p + "self_attn.v_proj.weight"].T).view(s, nkv, hd)
        rep = nh // nkv
        kh, vh = kh.repeat_interleave(rep, 1), vh.repeat_interleave(rep, 1)
        att = torch.softmax(torch.einsum("qhd,khd->hqk", qh, kh) / hd ** 0.5 + mask, -1)
        x = x + torch.einsum("hqk,khd->qhd", att, vh).reshape(s, -1) @ t[p + "self_attn.o_proj.weight"].T
        h = rms(x, t[p + "post_attention_layernorm.weight"])
        act = torch.nn.functional.silu(h @ t[p + "mlp.gate_proj.weight"].T) * (h @ t[p + "mlp.up_proj.weight"].T)
        x = x + act @ t[p + "mlp.down_proj.weight"].T
    return rms(x, t["model.norm.weight"]) @ t.get("lm_head.weight", t["model.embed_tokens.weight"]).T


def selftest():
    cfg = {"num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2,
           "head_dim": 16, "intermediate_size": 96}
    d, hd, ff, v = 64, 16, 96, 50
    gen = torch.Generator().manual_seed(1)

    def rnd(*shape, scale=0.2):
        return torch.randn(*shape, generator=gen, dtype=torch.float64) * scale

    base = {"model.embed_tokens.weight": rnd(v, d, scale=1.0), "model.norm.weight": 1 + rnd(d)}
    for i in range(2):
        p = f"model.layers.{i}."
        base.update({p + "input_layernorm.weight": 1 + rnd(d), p + "post_attention_layernorm.weight": 1 + rnd(d),
                     p + "self_attn.q_proj.weight": rnd(4 * hd, d), p + "self_attn.k_proj.weight": rnd(2 * hd, d),
                     p + "self_attn.v_proj.weight": rnd(2 * hd, d), p + "self_attn.o_proj.weight": rnd(d, 4 * hd),
                     p + "self_attn.q_norm.weight": 1 + rnd(hd), p + "self_attn.k_norm.weight": 1 + rnd(hd),
                     p + "mlp.gate_proj.weight": rnd(ff, d), p + "mlp.up_proj.weight": rnd(ff, d),
                     p + "mlp.down_proj.weight": rnd(d, ff)})
    ids = torch.randint(0, v, (12,), generator=gen)
    ref = tiny_forward(base, cfg, ids)
    worst = 0.0
    for kind in ("none", "had16", "full"):
        q = rotation(kind, d, seed=7)
        orth = (q @ q.T - torch.eye(d, dtype=torch.float64)).abs().max().item()
        for vo in (None, rotation("full", hd, seed=8)):
            t = {k: x.clone() for k, x in base.items()}
            transform(t, cfg, q, vo)
            err = ((tiny_forward(t, cfg, ids) - ref).abs().max() / ref.abs().max()).item()
            print(f"selftest {kind:5s} vo={vo is not None}: |QQ^T-I| {orth:.2e}, logits rel err {err:.2e}")
            worst = max(worst, orth, err)
    assert worst < 1e-10, worst
    print("selftest ok")


def load(src):
    t, files = {}, sorted(glob.glob(os.path.join(src, "*.safetensors")))
    for f in files:
        with safe_open(f, "pt") as fh:
            for k in fh.keys():
                t[k] = fh.get_tensor(k)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src")
    ap.add_argument("--out")
    ap.add_argument("--rotation", choices=("none", "had16", "full"), default="full")
    ap.add_argument("--vo", action="store_true", help="per-head Hadamard on v_proj rows / o_proj columns")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=("source", "f16"), default="source",
                    help="f16: one rounding step less, imp uploads BF16 weights as FP16 anyway")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    raw = load(a.src)
    dtype = torch.float16 if a.dtype == "f16" else raw["model.embed_tokens.weight"].dtype
    t = {k: x.double() for k, x in raw.items()}
    del raw
    hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    q = rotation(a.rotation, cfg["hidden_size"], a.seed)
    vo = rotation("full", hd, a.seed + 1) if a.vo else None
    transform(t, cfg, q, vo)
    amax = max(x.abs().max().item() for x in t.values())
    assert dtype != torch.float16 or amax < 65504, f"|w| max {amax} overflows F16"
    os.makedirs(a.out, exist_ok=True)
    out_st = os.path.join(a.out, "model.safetensors")
    save_file({k: x.to(dtype).contiguous() for k, x in t.items()}, out_st, metadata={"format": "pt"})
    os.chmod(out_st, 0o644)  # save_file writes 0600; the imp image reads as user imp
    for f in AUX:
        if os.path.exists(os.path.join(a.src, f)):
            shutil.copy(os.path.join(a.src, f), a.out)
    cfg["tie_word_embeddings"] = False
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    print(f"rotated {len(t)} tensors, rotation {a.rotation}, vo {a.vo}, seed {a.seed} -> {a.out}")


if __name__ == "__main__":
    main()
