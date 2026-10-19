"""DSpark / DFlash drafter reference forward (z-lab/dflash DFlashDraftModel, markov_rank 0), FP32 math.

Usage (torch + safetensors, e.g. the vllm/vllm-openai image):
  python3 -I tools/analysis/dspark_reference.py <drafter dir> <target dir> <out.json> [ctx_rows=24]

Inputs are deterministic so the C++ test rebuilds them: ctx feature [i][j] = 2 sin(0.37 i + 0.011 j) rounded
to BF16, block = [target token 785, mask x 6], positions ctx 0..C-1, block C..C+6. Writes the 7 draft argmax
ids (target lm_head; block row r predicts position C + r + 1, the anchor row included: on real Qwen3-8B features
all 7 rows match the greedy continuation), the final-norm hidden of every block row (first 16 dims + L2 norm), and the same
argmax ids when the context arrives in two parts (C/2, then the rest) through the draft KV cache.
"""
import json
import math
import sys

import torch
from safetensors import safe_open


def load(path, names=None):
    out = {}
    with safe_open(path, "pt") as f:
        for k in f.keys():
            if names is None or k in names:
                out[k] = f.get_tensor(k).float()
    return out


def rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope(x, pos, theta, hd):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
    ang = pos[:, None].double() * inv[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], -1).float()[:, None, :]
    sin = torch.cat([ang.sin(), ang.sin()], -1).float()[:, None, :]
    x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


class Draft:
    def __init__(self, ddir, cfg):
        self.w = load(f"{ddir}/model.safetensors")
        self.c = cfg
        self.cache = [None] * cfg["num_hidden_layers"]  # per layer (K, V) of ctx rows, roped, k-normed

    def ctx(self, feat, pos):
        """Append context rows: K/V of hidden_norm(fc(feat)) at every layer."""
        c, w = self.c, self.w
        hd, nkv = c["head_dim"], c["num_key_value_heads"]
        t = rms(feat @ w["fc.weight"].T, w["hidden_norm.weight"], c["rms_norm_eps"])
        for li in range(c["num_hidden_layers"]):
            p = f"layers.{li}.self_attn."
            k = (t @ w[p + "k_proj.weight"].T).view(-1, nkv, hd)
            k = rope(rms(k, w[p + "k_norm.weight"], c["rms_norm_eps"]), pos, c["rope_parameters"]["rope_theta"], hd)
            v = (t @ w[p + "v_proj.weight"].T).view(-1, nkv, hd)
            old = self.cache[li]
            self.cache[li] = (k, v) if old is None else (torch.cat([old[0], k]), torch.cat([old[1], v]))

    def block(self, emb, pos):
        """Non-causal block forward over cached ctx + block K/V; returns final-norm hidden [B, d]."""
        c, w = self.c, self.w
        hd, nh, nkv, eps = c["head_dim"], c["num_attention_heads"], c["num_key_value_heads"], c["rms_norm_eps"]
        theta = c["rope_parameters"]["rope_theta"]
        h = emb
        for li in range(c["num_hidden_layers"]):
            p = f"layers.{li}."
            x = rms(h, w[p + "input_layernorm.weight"], eps)
            q = (x @ w[p + "self_attn.q_proj.weight"].T).view(-1, nh, hd)
            q = rope(rms(q, w[p + "self_attn.q_norm.weight"], eps), pos, theta, hd)
            k = (x @ w[p + "self_attn.k_proj.weight"].T).view(-1, nkv, hd)
            k = rope(rms(k, w[p + "self_attn.k_norm.weight"], eps), pos, theta, hd)
            v = (x @ w[p + "self_attn.v_proj.weight"].T).view(-1, nkv, hd)
            kc, vc = self.cache[li]
            kk, vv = torch.cat([kc, k]), torch.cat([vc, v])
            g = nh // nkv
            kk, vv = kk.repeat_interleave(g, 1), vv.repeat_interleave(g, 1)
            s = torch.einsum("qhd,khd->hqk", q, kk) / math.sqrt(hd)
            a = torch.einsum("hqk,khd->qhd", s.softmax(-1), vv).reshape(-1, nh * hd)
            h = h + a @ w[p + "self_attn.o_proj.weight"].T
            x = rms(h, w[p + "post_attention_layernorm.weight"], eps)
            gate = x @ w[p + "mlp.gate_proj.weight"].T
            up = x @ w[p + "mlp.up_proj.weight"].T
            h = h + (torch.nn.functional.silu(gate) * up) @ w[p + "mlp.down_proj.weight"].T
        return rms(h, w["norm.weight"], eps)


def main():
    ddir, tdir, out = sys.argv[1], sys.argv[2], sys.argv[3]
    n_ctx = int(sys.argv[4]) if len(sys.argv) > 4 else 24
    cfg = json.load(open(f"{ddir}/config.json"))
    tw = load(f"{ddir}/model.safetensors", {"embed_tokens.weight", "lm_head.weight"})  # == target's (checked)
    d_model, n_feat = cfg["hidden_size"], len(cfg["target_layer_ids"])
    i = torch.arange(n_ctx, dtype=torch.float32)[:, None]
    j = torch.arange(n_feat * d_model, dtype=torch.float32)[None, :]
    feat = (2.0 * torch.sin(0.37 * i + 0.011 * j)).bfloat16().float()
    block_ids = torch.tensor([785] + [cfg["mask_token_id"]] * (cfg["block_size"] - 1))
    emb = tw["embed_tokens.weight"][block_ids]
    pos_ctx = torch.arange(n_ctx)
    pos_blk = torch.arange(n_ctx, n_ctx + cfg["block_size"])

    def run(parts):
        d = Draft(ddir, cfg)
        start = 0
        for n in parts:
            d.ctx(feat[start:start + n], pos_ctx[start:start + n])
            start += n
        hid = d.block(emb, pos_blk)
        ids = (hid @ tw["lm_head.weight"].T).argmax(-1)  # row r predicts position C + r + 1
        return hid, ids

    hid, ids = run([n_ctx])
    _, ids2 = run([n_ctx // 2, n_ctx - n_ctx // 2])
    json.dump({"ctx_rows": n_ctx, "block_ids": block_ids.tolist(), "draft_ids": ids.tolist(),
               "draft_ids_two_parts": ids2.tolist(),
               "hidden_head16": hid[:, :16].tolist(), "hidden_l2": hid.norm(dim=-1).tolist()},
              open(out, "w"), indent=1)
    print("draft ids", ids.tolist(), "two parts", ids2.tolist())


if __name__ == "__main__":
    main()
