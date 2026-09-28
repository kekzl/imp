#!/usr/bin/env python3
"""Reference draft logits of the Qwen4Exp (Qwen3.8-Flash-Next) MTP head, straight from the checkpoint.

Implements the vLLM math pinned in docs/plans/2026-09-28-qwen4exp-mtp.md in FP64/FP32 numpy, for
two chained draft steps from a fixed hc-stream input, and writes the fixture that
tests/test_mtp_qwen4exp_reference.cpp compares imp's draft step against.

A second arm rounds every intermediate to FP16 where imp stores one (buffers between kernels): the
gap between the two arms is the FP16 storage noise the test band is derived from (printed).

Run (numpy only, no torch):
  docker run --rm -v ~/models/Qwen3.8-Flash-Next-NVFP4:/m:ro -v $PWD:/src -w /src python:3.12-slim \
    sh -c 'pip -q install numpy && python tools/analysis/mtp_qwen4exp_reference.py /m tests/data/mtp_qwen4exp_ref.txt'
"""
import json
import struct
import sys

import numpy as np

D, HC, LR, NE, TOPK, EFF, NH, NKV, HD = 2560, 4, 320, 512, 10, 640, 24, 2, 256
ROPE_DIM, THETA, EPS = 64, 1.0e7, 1e-6
TOKENS = (9707, 1234)  # step 0, step 1 (fixed, so an argmax flip cannot decouple the chain)
N_IDS, N_TOP = 512, 16


def h_prev_input():
    """[HC*D] values k/1024, k in [-2048, 2048): exact in FP16, same hash in the C++ test."""
    i = np.arange(HC * D, dtype=np.uint64)
    u = (i * np.uint64(2654435761)) % np.uint64(1 << 32)
    k = (u >> np.uint64(20)).astype(np.int64) - 2048
    return k.astype(np.float64) / 1024.0


E4M3 = np.zeros(256, dtype=np.float64)
for b in range(256):
    s, e, m = b >> 7, (b >> 3) & 15, b & 7
    v = (m / 8.0) * 2.0 ** -6 if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7)
    E4M3[b] = np.nan if (e == 15 and m == 7) else (-v if s else v)


class Ckpt:
    def __init__(self, root):
        self.root = root
        self.where = json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]
        self.hdr = {}

    def _header(self, shard):
        if shard not in self.hdr:
            with open(f"{self.root}/{shard}", "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self.hdr[shard] = (json.loads(f.read(n)), 8 + n)
        return self.hdr[shard]

    def raw(self, name):
        shard = self.where[name]
        h, base = self._header(shard)
        t = h[name]
        b0, b1 = t["data_offsets"]
        dt = {"BF16": np.uint16, "F8_E4M3": np.uint8, "F32": np.float32}[t["dtype"]]
        a = np.memmap(f"{self.root}/{shard}", dtype=dt, mode="r", offset=base + b0, shape=((b1 - b0) // np.dtype(dt).itemsize,))
        return a.reshape(t["shape"]), t["dtype"]

    def get(self, name):
        a, dt = self.raw(name)
        if dt == "BF16":
            return (a.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        if dt == "F8_E4M3":
            return E4M3[a]
        return a.astype(np.float64)

    def rows(self, name, idx):
        a, dt = self.raw(name)
        assert dt == "BF16"
        return (np.asarray(a[idx]).astype(np.uint32) << 16).view(np.float32).astype(np.float64)

    def expert(self, e, proj):
        p = f"mtp.layers.0.mlp.experts.{e}.{proj}"
        w = self.get(p + ".weight")
        s = self.get(p + ".weight_scale_inv")
        # vllm fp8_utils.py:933: Bs.repeat_interleave(bn, 0).repeat_interleave(bk, 1)[:N, :K]
        return w * np.repeat(np.repeat(s, 128, axis=0), 128, axis=1)[: w.shape[0], : w.shape[1]]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


class Head:
    def __init__(self, ck, fp16):
        self.ck = ck
        self.r = (lambda x: np.asarray(x, dtype=np.float16).astype(np.float64)) if fp16 else (lambda x: x)
        g = ck.get
        self.w = {n: g("mtp." + n) for n in [
            "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight", "fc_embedding.weight", "fc_hidden.weight",
            "layers.0.self_attn.q_proj.weight", "layers.0.self_attn.k_proj.weight",
            "layers.0.self_attn.v_proj.weight", "layers.0.self_attn.o_proj.weight",
            "layers.0.self_attn.q_norm.weight", "layers.0.self_attn.k_norm.weight", "layers.0.mlp.gate.weight",
            "layers.0.mlp.shared_expert.gate_proj.weight", "layers.0.mlp.shared_expert.up_proj.weight",
            "layers.0.mlp.shared_expert.down_proj.weight", "layers.0.mlp.shared_expert_gate.weight"]}
        for m in ["layers.0.attn_hyper_connection", "layers.0.mlp_hyper_connection", "hyper_connection_mixer"]:
            for t in ["hc_norm", "input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"]:
                n = f"mtp.{m}.{t}.weight"
                if n in ck.where:
                    self.w[f"{m}.{t}.weight"] = g(n)
        self.kc, self.vc = [], []

    def rms(self, x, w):  # Gemma-style (1 + w), T:170-177
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * (1.0 + w)

    def hc_read(self, x, m):
        w, r = self.w, self.r
        xn = r(self.rms(x.reshape(HC, D), w[f"{m}.hc_norm.weight"].reshape(HC, D)).reshape(-1))
        low = r(silu(w[f"{m}.input_mix_weight_down.weight"] @ xn / HC))
        gate = r(w[f"{m}.input_mix_weight_up.weight"] @ low)
        bi = r(np.mean(sigmoid(gate.reshape(HC, D)) * xn.reshape(HC, D), axis=0))
        inj = None
        if f"{m}.block_inject_weight.weight" in w:
            inj = r(2.0 * sigmoid(w[f"{m}.block_inject_weight.weight"] @ xn / HC))
        return bi, inj

    def rope(self, v, pos):  # NeoX rotate_half over the first ROPE_DIM dims (partial_rotary_factor 0.25)
        inv = THETA ** (-np.arange(0, ROPE_DIM, 2, dtype=np.float64) / ROPE_DIM)
        c, s = np.cos(pos * inv), np.sin(pos * inv)
        a, b = v[..., : ROPE_DIM // 2], v[..., ROPE_DIM // 2: ROPE_DIM]
        out = v.copy()
        out[..., : ROPE_DIM // 2] = a * c - b * s
        out[..., ROPE_DIM // 2: ROPE_DIM] = b * c + a * s
        return out

    def step(self, token, h_prev, pos):
        w, r, p = self.w, self.r, "layers.0."
        emb = r(self.ck.rows("model.language_model.embed_tokens.weight", token))
        e = r(w["fc_embedding.weight"] @ r(self.rms(emb, w["pre_fc_norm_embedding.weight"])))
        hn = r(self.rms(h_prev, w["pre_fc_norm_hidden.weight"])).reshape(HC, D)
        x = r(r(hn @ w["fc_hidden.weight"].T) + e).reshape(-1)
        bi, inj = self.hc_read(x, "layers.0.attn_hyper_connection")
        qf = r(w[p + "self_attn.q_proj.weight"] @ bi).reshape(NH, 2 * HD)
        q, gate = qf[:, :HD], qf[:, HD:]
        k = r(w[p + "self_attn.k_proj.weight"] @ bi).reshape(NKV, HD)
        v = r(w[p + "self_attn.v_proj.weight"] @ bi).reshape(NKV, HD)
        q = r(self.rope(r(self.rms(q, w[p + "self_attn.q_norm.weight"])), pos))
        k = r(self.rope(r(self.rms(k, w[p + "self_attn.k_norm.weight"])), pos))
        self.kc.append(k)
        self.vc.append(v)
        K, V = np.stack(self.kc), np.stack(self.vc)  # [T, NKV, HD]
        o = np.zeros((NH, HD))
        for h in range(NH):
            g = h // (NH // NKV)
            sc = K[:, g, :] @ q[h] / np.sqrt(HD)
            pr = np.exp(sc - sc.max())
            o[h] = (pr / pr.sum()) @ V[:, g, :]
        o = r(r(o) * sigmoid(gate))
        attn = r(w[p + "self_attn.o_proj.weight"] @ o.reshape(-1))
        x = r(x + np.tile(attn, HC) * np.repeat(inj, D))
        bi, inj = self.hc_read(x, "layers.0.mlp_hyper_connection")
        logits = w[p + "mlp.gate.weight"] @ bi
        prob = np.exp(logits - logits.max())
        prob /= prob.sum()
        order = np.argsort(-prob, kind="stable")
        top = order[:TOPK]
        wt = prob[top] / prob[top].sum()
        routed = np.zeros(D)
        for e_id, e_w in zip(top, wt):
            gu = r(np.concatenate([self.ck.expert(e_id, "gate_proj") @ bi, self.ck.expert(e_id, "up_proj") @ bi]))
            act = r(silu(gu[:EFF]) * gu[EFF:])
            routed += e_w * r(self.ck.expert(e_id, "down_proj") @ act)
        routed = r(routed)
        sg = r(w[p + "mlp.shared_expert.gate_proj.weight"] @ bi)
        su = r(w[p + "mlp.shared_expert.up_proj.weight"] @ bi)
        so = r(w[p + "mlp.shared_expert.down_proj.weight"] @ r(silu(sg) * su))
        so = r(so * sigmoid(w[p + "mlp.shared_expert_gate.weight"] @ bi))
        mlp = r(routed + so)
        x = r(x + np.tile(mlp, HC) * np.repeat(inj, D))
        sample, _ = self.hc_read(x, "hyper_connection_mixer")
        return sample, x, top, prob[order[TOPK - 1]] - prob[order[TOPK]]


def lm_logits(ck, sample):
    a, _ = ck.raw("lm_head.weight")
    out = np.empty(a.shape[0], dtype=np.float64)
    for r0 in range(0, a.shape[0], 16384):
        blk = (np.asarray(a[r0:r0 + 16384]).astype(np.uint32) << 16).view(np.float32)
        out[r0:r0 + 16384] = blk.astype(np.float64) @ sample
    return out


def main():
    root, out_path = sys.argv[1], sys.argv[2]
    ck = Ckpt(root)
    arms = {}
    for name, fp16 in (("fp64", False), ("fp16", True)):
        head = Head(ck, fp16)
        h, steps = h_prev_input(), []
        for pos, tok in enumerate(TOKENS):
            sample, h, top, margin = head.step(tok, h, pos)
            lg = lm_logits(ck, sample)
            steps.append((tok, pos, sample, lg, top, margin))
        arms[name] = steps
    ids = (np.arange(N_IDS, dtype=np.int64) * 485) % 248320
    fmt = lambda xs: " ".join(f"{float(v):.6g}" for v in xs)  # noqa: E731
    with open(out_path, "w") as f:
        f.write(f"imp-qwen4exp-mtp-ref 1 {len(TOKENS)}\n")
        for (tok, pos, sample, lg, top, margin), s16 in zip(arms["fp64"], arms["fp16"]):
            order = np.argsort(-lg, kind="stable")
            noise_l = np.max(np.abs(s16[3] - lg))
            noise_h = np.max(np.abs(s16[2] - sample))
            print(f"step pos={pos} token={tok}: argmax {order[0]} (fp16 arm {int(np.argmax(s16[3]))}), "
                  f"top1-top2 {lg[order[0]] - lg[order[1]]:.4f}, |logit|max {np.abs(lg).max():.3f}, "
                  f"router experts {list(map(int, top))} (fp16 arm {list(map(int, s16[4]))}), "
                  f"p10-p11 {margin:.3e}, fp16-storage noise: logits {noise_l:.4f}, sample_hidden {noise_h:.4f}")
            f.write(f"step {tok} {pos} {int(order[0])} {noise_l:.6g} {noise_h:.6g}\n")
            f.write(f"sample {D} {fmt(sample)}\n")
            f.write(f"ids {N_IDS} " + " ".join(map(str, ids)) + "\n")
            f.write(f"logits {N_IDS} {fmt(lg[ids])}\n")
            f.write(f"top_ids {N_TOP} " + " ".join(map(str, order[:N_TOP])) + "\n")
            f.write(f"top_vals {N_TOP} {fmt(lg[order[:N_TOP]])}\n")
            f.write(f"experts {TOPK} " + " ".join(map(str, top)) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
