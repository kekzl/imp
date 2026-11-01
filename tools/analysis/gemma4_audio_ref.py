#!/usr/bin/env python3
# Reference tensors for tests/test_gemma4_audio_encoder.cu: HF Gemma4AudioFeatureExtractor,
# Gemma4AudioModel and the audio embedder on CPU in fp32 (transformers 5.22.1, torch CPU).
# No GPU. Container only, e.g. with torch/transformers/numpy installed under /pip/site:
#   docker run --rm -e PYTHONPATH=/pip/site -v <pip>:/pip -v $PWD/tools/analysis:/t:ro \
#     -v ~/models/gemma-4-E4B-it:/m:ro -v <out>:/o python:3.12-slim python /t/gemma4_audio_ref.py /m /o
# Writes features.f32, features_mask.u8, sscp/l0_*/layer0/tower_out/embeds.f32, signal.wav,
# small_golden.json, shapes.json; prints the chunked-attention re-derivation error.
import json
import math
import os
import struct
import sys

import numpy as np
import torch

from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig, Gemma4TextConfig
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel, Gemma4MultimodalEmbedder
from transformers.models.gemma4.processing_gemma4 import Gemma4Processor

model_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(0)


def signal(n_samples):
    # Same formula as tests/test_audio_features.cpp (double math, then float32).
    n = np.arange(n_samples, dtype=np.float64)
    t = n / 16000.0
    x = 0.30 * np.sin(2 * math.pi * 440.0 * t) + 0.20 * np.sin(2 * math.pi * 1234.5 * t + 0.7)
    x += 0.10 * np.sin(2 * math.pi * (200.0 + 1500.0 * t) * t)
    return x.astype(np.float32)


def save(name, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    arr.tofile(os.path.join(out_dir, name + ".f32"))
    return list(arr.shape)


shapes = {}
fe = Gemma4AudioFeatureExtractor()
print("fe frame/hop/fft", fe.frame_length, fe.hop_length, fe.fft_length)

# 1) Small golden for the CPU unit test: 0.37 s (5920 samples, not a multiple of 128).
small = signal(5920)
f = fe([small], return_tensors="np")
feat = f["input_features"][0]
mask = f["input_features_mask"][0]
print("small features", feat.shape, "valid", int(mask.sum()))
golden = {
    "n_samples": 5920,
    "frames": int(feat.shape[0]),
    "valid": int(mask.sum()),
    "row_sums": [float(v) for v in feat.sum(axis=1)],
    "rows": {str(r): [float(v) for v in feat[r]] for r in (0, 1, 17, int(mask.sum()) - 1)},
}
with open(os.path.join(out_dir, "small_golden.json"), "w") as fh:
    json.dump(golden, fh)

# 2) Encoder reference: 2.3 s signal.
n_big = 36800 + 77
x = signal(n_big)
f = fe([x], return_tensors="pt")
feats = f["input_features"].float()
fmask = f["input_features_mask"]
shapes["features"] = save("features", feats[0].numpy())
shapes["features_mask"] = [int(fmask.shape[1])]
np.asarray(fmask[0].numpy(), dtype=np.uint8).tofile(os.path.join(out_dir, "features_mask.u8"))
pcm16 = np.clip(np.round(x * 32767.0), -32768, 32767).astype("<i2")
with open(os.path.join(out_dir, "signal.wav"), "wb") as fh:
    data = pcm16.tobytes()
    fh.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE")
    fh.write(b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16))
    fh.write(b"data" + struct.pack("<I", len(data)) + data)

# Processor token count (no tokenizer needed for the arithmetic).
proc = Gemma4Processor.__new__(Gemma4Processor)
proc.feature_extractor = fe
proc.audio_seq_length = 750
n_tok = Gemma4Processor._compute_audio_num_tokens(proc, x, 16000)
print("processor audio tokens", n_tok)

cfg_all = json.load(open(os.path.join(model_dir, "config.json")))
acfg = Gemma4AudioConfig(**cfg_all["audio_config"])
tcfg = Gemma4TextConfig(**cfg_all["text_config"])
tower = Gemma4AudioModel(acfg).eval().float()
embed = Gemma4MultimodalEmbedder(acfg, tcfg).eval().float()

# Raw safetensors read (the file may still be downloading past the audio tensors).
st = os.path.join(model_dir, "model.safetensors")
with open(st, "rb") as fh:
    hlen = struct.unpack("<Q", fh.read(8))[0]
    header = json.loads(fh.read(hlen))
    base = 8 + hlen

    def read(name):
        meta = header[name]
        a, b = meta["data_offsets"]
        fh.seek(base + a)
        buf = bytearray(fh.read(b - a))
        assert meta["dtype"] == "BF16"
        t = torch.frombuffer(buf, dtype=torch.bfloat16).reshape(meta["shape"]).float()
        return t

    def load(module, prefix):
        sd = {}
        for k in module.state_dict().keys():
            name = prefix + k
            if name in header:
                sd[k] = read(name)
        missing, unexpected = module.load_state_dict(sd, strict=False)
        print(prefix, "loaded", len(sd), "missing", missing)

    load(tower, "model.audio_tower.")
    load(embed, "model.embed_audio.")

taps = {}
hooks = []
hooks.append(tower.subsample_conv_projection.register_forward_hook(lambda m, i, o: taps.__setitem__("sscp", o[0])))
hooks.append(tower.layers[0].register_forward_hook(lambda m, i, o: taps.__setitem__("layer0", o)))
hooks.append(tower.layers[0].feed_forward1.register_forward_hook(lambda m, i, o: taps.__setitem__("l0_ffw1", o)))
hooks.append(
    tower.layers[0].self_attn.register_forward_hook(lambda m, i, o: taps.__setitem__("l0_attn", o[0]))
)
hooks.append(tower.layers[0].lconv1d.register_forward_hook(lambda m, i, o: taps.__setitem__("l0_lconv", o)))
with torch.no_grad():
    out = tower(feats, fmask, return_dict=True)
    hidden = out.last_hidden_state
    omask = out.attention_mask
    emb = embed(hidden)
for k, v in taps.items():
    shapes[k] = save(k, v[0].numpy())
shapes["tower_out"] = save("tower_out", hidden[0].numpy())
valid = omask[0].bool()
print("tower out", hidden.shape, "valid tokens", int(valid.sum()))
shapes["embeds"] = save("embeds", emb[0][valid].numpy())
shapes["valid_tokens"] = int(valid.sum())
shapes["processor_tokens"] = int(n_tok)
shapes["n_samples"] = n_big


# Independent re-derivation of the chunked attention (what the CUDA kernel computes):
# key k attends from query q iff 0 <= q-k < context_left-1 and k is valid; logit =
# softcap(q.k + q.relk[q-k]).
def my_attention(attn, h, kv_valid):
    T = h.shape[0]
    H, D = attn.num_heads, attn.head_dim
    q = attn.q_proj(h).view(T, H, D) * attn.q_scale * torch.nn.functional.softplus(attn.per_dim_scale)
    k = attn.k_proj(h).view(T, H, D) * attn.k_scale
    v = attn.v_proj(h).view(T, H, D)
    pe = tower.rel_pos_enc(h[None])  # [13, hidden], row c = distance 12 - c
    relk = attn.relative_k_proj(pe).view(-1, H, D)
    n_pos = relk.shape[0]
    out = torch.zeros(T, H, D)
    cap = attn.attention_logits_soft_cap
    left = acfg.attention_context_left - 1
    for t in range(T):
        for hh in range(H):
            logits, vals = [], []
            for kk in range(max(0, t - left + 1), t + 1):
                if not kv_valid[kk]:
                    continue
                d = t - kk
                rel = relk[n_pos - 1 - d, hh]
                s = (q[t, hh] * (k[kk, hh] + rel)).sum()
                logits.append(torch.tanh(s / cap) * cap)
                vals.append(v[kk, hh])
            if logits:
                p = torch.softmax(torch.stack(logits), 0)
                out[t, hh] = (p[:, None] * torch.stack(vals)).sum(0)
    return attn.post(out.view(T, H * D))


with torch.no_grad():
    l0 = tower.layers[0]
    hin = taps["l0_ffw1"][0]
    xin = l0.norm_pre_attn(hin)
    mine = my_attention(l0.self_attn, xin, valid)
    ref = taps["l0_attn"][0]
    err = (mine - ref)[valid].abs().max().item()
    print("my attention vs HF (valid rows) max abs err", err, "ref max", ref[valid].abs().max().item())

with open(os.path.join(out_dir, "shapes.json"), "w") as fh:
    json.dump(shapes, fh, indent=1)
print(json.dumps(shapes))
