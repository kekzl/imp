"""Independent GPTQ PPL reference for #2249 (CPU, runs in tools/analysis/Dockerfile.gptqref).

Dequant follows AutoGPTQ auto_gptq/nn_modules/qlinear/qlinear_cuda_old.py (torch path, 4-bit), not imp.
PPL = exp(mean NLL over n-1 next-token predictions) on one causal sequence, as imp executor_perplexity.cu.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

BITS = 4
DTYPES = {"fp32": torch.float32, "fp16": torch.float16}


def autogptq_dequant(qweight, qzeros, scales, group_size, zero_add):
    """qlinear_cuda_old.py forward, bits in [2, 4, 8] branch; returns [K, N] fp16."""
    wf = torch.tensor(list(range(0, 32, BITS)), dtype=torch.int32).unsqueeze(0)
    zeros = torch.bitwise_right_shift(
        torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // BITS), wf.unsqueeze(0)
    ).to(torch.int8)
    zeros = torch.bitwise_and(zeros + zero_add, (2**BITS) - 1)
    zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])
    sc = scales.reshape(-1, 1, scales.shape[-1])
    weight = torch.bitwise_right_shift(
        torch.unsqueeze(qweight, 1).expand(-1, 32 // BITS, -1), wf.unsqueeze(-1)
    ).to(torch.int8)
    weight = torch.bitwise_and(weight, (2**BITS) - 1).reshape(-1, group_size, weight.shape[2])
    weight = sc * (weight - zeros)
    return weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])


def ppl(model, ids, chunk=1024):
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    past, prev_last, nll = None, None, 0.0
    with torch.no_grad():
        for s in range(0, x.shape[1], chunk):
            piece = x[:, s : s + chunk]
            out = model(input_ids=piece, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logp = torch.log_softmax(out.logits.double(), dim=-1)[0]
            if prev_last is not None:
                nll -= prev_last[piece[0, 0]].item()
            nll -= logp[:-1].gather(1, piece[0, 1:].unsqueeze(1)).sum().item()
            prev_last = logp[-1]
    return math.exp(nll / (len(ids) - 1))


def build(state, cfg_dir, dtype):
    cfg = AutoConfig.from_pretrained(cfg_dir)
    if hasattr(cfg, "quantization_config"):
        del cfg.quantization_config
    m = AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype, attn_implementation="sdpa")
    missing, unexpected = m.load_state_dict(state, strict=False)
    missing = [k for k in missing if not (k == "lm_head.weight" and cfg.tie_word_embeddings)]
    if missing or unexpected:
        raise SystemExit(f"state dict mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
    if cfg.tie_word_embeddings:
        m.tie_weights()
    return m.eval()


def fp8_head(m, dtype):
    """imp gemm.nvfp4_lm_head=auto on a 16-bit head: per-row E4M3, scale = row absmax / 448."""
    w = m.get_output_embeddings().weight.detach().float()
    s = (w.abs().amax(dim=1, keepdim=True) / 448.0).clamp_min(1e-12)
    wq = (w / s).to(torch.float8_e4m3fn).float() * s
    m.lm_head = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, dtype=dtype)
    m.lm_head.weight = torch.nn.Parameter(wq.to(dtype), requires_grad=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["dequant", "orig", "hf_gptq"], required=True)
    ap.add_argument("--model", required=True, help="snapshot dir (GPTQ for dequant/hf_gptq, original for orig)")
    ap.add_argument("--ids", required=True, help="one token id per line (imp diagnostics.dump_tokens)")
    ap.add_argument("--dtype", choices=list(DTYPES), default="fp32")
    ap.add_argument("--head", choices=["src", "fp8"], default="src")
    ap.add_argument("--dump-dir", help="dequant arm: write each projection as raw FP16 [N, K]")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    dtype = DTYPES[a.dtype]
    ids = [int(x) for x in open(a.ids)]
    t0 = time.time()
    if a.arm == "dequant":
        qc = json.load(open(f"{a.model}/config.json"))["quantization_config"]
        fmt = str(qc.get("checkpoint_format", qc.get("format", "gptq"))).lower()
        if fmt not in ("gptq", "gptq_v2") or qc.get("bits") != BITS:
            raise SystemExit(f"unsupported GPTQ config: bits={qc.get('bits')} format={fmt}")
        zero_add = 1 if fmt == "gptq" else 0
        raw = load_file(f"{a.model}/model.safetensors")
        state = {}
        for k, v in raw.items():
            if k.endswith(".qweight"):
                p = k[: -len(".qweight")]
                wt = autogptq_dequant(v, raw[p + ".qzeros"], raw[p + ".scales"], qc["group_size"], zero_add).t()
                if a.dump_dir:
                    os.makedirs(a.dump_dir, exist_ok=True)
                    wt.contiguous().numpy().astype(np.float16).tofile(f"{a.dump_dir}/{p}.f16")
                state[p + ".weight"] = wt.to(dtype)
            elif not k.endswith((".qzeros", ".scales", ".g_idx")):
                state[k] = v.to(dtype)
        print(f"dequantized {sum(k.endswith('.qweight') for k in raw)} projections, format {fmt}", flush=True)
        m = build(state, a.model, dtype)
    elif a.arm == "orig":
        m = build({k: v.to(dtype) for k, v in load_file(f"{a.model}/model.safetensors").items()}, a.model, dtype)
    else:
        m = AutoModelForCausalLM.from_pretrained(a.model, device_map="cpu", torch_dtype=torch.float16).eval()
        q = m.model.layers[0].self_attn.q_proj
        print(f"hf_gptq linear: {type(q).__module__}.{type(q).__name__}", flush=True)
        dtype = torch.float16
    if a.head == "fp8":
        fp8_head(m, dtype)
    dt = str(dtype).replace("torch.", "")
    print(f"ref_ppl arm={a.arm} head={a.head} dtype={dt} n_tokens={len(ids)} ppl={ppl(m, ids):.4f} "
          f"secs={time.time() - t0:.0f}", flush=True)


if __name__ == "__main__":
    main()
