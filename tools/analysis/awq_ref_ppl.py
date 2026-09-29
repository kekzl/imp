"""Independent AWQ PPL reference for #2205 (CPU, runs in tools/analysis/Dockerfile.gptqref).

Dequant follows AutoAWQ awq/utils/packing_utils.py dequantize_gemm (unpack_awq, reverse_awq_order), not imp.
PPL, model build and the FP8 head emulation are shared with gptq_ref_ppl.py (#2249).
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from safetensors.torch import load_file

from gptq_ref_ppl import DTYPES, build, fp8_head, ppl

AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def unpack_awq(qweight, qzeros, bits):
    shifts = torch.arange(0, 32, bits)
    iweights = torch.bitwise_right_shift(qweight[:, :, None], shifts[None, None, :]).to(torch.int8)
    iweights = iweights.view(iweights.shape[0], -1)
    izeros = torch.bitwise_right_shift(qzeros[:, :, None], shifts[None, None, :]).to(torch.int8)
    izeros = izeros.view(izeros.shape[0], -1)
    return iweights, izeros


def reverse_awq_order(iweights, izeros, bits):
    order = torch.arange(iweights.shape[-1], dtype=torch.int32).view(-1, 32 // bits)
    order = order[:, AWQ_REVERSE_ORDER].view(-1)
    return iweights[:, order], izeros[:, order]


def dequantize_gemm(qweight, qzeros, scales, bits, group_size):
    """AutoAWQ dequantize_gemm; returns [K, N] fp16."""
    iweight, izeros = unpack_awq(qweight, qzeros, bits)
    iweight, izeros = reverse_awq_order(iweight, izeros, bits)
    iweight = torch.bitwise_and(iweight, (2**bits) - 1)
    izeros = torch.bitwise_and(izeros, (2**bits) - 1)
    scales = scales.repeat_interleave(group_size, dim=0)
    izeros = izeros.repeat_interleave(group_size, dim=0)
    return (iweight - izeros) * scales


def load_dir(d):
    state = {}
    for f in sorted(glob.glob(f"{d}/*.safetensors")):
        state.update(load_file(f))
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["dequant", "orig"], required=True)
    ap.add_argument("--model", required=True, help="snapshot dir (AWQ for dequant, original for orig)")
    ap.add_argument("--ids", required=True, help="one token id per line (imp diagnostics.dump_tokens)")
    ap.add_argument("--dtype", choices=list(DTYPES), default="fp32")
    ap.add_argument("--head", choices=["src", "fp8"], default="src")
    ap.add_argument("--dump-dir", help="dequant arm: write each projection as raw FP16 [N, K]")
    ap.add_argument("--no-ppl", action="store_true", help="dequant (and dump) only")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    dtype = DTYPES[a.dtype]
    ids = [int(x) for x in open(a.ids)]
    t0 = time.time()
    raw = load_dir(a.model)
    if a.arm == "dequant":
        qc = json.load(open(f"{a.model}/config.json"))["quantization_config"]
        version = str(qc.get("version", "gemm")).lower()
        if qc.get("quant_method") != "awq" or qc.get("bits") != 4 or not qc.get("zero_point") or version != "gemm":
            raise SystemExit(f"unsupported AWQ config: {qc}")
        state = {}
        for k, v in raw.items():
            if k.endswith(".qweight"):
                p = k[: -len(".qweight")]
                wt = dequantize_gemm(v, raw[p + ".qzeros"], raw[p + ".scales"], 4, qc["group_size"]).t()
                if a.dump_dir:
                    os.makedirs(a.dump_dir, exist_ok=True)
                    wt.contiguous().numpy().astype(np.float16).tofile(f"{a.dump_dir}/{p}.f16")
                state[p + ".weight"] = wt.to(dtype)
            elif not k.endswith((".qzeros", ".scales")):
                state[k] = v.to(dtype)
        print(f"dequantized {sum(k.endswith('.qweight') for k in raw)} projections, awq gemm", flush=True)
    else:
        state = {k: v.to(dtype) for k, v in raw.items()}
    if a.no_ppl:
        return
    m = build(state, a.model, dtype)
    if a.head == "fp8":
        fp8_head(m, dtype)
    dt = str(dtype).replace("torch.", "")
    print(f"ref_ppl arm={a.arm} head={a.head} dtype={dt} n_tokens={len(ids)} ppl={ppl(m, ids):.4f} "
          f"secs={time.time() - t0:.0f}", flush=True)


if __name__ == "__main__":
    main()
