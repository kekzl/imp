#!/usr/bin/env python3
"""Generate golden reference outputs from HuggingFace models for testing imp.

Outputs, per prompt, into <out-dir>/<prompt-id>/:
  meta.json         : model id, prompt text, tokenizer info, tensor shapes/dtypes
  input_ids.npy     : int32 [seq]
  embeddings.npy    : fp32 [seq, d_model]            (post-embedding hidden state)
  hidden_<i>.npy    : fp32 [seq, d_model]            (output of transformer layer i)
  attn_<i>.npy      : fp32 [n_heads, seq, seq]       (attention probabilities, layer i)
  logits.npy        : fp32 [seq, vocab]              (pre-softmax final logits)
  greedy_tokens.npy : int32 [N]                      (N greedy-decoded tokens)

The imp test harness loads these .npy files and compares against imp's
internal tensors (token-by-token and layer-by-layer) to catch kernel
regressions that manifest only on real model data.

Usage:
  python gen_reference.py \\
      --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
      --out  tests/reference/data/tinyllama \\
      --prompts tests/reference/prompts.json \\
      --decode-tokens 16

The script is self-contained and CPU-only by default (use --device cuda
to run on GPU). `transformers` and `numpy` are the only runtime deps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any


DEFAULT_PROMPTS: list[dict[str, str]] = [
    {"id": "short",  "text": "Hello, world!"},
    {"id": "code",   "text": "def fibonacci(n):"},
    {"id": "chat",   "text": "The capital of France is"},
]


@dataclass
class Args:
    model: str
    out: str
    prompts: str | None
    decode_tokens: int
    device: str
    dtype: str
    trust_remote_code: bool


def parse_args(argv: list[str]) -> Args:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="HuggingFace model id or local path (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0)")
    p.add_argument("--out", required=True,
                   help="Output directory for reference data")
    p.add_argument("--prompts", default=None,
                   help="Path to JSON file with [{id, text}, ...] (default: builtin set)")
    p.add_argument("--decode-tokens", type=int, default=16,
                   help="Number of greedy-decoded continuation tokens (default 16)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="Device for the HuggingFace model (default cpu)")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"],
                   help="Model compute dtype. Reference tensors are always saved as fp32.")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="Pass trust_remote_code=True to HuggingFace loaders")
    ns = p.parse_args(argv)
    return Args(
        model=ns.model,
        out=ns.out,
        prompts=ns.prompts,
        decode_tokens=ns.decode_tokens,
        device=ns.device,
        dtype=ns.dtype,
        trust_remote_code=ns.trust_remote_code,
    )


def load_prompts(path: str | None) -> list[dict[str, str]]:
    if path is None:
        return DEFAULT_PROMPTS
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(p, dict) and "id" in p and "text" in p for p in data):
        raise ValueError(f"{path}: expected list of {{'id', 'text'}} objects")
    return data


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_cpu_fp32_numpy(t: Any):  # torch.Tensor -> numpy.ndarray
    import numpy as np
    return t.detach().to("cpu").to(dtype=_torch_fp32()).contiguous().numpy().astype(np.float32, copy=False)


def _torch_fp32():
    import torch
    return torch.float32


def _torch_dtype(name: str):
    import torch
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def run_prompt(model, tokenizer, prompt: dict[str, str], args: Args, out_root: str) -> dict:
    import numpy as np
    import torch

    prompt_dir = os.path.join(out_root, prompt["id"])
    ensure_dir(prompt_dir)

    enc = tokenizer(prompt["text"], return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(args.device)
    seq_len = int(input_ids.shape[1])

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    # hidden_states is a tuple of (n_layers + 1) tensors [1, seq, d_model].
    # Index 0 is post-embedding; 1..n_layers are transformer block outputs.
    hs = outputs.hidden_states
    if hs is None or len(hs) < 2:
        raise RuntimeError("Model did not return hidden_states; check output_hidden_states support")

    np.save(os.path.join(prompt_dir, "input_ids.npy"),
            input_ids[0].to("cpu").numpy().astype(np.int32, copy=False))

    np.save(os.path.join(prompt_dir, "embeddings.npy"), _to_cpu_fp32_numpy(hs[0][0]))
    for i, h in enumerate(hs[1:], start=0):
        np.save(os.path.join(prompt_dir, f"hidden_{i}.npy"), _to_cpu_fp32_numpy(h[0]))

    attns = outputs.attentions or ()
    for i, a in enumerate(attns):
        np.save(os.path.join(prompt_dir, f"attn_{i}.npy"), _to_cpu_fp32_numpy(a[0]))

    np.save(os.path.join(prompt_dir, "logits.npy"), _to_cpu_fp32_numpy(outputs.logits[0]))

    greedy = greedy_decode(model, input_ids, args.decode_tokens)
    np.save(os.path.join(prompt_dir, "greedy_tokens.npy"),
            greedy.to("cpu").numpy().astype(np.int32, copy=False))

    meta = {
        "model": args.model,
        "prompt_id": prompt["id"],
        "prompt_text": prompt["text"],
        "seq_len": seq_len,
        "n_layers": len(hs) - 1,
        "n_attn_layers": len(attns),
        "hidden_dim": int(hs[0].shape[-1]),
        "vocab_size": int(outputs.logits.shape[-1]),
        "decode_tokens": args.decode_tokens,
        "dtype": args.dtype,
    }
    with open(os.path.join(prompt_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def greedy_decode(model, input_ids, n_new: int):
    import torch
    ids = input_ids
    for _ in range(n_new):
        with torch.inference_mode():
            out = model(input_ids=ids, use_cache=False, return_dict=True)
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    return ids[0, input_ids.shape[1]:]


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    try:
        import numpy  # noqa: F401
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"gen_reference.py: missing dependency ({e}).\n"
              f"Install with: pip install torch transformers numpy", file=sys.stderr)
        return 2

    prompts = load_prompts(args.prompts)
    ensure_dir(args.out)

    print(f"Loading tokenizer: {args.model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

    print(f"Loading model:     {args.model} ({args.dtype} on {args.device})", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=_torch_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
    )
    model.to(args.device)
    model.eval()

    index: list[dict] = []
    for prompt in prompts:
        print(f"  prompt[{prompt['id']}]: {prompt['text']!r}", file=sys.stderr)
        meta = run_prompt(model, tokenizer, prompt, args, args.out)
        index.append(meta)

    with open(os.path.join(args.out, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "prompts": index}, f, indent=2)

    print(f"Reference data written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
