#!/usr/bin/env python3
"""Recursive's official Flint NVFP4 GGUF -> SafeTensors in the ModelOpt NVFP4 layout (lossless repack).

Runs inside the vLLM >= 0.33.0 image: vllm.models.flint.gguf_nvfp4 already maps the GGUF tensors to
`<name>.weight` (U8, two E2M1 nibbles per byte), `<name>.weight_scale` (F8_E4M3 per 16 columns) and
`<name>.weight_scale_2` (F32). lm_head is written as BF16 = e2m1 * e4m3 * scale_2 (one rounding to BF16), the
form every loader takes; --lm-head nvfp4 keeps its packed form instead (imp only).

Usage: gguf_to_modelopt.py --gguf FILE --src BF16_REPO_DIR --out DIR [--lm-head bf16|nvfp4] [--shard-gb 1.5]
"""
import argparse
import json
import pathlib
import shutil

import torch
from safetensors.torch import save_file
from vllm.models.flint.gguf_nvfp4 import dequantize_nvfp4, gguf_nvfp4_weights_iterator

ap = argparse.ArgumentParser()
ap.add_argument("--gguf", required=True)
ap.add_argument("--src", required=True, help="RecursiveSI/flint dir: config.json, tokenizer, chat template")
ap.add_argument("--out", required=True)
ap.add_argument("--lm-head", default="bf16", choices=["bf16", "nvfp4"])
ap.add_argument("--shard-gb", type=float, default=1.5)
a = ap.parse_args()
out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)

shard, shard_bytes, shards, index = {}, 0, [], {}
quantized, plain = set(), set()
pending_head = {}


def flush():
    global shard, shard_bytes
    if not shard:
        return
    name = f"model-{len(shards) + 1:05d}.safetensors"
    save_file(shard, str(out / name), metadata={"format": "pt"})
    for k in shard:
        index[k] = name
    shards.append(name)
    shard, shard_bytes = {}, 0


def add(name, t):
    global shard_bytes
    t = t.contiguous()
    shard[name] = t
    shard_bytes += t.numel() * t.element_size()
    if shard_bytes >= a.shard_gb * 2 ** 30:
        flush()


n = 0
for name, t in gguf_nvfp4_weights_iterator(a.gguf, device="cpu"):
    n += 1
    if a.lm_head == "bf16" and name.startswith("lm_head."):
        pending_head[name] = t.clone()
        continue
    if name.endswith(".weight_scale"):
        quantized.add(name.removesuffix(".weight_scale"))
    add(name, t.clone())
if pending_head:
    w = dequantize_nvfp4(pending_head["lm_head.weight"], pending_head["lm_head.weight_scale"],
                         pending_head["lm_head.weight_scale_2"], out_dtype=torch.float32)
    add("lm_head.weight", w.to(torch.bfloat16))
flush()
for f in out.glob('model-*.safetensors'):
    f.chmod(0o644)  # save_file writes 0600; imp's container user reads the mount as another uid
for k in index:
    if k.endswith(".weight") and k.removesuffix(".weight") not in quantized:
        plain.add(k.removesuffix(".weight"))

# shard names carry the total, as HF writes them
total = len(shards)
final = {}
for i, old in enumerate(shards):
    new = f"model-{i + 1:05d}-of-{total:05d}.safetensors"
    (out / old).rename(out / new)
    final[old] = new
(out / "model.safetensors.index.json").write_text(json.dumps(
    {"metadata": {"format": "pt"}, "weight_map": {k: final[v] for k, v in sorted(index.items())}}, indent=1))

src = pathlib.Path(a.src)
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "LICENSE"):
    if (src / f).exists():
        shutil.copy(src / f, out / f)
# 2-D modules kept at source precision: what a ModelOpt reader must not treat as quantized.
exclude = sorted(m for m in plain if m.endswith((".mlp.gate", "embed_tokens")) or m == "lm_head")
cfg = json.loads((src / "config.json").read_text())
cfg["quantization_config"] = {
    "config_groups": {"group_0": {"weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
                                  "targets": ["Linear"]}},
    "ignore": exclude,
    "quant_algo": "W4A16_NVFP4",
    "group_size": 16,
    "producer": {"name": "RecursiveSI/flint-nvfp4-gguf", "version": "repacked"},
    "quant_method": "modelopt",
}
(out / "config.json").write_text(json.dumps(cfg, indent=2))
(out / "hf_quant_config.json").write_text(json.dumps({
    "producer": {"name": "RecursiveSI/flint-nvfp4-gguf", "version": "repacked",
                 "note": "lossless repack of the official NVFP4 GGUF tensors into the ModelOpt layout"},
    "quantization": {"quant_algo": "W4A16_NVFP4", "kv_cache_quant_algo": None, "group_size": 16,
                     "exclude_modules": exclude}}, indent=2))
print(f"tensors read {n}, written {len(index)} in {total} shards, quantized modules {len(quantized)}, "
      f"source-precision 2-D modules {len(exclude)}: {exclude[:3]} ... lm_head {a.lm_head}")
