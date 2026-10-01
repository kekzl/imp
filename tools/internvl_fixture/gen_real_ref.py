"""InternVL3.5 real-tower reference: projector output for one image, HF FP32 (and BF16 for context), CPU.

argv: MODEL_DIR IMAGE OUT_DIR. Writes OUT_DIR/<stem>_projector_fp32.f16 ([256, out_hidden], the FP32
output rounded to FP16, raw LE) and OUT_DIR/<stem>_ref.txt (image sha256, shape, relL2 of BF16 and of the
FP16 rounding vs FP32). Preprocessing: the pinned HF processor, crop_to_patches False (one 448 tile).
Only vision_tower.* and multi_modal_projector.* tensors are read.
"""

import hashlib
import os
import sys

import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoProcessor, InternVLConfig
from transformers.models.internvl.modeling_internvl import (InternVLModel, InternVLMultiModalProjector,
                                                            InternVLVisionModel)

model_dir, image_path, out = sys.argv[1:4]
cfg = InternVLConfig.from_pretrained(model_dir)
cfg.vision_config._attn_implementation = "eager"


def load(dtype):
    tower = InternVLVisionModel(cfg.vision_config)
    proj = InternVLMultiModalProjector(cfg)
    ts, ps = {}, {}
    with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt") as sf:
        for k in sf.keys():
            if k.startswith("vision_tower."):
                ts[k[len("vision_tower."):]] = sf.get_tensor(k)
            elif k.startswith("multi_modal_projector."):
                ps[k[len("multi_modal_projector."):]] = sf.get_tensor(k)
    for mod, sd in ((tower, ts), (proj, ps)):
        missing, unexpected = mod.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
    return tower.to(dtype).eval(), proj.to(dtype).eval()


proc = AutoProcessor.from_pretrained(model_dir)
img = Image.open(image_path).convert("RGB")
pv = proc.image_processor(images=[img], crop_to_patches=False, return_tensors="pt")["pixel_values"]
outs = {}
for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16), ("fp16", torch.float16)):
    tower, proj = load(dtype)
    with torch.no_grad():
        h = tower(pixel_values=pv.to(dtype)).last_hidden_state[:, 1:, :]
        side = int(h.shape[1] ** 0.5)
        h = InternVLModel.pixel_shuffle(None, h.reshape(1, side, side, -1), scale_factor=cfg.downsample_ratio)
        outs[name] = proj(h.reshape(1, -1, h.shape[-1]))[0].float()
    del tower, proj
ref = outs["fp32"]
# HF's 448 tile on the u8 grid [3, 448, 448]: separates preprocessing (decode, resize) from encoder error.
mean = torch.tensor(proc.image_processor.image_mean, dtype=torch.float64).view(3, 1, 1)
std = torch.tensor(proc.image_processor.image_std, dtype=torch.float64).view(3, 1, 1)
tile = ((pv[0].double() * std + mean) * 255.0).round().clamp(0, 255).to(torch.uint8)
tile.numpy().tofile(os.path.join(out, f"{os.path.splitext(os.path.basename(image_path))[0]}_pixels_448.u8"))


def rel(a):
    return ((a - ref).norm() / ref.norm()).item()


stem = os.path.splitext(os.path.basename(image_path))[0]
ref.half().view(torch.int16).numpy().astype("<i2").tofile(os.path.join(out, f"{stem}_projector_fp32.f16"))
sha = hashlib.sha256(open(image_path, "rb").read()).hexdigest()
with open(os.path.join(out, f"{stem}_ref.txt"), "w") as fh:
    fh.write(f"image_sha256 {sha}\n")
    fh.write(f"shape {ref.shape[0]} {ref.shape[1]}\n")
    fh.write(f"relL2_hf_bf16_vs_fp32 {rel(outs['bf16']):.6e}\n")
    fh.write(f"relL2_hf_fp16_vs_fp32 {rel(outs['fp16']):.6e}\n")
    fh.write(f"relL2_stored_fp32_as_f16 {rel(ref.half().float()):.6e}\n")
print("shape", tuple(ref.shape), "bf16 vs fp32", rel(outs["bf16"]), "sha256", sha)
