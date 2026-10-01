"""Qwen3-VL vision encoder reference for a client-frames video (run_encoder_ref.sh runs this in a container).

argv: MODEL_DIR FRAMES_DIR OUT_DIR. FRAMES_DIR holds frame_1.png .. frame_N.png (same size).
Writes OUT_DIR/encoder_ref.bf16 (BF16 run) and encoder_ref_fp32.f16 (FP32 run rounded to FP16), merger
output [tokens, out_hidden], raw LE, and
OUT_DIR/encoder_ref.txt (grid_thw, shape, relL2 of each run vs the FP32 run).
Only `model.visual.*` tensors are read.
"""

import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

VIDEO_SIZE = {"shortest_edge": 4096, "longest_edge": 25165824}


def load_tower(model_dir, dtype):
    cfg = Qwen3VLConfig.from_pretrained(model_dir).vision_config
    cfg._attn_implementation = "eager"
    tower = Qwen3VLVisionModel(cfg)
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as fh:
        index = json.load(fh)["weight_map"]
    state = {}
    for shard in sorted({f for k, f in index.items() if k.startswith("model.visual.")}):
        with safe_open(os.path.join(model_dir, shard), framework="pt") as sf:
            for k in sf.keys():
                if k.startswith("model.visual."):
                    state[k[len("model.visual."):]] = sf.get_tensor(k)
    missing, unexpected = tower.load_state_dict(state, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return tower.to(dtype).eval()


def main(model_dir, frames_dir, out):
    paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")),
                   key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    frames = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])
    proc = Qwen3VLVideoProcessor(size=VIDEO_SIZE, image_mean=[0.5] * 3, image_std=[0.5] * 3,
                                 cap_pixels_per_frame=False, do_sample_frames=False)
    res = proc(videos=[frames], return_tensors="pt", do_sample_frames=False)
    px, grid = res["pixel_values_videos"], res["video_grid_thw"]
    outs = {}
    for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", torch.float32)):
        tower = load_tower(model_dir, dtype)
        with torch.no_grad():
            outs[name] = tower(px.to(dtype), grid_thw=grid).pooler_output.float()
        del tower
    fp32 = outs["fp32"]

    def rel(a):
        return ((a - fp32).norm() / fp32.norm()).item()

    # bf16: the BF16 run as computed. fp32: the FP32 run rounded to FP16 (relL2 of that rounding is
    # written too, so the stored file's own error is on record).
    outs["bf16"].to(torch.bfloat16).view(torch.int16).numpy().astype("<i2").tofile(
        os.path.join(out, "encoder_ref.bf16"))
    fp32.to(torch.float16).view(torch.int16).numpy().astype("<i2").tofile(os.path.join(out, "encoder_ref_fp32.f16"))
    with open(os.path.join(out, "encoder_ref.txt"), "w") as fh:
        fh.write(f"frames {len(paths)} {frames.shape[1]} {frames.shape[2]}\n")
        fh.write(f"grid_thw {' '.join(str(int(v)) for v in grid[0].tolist())}\n")
        fh.write(f"shape {fp32.shape[0]} {fp32.shape[1]}\n")
        fh.write(f"relL2_hf_bf16_vs_fp32 {rel(outs['bf16']):.6e}\n")
        fh.write(f"relL2_hf_fp16_vs_fp32 {rel(outs['fp16']):.6e}\n")
        fh.write(f"relL2_stored_fp32_as_f16 {rel(fp32.to(torch.float16).float()):.6e}\n")
    print("grid", grid.tolist(), "shape", tuple(fp32.shape), "relL2 vs fp32: bf16", rel(outs["bf16"]),
          "fp16", rel(outs["fp16"]))


if __name__ == "__main__":
    main(*sys.argv[1:4])
