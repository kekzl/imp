"""Qwen3-VL video reference fixtures from the pinned HF processor (run.sh runs this in a container).

Writes into OUT (argv[2]):
  v1_pixels.f32  float32 LE pixel_values_videos of 4 synthetic 32x64 frames, [tokens, 1536]
  v1_meta.txt    grid_thw of that video
  v2_layout.txt  1 image + 1 video (2 frame-pair groups): prompt ids before/after expansion,
                 mm_token_type_ids, get_rope_index positions, grids, timestamp token ids
MODEL (argv[1]) supplies tokenizer, chat template, config; no weights are read.
"""

import sys
import types

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3VLConfig, Qwen3VLProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor, smart_resize
from transformers.video_utils import VideoMetadata

# Qwen/Qwen3-VL-4B-Instruct video_preprocessor_config.json (the local model dir lacks it).
VIDEO_SIZE = {"shortest_edge": 4096, "longest_edge": 25165824}
# Covers: no-op, min branch, max branch, side < factor upscale, t ties (3 -> 4, 5 -> 4), 200:1 edge.
RESIZE_CASES = [(4, 32, 64), (4, 64, 96), (2, 20, 30), (768, 1080, 1920), (64, 720, 1280),
                (2, 10, 500), (3, 48, 80), (5, 48, 80), (4, 33, 47), (2, 16, 3200)]


def synth_frame(f, h, w):
    """Must match synth_pixel() in tests/test_qwen3vl_video.cpp bit for bit."""
    y, x, c = np.meshgrid(np.arange(h), np.arange(w), np.arange(3), indexing="ij")
    v = (f * 131 + y * 31 + x * 17 + c * 7 + 1).astype(np.uint64) * 2654435761
    return ((v & 0xFFFFFFFF) >> 13 & 0xFF).astype(np.uint8)


def video_processor():
    return Qwen3VLVideoProcessor(
        size=VIDEO_SIZE,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        cap_pixels_per_frame=False,
        do_sample_frames=False,
    )


def ints(xs):
    return " ".join(str(int(v)) for v in xs)


def write_v1(out):
    frames = np.stack([synth_frame(f, 32, 64) for f in range(4)])  # (T, H, W, C)
    res = video_processor()(videos=[frames], return_tensors="pt", do_sample_frames=False)
    px = res["pixel_values_videos"].to(torch.float32).contiguous().numpy()
    px.astype("<f4").tofile(f"{out}/v1_pixels.f32")
    with open(f"{out}/v1_meta.txt", "w") as fh:
        fh.write(f"grid_thw {ints(res['video_grid_thw'][0].tolist())}\n")
        fh.write(f"shape {px.shape[0]} {px.shape[1]}\n")
        # smart_resize(t, h, w) -> (h_bar, w_bar) at the video defaults, factor 32, temporal 2.
        for t, h, w in RESIZE_CASES:
            hb, wb = smart_resize(t, h, w, temporal_factor=2, factor=32,
                                  min_pixels=VIDEO_SIZE["shortest_edge"], max_pixels=VIDEO_SIZE["longest_edge"])
            fh.write(f"resize {t} {h} {w} {hb} {wb}\n")
    print("v1", px.shape, res["video_grid_thw"].tolist())


def write_v2(out, model):
    tok = AutoTokenizer.from_pretrained(model)
    proc = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor.from_pretrained(model),
        tokenizer=tok,
        video_processor=video_processor(),
        chat_template=tok.chat_template or Qwen3VLProcessor.from_pretrained(model).chat_template,
    )
    image = synth_frame(9, 300, 200)
    frames = np.stack([synth_frame(f, 64, 96) for f in range(4)])
    fps = 30.0
    indices = [0, 7, 30, 45]
    meta = VideoMetadata(total_num_frames=46, fps=fps, frames_indices=indices)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "video"},
                                         {"type": "text", "text": "Describe both."}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    res = proc(text=[text], images=[image], videos=[frames], video_metadata=[meta],
               do_sample_frames=False, return_tensors="pt", return_mm_token_type_ids=True)
    ids = res["input_ids"][0]
    cfg = Qwen3VLConfig.from_pretrained(model)
    fake = types.SimpleNamespace(config=cfg,
                                 get_vision_position_ids=lambda *a, **k: Qwen3VLModel.get_vision_position_ids(None, *a, **k))
    pos, delta = Qwen3VLModel.get_rope_index(fake, res["input_ids"], res["mm_token_type_ids"],
                                             image_grid_thw=res["image_grid_thw"],
                                             video_grid_thw=res["video_grid_thw"].clone())
    stamps = proc._calculate_timestamps(list(indices), fps, proc.video_processor.temporal_patch_size)
    with open(f"{out}/v2_layout.txt", "w") as fh:
        fh.write(f"image_hw {image.shape[0]} {image.shape[1]}\n")
        fh.write(f"video_thw {frames.shape[0]} {frames.shape[1]} {frames.shape[2]}\n")
        fh.write(f"frame_seconds {' '.join(repr(i / fps) for i in indices)}\n")
        fh.write(f"image_grid_thw {ints(res['image_grid_thw'][0].tolist())}\n")
        fh.write(f"video_grid_thw {ints(res['video_grid_thw'][0].tolist())}\n")
        fh.write(f"pad_ids {proc.image_token_id} {proc.video_token_id} "
                 f"{proc.vision_start_token_id} {proc.vision_end_token_id}\n")
        for s in stamps:
            label = f"<{s:.1f} seconds>"
            fh.write(f"stamp {label.replace(' ', '_')} {ints(tok(label, add_special_tokens=False)['input_ids'])}\n")
        fh.write(f"prompt_ids {ints(tok(text, add_special_tokens=False)['input_ids'])}\n")
        fh.write(f"input_ids {ints(ids.tolist())}\n")
        fh.write(f"mm_token_type_ids {ints(res['mm_token_type_ids'][0].tolist())}\n")
        for axis in range(3):
            fh.write(f"pos{axis} {ints(pos[axis, 0].tolist())}\n")
        fh.write(f"next_pos {int(pos.max()) + 1}\n")
        fh.write(f"rope_delta {int(delta[0, 0])}\n")
    print("v2", len(ids), res["image_grid_thw"].tolist(), res["video_grid_thw"].tolist(), stamps)


if __name__ == "__main__":
    write_v1(sys.argv[2])
    write_v2(sys.argv[2], sys.argv[1])
