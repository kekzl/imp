"""InternVL3.5 preprocessing + prompt fixture from the pinned HF processor (run_preprocess.sh).

argv: MODEL_DIR OUT_DIR. Writes:
  synth_300x200.png    synthetic RGB test image (PNG: identical decode in PIL and stb)
  pixels_448.u8        HF pixel_values mapped back to the uint8 grid, [3, 448, 448] CHW
  prompt.txt           pad/start/end ids, template ids before expansion, HF input_ids
crop_to_patches=False (preprocessor_config.json; the processor's own default tiles up to 12 crops).
"""

import sys

import numpy as np
from PIL import Image
from transformers import AutoProcessor

model, out = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(7)
h, w = 300, 200
y, x = np.mgrid[0:h, 0:w]
img = np.stack([(x * 255 // (w - 1)), (y * 255 // (h - 1)), ((x + y) * 3) % 256], axis=-1).astype(np.int32)
img[60:140, 40:120] = [220, 30, 30]          # a block with hard edges
img += rng.integers(-12, 13, size=img.shape)  # texture
img = np.clip(img, 0, 255).astype(np.uint8)
Image.fromarray(img).save(f"{out}/synth_300x200.png")

proc = AutoProcessor.from_pretrained(model)
pil = Image.open(f"{out}/synth_300x200.png").convert("RGB")
msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the image."}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
res = proc(text=[text], images=[pil], crop_to_patches=False, return_tensors="np")
pv = res["pixel_values"][0]  # [3, 448, 448]
mean = np.array(proc.image_processor.image_mean, dtype=np.float64).reshape(3, 1, 1)
std = np.array(proc.image_processor.image_std, dtype=np.float64).reshape(3, 1, 1)
u = (pv.astype(np.float64) * std + mean) * 255.0
print("pixel_values", pv.shape, "u8 reconstruction max |frac|", float(np.abs(u - np.round(u)).max()))
np.round(u).clip(0, 255).astype(np.uint8).tofile(f"{out}/pixels_448.u8")

tok = proc.tokenizer
ids = res["input_ids"][0].tolist()
with open(f"{out}/prompt.txt", "w") as fh:
    fh.write(f"pad_ids {tok.context_image_token_id} {tok.start_image_token_id} {tok.end_image_token_id}\n")
    fh.write(f"template_text_len {len(text)}\n")
    fh.write(f"prompt_ids {' '.join(map(str, tok(text, add_special_tokens=False)['input_ids']))}\n")
    fh.write(f"input_ids {' '.join(map(str, ids))}\n")
print("input_ids", len(ids), "num_patches", res.get("num_patches"))
