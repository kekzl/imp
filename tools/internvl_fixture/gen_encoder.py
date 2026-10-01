"""Tiny InternVL vision tower + projector from the pinned HF modules, FP32, per-stage outputs.

run_encoder.sh runs this in a container. Writes into argv[1]:
  tiny_tower.st   weights under the checkpoint's own names (vision_tower.*, multi_modal_projector.*)
  tiny_stages.st  pixel_values [1,3,56,56] and FP32 stage outputs: embeddings [17,64],
                           layer_0 / layer_1 [17,64], shuffled [4,256], projected [4,32]
Weights and pixels are FP16-representable (imp stores FP16). Magnitudes are set by hand (HF init uses initializer_range 1e-10 and lambda 0.1): LN gain ~1,
Linear ~1/sqrt(fan_in), lambda 0.5..1.5, so attention, layer scale and the shuffle order all move
the output (docs/plans/2026-07-31-qwen3-vl-vision.md, realistic-magnitude trap).
"""

import sys

import torch
from safetensors.torch import save_file
from transformers import InternVLConfig
from transformers.models.internvl.modeling_internvl import (InternVLModel, InternVLMultiModalProjector,
                                                            InternVLVisionModel)

torch.manual_seed(1234)
cfg = InternVLConfig(
    vision_config=dict(hidden_size=64, num_attention_heads=4, intermediate_size=128, num_hidden_layers=2,
                       image_size=[56, 56], patch_size=[14, 14], hidden_act="gelu", norm_type="layer_norm",
                       layer_norm_eps=1e-6, use_qk_norm=False, attention_bias=True, use_mean_pooling=True,
                       use_absolute_position_embeddings=True, layer_scale_init_value=0.1),
    text_config=dict(model_type="qwen3", hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                     num_key_value_heads=1, intermediate_size=64, vocab_size=64),
    downsample_ratio=0.5, projector_hidden_act="gelu")
cfg.vision_config._attn_implementation = "eager"

tower = InternVLVisionModel(cfg.vision_config).eval()
proj = InternVLMultiModalProjector(cfg).eval()


def realistic_(module):
    for name, p in module.named_parameters():
        with torch.no_grad():
            if name.endswith("lambda_1") or name.endswith("lambda_2"):
                p.uniform_(0.5, 1.5)
            elif "layernorm" in name or "layer_norm" in name:
                if name.endswith("weight"):
                    p.copy_(1.0 + 0.1 * torch.randn_like(p))
                else:
                    p.normal_(0.0, 0.05)
            elif name.endswith("bias"):
                p.normal_(0.0, 0.02)
            elif name.endswith("cls_token"):
                p.normal_(0.0, 1.0)
            elif name.endswith("position_embeddings"):
                p.normal_(0.0, 0.5)
            else:  # Linear [out, in] and Conv2d [out, c, kh, kw]
                fan_in = p[0].numel()
                p.normal_(0.0, fan_in ** -0.5)


realistic_(tower)
realistic_(proj)
# imp holds the tower in FP16: weights and pixels are made FP16-representable, so the FP32
# reference measures activation error only, not the weight rounding imp cannot avoid.
with torch.no_grad():
    for p in list(tower.parameters()) + list(proj.parameters()):
        p.copy_(p.half().float())

pixels = torch.randn(1, 3, 56, 56).half().float()
stages = {"pixel_values": pixels}
with torch.no_grad():
    h = tower.embeddings(pixels)
    stages["embeddings"] = h[0].clone()
    for i, layer in enumerate(tower.encoder.layer):
        h = layer(h)
        stages[f"layer_{i}"] = h[0].clone()
    feats = h[:, 1:, :]
    side = int(feats.shape[1] ** 0.5)
    feats = feats.reshape(1, side, side, -1)
    shuffled = InternVLModel.pixel_shuffle(None, feats, scale_factor=0.5)
    shuffled = shuffled.reshape(1, -1, shuffled.shape[-1])
    stages["shuffled"] = shuffled[0].clone()
    stages["projected"] = proj(shuffled)[0].clone()

weights = {f"vision_tower.{k}": v.contiguous() for k, v in tower.state_dict().items()}
weights.update({f"multi_modal_projector.{k}": v.contiguous() for k, v in proj.state_dict().items()})
out = sys.argv[1]
save_file(weights, f"{out}/tiny_tower.st")
save_file({k: v.contiguous() for k, v in stages.items()}, f"{out}/tiny_stages.st")
for k, v in stages.items():
    print(k, tuple(v.shape), f"rms={v.pow(2).mean().sqrt().item():.4f}")
