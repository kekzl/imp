#pragma once

// config.json of the tiny tower in tests/fixtures/internvl (tools/internvl_fixture/gen_encoder.py).

namespace imp_test {

inline constexpr const char* kInternVLTinyConfig = R"({
  "architectures": ["InternVLForConditionalGeneration"], "model_type": "internvl",
  "downsample_ratio": 0.5, "image_token_id": 151671, "projector_hidden_act": "gelu",
  "text_config": {"model_type": "qwen3", "hidden_size": 32},
  "vision_config": {"model_type": "internvl_vision", "hidden_size": 64, "num_attention_heads": 4,
    "intermediate_size": 128, "num_hidden_layers": 2, "image_size": [56, 56], "patch_size": [14, 14],
    "hidden_act": "gelu", "norm_type": "layer_norm", "layer_norm_eps": 1e-06, "use_qk_norm": false,
    "use_absolute_position_embeddings": true, "attention_bias": true}
})";

}  // namespace imp_test
