#pragma once

// Multimodal "ForConditionalGeneration" wrappers around a text LM: which LM class they wrap and
// how their tensor names map onto the bare text model. Own TU: hf_config_loader.cpp and
// weight_map.cpp are at their size ceilings.

#include "model/json_util.h"

#include <string>

namespace imp {

// InternVLForConditionalGeneration wraps whatever text_config.architectures[0] names (InternVL3.5
// ships Qwen3 and GPT-OSS LMs under one class); every other class is returned unchanged.
std::string wrapped_lm_architecture(const JValue& root, const std::string& top_arch);

// Wrapper tensors that are not LM weights: model.vision_tower. (Gemma-4), model.visual. (Qwen3-VL,
// Qwen3.6), model.embed_vision. (Gemma-4 unified), vision_tower. and multi_modal_projector.
// (InternVL, HF layout).
bool is_wrapper_vision_name(const std::string& name);

// The wrapper's LM names -> bare text-model names: model.language_model.X (Qwen3-VL, Gemma-4) and
// language_model.model.X (InternVL) -> model.X; language_model.lm_head.* -> lm_head.*;
// language_model.embed_tokens.* -> model.embed_tokens.*. Other names unchanged.
std::string strip_wrapper_lm_prefix(const std::string& name);

}  // namespace imp
