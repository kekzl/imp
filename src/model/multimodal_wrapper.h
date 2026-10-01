#pragma once

// Multimodal "ForConditionalGeneration" wrappers around a text LM: which LM class they wrap and
// how their tensor names map onto the bare text model. Own TU: hf_config_loader.cpp and
// weight_map.cpp are at their size ceilings.

#include "core/tensor.h"
#include "model/json_util.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace imp {

class Model;
struct VisionModel;

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

// vision_config -> *out_vision_tower through the family registry (vision_family.h). A tower that is
// not parsed (unknown family, rejected geometry, no out pointer) says so: no silent text-only degrade.
void load_vision_tower_config(const JValue& root, std::unique_ptr<VisionModel>* out_vision_tower);

// The vision tower's tensors through its family's loader; an incomplete tower is dropped
// (text-only) rather than handed to the encoder with null slots.
void load_vision_tower_weights(const std::unordered_map<std::string, Tensor>& tensors, Model& model);

}  // namespace imp
