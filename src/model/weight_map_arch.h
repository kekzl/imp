#pragma once

#include "model/model_config.h"
#include "model/model_arch.h"

#include <string>

namespace imp {

// Arch-specific name translation before the generic matchers (LFM2 names, DeepSeek/GLM q_b_proj and
// e_score_correction_bias). false = drop the tensor as an MTP sidecar: DeepSeek-style checkpoints
// (GLM-4.7-Flash) append num_nextn_predict_layers as model.layers.<n_layers + i>.*.
[[nodiscard]] bool arch_canonical_name(ModelArch arch, int declared_layers, std::string& name);

// MLA self_attn.<proj>.weight (kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj, q_a_proj, q_a_layernorm).
// false = not an MLA tensor.
[[nodiscard]] bool assign_mla_tensor(TransformerLayer& layer, const std::string& proj, const Tensor& t);

}  // namespace imp
