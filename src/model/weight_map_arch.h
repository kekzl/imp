#pragma once

#include "model/model_config.h"
#include "model/model_arch.h"

#include <string>
#include <vector>

namespace imp {

// Arch-specific name translation before the generic matchers (LFM2 names, DeepSeek/GLM q_b_proj and
// e_score_correction_bias). false = drop the tensor as an MTP sidecar: DeepSeek-style checkpoints
// (GLM-4.7-Flash) append num_nextn_predict_layers as model.layers.<n_layers + i>.*.
[[nodiscard]] bool arch_canonical_name(ModelArch arch, int declared_layers, std::string& name);

// Base-model (AutoModel) exports drop the `model.` prefix (Qwen3-Embedding: layers.N.self_attn.*,
// embed_tokens, norm); adds it back. HF names only: Mistral-native `.attention.` stays untouched.
void base_model_canonical_name(std::string& name);

// MLA self_attn.<proj>.weight (kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj, q_a_proj, q_a_layernorm).
// false = not an MLA tensor.
[[nodiscard]] bool assign_mla_tensor(TransformerLayer& layer, const std::string& proj, const Tensor& t);

class Model;

// Gemma-4 E-variant tensors (wrapper prefix stripped; Other unless gemma4_layout):
//   Assigned: embed_tokens_per_layer, per_layer_model_projection, per_layer_projection_norm and
//             layers.N.{per_layer_input_gate, per_layer_projection, post_per_layer_input_norm}.
//   Unused:   layers.N.self_attn.{k_proj,v_proj,k_norm} of a KV-shared layer (HF ignores them too).
//   Other:    the generic matchers take it.
enum class Gemma4ETensor { Other, Assigned, Unused };
[[nodiscard]] Gemma4ETensor assign_gemma4_e_tensor(bool gemma4_layout, Model& model, const std::string& name,
                                                   const Tensor& t);

}  // namespace imp
