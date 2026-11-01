#pragma once

#include "model/model_config.h"
#include "model/model_arch.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace imp {

// Whole-map rewrite before WeightMap::apply_weights for archs whose layout is not one tensor per slot.
// bailing_hybrid (Ling-3.0): attention.* renamed to linear_attn.* (KDA) / self_attn.* (MLA), KDA q/k/v
// projections and short convs row-concatenated into in_proj_qkv / conv1d (bytes owned by `owned`).
// No-op for other archs. Throws std::runtime_error on a checkpoint it cannot serve.
void arch_prepare_tensor_map(const ModelConfig& cfg, std::unordered_map<std::string, Tensor>& tensors,
                             std::vector<void*>& owned);

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

}  // namespace imp
