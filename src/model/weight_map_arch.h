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
// kimi_linear: KDA layers' self_attn.* renamed to linear_attn.* (dense q/k/v fused like bailing_hybrid,
// NVFP4 q/k/v kept as in_proj_q/k/v), block_sparse_moe.* to the DeepSeek mlp.* spelling.
// No-op for other archs. Throws std::runtime_error on a checkpoint it cannot serve.
void arch_prepare_tensor_map(const ModelConfig& cfg, std::unordered_map<std::string, Tensor>& tensors,
                             std::vector<void*>& owned);
void bailing_hybrid_prepare_tensor_map(const ModelConfig& cfg,
                                       std::unordered_map<std::string, Tensor>& tensors,
                                       std::vector<void*>& owned);
void kimi_linear_prepare_tensor_map(const ModelConfig& cfg, std::unordered_map<std::string, Tensor>& tensors,
                                    std::vector<void*>& owned);

// F32/F16/BF16, no scales, >= 2-D: a tensor kda_concat_rows can fuse.
[[nodiscard]] bool kda_dense_tensor(const Tensor& t);
// Row-concatenation of three dense tensors with equal trailing dims and dtype, bytes owned by `owned`.
// Throws std::runtime_error("<what> ...") otherwise.
Tensor kda_concat_rows(const Tensor* const parts[3], std::vector<void*>& owned, const std::string& what);

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

// Kimi-Linear linear_attn.<proj> slots kimi_linear_tensors.cpp emits: in_proj_q -> ssm_in,
// in_proj_k / in_proj_v -> kda_k_proj / kda_v_proj, f_a_proj / g_a_proj -> kda_f_a / kda_g_a.
// nullptr = not one of them.
[[nodiscard]] Tensor* kimi_kda_slot(TransformerLayer& layer, const std::string& proj);
// The nvfp4_scratch_ slot name ("ssm_in", "kda_k_proj", ...) for the same projections; nullptr otherwise.
[[nodiscard]] const char* kimi_kda_scale_slot(const std::string& proj);

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
