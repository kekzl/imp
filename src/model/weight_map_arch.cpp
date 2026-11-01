#include "model/weight_map_arch.h"

#include "model/lfm2_names.h"
#include "model/model.h"
#include "model/model_limits.h"

#include <string_view>

namespace imp {

// Arch-specific name translation before the generic matchers. false = drop the tensor as an MTP
// sidecar: DeepSeek-style checkpoints (GLM-4.7-Flash) append num_nextn_predict_layers as
// model.layers.<n_layers + i>.*, which would otherwise grow the decoder stack.
void base_model_canonical_name(std::string& name) {
    const bool layer = name.starts_with("layers.") && (name.contains(".self_attn.") ||
                                                       name.contains(".mlp.") || name.contains("layernorm."));
    if (layer || name == "embed_tokens.weight" || name == "norm.weight")
        name = "model." + name;
}

bool arch_canonical_name(ModelArch arch, int declared_layers, std::string& name) {
    if (arch == ModelArch::LFM2) {
        name = lfm2_canonical_name(name);
        return true;
    }
    constexpr std::string_view kLayers = "model.layers.";
    if (arch != ModelArch::DEEPSEEK || declared_layers <= 0 || !name.starts_with(kLayers))
        return true;
    // q_lora: q_b_proj is the Q projection proper (input q_lora_rank), weight and NVFP4 scales alike.
    if (const size_t qb = name.find(".self_attn.q_b_proj."); qb != std::string::npos)
        name.replace(qb, 20, ".self_attn.q_proj.");
    // DeepSeek-V3 / GLM noaux_tc score correction is the router bias the MoE routing adds before top-k.
    constexpr std::string_view kScoreBias = ".mlp.gate.e_score_correction_bias";
    if (name.size() > kScoreBias.size() && name.ends_with(kScoreBias))
        name.replace(name.size() - kScoreBias.size(), kScoreBias.size(), ".mlp.gate.bias");
    return parse_index(name.substr(kLayers.size(), name.find('.', kLayers.size()) - kLayers.size())) <
           declared_layers;
}

// MLA self_attn.<proj>.weight. kv_a_proj_with_mqa packs latent (kv_lora_rank) + rope dims, split at
// projection time; kv_a / q_a layernorms are RMSNorm weights, never quantized. q_a_proj (V3, GLM
// q_lora_rank) feeds q_b_proj, which arch_canonical_name renames to q_proj (wq).
bool assign_mla_tensor(TransformerLayer& layer, const std::string& proj, const Tensor& t) {
    if (proj == "kv_a_proj_with_mqa")
        layer.kv_a_proj = t;
    else if (proj == "kv_a_layernorm")
        layer.kv_a_layernorm = t;
    else if (proj == "kv_b_proj")
        layer.kv_b_proj = t;
    else if (proj == "q_a_proj")
        layer.q_a_proj = t;
    else if (proj == "q_a_layernorm")
        layer.q_a_layernorm = t;
    else
        return false;
    return true;
}

namespace {

Tensor* gemma4_pli_model_slot(Model& model, const std::string& name) {
    if (name == "model.embed_tokens_per_layer.weight")
        return &model.pli_table_;
    if (name == "model.per_layer_model_projection.weight")
        return &model.pli_model_proj_;
    if (name == "model.per_layer_projection_norm.weight")
        return &model.pli_proj_norm_;
    return nullptr;
}

// rest = the name after "model.layers.<idx>."
Gemma4ETensor gemma4_e_layer_tensor(Model& model, int idx, std::string_view rest, const Tensor& t) {
    if (idx < 0 || idx >= kMaxModelLayers)
        return Gemma4ETensor::Other;
    Tensor* slot = nullptr;
    if (rest == "per_layer_input_gate.weight" || rest == "per_layer_projection.weight" ||
        rest == "post_per_layer_input_norm.weight") {
        if (idx >= static_cast<int>(model.layers_.size()))
            model.layers_.resize(static_cast<size_t>(idx) + 1);
        TransformerLayer& L = model.layers_[static_cast<size_t>(idx)];
        slot = rest.starts_with("per_layer_input_gate")   ? &L.pli_gate
               : rest.starts_with("per_layer_projection") ? &L.pli_proj
                                                          : &L.pli_norm;
        *slot = t;
        return Gemma4ETensor::Assigned;
    }
    const bool kv_tensor = rest.starts_with("self_attn.k_proj.") || rest.starts_with("self_attn.v_proj.") ||
                           rest.starts_with("self_attn.k_norm.");
    return kv_tensor && model.config_.layer_kv_shared(idx) ? Gemma4ETensor::Unused : Gemma4ETensor::Other;
}

}  // namespace

Gemma4ETensor assign_gemma4_e_tensor(bool gemma4_layout, Model& model, const std::string& name,
                                     const Tensor& t) {
    if (!gemma4_layout)
        return Gemma4ETensor::Other;
    if (Tensor* slot = gemma4_pli_model_slot(model, name)) {
        *slot = t;
        return Gemma4ETensor::Assigned;
    }
    constexpr std::string_view kLayers = "model.layers.";
    if (!name.starts_with(kLayers))
        return Gemma4ETensor::Other;
    const size_t dot = name.find('.', kLayers.size());
    if (dot == std::string::npos || dot == kLayers.size())
        return Gemma4ETensor::Other;
    int idx = 0;
    for (size_t i = kLayers.size(); i < dot; ++i) {
        if (name[i] < '0' || name[i] > '9' || idx >= kMaxModelLayers)
            return Gemma4ETensor::Other;
        idx = (idx * 10) + (name[i] - '0');
    }
    return gemma4_e_layer_tensor(model, idx, std::string_view(name).substr(dot + 1), t);
}

}  // namespace imp
