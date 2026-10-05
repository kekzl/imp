#include "model/weight_map_arch.h"

#include "model/lfm2_names.h"
#include "model/model_limits.h"

#include <string_view>

namespace imp {

// Arch-specific name translation before the generic matchers. false = drop the tensor as an MTP
// sidecar: DeepSeek-style checkpoints (GLM-4.7-Flash) append num_nextn_predict_layers as
// model.layers.<n_layers + i>.*, which would otherwise grow the decoder stack.
bool arch_canonical_name(ModelArch arch, int declared_layers, std::string& name) {
    if (arch == ModelArch::LFM2) {
        name = lfm2_canonical_name(name);
        return true;
    }
    constexpr std::string_view kLayers = "model.layers.";
    if (arch != ModelArch::DEEPSEEK || declared_layers <= 0 || name.compare(0, kLayers.size(), kLayers) != 0)
        return true;
    // q_lora: q_b_proj is the Q projection proper (input q_lora_rank), weight and NVFP4 scales alike.
    if (const size_t qb = name.find(".self_attn.q_b_proj."); qb != std::string::npos)
        name.replace(qb, 20, ".self_attn.q_proj.");
    // DeepSeek-V3 / GLM noaux_tc score correction is the router bias the MoE routing adds before top-k.
    constexpr std::string_view kScoreBias = ".mlp.gate.e_score_correction_bias";
    if (name.size() > kScoreBias.size() &&
        name.compare(name.size() - kScoreBias.size(), kScoreBias.size(), kScoreBias) == 0)
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

}  // namespace imp
