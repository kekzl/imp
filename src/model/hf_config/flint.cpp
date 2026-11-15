#include "core/logging.h"
#include "model/hf_config_hooks.h"
#include "model/json_util.h"
#include "model/model_config.h"

#include <string>

namespace imp {

// Flint (RecursiveSI/flint): the Qwen3-MoE block with learned attention sinks, layer_types
// sliding_attention (window, RoPE) / full_attention, no positional encoding on no_rope_layer_indices,
// dense SwiGLU on mlp_only_layers (tensor-driven), sigmoid top-k router renormalized to 1, ungated
// shared expert.
bool parse_flint_config(const JValue& /*root*/, const JValue& eff, ModelConfig& cfg) {
    const JValue* lt = jobj_find(eff, "layer_types");
    if (!lt || lt->type != JType::ARRAY || static_cast<int>(lt->arr.size()) != cfg.n_layers) {
        IMP_LOG_ERROR("config.json: Flint needs layer_types for all %d layers", cfg.n_layers);
        return false;
    }
    std::string softmax;
    jobj_opt_string(eff, "softmax_type", softmax);
    if (softmax != "learnable") {
        IMP_LOG_ERROR("config.json: Flint softmax_type \"%s\" is not supported (learnable sinks only)",
                      softmax.c_str());
        return false;
    }
    std::string score;
    jobj_opt_string(eff, "router_score_function", score);
    if (score != "sigmoid" && score != "softmax") {
        IMP_LOG_ERROR("config.json: Flint router_score_function \"%s\" is not supported", score.c_str());
        return false;
    }
    cfg.moe_sigmoid_gating = score == "sigmoid";
    const JValue* ntp = jobj_find(eff, "norm_topk_prob");
    cfg.expert_weights_norm = ntp && ntp->type == JType::NUMBER && ntp->num_val != 0.0;

    int n_swa = 0;
    cfg.swa_layers.assign(static_cast<size_t>(cfg.n_layers), 0);
    cfg.nope_layers.assign(static_cast<size_t>(cfg.n_layers), 0);
    for (int i = 0; i < cfg.n_layers; ++i) {
        const bool swa = lt->arr[static_cast<size_t>(i)].str_val == "sliding_attention";
        cfg.swa_layers[static_cast<size_t>(i)] = swa ? 1 : 0;
        n_swa += swa ? 1 : 0;
    }
    int n_nope = 0;
    if (const JValue* nr = jobj_find(eff, "no_rope_layer_indices"); nr && nr->type == JType::ARRAY) {
        for (const JValue& v : nr->arr) {
            const int l = v.type == JType::NUMBER ? static_cast<int>(v.num_val) : -1;
            if (l < 0 || l >= cfg.n_layers) {
                IMP_LOG_ERROR("config.json: Flint no_rope_layer_indices entry %d outside [0, %d)", l,
                              cfg.n_layers);
                return false;
            }
            n_nope += cfg.nope_layers[static_cast<size_t>(l)] == 0 ? 1 : 0;
            cfg.nope_layers[static_cast<size_t>(l)] = 1;
        }
    }
    if (n_swa > 0 && cfg.sliding_window <= 0) {
        IMP_LOG_ERROR("config.json: Flint has %d sliding layers but no sliding_window", n_swa);
        return false;
    }
    int dense = 0;
    if (const JValue* mo = jobj_find(eff, "mlp_only_layers"); mo && mo->type == JType::ARRAY)
        dense = static_cast<int>(mo->arr.size());
    int shared_d_ff = 0;
    jobj_opt_int(eff, "shared_expert_intermediate_size", shared_d_ff);
    if (shared_d_ff > 0) {
        cfg.n_experts_shared = 1;
        cfg.expert_shared_d_ff = shared_d_ff;
    }
    IMP_LOG_INFO(
        "  Flint: %d SWA + %d full layers, window %d, %d NoPE layers, %d dense layers (d_ff %d), "
        "router %s norm=%d, shared expert d_ff %d",
        n_swa, cfg.n_layers - n_swa, cfg.sliding_window, n_nope, dense, cfg.d_ff, score.c_str(),
        static_cast<int>(cfg.expert_weights_norm), shared_d_ff);
    return true;
}

}  // namespace imp
