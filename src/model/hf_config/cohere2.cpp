#include "core/logging.h"
#include "model/hf_config_hooks.h"
#include "model/json_util.h"
#include "model/model_config.h"

#include <string>

namespace imp {

// Cohere2-MoE (North-Mini-Code): layer_types sliding_attention (window, RoPE) / full_attention (NoPE,
// except a dense prefix layer when prefix_dense_sliding_window_pattern == 1, which keeps RoPE);
// use_parallel_block: attention and MLP both read input_layernorm(h). First first_k_dense_replace
// layers are dense with prefix_dense_intermediate_size; experts use intermediate_size.
bool parse_cohere2_config(const JValue& /*root*/, const JValue& eff, ModelConfig& cfg) {
    const JValue* lt = jobj_find(eff, "layer_types");
    if (!lt || lt->type != JType::ARRAY || static_cast<int>(lt->arr.size()) != cfg.n_layers) {
        IMP_LOG_ERROR("config.json: Cohere2 needs layer_types for all %d layers", cfg.n_layers);
        return false;
    }
    int dense_prefix = 0, prefix_pattern = 0, prefix_d_ff = 0, expert_d_ff = 0;
    jobj_opt_int(eff, "first_k_dense_replace", dense_prefix);
    jobj_opt_int(eff, "prefix_dense_sliding_window_pattern", prefix_pattern);
    jobj_opt_int(eff, "prefix_dense_intermediate_size", prefix_d_ff);
    jobj_opt_int(eff, "intermediate_size", expert_d_ff);
    cfg.first_k_dense_replace = dense_prefix;
    if (prefix_d_ff > 0)
        cfg.d_ff = prefix_d_ff;
    if (expert_d_ff > 0)
        cfg.expert_d_ff = expert_d_ff;
    int n_swa = 0;
    cfg.swa_layers.assign(static_cast<size_t>(cfg.n_layers), 0);
    cfg.nope_layers.assign(static_cast<size_t>(cfg.n_layers), 0);
    for (int i = 0; i < cfg.n_layers; ++i) {
        const bool swa = lt->arr[static_cast<size_t>(i)].str_val == "sliding_attention";
        const bool force_rope = i < dense_prefix && prefix_pattern == 1;
        cfg.swa_layers[static_cast<size_t>(i)] = swa ? 1 : 0;
        cfg.nope_layers[static_cast<size_t>(i)] = (!swa && !force_rope) ? 1 : 0;
        n_swa += swa ? 1 : 0;
    }
    const JValue* pb = jobj_find(eff, "use_parallel_block");
    cfg.parallel_block = pb && pb->type == JType::NUMBER && pb->num_val != 0.0;
    std::string sel;
    jobj_opt_string(eff, "expert_selection_fn", sel);
    cfg.moe_sigmoid_gating = sel == "sigmoid";
    const JValue* ntp = jobj_find(eff, "norm_topk_prob");
    cfg.expert_weights_norm = ntp && ntp->type == JType::NUMBER && ntp->num_val != 0.0;
    int n_shared = 0;
    jobj_opt_int(eff, "num_shared_experts", n_shared);
    if (n_shared > 0) {
        IMP_LOG_ERROR("config.json: Cohere2 num_shared_experts=%d (average combination) is not supported",
                      n_shared);
        return false;
    }
    IMP_LOG_INFO(
        "  Cohere2: %d SWA (RoPE) + %d full layers, window %d, parallel=%d, dense prefix %d (d_ff %d), "
        "router %s",
        n_swa, cfg.n_layers - n_swa, cfg.sliding_window, static_cast<int>(cfg.parallel_block), dense_prefix,
        cfg.d_ff, sel.c_str());
    return true;
}

}  // namespace imp
