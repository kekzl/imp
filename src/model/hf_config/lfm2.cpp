#include "core/logging.h"
#include "model/hf_config_hooks.h"
#include "model/json_util.h"
#include "model/model_config.h"

#include <vector>

namespace imp {

namespace {

// Attention layers: lfm2_moe names them in layer_types ("full_attention" / "conv"), lfm2 lists
// full_attn_idxs. Every other layer is a short-conv layer.
std::vector<bool> lfm2_attention_layers(const JValue& eff, int n_layers) {
    std::vector<bool> attn(static_cast<size_t>(n_layers), false);
    const JValue* lt = jobj_find(eff, "layer_types");
    if (lt && lt->type == JType::ARRAY) {
        for (size_t i = 0; i < lt->arr.size() && i < attn.size(); ++i)
            attn[i] = lt->arr[i].str_val == "full_attention";
        return attn;
    }
    const JValue* idx = jobj_find(eff, "full_attn_idxs");
    if (idx && idx->type == JType::ARRAY)
        for (const auto& v : idx->arr)
            if (v.type == JType::NUMBER && v.as_int() >= 0 && v.as_int() < n_layers)
                attn[static_cast<size_t>(v.as_int())] = true;
    return attn;
}

}  // namespace

// LFM2: short-conv layers (in_proj -> B, C, x; y = C * causal_conv3(B * x); out_proj) keep a
// [hidden, conv_L_cache] window in the SSM state pool (inner = hidden, no SSM scan state).
// MoE: sigmoid router, expert_bias for selection only, norm_topk_prob, first num_dense_layers dense.
bool parse_lfm2_config(const JValue& /*root*/, const JValue& eff, ModelConfig& cfg) {
    int conv_l = 3;
    jobj_opt_int(eff, "conv_L_cache", conv_l);
    if (!jobj_get_float(eff, "rms_norm_eps", cfg.rms_norm_eps))
        jobj_opt_float(eff, "norm_eps", cfg.rms_norm_eps);
    jobj_opt_int(eff, "num_dense_layers", cfg.first_k_dense_replace);
    jobj_opt_float(eff, "routed_scaling_factor", cfg.expert_weights_scale);
    const JValue* ntp = jobj_find(eff, "norm_topk_prob");
    if (ntp && ntp->type == JType::NUMBER)
        cfg.expert_weights_norm = (ntp->num_val != 0.0);

    const std::vector<bool> attn = lfm2_attention_layers(eff, cfg.n_layers);
    cfg.n_kv_heads_per_layer.clear();
    int n_attn = 0;
    for (bool a : attn) {
        cfg.n_kv_heads_per_layer.push_back(a ? cfg.n_kv_heads : 0);
        n_attn += a ? 1 : 0;
    }
    if (n_attn == 0) {
        IMP_LOG_ERROR("config.json: LFM2 names no attention layer (layer_types / full_attn_idxs)");
        return false;
    }
    cfg.ssm_inner_size = cfg.d_model;
    cfg.ssm_conv_kernel = conv_l;
    cfg.ssm_group_count = 0;
    cfg.ssm_state_size = 0;
    cfg.ssm_dt_rank = 0;
    cfg.ssm_short_conv = true;
    IMP_LOG_INFO("  LFM2: %d conv + %d attention layers, conv kernel %d, experts %d top-%d (dense %d)",
                 cfg.n_layers - n_attn, n_attn, conv_l, cfg.n_experts, cfg.n_experts_active,
                 cfg.first_k_dense_replace);
    return true;
}

}  // namespace imp
