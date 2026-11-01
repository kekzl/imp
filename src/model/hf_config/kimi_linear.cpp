#include "core/logging.h"
#include "model/hf_config_hooks.h"
#include "model/json_util.h"
#include "model/model_config.h"

#include <algorithm>
#include <string>
#include <vector>

namespace imp {

namespace {

// 1-based layer list of linear_attn_config (kda_layers / full_attn_layers) into a 0-based mask.
bool read_layer_list(const JValue& lac, const char* key, int n_layers, std::vector<uint8_t>& mask) {
    const JValue* v = jobj_find(lac, key);
    if (!v || v->type != JType::ARRAY) {
        IMP_LOG_ERROR("config.json: kimi_linear linear_attn_config.%s missing or not a list", key);
        return false;
    }
    mask.assign(static_cast<size_t>(n_layers), 0);
    for (const JValue& e : v->arr) {
        const int64_t one_based = e.type == JType::NUMBER ? e.as_int() : 0;
        if (one_based < 1 || one_based > n_layers || mask[static_cast<size_t>(one_based - 1)]) {
            IMP_LOG_ERROR(
                "config.json: kimi_linear linear_attn_config.%s entry %lld outside 1..%d or repeated", key,
                static_cast<long long>(one_based), n_layers);
            return false;
        }
        mask[static_cast<size_t>(one_based - 1)] = 1;
    }
    return true;
}

}  // namespace

// Kimi-Linear (KimiLinearForCausalLM, model_type kimi_linear). Layer i is KDA when i + 1 is in
// linear_attn_config.kda_layers, MLA when in full_attn_layers (both 1-based, together a partition).
// KDA dims live in linear_attn_config (the top-level head_dim 72 belongs to neither block):
//   num_heads x head_dim (128) -> ssm_dt_rank x ssm_state_size, ssm_inner_size = H * 128, ssm_group_count =
//   H, short_conv_kernel_size -> ssm_conv_kernel; f and the output gate are rank-head_dim LoRA pairs.
// MLA through the DeepSeek parser (q_lora null -> 0, head_dim 128 + 64); mla_use_nope marks every MLA layer
// NoPE. Router: sigmoid + e_score_correction_bias, moe_renormalize, routed_scaling_factor.
bool parse_kimi_linear_config(const JValue& root, const JValue& eff, ModelConfig& cfg) {
    const JValue* lac = jobj_find(eff, "linear_attn_config");
    if (!lac || lac->type != JType::OBJECT) {
        IMP_LOG_ERROR("config.json: kimi_linear without linear_attn_config");
        return false;
    }
    int kda_heads = 0, kda_head_dim = 0, conv = 0;
    jobj_opt_int(*lac, "num_heads", kda_heads);
    jobj_opt_int(*lac, "head_dim", kda_head_dim);
    jobj_opt_int(*lac, "short_conv_kernel_size", conv);
    if (kda_heads <= 0 || kda_head_dim != 128 || conv <= 0) {
        IMP_LOG_ERROR(
            "config.json: kimi_linear KDA %d heads x %d, conv %d: the KDA scans serve head_dim 128 only",
            kda_heads, kda_head_dim, conv);
        return false;
    }
    std::vector<uint8_t> full;
    if (!read_layer_list(*lac, "kda_layers", cfg.n_layers, cfg.kda_layers) ||
        !read_layer_list(*lac, "full_attn_layers", cfg.n_layers, full))
        return false;
    for (int i = 0; i < cfg.n_layers; ++i) {
        if (cfg.kda_layers[static_cast<size_t>(i)] == full[static_cast<size_t>(i)]) {
            IMP_LOG_ERROR("config.json: kimi_linear layer %d is %s", i,
                          full[static_cast<size_t>(i)] ? "both KDA and full attention"
                                                       : "neither KDA nor full attention");
            return false;
        }
    }
    int moe_layer_freq = 1;
    jobj_opt_int(eff, "moe_layer_freq", moe_layer_freq);
    if (moe_layer_freq != 1) {
        IMP_LOG_ERROR("config.json: kimi_linear moe_layer_freq %d (only 1 is modelled)", moe_layer_freq);
        return false;
    }

    cfg.gdn_kda = true;
    cfg.kda_softplus_gate = true;
    cfg.kda_lora_rank = kda_head_dim;  // f_a_proj / g_a_proj: [head_dim, hidden_size]
    cfg.gdn_gate_sigmoid = true;       // FusedRMSNormGated(activation='sigmoid')
    cfg.gdn_grouped_head_layout = true;
    cfg.ssm_conv_kernel = conv;
    cfg.ssm_state_size = kda_head_dim;
    cfg.ssm_group_count = kda_heads;
    cfg.ssm_dt_rank = kda_heads;
    cfg.ssm_inner_size = kda_heads * kda_head_dim;

    // Kimi spellings of the MoE keys the generic parser reads under DeepSeek / Qwen names.
    jobj_opt_int(eff, "num_experts_per_token", cfg.n_experts_active);
    int n_shared = 0;
    if (jobj_get_int(eff, "num_shared_experts", n_shared) && n_shared > 0) {
        cfg.n_experts_shared = n_shared;
        cfg.expert_shared_d_ff = n_shared * cfg.expert_d_ff;  // KimiMLP(moe_intermediate_size * n_shared)
    }
    std::string act;
    jobj_opt_string(eff, "moe_router_activation_func", act);
    if (act != "sigmoid") {
        IMP_LOG_ERROR("config.json: kimi_linear moe_router_activation_func '%s' (only sigmoid is modelled)",
                      act.c_str());
        return false;
    }
    if (!parse_deepseek_config(root, eff, cfg))
        return false;
    cfg.moe_sigmoid_gating = true;
    const JValue* renorm = jobj_find(eff, "moe_renormalize");
    cfg.expert_weights_norm = !renorm || renorm->num_val != 0.0;
    int n_group = 1, topk_group = 0;
    jobj_opt_int(eff, "num_expert_group", n_group);
    jobj_opt_int(eff, "topk_group", topk_group);
    if (n_group > 1 && topk_group > 0 && topk_group < n_group) {
        cfg.moe_n_group = n_group;
        cfg.moe_topk_group = topk_group;
    }
    if (!cfg.is_mla()) {
        IMP_LOG_ERROR("config.json: kimi_linear without kv_lora_rank (its full-attention layers are MLA)");
        return false;
    }
    const JValue* nope = jobj_find(eff, "mla_use_nope");
    if (nope && nope->num_val != 0.0)
        cfg.nope_layers = full;
    IMP_LOG_INFO(
        "  Kimi-Linear: KDA %d heads x %d (conv %d, softplus gate, LoRA rank %d), %zu MLA layers%s, %d "
        "shared "
        "expert(s), routed scale %.3f",
        kda_heads, kda_head_dim, conv, cfg.kda_lora_rank,
        static_cast<size_t>(std::count(full.begin(), full.end(), 1)),
        cfg.nope_layers.empty() ? "" : " (NoPE)", cfg.n_experts_shared, cfg.expert_weights_scale);
    return true;
}

}  // namespace imp
