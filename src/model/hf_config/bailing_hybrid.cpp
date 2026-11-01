#include "core/logging.h"
#include "model/hf_config_hooks.h"
#include "model/json_util.h"
#include "model/model_config.h"

#include <string>

namespace imp {

// Ling-3.0 (BailingMoeV3, model_type bailing_hybrid): layer i is MLA when (i + 1) % layer_group_size
// == 0 (tensor presence decides at run time), else Kimi Delta Attention on the GDN slots:
//   num_attention_heads x head_dim (128) -> ssm_dt_rank x ssm_state_size, ssm_inner_size = H * 128,
//   ssm_group_count = H (H_k = H_v), short_conv_kernel_size -> ssm_conv_kernel.
// MLA, router (sigmoid + expert_bias, n_group / topk_group, routed_scaling_factor) and
// first_k_dense_replace come from the DeepSeek parser; MLA overrides head_dim to qk 128 + 64.
bool parse_bailing_hybrid_config(const JValue& root, const JValue& eff, ModelConfig& cfg) {
    int kda_head_dim = 0, conv = 0, group = 4, n_shared = 0, shared_d_ff = 0;
    jobj_opt_int(eff, "head_dim", kda_head_dim);
    jobj_opt_int(eff, "short_conv_kernel_size", conv);
    jobj_opt_int(eff, "layer_group_size", group);
    jobj_opt_int(eff, "num_shared_experts", n_shared);
    jobj_opt_int(eff, "moe_shared_expert_intermediate_size", shared_d_ff);
    jobj_opt_float(eff, "kda_lower_bound", cfg.kda_lower_bound);
    const JValue* lora = jobj_find(eff, "no_kda_lora");
    if (lora && lora->type == JType::NUMBER && lora->num_val == 0.0) {  // JSON false
        IMP_LOG_ERROR("config.json: bailing_hybrid with KDA LoRA gates (no_kda_lora=false) is not supported");
        return false;
    }
    if (cfg.kda_lower_bound >= 0.0f || cfg.kda_lower_bound < -5.0f) {
        IMP_LOG_ERROR("config.json: kda_lower_bound %.3f outside the safe range [-5, 0)",
                      cfg.kda_lower_bound);
        return false;
    }
    if (kda_head_dim <= 0)
        kda_head_dim = 128;
    cfg.gdn_kda = true;
    cfg.gdn_gate_sigmoid = true;  // FusedRMSNormGated(activation='sigmoid')
    cfg.gdn_grouped_head_layout = true;
    cfg.ssm_conv_kernel = conv > 0 ? conv : 4;
    cfg.ssm_state_size = kda_head_dim;
    cfg.ssm_group_count = cfg.n_heads;
    cfg.ssm_dt_rank = cfg.n_heads;
    cfg.ssm_inner_size = cfg.n_heads * kda_head_dim;
    if (n_shared > 0) {
        cfg.n_experts_shared = n_shared;
        cfg.expert_shared_d_ff = n_shared * shared_d_ff;
    }
    if (!parse_deepseek_config(root, eff, cfg))
        return false;
    IMP_LOG_INFO(
        "  Ling-3.0: KDA %d heads x %d (conv %d, lower bound %.1f), MLA every %d layers, %d shared expert(s)",
        cfg.n_heads, kda_head_dim, cfg.ssm_conv_kernel, cfg.kda_lower_bound, group, n_shared);
    return true;
}

}  // namespace imp
