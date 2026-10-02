#include "core/logging.h"
#include "model/hf_config_hooks.h"
#include "model/json_util.h"
#include "model/model_config.h"

#include <string>

namespace imp {

namespace {

// hybrid_override_pattern: one char per layer. M=Mamba2 (no attention), E=MoE expert FFN
// (no attention), *=Attention layer.
void parse_hybrid_pattern(const JValue& eff, ModelConfig& cfg) {
    std::string pat;
    if (!jobj_get_string(eff, "hybrid_override_pattern", pat) || pat.empty())
        return;
    cfg.n_kv_heads_per_layer.clear();
    cfg.n_kv_heads_per_layer.reserve(pat.size());
    int n_attn = 0, n_ssm = 0, n_moe = 0;
    for (char c : pat) {
        if (c == '*') {
            cfg.n_kv_heads_per_layer.push_back(cfg.n_kv_heads);
            ++n_attn;
        } else {
            cfg.n_kv_heads_per_layer.push_back(0);
            if (c == 'M')
                ++n_ssm;
            else if (c == 'E')
                ++n_moe;
        }
    }
    IMP_LOG_INFO("  Nemotron-H hybrid pattern: %d Mamba2 + %d MoE + %d Attention = %d layers", n_ssm, n_moe,
                 n_attn, (int)pat.size());
}

// Extended MoE config (DeepSeek-style routing): n_routed_experts is the
// total expert count, num_experts_per_tok is top_k (already read generically),
// moe_shared_expert_intermediate_size sizes the always-on shared expert.
void parse_nemotron_moe(const JValue& eff, ModelConfig& cfg) {
    int n_routed = 0, n_shared = 0, shared_d_ff = 0;
    jobj_opt_int(eff, "n_routed_experts", n_routed);
    jobj_opt_int(eff, "n_shared_experts", n_shared);
    // moe_shared_expert_intermediate_size: DeepSeek / Qwen2-MoE naming.
    // shared_expert_intermediate_size: Qwen3.5 / 3.6 naming.
    jobj_opt_int(eff, "moe_shared_expert_intermediate_size", shared_d_ff);
    if (shared_d_ff == 0)
        jobj_opt_int(eff, "shared_expert_intermediate_size", shared_d_ff);
    if (n_routed > 0 && cfg.n_experts == 0)
        cfg.n_experts = n_routed;
    if (n_shared > 0)
        cfg.n_experts_shared = n_shared;
    if (shared_d_ff > 0)
        cfg.expert_shared_d_ff = shared_d_ff;
    jobj_opt_float(eff, "routed_scaling_factor", cfg.expert_weights_scale);
    // norm_topk_prob arrives as a bool (number 0/1 in our JSON parser)
    const JValue* ntp = jobj_find(eff, "norm_topk_prob");
    if (ntp && ntp->type == JType::NUMBER)
        cfg.expert_weights_norm = (ntp->num_val != 0.0);
}

}  // namespace

// Nemotron-H MoE: hybrid Mamba2+MoE+Attention. Parses hybrid_override_pattern to fill
// n_kv_heads_per_layer so executor_workspace sizes buffers per layer. Missing this leaves
// ssm_inner_size=0: SSM kernels read past allocated state, IMA on first prefill.
//   mamba_head_dim x mamba_num_heads -> ssm_inner_size
//   ssm_state_size                   -> ssm_state_size
//   n_groups                         -> ssm_group_count
//   mamba_num_heads                  -> ssm_dt_rank (n_heads)
//   conv_kernel                      -> ssm_conv_kernel
//   hybrid_override_pattern (M/E/*)  -> n_kv_heads_per_layer (0/0/n_kv)
bool parse_nemotron_h_config(const JValue& /*root*/, const JValue& eff, ModelConfig& cfg) {
    int mamba_head_dim = 0, mamba_num_heads = 0, n_groups_v = 0;
    int ssm_state = 0, conv_k = 0;
    jobj_opt_int(eff, "mamba_head_dim", mamba_head_dim);
    jobj_opt_int(eff, "mamba_num_heads", mamba_num_heads);
    jobj_opt_int(eff, "n_groups", n_groups_v);
    jobj_opt_int(eff, "ssm_state_size", ssm_state);
    jobj_opt_int(eff, "conv_kernel", conv_k);
    if (mamba_head_dim > 0 && mamba_num_heads > 0)
        cfg.ssm_inner_size = mamba_head_dim * mamba_num_heads;
    if (ssm_state > 0)
        cfg.ssm_state_size = ssm_state;
    if (n_groups_v > 0)
        cfg.ssm_group_count = n_groups_v;
    if (mamba_num_heads > 0)
        cfg.ssm_dt_rank = mamba_num_heads;
    if (conv_k > 0)
        cfg.ssm_conv_kernel = conv_k;

    parse_nemotron_moe(eff, cfg);
    parse_hybrid_pattern(eff, cfg);

    IMP_LOG_INFO("  Nemotron-H SSM: inner=%d state=%d groups=%d n_heads=%d conv_kernel=%d",
                 cfg.ssm_inner_size, cfg.ssm_state_size, cfg.ssm_group_count, cfg.ssm_dt_rank,
                 cfg.ssm_conv_kernel);
    IMP_LOG_INFO(
        "  Nemotron-H MoE: n_experts=%d top_k=%d n_shared=%d shared_d_ff=%d "
        "scale=%.2f norm_topk=%d",
        cfg.n_experts, cfg.n_experts_active, cfg.n_experts_shared, cfg.expert_shared_d_ff,
        cfg.expert_weights_scale, (int)cfg.expert_weights_norm);
    return true;
}

}  // namespace imp
