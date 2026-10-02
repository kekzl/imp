#include "core/logging.h"
#include "model/hf_config_hooks.h"
#include "model/json_util.h"
#include "model/model_config.h"

#include <cmath>

namespace imp {

namespace {

// YaRN mscale from rope_scaling. HF DeepSeek-V2 uses two mscale fields differently, must
// not conflate: softmax scale gets yarn_get_mscale(factor,mscale_all_dim)^2; rope cos/sin
// gets the ratio yarn_get_mscale(factor,mscale)/yarn_get_mscale(factor,mscale_all_dim).
// For V2-Lite both are 0.707 so the rope ratio is exactly 1.0; V3 may differ.
void parse_mla_mscale(const JValue& eff, ModelConfig& cfg) {
    const JValue* rs = jobj_find(eff, "rope_scaling");
    if (!rs || rs->type != JType::OBJECT)
        return;
    float mscale = 1.0f, mscale_all_dim = 1.0f;
    bool has_mscale = jobj_get_float(*rs, "mscale", mscale);
    bool has_all_dim = jobj_get_float(*rs, "mscale_all_dim", mscale_all_dim);
    // Softmax scale prefers mscale_all_dim, falls back to mscale.
    cfg.mla_mscale = has_all_dim ? mscale_all_dim : (has_mscale ? mscale : 1.0f);
    // RoPE ratio numerator is the raw mscale (fallback: mscale_all_dim
    // → ratio 1.0, i.e. no rope scaling, which is HF's default when
    // the two coincide).
    cfg.mla_mscale_num = has_mscale ? mscale : cfg.mla_mscale;
}

// imp's rope_yarn kernel scales cos/sin by yarn_attn_factor*(1+0.1*log(rope_freq_scale)).
// HF DeepseekV2YarnRotaryEmbedding instead scales by the mscale RATIO
//   yarn_get_mscale(factor,mscale)/yarn_get_mscale(factor,mscale_all_dim),
//   where yarn_get_mscale(f,m)=0.1*m*ln(f)+1. Set yarn_attn_factor to match this ratio.
// Softmax scale is separate and unchanged: yarn_get_mscale(factor,mscale_all_dim)^2 via
// mla_attention_scale_multiplier.
void adjust_mla_yarn_attn_factor(ModelConfig& cfg) {
    if (cfg.yarn_ext_factor <= 0.0f || cfg.rope_freq_scale <= 1.0f)
        return;
    const float log_scale = std::log(cfg.rope_freq_scale);
    const float ms_num = 0.1f * cfg.mla_mscale_num * log_scale + 1.0f;
    const float ms_den = 0.1f * cfg.mla_mscale * log_scale + 1.0f;
    const float hf_rope_mscale = ms_num / ms_den;
    const float imp_mscale = 0.1f * log_scale + 1.0f;
    cfg.yarn_attn_factor = hf_rope_mscale / imp_mscale;
    IMP_LOG_INFO(
        "  MLA YaRN rope-mscale adjust: yarn_attn_factor=%.4f "
        "(hf_rope_mscale=%.4f, softmax_scale_mult=%.4f)",
        cfg.yarn_attn_factor, hf_rope_mscale, ms_den * ms_den);
}

}  // namespace

// DeepSeek V2/V3 Multi-head Latent Attention (MLA) config.
// kv_lora_rank > 0 is the unambiguous MLA indicator. When present, override
// head_dim and rope_dim to match the decoupled-head layout.
bool parse_deepseek_config(const JValue& /*root*/, const JValue& eff, ModelConfig& cfg) {
    jobj_opt_int(eff, "kv_lora_rank", cfg.kv_lora_rank);
    jobj_opt_int(eff, "q_lora_rank", cfg.q_lora_rank);  // absent/null -> stays 0
    jobj_opt_int(eff, "qk_rope_head_dim", cfg.qk_rope_head_dim);
    jobj_opt_int(eff, "qk_nope_head_dim", cfg.qk_nope_head_dim);
    jobj_opt_int(eff, "v_head_dim", cfg.v_head_dim);
    jobj_opt_int(eff, "first_k_dense_replace", cfg.first_k_dense_replace);
    if (!cfg.is_mla())
        return true;
    // Decoupled-head layout: each attention head has qk_nope_head_dim
    // non-RoPE dims plus qk_rope_head_dim RoPE dims.
    cfg.head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
    cfg.rope_dim = cfg.qk_rope_head_dim;
    parse_mla_mscale(eff, cfg);
    IMP_LOG_INFO(
        "  MLA: kv_lora_rank=%d q_lora_rank=%d "
        "qk_rope=%d qk_nope=%d v_head=%d head_dim=%d mla_mscale=%.4f",
        cfg.kv_lora_rank, cfg.q_lora_rank, cfg.qk_rope_head_dim, cfg.qk_nope_head_dim, cfg.v_head_dim,
        cfg.head_dim, cfg.mla_mscale);
    adjust_mla_yarn_attn_factor(cfg);
    return true;
}

}  // namespace imp
