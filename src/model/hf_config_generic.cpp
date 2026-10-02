// Arch-independent config.json parsers for HFConfigLoader::load_config (#2537).

#include "core/logging.h"
#include "model/hf_config_hooks.h"
#include "model/json_util.h"
#include "model/model_config.h"
#include "model/model_limits.h"

#include <cmath>
#include <string>

namespace imp {

namespace {

// Granite multipliers (HF modeling_granite.py): embeddings * embedding_multiplier, residual
// x + residual_multiplier * out, attention scale attention_multiplier, logits / logits_scaling.
// RMSNorm is scale-invariant, so a residual stream run at 1/residual_multiplier with the plain
// add is exact: both stream factors go into embed_scale. Absent keys are no-ops, so every arch
// reads them (GraniteMoe maps to GENERIC).
void parse_granite_multipliers(const JValue& eff, ModelConfig& cfg) {
    float attn = 0.0f, emb = 1.0f, res = 1.0f, logits = 1.0f;
    jobj_opt_float(eff, "attention_multiplier", attn);
    jobj_opt_float(eff, "embedding_multiplier", emb);
    jobj_opt_float(eff, "residual_multiplier", res);
    jobj_opt_float(eff, "logits_scaling", logits);
    set_granite_multipliers(cfg, attn, emb, res, logits);
}

// RoPE scaling block: `rope_scaling`, else transformers 5 `rope_parameters` when it names a type
// (Ministral 3 carries YaRN only there; a theta-only rope_parameters is not a scaling block).
const JValue* rope_scaling_object(const JValue& eff) {
    const JValue* rs = jobj_find(eff, "rope_scaling");
    if (rs && rs->type == JType::OBJECT)
        return rs;
    const JValue* rp = jobj_find(eff, "rope_parameters");
    if (rp && rp->type == JType::OBJECT && (jobj_find(*rp, "rope_type") || jobj_find(*rp, "type")))
        return rp;
    return nullptr;
}

// YaRN: factor, beta, original context, mscale + mscale_all_dim (cos/sin factor ratio) and
// Ministral 3 llama_4_scaling_beta (query temperature past original_max_position_embeddings).
void parse_yarn(const JValue& rs, float factor, ModelConfig& cfg) {
    cfg.rope_freq_scale = factor;
    jobj_opt_float(rs, "attn_factor", cfg.yarn_attn_factor);
    jobj_opt_float(rs, "beta_fast", cfg.yarn_beta_fast);
    jobj_opt_float(rs, "beta_slow", cfg.yarn_beta_slow);
    jobj_opt_int(rs, "original_max_position_embeddings", cfg.rope_n_ctx_orig);
    cfg.yarn_ext_factor = 1.0f;
    float mscale = 0.0f, mscale_all_dim = 0.0f;
    jobj_opt_float(rs, "mscale", mscale);
    jobj_opt_float(rs, "mscale_all_dim", mscale_all_dim);
    set_yarn_mscale_ratio(cfg, mscale, mscale_all_dim);
    jobj_opt_float(rs, "llama_4_scaling_beta", cfg.attn_temp_scale);
    cfg.attn_temp_floor = cfg.rope_n_ctx_orig;
}

void append_floats(const JValue* arr, std::vector<float>& out) {
    if (!arr || arr->type != JType::ARRAY)
        return;
    out.reserve(arr->arr.size());
    for (const auto& v : arr->arr)
        out.push_back(static_cast<float>(v.num_val));
}

// LongRoPE: per-dimension frequency scaling factors.
void parse_longrope(const JValue& rs, ModelConfig& cfg) {
    jobj_opt_int(rs, "original_max_position_embeddings", cfg.rope_scaling_orig_max_pos);
    append_floats(jobj_find(rs, "short_factor"), cfg.rope_short_factor);
    append_floats(jobj_find(rs, "long_factor"), cfg.rope_long_factor);
}

// Llama-3.x per-frequency RoPE scaling reuses LongRoPE infra: one factor per rope-pair so
// freqs[i] = base_freq[i]/factor[i], matching the HF algorithm. Independent of sequence
// length, so short and long arrays carry identical values.
bool parse_llama3_rope(const JValue& rs, const JValue& eff, float factor, ModelConfig& cfg) {
    float low_freq_factor = 1.0f, high_freq_factor = 4.0f;
    int orig_max_pos = 0;
    jobj_opt_float(rs, "low_freq_factor", low_freq_factor);
    jobj_opt_float(rs, "high_freq_factor", high_freq_factor);
    jobj_opt_int(rs, "original_max_position_embeddings", orig_max_pos);
    if (orig_max_pos <= 0)
        jobj_opt_int(eff, "original_max_position_embeddings", orig_max_pos);

    int hd = cfg.head_dim > 0 ? cfg.head_dim : (cfg.n_heads > 0 ? cfg.d_model / cfg.n_heads : 0);
    int rd = (cfg.rope_dim > 0) ? cfg.rope_dim : hd;
    // rope_dim is head_dim scaled by a file-supplied partial_rotary_factor: own ceiling.
    if (rd < 0 || rd > kMaxHeadDim) {
        IMP_LOG_ERROR("config.json: rope dimension is %d, which exceeds the limit of %d", rd, kMaxHeadDim);
        return false;
    }
    int pairs = rd / 2;
    if (!(pairs > 0 && orig_max_pos > 0 && factor > 1.0f && high_freq_factor > low_freq_factor)) {
        IMP_LOG_WARN("Llama-3 RoPE: skipping (pairs=%d orig_max_pos=%d factor=%.2f low=%.2f high=%.2f)",
                     pairs, orig_max_pos, factor, low_freq_factor, high_freq_factor);
        return true;
    }
    const float low_wavelen = static_cast<float>(orig_max_pos) / low_freq_factor;
    const float high_wavelen = static_cast<float>(orig_max_pos) / high_freq_factor;
    const float two_pi = 6.28318530717958647692f;
    cfg.rope_short_factor.resize(pairs);
    cfg.rope_long_factor.resize(pairs);
    for (int i = 0; i < pairs; i++) {
        const float base_freq = 1.0f / std::pow(cfg.rope_theta, (2.0f * i) / static_cast<float>(rd));
        const float wavelen = two_pi / base_freq;
        float pair_factor;
        if (wavelen < high_wavelen) {
            pair_factor = 1.0f;
        } else if (wavelen > low_wavelen) {
            pair_factor = factor;
        } else {
            const float smooth = (static_cast<float>(orig_max_pos) / wavelen - low_freq_factor) /
                                 (high_freq_factor - low_freq_factor);
            pair_factor = factor / (1.0f - smooth + smooth * factor);
        }
        cfg.rope_short_factor[i] = pair_factor;
        cfg.rope_long_factor[i] = pair_factor;
    }
    cfg.rope_scaling_orig_max_pos = orig_max_pos;
    IMP_LOG_INFO("Llama-3 RoPE: factor=%.1f low=%.1f high=%.1f orig_max_pos=%d → %d freq pairs", factor,
                 low_freq_factor, high_freq_factor, orig_max_pos, pairs);
    return true;
}

// imp convention (matches the GGUF loader / rope_forward): rope_freq_scale stores the
// FACTOR (>1), the kernel applies 1/factor itself. Storing 1/factor here double-inverted
// YaRN for gpt-oss (#547): dims rotated factor^2=1024x too fast, mscale flipped 0.653 vs 1.347.
bool parse_rope_scaling(const JValue& eff, ModelConfig& cfg) {
    const JValue* rs = rope_scaling_object(eff);
    if (!rs)
        return true;
    std::string rope_type;
    jobj_opt_string(*rs, "type", rope_type);
    if (rope_type.empty())  // some HF configs spell it rope_type
        jobj_opt_string(*rs, "rope_type", rope_type);
    float factor = 1.0f;
    jobj_opt_float(*rs, "factor", factor);

    if (rope_type == "linear") {
        cfg.rope_freq_scale = factor;
    } else if (rope_type == "yarn") {
        parse_yarn(*rs, factor, cfg);
    } else if (rope_type == "longrope" || rope_type == "long_rope") {
        parse_longrope(*rs, cfg);
    } else if (rope_type == "llama3") {
        return parse_llama3_rope(*rs, eff, factor, cfg);
    } else if (rope_type == "dynamic") {  // same factor as linear at runtime
        cfg.rope_freq_scale = 1.0f / factor;
    } else if (!rope_type.empty() && rope_type != "default" && rope_type != "none") {
        // An unhandled spelling (su/su_scaled, dynamic_ntk/ntk) would rotate unscaled while reporting
        // full context. "default"/"none" mean no scaling (Qwen3-VL-4B declares "default").
        cfg.rope_scaling_unhandled = true;
        IMP_LOG_WARN(
            "rope_scaling type \"%s\" is not one of linear/yarn/longrope/llama3/dynamic: "
            "ignored, so this model rotates UNSCALED and will degrade past its native "
            "context window (declared max_position_embeddings=%d, factor=%.3f)",
            rope_type.c_str(), cfg.max_seq_len, factor);
    }
    return true;
}

// M-RoPE section split. It lives under `rope_scaling` in older configs and under
// `rope_parameters` in newer ones (Qwen3-VL): both are checked, or a multimodal model
// silently runs single-axis RoPE.
void parse_mrope(const JValue& eff, ModelConfig& cfg) {
    for (const char* key : {"rope_scaling", "rope_parameters"}) {
        const JValue* obj = jobj_find(eff, key);
        if (!obj || obj->type != JType::OBJECT)
            continue;
        const JValue* sec = jobj_find(*obj, "mrope_section");
        if (sec && sec->type == JType::ARRAY && sec->arr.size() == 3) {
            bool ok = true;
            int parsed[3] = {0, 0, 0};
            for (size_t i = 0; i < 3; ++i) {
                if (sec->arr[i].type != JType::NUMBER || sec->arr[i].as_int() < 0)
                    ok = false;
                else
                    parsed[i] = static_cast<int>(sec->arr[i].as_int());
            }
            if (ok) {
                for (int i = 0; i < 3; ++i)
                    cfg.mrope_section[i] = parsed[i];
            } else {
                IMP_LOG_WARN("mrope_section under '%s' is malformed, ignoring it", key);
            }
        }
        // Booleans arrive as NUMBER 0.0/1.0 from this parser.
        const JValue* inter = jobj_find(*obj, "mrope_interleaved");
        if (inter && inter->type == JType::NUMBER)
            cfg.mrope_interleaved = inter->num_val != 0.0;
    }
    if (cfg.has_mrope()) {
        IMP_LOG_INFO("M-RoPE section [%d, %d, %d]%s", cfg.mrope_section[0], cfg.mrope_section[1],
                     cfg.mrope_section[2], cfg.mrope_interleaved ? " (interleaved)" : "");
    }
}

// Present as NUMBER (booleans arrive that way): out = 0/1, returns true. Absent: out untouched.
bool read_tristate(const JValue& obj, const char* key, int& out) {
    const JValue* v = jobj_find(obj, key);
    if (!v || v->type != JType::NUMBER)
        return false;
    out = (v->num_val != 0.0) ? 1 : 0;
    return true;
}

}  // namespace

bool parse_hf_core_dims(const JValue& eff, ModelConfig& cfg) {
    jobj_opt_int(eff, "hidden_size", cfg.d_model);
    jobj_opt_int(eff, "num_attention_heads", cfg.n_heads);
    jobj_opt_int(eff, "intermediate_size", cfg.d_ff);
    jobj_opt_int(eff, "num_hidden_layers", cfg.n_layers);
    jobj_opt_int(eff, "vocab_size", cfg.vocab_size);
    jobj_opt_int(eff, "max_position_embeddings", cfg.max_seq_len);
    jobj_opt_int(eff, "head_dim", cfg.head_dim);
    // KV heads: default to n_heads (MHA) if not specified
    if (!jobj_get_int(eff, "num_key_value_heads", cfg.n_kv_heads))
        cfg.n_kv_heads = cfg.n_heads;

    // Every count above came out of the file and `head_dim` sizes the RoPE factor tables
    // (AUDIT_arch_2026 F1-10); the SafeTensors loader checks again once the config is complete.
    std::string dim_err;
    if (!validate_declared_dimensions(cfg, &dim_err)) {
        IMP_LOG_ERROR("config.json: %s", dim_err.c_str());
        return false;
    }
    if (!jobj_get_float(eff, "rms_norm_eps", cfg.rms_norm_eps))
        jobj_opt_float(eff, "layer_norm_eps", cfg.rms_norm_eps);
    return true;
}

bool parse_hf_rope(const JValue& eff, ModelConfig& cfg) {
    // Newer HF configs (Qwen3.5/3.6, Qwen3-Next) nest rope_theta/partial_rotary_factor under
    // rope_parameters, read after the top-level keys. Otherwise Qwen3.6 silently runs
    // theta=10000 (1000x too small) with full-dim RoPE.
    jobj_opt_float(eff, "rope_theta", cfg.rope_theta);
    float partial_factor = 0.0f;
    jobj_opt_float(eff, "partial_rotary_factor", partial_factor);
    const JValue* rope_params = jobj_find(eff, "rope_parameters");
    if (rope_params && rope_params->type == JType::OBJECT) {
        jobj_opt_float(*rope_params, "rope_theta", cfg.rope_theta);
        jobj_opt_float(*rope_params, "partial_rotary_factor", partial_factor);
    }
    parse_mrope(eff, cfg);
    if (partial_factor > 0.0f && partial_factor < 1.0f) {
        int hd_for_rope = (cfg.head_dim > 0) ? cfg.head_dim
                                             : (cfg.n_heads > 0 ? cfg.d_model / cfg.n_heads : 0);
        if (hd_for_rope > 0) {
            cfg.rope_dim = static_cast<int>(hd_for_rope * partial_factor);
            IMP_LOG_INFO("Partial RoPE: partial_rotary_factor=%.3f head_dim=%d → rope_dim=%d", partial_factor,
                         hd_for_rope, cfg.rope_dim);
        }
    }
    return parse_rope_scaling(eff, cfg);
}

void parse_hf_ffn(const JValue& root, const JValue& eff, ModelConfig& cfg) {
    jobj_opt_int(eff, "sliding_window", cfg.sliding_window);
    // Softcapping (Gemma-2/3); Gemma 4 also carries final_logit_softcapping on the root.
    jobj_opt_float(eff, "attn_logit_softcapping", cfg.attn_logit_softcap);
    jobj_opt_float(eff, "final_logit_softcapping", cfg.final_logit_softcap);
    jobj_opt_float(root, "final_logit_softcapping", cfg.final_logit_softcap);
    parse_granite_multipliers(eff, cfg);

    std::string hidden_act;
    if (jobj_get_string(eff, "hidden_act", hidden_act) ||
        jobj_get_string(eff, "hidden_activation", hidden_act)) {
        if (hidden_act == "silu" || hidden_act == "swiglu")
            cfg.ffn_activation = FFNActivation::SWIGLU;
        else if (hidden_act == "gelu" || hidden_act == "gelu_pytorch_tanh" || hidden_act == "geglu")
            cfg.ffn_activation = FFNActivation::GEGLU;
    }
}

void parse_hf_moe(const JValue& eff, ModelConfig& cfg) {
    if (!jobj_get_int(eff, "num_local_experts", cfg.n_experts) &&
        !jobj_get_int(eff, "num_experts", cfg.n_experts))
        jobj_opt_int(eff, "n_routed_experts", cfg.n_experts);  // DeepSeek-V2/V3 routed count
    if (!jobj_get_int(eff, "num_experts_per_tok", cfg.n_experts_active))
        jobj_opt_int(eff, "top_k_experts", cfg.n_experts_active);
    if (!jobj_get_int(eff, "moe_intermediate_size", cfg.expert_d_ff))
        jobj_opt_int(eff, "expert_intermediate_size", cfg.expert_d_ff);
    // Shared (always-active) experts alongside routed experts (DeepSeek-V2/V3, Nemotron-H).
    int n_shared = 0;
    if (jobj_get_int(eff, "n_shared_experts", n_shared) && n_shared > 0)
        cfg.n_experts_shared = n_shared;
}

// Tri-state so the SafeTensors loader cross-checks the flags against the tensors present
// (lm_head.weight, bias tensors) instead of trusting either side alone.
void parse_hf_tristate_flags(const JValue& root, const JValue& eff, ModelConfig& cfg) {
    if (read_tristate(root, "tie_word_embeddings", cfg.tie_word_embeddings))
        IMP_LOG_INFO("  tie_word_embeddings = %s", cfg.tie_word_embeddings == 1 ? "true" : "false");
    read_tristate(eff, "attention_bias", cfg.attention_bias);
    read_tristate(eff, "mlp_bias", cfg.mlp_bias);
}

}  // namespace imp
