#include "core/logging.h"
#include "model/hf_config_hooks.h"
#include "model/json_util.h"
#include "model/model_config.h"

#include <string>

namespace imp {

namespace {

// Gemma-3 vs Gemma 1/2 (all ModelArch::GEMMA3): architectures[0] or model_type names gemma3.
bool is_gemma3_config(const JValue& root, const JValue& eff) {
    const JValue* archs = jobj_find(root, "architectures");
    if (archs && archs->type == JType::ARRAY && !archs->arr.empty() &&
        archs->arr[0].str_val.rfind("Gemma3", 0) == 0)
        return true;
    std::string mt;
    return (jobj_get_string(eff, "model_type", mt) || jobj_get_string(root, "model_type", mt)) &&
           mt.rfind("gemma3", 0) == 0;
}

// transformers 5 rope_parameters.{full,sliding}_attention: global theta + scaling, local theta.
void parse_gemma3_rope_parameters(const JValue& rp, ModelConfig& cfg, float& theta_global,
                                  float& theta_local) {
    const JValue* fa = jobj_find(rp, "full_attention");
    if (fa && fa->type == JType::OBJECT) {
        jobj_opt_float(*fa, "rope_theta", theta_global);
        std::string type;
        float factor = 1.0f;
        jobj_opt_string(*fa, "rope_type", type);
        jobj_opt_float(*fa, "factor", factor);
        if (type == "linear") {
            cfg.rope_freq_scale = factor;
        } else if (!type.empty() && type != "default") {
            cfg.rope_scaling_unhandled = true;
            IMP_LOG_WARN("Gemma-3 rope_type \"%s\" unhandled: global layers UNSCALED", type.c_str());
        }
    }
    const JValue* sa = jobj_find(rp, "sliding_attention");
    if (sa && sa->type == JType::OBJECT)
        jobj_opt_float(*sa, "rope_theta", theta_local);
}

// Period N of the global layer (i % N == N-1); 0 when layer_types is not periodic.
int gemma3_global_period(const JValue& eff) {
    int pattern = 6;
    jobj_opt_int(eff, "sliding_window_pattern", pattern);
    const JValue* lt = jobj_find(eff, "layer_types");
    if (!lt || lt->type != JType::ARRAY || lt->arr.empty())
        return pattern;
    pattern = 0;
    for (size_t i = 0; i < lt->arr.size() && pattern == 0; ++i)
        if (lt->arr[i].str_val == "full_attention")
            pattern = static_cast<int>(i) + 1;
    if (pattern == 0)  // no full_attention entry: period n+1 keeps every layer local
        pattern = static_cast<int>(lt->arr.size()) + 1;
    bool periodic = true;
    for (size_t i = 0; i < lt->arr.size() && periodic; ++i) {
        const bool full = static_cast<int>(i % pattern) == pattern - 1;
        periodic = lt->arr[i].str_val == (full ? "full_attention" : "sliding_attention");
    }
    return periodic ? pattern : 0;
}

}  // namespace

// Gemma-3: every sliding_window_pattern-th layer (default 6) is global at rope_theta (1e6) plus
// rope_scaling; the rest run rope_local_base_freq (1e4) unscaled. Gemma3TextConfig defaults apply;
// Gemma 1/2 share the arch and are skipped. layer_types must be periodic: the executor assumes it.
bool parse_gemma3_config(const JValue& root, const JValue& eff, ModelConfig& cfg) {
    if (!is_gemma3_config(root, eff))
        return true;
    float theta_global = 1e6f, theta_local = 1e4f;
    jobj_opt_float(eff, "rope_theta", theta_global);
    jobj_opt_float(eff, "rope_local_base_freq", theta_local);
    const JValue* rp = jobj_find(eff, "rope_parameters");
    if (rp && rp->type == JType::OBJECT)
        parse_gemma3_rope_parameters(*rp, cfg, theta_global, theta_local);
    cfg.rope_theta = theta_global;
    cfg.rope_local_theta = theta_local;

    const int pattern = gemma3_global_period(eff);
    if (pattern <= 0) {
        IMP_LOG_ERROR("config.json: Gemma-3 layer_types / sliding_window_pattern is not every-Nth-global");
        return false;
    }
    cfg.sliding_window_pattern = pattern;
    IMP_LOG_INFO("  Gemma-3 RoPE: global theta %.0f (scale %.2f), local theta %.0f, every %d-th layer global",
                 cfg.rope_theta, cfg.rope_freq_scale, cfg.rope_local_theta, pattern);
    return true;
}

}  // namespace imp
