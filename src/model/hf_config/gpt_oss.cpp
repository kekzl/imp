#include "model/hf_config_hooks.h"
#include "model/json_util.h"
#include "model/model_config.h"

namespace imp {

// gpt-oss: layer_types[] marks "sliding_attention" (window 128) on even layers,
// "full_attention" on odd. Uniform head_dim=64; per-head sink logits + per-expert biases
// are tensor-level (weight_map.cpp). Router: topk-then-softmax.
bool parse_gpt_oss_config(const JValue& /*root*/, const JValue& eff, ModelConfig& cfg) {
    // No separate MoE intermediate key: intermediate_size IS the per-expert FFN width (2880).
    if (cfg.expert_d_ff == 0 && cfg.n_experts > 0)
        cfg.expert_d_ff = cfg.d_ff;
    const JValue* lt = jobj_find(eff, "layer_types");
    if (lt && lt->type == JType::ARRAY) {
        cfg.swa_layers.clear();
        cfg.swa_layers.reserve(lt->arr.size());
        for (const auto& v : lt->arr)
            cfg.swa_layers.push_back(v.str_val == "sliding_attention" ? 1 : 0);
    }
    if (cfg.sliding_window <= 0)
        cfg.sliding_window = 128;
    return true;
}

}  // namespace imp
