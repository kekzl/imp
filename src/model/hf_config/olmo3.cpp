#include "core/logging.h"
#include "model/hf_config_hooks.h"
#include "model/json_util.h"
#include "model/model_config.h"

namespace imp {

// Olmo-3: layer_types repeats 3 sliding + 1 full (HF default when absent: (i + 1) % 4 == 0 is full).
// Both layer kinds rotate at rope_theta; rope_scaling (YaRN) applies to full layers only.
// The executor's sliding_window_pattern path runs local layers at rope_local_theta, unscaled.
bool parse_olmo3_config(const JValue& /*root*/, const JValue& eff, ModelConfig& cfg) {
    int period = 4;
    const JValue* lt = jobj_find(eff, "layer_types");
    if (lt && lt->type == JType::ARRAY && !lt->arr.empty()) {
        period = 0;
        for (size_t i = 0; i < lt->arr.size() && period == 0; ++i)
            if (lt->arr[i].str_val == "full_attention")
                period = static_cast<int>(i) + 1;
        for (size_t i = 0; i < lt->arr.size() && period > 0; ++i) {
            const bool full = static_cast<int>(i % period) == period - 1;
            if (lt->arr[i].str_val != (full ? "full_attention" : "sliding_attention"))
                period = 0;
        }
    }
    if (period <= 0) {
        IMP_LOG_ERROR("config.json: Olmo-3 layer_types is not every-Nth-full");
        return false;
    }
    cfg.sliding_window_pattern = period;
    cfg.rope_local_theta = cfg.rope_theta;
    IMP_LOG_INFO("  Olmo-3: window %d, every %d-th layer full (YaRN factor %.1f), theta %.0f",
                 cfg.sliding_window, period, cfg.rope_freq_scale, cfg.rope_theta);
    return true;
}

}  // namespace imp
