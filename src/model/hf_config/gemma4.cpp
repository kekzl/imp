#include "core/logging.h"
#include "model/hf_config_hooks.h"
#include "model/hf_config_loader.h"
#include "model/json_util.h"
#include "model/model_config.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace imp {

std::vector<float> proportional_rope_inv_freqs(float theta, int head_dim, float partial_factor) {
    const int n_pairs = head_dim / 2;
    const int n_rot = std::min(n_pairs,
                               static_cast<int>(partial_factor * static_cast<float>(head_dim) / 2.0f));
    std::vector<float> f(static_cast<size_t>(n_pairs), 0.0f);
    for (int p = 0; p < n_rot; ++p)
        f[p] = static_cast<float>(std::pow(static_cast<double>(theta), -2.0 * p / head_dim));
    return f;
}

namespace {

// rope params nested under rope_parameters.{full_attention,sliding_attention}
void parse_gemma4_rope(const JValue& eff, int global_head_dim, ModelConfig& cfg) {
    const JValue* rp = jobj_find(eff, "rope_parameters");
    float theta_full = cfg.rope_theta > 0.0f ? cfg.rope_theta : 1e6f;
    float theta_swa = 1e4f;
    float partial_full = 0.0f;
    if (rp && rp->type == JType::OBJECT) {
        const JValue* fa = jobj_find(*rp, "full_attention");
        if (fa && fa->type == JType::OBJECT) {
            jobj_opt_float(*fa, "rope_theta", theta_full);
            jobj_opt_float(*fa, "partial_rotary_factor", partial_full);
        }
        const JValue* sa = jobj_find(*rp, "sliding_attention");
        if (sa && sa->type == JType::OBJECT)
            jobj_opt_float(*sa, "rope_theta", theta_swa);
    }
    cfg.rope_theta = theta_full;
    cfg.rope_theta_swa = theta_swa;
    // partial_rotary_factor < 1 on full_attention: only the first factor*hd/2 pairs rotate.
    // Unhandled, all 256 pairs of the hd=512 layers rotate and 16k NIAH drops to 0/5 (#2519).
    const int hd_full = global_head_dim > 0 ? global_head_dim : cfg.head_dim;
    cfg.rope_inv_freqs_global.clear();
    if (partial_full > 0.0f && partial_full < 1.0f && hd_full > 0)
        cfg.rope_inv_freqs_global = proportional_rope_inv_freqs(theta_full, hd_full, partial_full);
}

// Scalar head_dim / n_kv_heads = max per-layer value, so KV-cache/attention workspace sizes for the
// largest layer, not the SWA-only value (matches the GGUF loader). Otherwise full-attention layers
// (head_dim=512) write past their stride into adjacent layer slots.
void widen_scalar_geometry(ModelConfig& cfg) {
    int max_hd = 0;
    for (int v : cfg.head_dim_per_layer)
        max_hd = std::max(max_hd, v);
    if (max_hd > cfg.head_dim) {
        IMP_LOG_INFO("Gemma 4 (HF): scalar head_dim %d → %d (max of per-layer)", cfg.head_dim, max_hd);
        cfg.head_dim = max_hd;
    }
    int max_nkv = 0;
    for (int v : cfg.n_kv_heads_per_layer)
        max_nkv = std::max(max_nkv, v);
    if (max_nkv > cfg.n_kv_heads) {
        IMP_LOG_INFO("Gemma 4 (HF): scalar n_kv_heads %d → %d (max of per-layer)", cfg.n_kv_heads, max_nkv);
        cfg.n_kv_heads = max_nkv;
    }
}

}  // namespace

// Gemma-4: layer_types[] gives SWA vs global; head_dim/global_head_dim and
// num_key_value_heads/num_global_key_value_heads define the dual geometry. Builds
// per-layer vectors so executor_attention.cu picks the right shape/theta per layer.
bool parse_gemma4_config(const JValue& /*root*/, const JValue& eff, ModelConfig& cfg) {
    int global_head_dim = 0;
    int num_global_kv = 0;
    jobj_opt_int(eff, "global_head_dim", global_head_dim);
    jobj_opt_int(eff, "num_global_key_value_heads", num_global_kv);
    parse_gemma4_rope(eff, global_head_dim, cfg);

    const JValue* lt = jobj_find(eff, "layer_types");
    if (!lt || lt->type != JType::ARRAY)
        return true;
    cfg.swa_layers.clear();
    cfg.head_dim_per_layer.clear();
    cfg.n_kv_heads_per_layer.clear();
    cfg.swa_layers.reserve(lt->arr.size());
    cfg.head_dim_per_layer.reserve(lt->arr.size());
    cfg.n_kv_heads_per_layer.reserve(lt->arr.size());
    for (const auto& v : lt->arr) {
        bool is_swa = (v.str_val == "sliding_attention");
        cfg.swa_layers.push_back(is_swa ? 1 : 0);
        cfg.head_dim_per_layer.push_back(is_swa ? cfg.head_dim
                                                : (global_head_dim > 0 ? global_head_dim : cfg.head_dim));
        cfg.n_kv_heads_per_layer.push_back(is_swa ? cfg.n_kv_heads
                                                  : (num_global_kv > 0 ? num_global_kv : cfg.n_kv_heads));
    }
    widen_scalar_geometry(cfg);
    return true;
}

}  // namespace imp
