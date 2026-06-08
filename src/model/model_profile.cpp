#include "model/model_profile.h"

#include "model/model.h"

namespace imp {

ModelProfile derive_model_profile(const Model& model, const ModelConfig& cfg) {
    ModelProfile p;

    p.is_moe = cfg.n_experts > 0;
    p.is_dense = !p.is_moe;

    // Recurrent / hybrid classification: scan the layers ONCE here (the ≥6 sites
    // that re-derive this — engine_init_resolver, engine_kv_cache_init,
    // vram_budget, engine_weight_upload — read these flags instead).
    bool any_gdn = false, any_ssm = false, any_attn = false, any_pure_ssm = false;
    const int n = model.n_layers();
    for (int i = 0; i < n; i++) {
        const auto& L = model.layer(i);
        const bool has_gdn = L.gdn_gate.data != nullptr;
        const bool has_ssm = L.ssm_in.data != nullptr;
        if (has_gdn)
            any_gdn = true;
        if (has_ssm)
            any_ssm = true;
        if (has_ssm && !has_gdn)
            any_pure_ssm = true;
        if (L.wq.data != nullptr)
            any_attn = true;
    }
    p.is_gdn = any_gdn;
    p.is_ssm = any_ssm;
    p.has_pure_ssm = any_pure_ssm;
    p.is_hybrid = (any_gdn || any_ssm) && any_attn;

    // Architecture identity: the single mapping from the arch enum to the
    // kernel/norm-selection booleans the executors read.
    p.is_gemma3 = cfg.arch == ModelArch::GEMMA3;
    p.is_gemma4 = cfg.arch == ModelArch::GEMMA4;
    p.is_gpt_oss = cfg.arch == ModelArch::GPT_OSS;
    p.is_llama4 = cfg.arch == ModelArch::LLAMA4;

    return p;
}

}  // namespace imp
