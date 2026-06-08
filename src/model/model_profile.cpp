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
    bool any_gdn = false, any_ssm = false, any_attn = false;
    const int n = model.n_layers();
    for (int i = 0; i < n; i++) {
        const auto& L = model.layer(i);
        if (L.gdn_gate.data != nullptr)
            any_gdn = true;
        if (L.ssm_in.data != nullptr)
            any_ssm = true;
        if (L.wq.data != nullptr)
            any_attn = true;
    }
    p.is_gdn = any_gdn;
    p.is_ssm = any_ssm;
    p.is_hybrid = (any_gdn || any_ssm) && any_attn;

    // attn_variant / attn_* flags / eligibility are filled by their respective
    // migration steps (B + eligibility); defaults until then.

    return p;
}

}  // namespace imp
