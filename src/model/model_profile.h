#pragma once

#include "model/model_config.h"  // ModelArch

namespace imp {

class Model;

// Architecture-derived facts, decided ONCE at init and read everywhere — the
// single source of truth that replaces the scattered inline `if (arch==…)`
// branches and the repeated "loop the layers, check gdn_gate/ssm_in" detection
// (D1 in docs/audit/structural_debt_2026_06_08.md; the #514/#516 bug class).
//
// ModelConfig holds STATIC metadata (loaded from the file). ModelProfile holds
// the DERIVED classification/dispatch decisions computed from that metadata +
// the loaded layers. Filled by derive_model_profile() once, before any forward
// pass or the engine-init resolvers that currently re-derive it inline.
struct ModelProfile {
    // --- classification ---
    bool is_moe = false;     // n_experts > 0
    bool is_gdn = false;     // any layer carries a gdn_gate (Gated DeltaNet)
    bool is_ssm = false;     // any layer carries an ssm_in (Mamba2 / GDN)
    bool is_hybrid = false;  // recurrent (gdn/ssm) AND attention layers coexist
    bool is_dense = true;    // !is_moe

    // --- attention variant + flags (drives executor_attention dispatch) ---
    // Filled in migration step B; STANDARD until then.
    enum class AttnVariant { STANDARD, GEMMA4_SWA, GPTOSS_SWA, NOPE };
    AttnVariant attn_variant = AttnVariant::STANDARD;
    bool attn_qk_norm = false;            // gemma-4 per-head q/k RMSNorm
    bool attn_v_eq_k = false;             // gemma-4 V=K layers (wv absent)
    bool attn_fp32_accum_gemma4 = false;  // gemma-4 fp32 attention accumulation

    // --- eligibility (centralizes engine_init_resolver decisions) ---
    // Filled in the eligibility migration step; false until then.
    bool fp8_eligible = false;
    bool graphs_eligible = false;
};

// Pure: no side effects, no allocation. Reads the model's layers + config once.
ModelProfile derive_model_profile(const Model& model, const ModelConfig& cfg);

}  // namespace imp
