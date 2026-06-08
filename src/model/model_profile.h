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
    // --- classification (scanned from the layers) ---
    bool is_moe = false;       // n_experts > 0
    bool is_gdn = false;       // any layer carries a gdn_gate (Gated DeltaNet)
    bool is_ssm = false;       // any layer carries an ssm_in (Mamba2 / GDN)
    bool has_pure_ssm = false; // any layer is SSM WITHOUT a gdn_gate (Mamba2,
                               // e.g. Nemotron-H) — these disable CUDA graphs
    bool is_hybrid = false;    // recurrent (gdn/ssm) AND attention layers coexist
    bool is_dense = true;      // !is_moe

    // --- architecture identity (mirrors ModelConfig::arch) ---
    // The hot path (executor_attention / executor_forward_moe / …) keys many
    // kernel- and norm-selection branches off the architecture. These booleans
    // are the ONE place that maps the arch enum to those branches: every
    // `cfg.arch == ModelArch::X` in the executors reads the matching flag here
    // instead of re-comparing the enum inline.
    bool is_gemma3 = false;
    bool is_gemma4 = false;
    bool is_gpt_oss = false;
    bool is_llama4 = false;
};

// Pure: no side effects, no allocation. Reads the model's layers + config once.
ModelProfile derive_model_profile(const Model& model, const ModelConfig& cfg);

}  // namespace imp
