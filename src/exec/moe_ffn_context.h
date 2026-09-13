#pragma once

#include "core/tensor.h"
#include "core/qtype.h"          // QType
#include "compute/moe_routing.h"  // MoeRoutingResult
#include "exec/expert_cache.h"           // kExpertProjCount
#include "exec/nvfp4_expert_offload.h"  // StagedProj
#include <cstddef>

namespace imp {

// Per-call state for GraphExecutor::run_moe_ffn(): bundles locals previously captured by
// the monolithic body so each MoE phase helper takes one MoeFfnContext& instead of a
// 20-arg parameter list. Populated by moe_ffn_phase1/2/...; later phases read/mutate it.
struct MoeFfnContext {
    // Shape / dtype parameters
    int n = 0;
    int d = 0;
    int ne = 0;
    int top_k = 0;
    int eff = 0;
    int expanded = 0;
    float eps = 0.f;
    size_t es = 0;

    // Tensor views (point at hidden_, residual_, norm_out_ for the current step)
    Tensor h{};
    Tensor r{};
    Tensor no{};

    // Path-selection flags computed during setup/routing
    bool nvfp4_covers_layer = false;
    bool will_skip_residual_copy = false;
    bool gemma4_fp32_norm = false;
    bool moe_use_fp32_residual = false;
    bool moe_fused_norm_q8 = false;
    bool fp32_gate_logits_ready = false;
    bool will_decode_fast = false;
    bool non_gated_experts = false;
    bool use_packed_dequant = false;
    QType up_qtype = QType::F16;

    // Routing result + transient buffers carried across phases
    MoeRoutingResult routing{};
    bool residual_fused = false;  // true when decode-fast / fused scatter already added residual

    // True if moe_gather already populated moe_.gathered for this call. Set false when the
    // CUTLASS3x device-args path will fire (it consumes sorted_token_ids directly, no
    // gathered intermediate needed); if that path falls back to legacy, moe_gather runs
    // lazily and flips this back to true. Default true so unchecked paths see it populated.
    bool moe_gather_done = true;

    // Host-resident NVFP4 experts staged into the device buffer for THIS layer. Filled by
    // whichever prefill path reaches the layer first, reused by later ones: a layer
    // transfers once even when the CUTLASS attempt falls through to legacy.
    StagedProj staged[kExpertProjCount]{};
    bool staged_done = false;
};

}  // namespace imp
