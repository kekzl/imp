#pragma once

#include <cstddef>
#include <vector>

namespace imp {

// Does this MoE expert placement depend on the NVFP4 host-offload path?
//
// History, because it explains where the real gate now lives. NVFP4-prequant
// experts used to have no host path at all, and the failure was silent rather
// than loud — three places disagreed about it, which is why it reached a
// released build:
//
//   1. Phase 0 (`pre_dequant_phase0_nvfp4_loader.cu`) promoted the
//      `weight_scale` / `weight_scale_2` sidecars onto a weight only when that
//      weight was already `on_device`. A host-resident expert therefore kept
//      `scales == nullptr`, and Phase 0 said so at IMP_LOG_DEBUG, i.e. not at
//      all under the default log level.
//   2. `gemm()` recognised the scale-less packed weight, logged an ERROR and
//      RETURNED without multiplying (bounded to 20 lines). The output buffer is
//      pre-zeroed, so the forward continued with those experts contributing
//      nothing.
//   3. `decide_expert_layer_placement_` warned about this, but only on the
//      budget route (`budget < total_expert_bytes`). `moe.force_host_experts`
//      reached the same placement without passing that branch at all.
//
// Measured on Qwen3-30B-A3B-NVFP4-Modelopt, 2026-08-13, greedy, same prompt:
// resident was coherent at 361.97 tok/s; 8 of 48 layers on host answered "the
// capital of France is the city of the same name, France itself" at 88.77
// tok/s; all 48 on host repeated "ftp" forever. Every one of the three exited 0.
//
// #1403 refused the placement outright. The path itself now exists — Phase 0
// promotes host-resident experts too, and the expert cache stages them into a
// slot pool the fused NVFP4 kernels can address (see
// `exec/nvfp4_expert_offload.h`). What is still undecidable HERE is whether the
// cache will be large enough, because it is sized after weight upload. So this
// predicate answers only "does this placement rely on that path", and the
// refusal moved to `GraphExecutor::verify_host_expert_placement()`, which runs
// once the promotion and the real cache both exist. Kept pure so it can be
// tested without a GPU or a checkpoint.
//
// `layer_expert_bytes[i] > 0` marks layer i as an MoE layer;
// `experts_upload_layer[i]` is true when its experts go to the device.
inline bool expert_placement_needs_host_path(bool is_nvfp4_prequant,
                                             const std::vector<size_t>& layer_expert_bytes,
                                             const std::vector<bool>& experts_upload_layer) {
    if (!is_nvfp4_prequant)
        return false;  // GGUF-class experts have their own host path (#1370).
    const size_t n = layer_expert_bytes.size() < experts_upload_layer.size() ? layer_expert_bytes.size()
                                                                             : experts_upload_layer.size();
    for (size_t i = 0; i < n; ++i) {
        if (layer_expert_bytes[i] > 0 && !experts_upload_layer[i])
            return true;
    }
    return false;
}

// How many MoE layers the placement leaves on host, for the error message.
inline int expert_placement_host_layers(const std::vector<size_t>& layer_expert_bytes,
                                        const std::vector<bool>& experts_upload_layer) {
    const size_t n = layer_expert_bytes.size() < experts_upload_layer.size() ? layer_expert_bytes.size()
                                                                             : experts_upload_layer.size();
    int host = 0;
    for (size_t i = 0; i < n; ++i) {
        if (layer_expert_bytes[i] > 0 && !experts_upload_layer[i])
            ++host;
    }
    return host;
}

}  // namespace imp
