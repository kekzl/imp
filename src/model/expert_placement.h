#pragma once

#include <cstddef>
#include <vector>

namespace imp {

// Can this MoE expert placement actually be served?
//
// For NVFP4-prequant checkpoints the answer is no as soon as one expert layer
// stays on host, and the failure is silent rather than loud. Three places
// disagree about it today, which is why it reached a released build:
//
//   1. Phase 0 (`pre_dequant_phase0_nvfp4_loader.cu`) promotes the
//      `weight_scale` / `weight_scale_2` sidecars onto a weight only when that
//      weight is already `on_device`. A host-resident expert therefore keeps
//      `scales == nullptr`, and Phase 0 says so at IMP_LOG_DEBUG, i.e. not at
//      all under the default log level.
//   2. `gemm()` recognises the scale-less packed weight, logs an ERROR and
//      RETURNS without multiplying (bounded to 20 lines). The output buffer is
//      pre-zeroed, so the forward continues with those experts contributing
//      nothing.
//   3. `decide_expert_layer_placement_` warns about this, but only on the
//      budget route (`budget < total_expert_bytes`). `moe.force_host_experts`
//      reaches the same placement without passing that branch at all.
//
// Measured on Qwen3-30B-A3B-NVFP4-Modelopt, 2026-08-13, greedy, same prompt:
// resident is coherent at 361.97 tok/s; 8 of 48 layers on host answers "the
// capital of France is the city of the same name, France itself" at 88.77
// tok/s; all 48 on host repeats "ftp" forever. Every one of the three exits 0.
//
// So the placement has to be rejected where it is decided, and by a predicate
// that does not care which route produced it. This is that predicate, kept
// pure so it can be tested without a GPU or a checkpoint.
//
// `layer_expert_bytes[i] > 0` marks layer i as an MoE layer;
// `experts_upload_layer[i]` is true when its experts go to the device.
inline bool expert_placement_is_serveable(bool is_nvfp4_prequant,
                                          const std::vector<size_t>& layer_expert_bytes,
                                          const std::vector<bool>& experts_upload_layer) {
    if (!is_nvfp4_prequant)
        return true;  // GGUF-class experts have a working host path (#1370).
    const size_t n = layer_expert_bytes.size() < experts_upload_layer.size() ? layer_expert_bytes.size()
                                                                             : experts_upload_layer.size();
    for (size_t i = 0; i < n; ++i) {
        if (layer_expert_bytes[i] > 0 && !experts_upload_layer[i])
            return false;
    }
    return true;
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
