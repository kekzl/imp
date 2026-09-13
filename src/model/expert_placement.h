#pragma once

#include <cstddef>
#include <vector>

namespace imp {

// True if this MoE expert placement depends on the NVFP4 host-offload path. Answers only
// "does placement rely on that path"; the refusal for an undersized cache lives in
// GraphExecutor::verify_host_expert_placement() (cache is sized after weight upload).
// layer_expert_bytes[i]>0 marks layer i as MoE; experts_upload_layer[i] means device-resident.
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
