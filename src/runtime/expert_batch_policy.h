#pragma once

// max_batch_size with host-resident MoE experts. Header-only, CUDA-free: the CPU lane pins it.
// Each extra decode row widens the expert set a step streams from the host, and the per-slot
// state/KV of a larger batch comes out of the expert cache budget. Qwen3.8-Flash-Next, batch 32
// vs 1: out tok/s 5.6 vs 7.2 at c=8, ITL p50 934 vs 74 ms.

#include <cstddef>
#include <format>
#include <string>

namespace imp {

// Auto max_batch_size: 1 when any MoE layer is expected on the host, else the VRAM-derived value.
inline int auto_batch_for_host_experts(int auto_batch, int host_expert_layers) {
    return host_expert_layers > 0 ? 1 : auto_batch;
}

// WARN for a batch > 1 with experts on the host (explicit runtime.max_batch_size / --max-batch).
// Empty when it does not apply.
inline std::string host_expert_batch_warning(int max_batch, bool experts_on_host, size_t cache_bytes,
                                             int slots_per_layer) {
    if (!experts_on_host || max_batch <= 1)
        return {};
    return std::format(
        "max_batch_size {} with host-resident experts: the expert cache budget is {:.2f} GiB, {} "
        "slots/layer. Decode rows share it; auto resolves to 1 on this model.",
        max_batch, static_cast<double>(cache_bytes) / (1024.0 * 1024.0 * 1024.0), slots_per_layer);
}

}  // namespace imp
