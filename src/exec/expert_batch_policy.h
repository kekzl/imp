#pragma once

// max_batch_size with host-resident MoE experts. Header-only, CUDA-free: the CPU lane pins it.
// Pinned NVFP4 experts: the device expert cache runs up to kHostExpertAutoBatch decode rows in one
// resolve + GEMV set, the KV ceiling stays one context. Qwen3.8-Flash-Next 8 streams, batch 1 -> 8:
// 81.0 -> 162.2 tok/s; single stream 86.5 -> 86.1. Host expert tier: one row at a time.

#include <algorithm>
#include <cstddef>
#include <format>
#include <string>

namespace imp {

// Auto batch with pinned host experts; <= kDevExpertMaxRows (static_assert in exec).
inline constexpr int kHostExpertAutoBatch = 8;
// KV pool ceiling with host NVFP4 experts, in max_seq_len sequences: rows share one context
// (admission bounds them), the VRAM a batch-sized ceiling held goes to the expert cache.
inline constexpr int kHostExpertKvSeqs = 1;

// Auto max_batch_size: min(auto, kHostExpertAutoBatch) when MoE layers are expected on the host
// and the host can pin them, 1 when they are expected on the host expert tier, else unchanged.
inline int auto_batch_for_host_experts(int auto_batch, int host_expert_layers, bool host_pins_experts) {
    if (host_expert_layers <= 0)
        return auto_batch;
    return host_pins_experts ? std::clamp(auto_batch, 1, kHostExpertAutoBatch) : 1;
}

// Batch after the upload: an auto batch drops to 1 when the upload left the host experts unpinned
// (the resolver's RAM reading can still pass, 64 GiB host: 62.11 GiB free at upload for 62.65 needed).
inline int batch_after_host_expert_pin(int batch, bool auto_batch, bool host_experts_unpinned) {
    return auto_batch && host_experts_unpinned ? std::min(batch, 1) : batch;
}

// KV pool ceiling in max_seq_len sequences: kHostExpertKvSeqs with host NVFP4 experts, else the batch.
inline int kv_ceiling_seqs(int max_batch, bool host_nvfp4_experts) {
    return host_nvfp4_experts ? std::min(max_batch, kHostExpertKvSeqs) : max_batch;
}

// WARN for a batch the device expert cache cannot run as one set of rows: above
// kHostExpertAutoBatch, or with layers on the host expert tier. Empty when it does not apply.
inline std::string host_expert_batch_warning(int max_batch, bool experts_on_host, bool tiered,
                                             size_t cache_bytes, int slots_per_layer) {
    if (!experts_on_host || max_batch <= 1 || (max_batch <= kHostExpertAutoBatch && !tiered))
        return {};
    return std::format(
        "max_batch_size {} with host-resident experts: decode rows {} run one at a time (expert cache "
        "{:.2f} GiB, {} slots/layer).",
        max_batch, tiered ? "on host expert tier layers" : std::format("past {}", kHostExpertAutoBatch),
        static_cast<double>(cache_bytes) / (1024.0 * 1024.0 * 1024.0), slots_per_layer);
}

}  // namespace imp
