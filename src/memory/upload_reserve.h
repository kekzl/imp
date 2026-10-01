#pragma once

// The VRAM reserve weight upload leaves free (Engine::init_weights -> Model::upload_weights_gpu)
// as a function of max_batch_size, and the largest batch whose reserve fits the weights
// (#2393). Header-only and CUDA-free so the CPU lane pins it.

#include <algorithm>
#include <cstddef>
#include <format>
#include <string>

namespace imp {

// Fixed and batch-shaped charges of the upload reserve.
struct UploadReserve {
    size_t fixed_bytes = 0;           // workspace + recurrent snapshot store + safety
    size_t kv_bytes_per_seq = 0;      // one sequence at max_seq_len
    size_t kv_cap_bytes = 0;          // total VRAM / 5 (MoE > 16 experts: / 10)
    size_t state_bytes_per_slot = 0;  // SSM/GDN state slot, 0 without recurrent layers
    int reserved_state_slots = 0;     // spec multi-candidate verify slots past the batch
};

// Engine -> upload: the reserve and the configured batch; upload -> Engine: the batch Pass 1 fits.
struct UploadBatchFit {
    UploadReserve reserve;
    int configured = 0;  // <= 1: no clamp, the reserve stays fixed
    int fitted = 0;      // configured, or the clamp
};

inline size_t upload_reserve_kv_bytes(const UploadReserve& r, int batch) {
    return std::min(r.kv_bytes_per_seq * static_cast<size_t>(std::max(batch, 0)), r.kv_cap_bytes);
}

inline size_t upload_reserve_bytes(const UploadReserve& r, int batch) {
    const int slots = std::max(batch, 0) + std::max(r.reserved_state_slots, 0);
    return r.fixed_bytes + upload_reserve_kv_bytes(r, batch) +
           r.state_bytes_per_slot * static_cast<size_t>(slots);
}

// Largest batch in [1, configured] with weights + reserve <= free, the test checked_cuda_malloc
// applies; configured when it fits, 0 when batch 1 does not.
inline int upload_fitting_batch(const UploadReserve& r, int configured, size_t free_bytes,
                                size_t weight_bytes) {
    for (int b = configured; b >= 1; --b)
        if (weight_bytes + upload_reserve_bytes(r, b) <= free_bytes)
            return b;
    return 0;
}

inline std::string upload_batch_clamp_message(const UploadReserve& r, int configured, int fitted,
                                              size_t free_bytes, size_t weight_bytes) {
    constexpr double kMiB = 1024.0 * 1024.0;
    return std::format(
        "max_batch_size clamped {} -> {} at weight upload: {:.0f} MiB weights + {:.0f} MiB reserve at {} "
        "slots "
        "exceed {:.0f} MiB free; each slot costs {:.1f} MiB SSM/GDN state, KV reserve {:.0f} MiB. "
        "Set runtime.max_batch_size={} to silence this, or free VRAM to raise it.",
        configured, fitted, weight_bytes / kMiB, upload_reserve_bytes(r, configured) / kMiB, configured,
        free_bytes / kMiB, r.state_bytes_per_slot / kMiB, upload_reserve_kv_bytes(r, configured) / kMiB,
        fitted);
}

}  // namespace imp
