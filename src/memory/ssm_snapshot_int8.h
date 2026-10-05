#pragma once

#include "memory/ssm_state_size.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace imp {

// Packed form of one SSM/GDN slab for the recurrent snapshot store (#2419): per layer the conv
// window raw, the h state as int8 groups of 32 with one FP32 absmax scale per group; the
// per-slot tail raw. Every part 256-aligned like the slab.
struct SsmSnapshotInt8Layout {
    size_t conv = 0;       // per layer, raw FP32 conv window
    size_t h_q = 0;        // per layer, int8 groups
    size_t h_scale = 0;    // per layer, FP32 group scales
    size_t per_layer = 0;  // conv + h_q + h_scale
    size_t extra = 0;      // per-slot tail
    size_t total = 0;
    int64_t rows = 0;  // scale groups: n_heads * head_dim * state_size / row_len
    int row_len = 0;   // 32, or state_size when it is not a multiple of 32
};

// total == 0 when the geometry has no h state or an h dtype the codec does not read.
[[nodiscard]] inline SsmSnapshotInt8Layout ssm_snapshot_int8_layout(const SsmStateGeometry& g) {
    SsmSnapshotInt8Layout l;
    const bool dtype_ok = g.h_dtype == QType::F32 || g.h_dtype == QType::BF16 || g.h_dtype == QType::F16;
    if (!dtype_ok || g.n_ssm_layers <= 0 || g.n_heads <= 0 || g.head_dim <= 0 || g.state_size <= 0)
        return l;
    // Groups of 32 along state_size (one scale per row of 128 flipped top-1 in 3 of 7 restores).
    l.row_len = g.state_size % 32 == 0 ? 32 : g.state_size;
    l.rows = static_cast<int64_t>(g.n_heads) * g.head_dim * (g.state_size / l.row_len);
    l.conv = ssm_conv_bytes_per_layer(g);
    l.h_q = ssm_align256(static_cast<size_t>(l.rows) * static_cast<size_t>(l.row_len));
    l.h_scale = ssm_align256(static_cast<size_t>(l.rows) * sizeof(float));
    l.per_layer = l.conv + l.h_q + l.h_scale;
    l.extra = ssm_align256(g.extra_bytes_per_slot);
    l.total = l.per_layer * static_cast<size_t>(g.n_ssm_layers) + l.extra;
    return l;
}

// slab (ssm_bytes_per_slot) -> packed (layout.total) and back, both device pointers, on `stream`.
[[nodiscard]] bool ssm_snapshot_int8_encode(const void* slab, void* packed, const SsmStateGeometry& g,
                                            cudaStream_t stream);
[[nodiscard]] bool ssm_snapshot_int8_decode(const void* packed, void* slab, const SsmStateGeometry& g,
                                            cudaStream_t stream);

}  // namespace imp
