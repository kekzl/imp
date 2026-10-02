#pragma once

#include "core/qtype.h"

#include <cstddef>
#include <cstdint>

namespace imp {

// Scale plane layout of a quantized KV dtype.
enum class KvScale : uint8_t {
    None,      // no scale plane
    PerHead,   // one scale_bytes entry per (token, KV head)
    PerGroup,  // one scale_bytes entry per scale_group elements along head_dim
};

// Storage facts of one KV cache dtype. Adding a KV dtype = one row in kv_dtype_info().
struct KvDtypeInfo {
    bool kv;      // a KV cache dtype (has a row below)
    bool packed;  // two elements per byte
    KvScale scale;
    uint8_t scale_bytes;  // bytes per scale entry
    uint8_t scale_group;  // elements per scale entry along head_dim (PerGroup only)

    [[nodiscard]] constexpr bool has_scales() const { return scale != KvScale::None; }
};

// FP4 micro-block: 16 elements share one scale byte (E4M3 for NVFP4, UE8M0 for MXFP4_KV).
inline constexpr int kKvFp4Group = 16;

[[nodiscard]] constexpr KvDtypeInfo kv_dtype_info(QType q) {
    switch (q) {
        case QType::F16:
        case QType::FP8_E4M3:
            return {true, false, KvScale::None, 0, 0};
        case QType::INT8:
            return {true, false, KvScale::PerHead, 2, 0};  // FP16 scale
        case QType::INT4:
            return {true, true, KvScale::PerHead, 2, 0};  // FP16 scale
        case QType::NVFP4:
        case QType::MXFP4_KV:
            return {true, true, KvScale::PerGroup, 1, kKvFp4Group};
        default:
            return {false, false, KvScale::None, 0, 0};
    }
}

// Data bytes of `elems` KV elements; unpacked dtypes cost qtype_elem_bytes() each.
[[nodiscard]] inline size_t kv_data_bytes(QType q, size_t elems) {
    return kv_dtype_info(q).packed ? elems / 2 : elems * qtype_elem_bytes(q);
}

// Data bytes of one K (or V) block: block_size tokens x n_kv_heads x head_dim.
[[nodiscard]] inline size_t kv_block_data_bytes(QType q, int block_size, int n_kv_heads, int head_dim) {
    return kv_data_bytes(q, static_cast<size_t>(block_size) * n_kv_heads * head_dim);
}

// Scale bytes of one K (or V) block; 0 without a scale plane. PerGroup needs
// head_dim % scale_group == 0 (kv_scale_head_dim_ok()).
[[nodiscard]] constexpr size_t kv_block_scale_bytes(QType q, int block_size, int n_kv_heads, int head_dim) {
    const KvDtypeInfo d = kv_dtype_info(q);
    const size_t per_token_head = d.scale == KvScale::PerHead ? d.scale_bytes
                                  : d.scale == KvScale::PerGroup
                                      ? static_cast<size_t>(head_dim / d.scale_group) * d.scale_bytes
                                      : 0;
    return static_cast<size_t>(block_size) * n_kv_heads * per_token_head;
}

// Scale entries (not bytes) of one K (or V) block: the kernels' scale stride. 0 without scales.
[[nodiscard]] constexpr size_t kv_block_scale_entries(QType q, int block_size, int n_kv_heads, int head_dim) {
    const KvDtypeInfo d = kv_dtype_info(q);
    return d.has_scales() ? kv_block_scale_bytes(q, block_size, n_kv_heads, head_dim) / d.scale_bytes : 0;
}

[[nodiscard]] constexpr bool kv_scale_head_dim_ok(QType q, int head_dim) {
    const KvDtypeInfo d = kv_dtype_info(q);
    return d.scale != KvScale::PerGroup || head_dim % d.scale_group == 0;
}

}  // namespace imp
