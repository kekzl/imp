#pragma once

// Internal shared device helpers for the executor kernel translation units.
// Split out of executor_kernels.cu so the KV-write / elementwise kernel files
// can share these symbols verbatim. Not part of the public kernel API.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace imp {

// ---------------------------------------------------------------------------
// Device helpers for paged KV cache block table lookup
// ---------------------------------------------------------------------------

// Resolve the physical block ID from the block table for a given token.
// For batched decode (n_sequences > 1), each token maps to its own sequence row.
// For single-sequence or legacy mode, uses a flat block table.
__device__ __forceinline__ int kv_get_block_id(const int* block_tables, int block_idx, int token_idx,
                                               int max_blocks_per_seq, int n_sequences) {
    if (max_blocks_per_seq > 0 && n_sequences > 1)
        return block_tables[token_idx * max_blocks_per_seq + block_idx];
    return block_tables[block_idx];
}

// Compute block index and slot within block from a token's position.
// Returns the physical block ID via kv_get_block_id.
__device__ __forceinline__ int kv_resolve_slot(const int* block_tables, int pos, int block_size,
                                               int token_idx, int max_blocks_per_seq, int n_sequences,
                                               int& slot_in_block) {
    int block_idx = pos / block_size;
    slot_in_block = pos % block_size;
    return kv_get_block_id(block_tables, block_idx, token_idx, max_blocks_per_seq, n_sequences);
}

// ---------------------------------------------------------------------------
// NVFP4 / MXFP4 KV cache write: FP4 E2M1 nibble quantizer (shared by the
// nvfp4 and mxfp4_kv write kernels).
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint8_t e2m1_quantize(float v, float inv_scale) {
    float n = v * inv_scale;
    uint8_t sign = (n < 0.0f) ? 0x8u : 0u;
    float m = fabsf(n);
    // Nearest in {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}.
    // Tested midpoint boundaries: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
    uint8_t mag;
    if (m < 0.25f)
        mag = 0;
    else if (m < 0.75f)
        mag = 1;
    else if (m < 1.25f)
        mag = 2;
    else if (m < 1.75f)
        mag = 3;
    else if (m < 2.5f)
        mag = 4;
    else if (m < 3.5f)
        mag = 5;
    else if (m < 5.0f)
        mag = 6;
    else
        mag = 7;
    return sign | mag;
}

}  // namespace imp

// FP4/UE8M0 helpers — shared by write_kv_cache_mxfp4_kv_kernel and
// attention_paged_nvfp4.cu (decode_kv_scale<UE8M0> specialization).
// Close namespace before including (header defines in imp::).
#include "quant/turboquant_fp4.cuh"

namespace imp {

// Aliases for shorter names in write_kv_cache_mxfp4_kv_kernel below.
#define tq_float_to_ue8m0 tq_fp4_float_to_ue8m0
#define tq_ue8m0_to_float tq_fp4_ue8m0_to_float

}  // namespace imp
