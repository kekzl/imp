#include "exec/executor_kernels.h"
#include "exec/executor_kernels_internal.cuh"
#include "compute/ptx92_utils.cuh"
#include "compute/warp_reduce.cuh"  // kWarpSize

namespace imp {

// Copy K/V for a set of tokens into paged KV cache blocks.
// Each token's K (or V) slice is copied to the correct slot in the right block.
//
// data_in:          [n_tokens, n_kv_heads * head_dim] contiguous
// positions:        [n_tokens] position of each token in the sequence
// block_tables:     [n_sequences, max_blocks_per_seq] or [max_blocks] block IDs
// cache_base:       base pointer of the KV pool for this layer (block 0)
// block_stride:     elements per block = kKVBlockSize * n_kv_heads * head_dim
// row_elems:        n_kv_heads * head_dim (elements per token)
// max_blocks_per_seq: stride for 2D block table (0 = legacy flat)
// n_sequences:      number of sequences in the batch
__global__ __launch_bounds__(256) void write_kv_cache_kernel(const half* __restrict__ data_in,
                                                             const int* __restrict__ positions,
                                                             const int* __restrict__ block_tables,
                                                             half* __restrict__ cache_base, int block_stride,
                                                             int row_elems, int block_size, int n_tokens,
                                                             int max_blocks_per_seq, int n_sequences) {
    int token_idx = blockIdx.x;
    if (token_idx >= n_tokens)
        return;

    int pos = positions[token_idx];
    int slot_in_block;
    int block_id = kv_resolve_slot(block_tables, pos, block_size, token_idx, max_blocks_per_seq, n_sequences,
                                   slot_in_block);
    // Defense-in-depth (F-A12): a negative block_id — a freed StreamingLLM -1
    // sentinel, or a future block-table bug — would index the KV pool OOB.
    // block_id is uniform across the block (derived from blockIdx.x), so this
    // skips the whole write without divergence. Never fires today (host-side
    // admission guarantees every written position has a real block).
    if (block_id < 0)
        return;

    half* dst = cache_base + static_cast<int64_t>(block_id) * block_stride +
                static_cast<int64_t>(slot_in_block) * row_elems;
    const half* src = data_in + static_cast<int64_t>(token_idx) * row_elems;

    // Vectorized 128-bit copy (8 FP16 per store) — row_elems is always a
    // multiple of 8 (n_kv_heads * head_dim, where head_dim is power of 2).
    const int vec_elems = row_elems / 8;
    const float4* src4 = reinterpret_cast<const float4*>(src);
    float4* dst4 = reinterpret_cast<float4*>(dst);
    for (int i = threadIdx.x; i < vec_elems; i += blockDim.x) {
        dst4[i] = src4[i];
    }
}

// Fused K+V write to paged KV cache in a single launch.
// blockIdx.x = token index, blockIdx.y = 0 (K) or 1 (V).
// Saves one kernel launch per attention layer.
__global__ __launch_bounds__(256) void write_kv_cache_fused_kernel(
    const half* __restrict__ k_in,  // [n_tokens, n_kv_heads * head_dim]
    const half* __restrict__ v_in,  // [n_tokens, n_kv_heads * head_dim]
    const int* __restrict__ positions, const int* __restrict__ block_tables, half* __restrict__ k_cache_base,
    half* __restrict__ v_cache_base, int block_stride, int row_elems, int block_size, int n_tokens,
    int max_blocks_per_seq, int n_sequences) {
    int token_idx = blockIdx.x;
    if (token_idx >= n_tokens)
        return;

    int pos = positions[token_idx];
    int slot_in_block;
    int block_id = kv_resolve_slot(block_tables, pos, block_size, token_idx, max_blocks_per_seq, n_sequences,
                                   slot_in_block);
    // Defense-in-depth (F-A12): a negative block_id — a freed StreamingLLM -1
    // sentinel, or a future block-table bug — would index the KV pool OOB.
    // block_id is uniform across the block (derived from blockIdx.x), so this
    // skips the whole write without divergence. Never fires today (host-side
    // admission guarantees every written position has a real block).
    if (block_id < 0)
        return;

    // blockIdx.y selects K (0) or V (1)
    const half* src;
    half* dst_base;
    if (blockIdx.y == 0) {
        src = k_in + static_cast<int64_t>(token_idx) * row_elems;
        dst_base = k_cache_base;
    } else {
        src = v_in + static_cast<int64_t>(token_idx) * row_elems;
        dst_base = v_cache_base;
    }

    half* dst = dst_base + static_cast<int64_t>(block_id) * block_stride +
                static_cast<int64_t>(slot_in_block) * row_elems;

    // Vectorized 128-bit copy (8 FP16 per store)
    const int vec_elems = row_elems / 8;
    const float4* src4 = reinterpret_cast<const float4*>(src);
    float4* dst4 = reinterpret_cast<float4*>(dst);
    for (int i = threadIdx.x; i < vec_elems; i += blockDim.x) {
        dst4[i] = src4[i];
    }
}

// FP16 -> FP8 E4M3 quantization + write to paged KV cache
__global__ __launch_bounds__(256) void write_kv_cache_fp8_kernel(
    const half* __restrict__ data_in, const int* __restrict__ positions, const int* __restrict__ block_tables,
    __nv_fp8_e4m3* __restrict__ cache_base,  // FP8 cache
    float inv_scale,                         // 1.0 / kv_scale
    int block_stride, int row_elems, int block_size, int n_tokens, int max_blocks_per_seq, int n_sequences) {
    int token_idx = blockIdx.x;
    if (token_idx >= n_tokens)
        return;

    int pos = positions[token_idx];
    int slot_in_block;
    int block_id = kv_resolve_slot(block_tables, pos, block_size, token_idx, max_blocks_per_seq, n_sequences,
                                   slot_in_block);
    // Defense-in-depth (F-A12): a negative block_id — a freed StreamingLLM -1
    // sentinel, or a future block-table bug — would index the KV pool OOB.
    // block_id is uniform across the block (derived from blockIdx.x), so this
    // skips the whole write without divergence. Never fires today (host-side
    // admission guarantees every written position has a real block).
    if (block_id < 0)
        return;

    __nv_fp8_e4m3* dst = cache_base + static_cast<int64_t>(block_id) * block_stride +
                         static_cast<int64_t>(slot_in_block) * row_elems;
    const half* src = data_in + static_cast<int64_t>(token_idx) * row_elems;

    // Packed PTX cvt: 2 paired conversions per 4 elements (half→e4m3x2).
    // Scale applied in FP16 before conversion — sufficient precision for E4M3.
    const half inv_scale_h = __float2half(inv_scale);
    const half2 inv_scale_h2 = make_half2(inv_scale_h, inv_scale_h);
    const int vec_elems = row_elems / 4;
    const half2* src2 = reinterpret_cast<const half2*>(src);
    uint32_t* dst4 = reinterpret_cast<uint32_t*>(dst);
    for (int i = threadIdx.x; i < vec_elems; i += blockDim.x) {
        half2 lo = __hmul2(src2[2 * i], inv_scale_h2);
        half2 hi = __hmul2(src2[2 * i + 1], inv_scale_h2);
        uint16_t e4m3_lo = cvt_f16x2_to_e4m3x2(*reinterpret_cast<uint32_t*>(&lo));
        uint16_t e4m3_hi = cvt_f16x2_to_e4m3x2(*reinterpret_cast<uint32_t*>(&hi));
        dst4[i] = static_cast<uint32_t>(e4m3_lo) | (static_cast<uint32_t>(e4m3_hi) << 16);
    }
    // Scalar tail for non-aligned remainder
    for (int i = vec_elems * 4 + threadIdx.x; i < row_elems; i += blockDim.x) {
        dst[i] = __nv_fp8_e4m3(__half2float(src[i]) * inv_scale);
    }
}

// ---------------------------------------------------------------------------
// FP16 -> INT8 quantization + write to paged KV cache with per-head scales.
// Each warp processes one KV head independently: compute absmax via warp shuffle,
// then quantize and write int8 data + half scale.
//
// blockIdx.x = token_idx, blockIdx.y = 0 (K) or 1 (V).
// blockDim.x = 256 (8 warps). Each warp loops over heads.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void write_kv_cache_int8_kernel(
    const half* __restrict__ k_in,  // [n_tokens, n_kv_heads * head_dim]
    const half* __restrict__ v_in, const int* __restrict__ positions, const int* __restrict__ block_tables,
    int8_t* __restrict__ k_cache_base, int8_t* __restrict__ v_cache_base,
    half* __restrict__ k_scale_base,  // [total_blocks, kKVBlockSize, n_kv_heads]
    half* __restrict__ v_scale_base,
    int block_stride,        // kKVBlockSize * n_kv_heads * head_dim (int8 elems)
    int scale_block_stride,  // kKVBlockSize * n_kv_heads (half elems)
    int n_kv_heads, int head_dim, int block_size, int n_tokens, int max_blocks_per_seq, int n_sequences) {
    const int token_idx = blockIdx.x;
    if (token_idx >= n_tokens)
        return;

    const int pos = positions[token_idx];
    int slot_in_block;
    int block_id = kv_resolve_slot(block_tables, pos, block_size, token_idx, max_blocks_per_seq, n_sequences,
                                   slot_in_block);
    // Defense-in-depth (F-A12): a negative block_id — a freed StreamingLLM -1
    // sentinel, or a future block-table bug — would index the KV pool OOB.
    // block_id is uniform across the block (derived from blockIdx.x), so this
    // skips the whole write without divergence. Never fires today (host-side
    // admission guarantees every written position has a real block).
    if (block_id < 0)
        return;

    // Select K or V based on blockIdx.y
    const half* src_base = (blockIdx.y == 0) ? k_in : v_in;
    int8_t* cache_base = (blockIdx.y == 0) ? k_cache_base : v_cache_base;
    half* scale_base = (blockIdx.y == 0) ? k_scale_base : v_scale_base;

    const int row_elems = n_kv_heads * head_dim;
    const half* src = src_base + static_cast<int64_t>(token_idx) * row_elems;
    int8_t* dst = cache_base + static_cast<int64_t>(block_id) * block_stride +
                  static_cast<int64_t>(slot_in_block) * row_elems;
    half* scale_dst = scale_base + static_cast<int64_t>(block_id) * scale_block_stride +
                      static_cast<int64_t>(slot_in_block) * n_kv_heads;

    const int warp_id = threadIdx.x / kWarpSize;
    const int lane_id = threadIdx.x % kWarpSize;
    const int num_warps = blockDim.x / kWarpSize;

    // Each warp processes one head at a time, looping over heads
    for (int h = warp_id; h < n_kv_heads; h += num_warps) {
        const int head_offset = h * head_dim;

        // Step 1: Load FP16 values and compute per-head absmax
        // Vectorized: load 2 FP16 per iteration via half2 for better coalescing.
        // head_dim is always even (64, 128, 256).
        float amax = 0.0f;
        const half2* src2 = reinterpret_cast<const half2*>(src + head_offset);
        for (int d = lane_id; d < head_dim / 2; d += kWarpSize) {
            half2 h2 = src2[d];
            amax = fmaxf(amax, fabsf(__half2float(h2.x)));
            amax = fmaxf(amax, fabsf(__half2float(h2.y)));
        }
// Warp-level absmax reduction
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset));

        // Step 2: Compute scale
        float sc = amax / 127.0f;
        float inv_sc = (amax > 1e-8f) ? (127.0f / amax) : 0.0f;

        // Step 3: Quantize and write int8 data (vectorized: load 4 FP16 via float2,
        // store 4 INT8 via uint32_t). head_dim is always a multiple of 4 (64, 128, 256).
        // Each lane processes 4 consecutive elements per iteration, stride = 32*4 = 128.
        const float2* src_head4 = reinterpret_cast<const float2*>(src + head_offset);
        for (int d4 = lane_id; d4 < head_dim / 4; d4 += 32) {
            float2 h2 = src_head4[d4];
            const half* hp = reinterpret_cast<const half*>(&h2);
            uint32_t packed;
            int8_t* p = reinterpret_cast<int8_t*>(&packed);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                p[j] = static_cast<int8_t>(__float2int_rn(__half2float(hp[j]) * inv_sc));
            }
            reinterpret_cast<uint32_t*>(dst + head_offset)[d4] = packed;
        }

        // Step 4: Write scale (one half per head per token)
        if (lane_id == 0) {
            scale_dst[h] = __float2half(sc);
        }
    }
}

// ---------------------------------------------------------------------------
// INT4 KV cache write: FP16 → 4-bit symmetric quantization with per-head scales.
// Two INT4 values packed into one byte (low nibble = even index, high nibble = odd).
// Range: [-8, 7] symmetric. Scale = absmax / 7.0.
// blockIdx.x = token, blockIdx.y = 0 (K) or 1 (V).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void write_kv_cache_int4_kernel(
    const half* __restrict__ k_in, const half* __restrict__ v_in, const int* __restrict__ positions,
    const int* __restrict__ block_tables,
    uint8_t* __restrict__ k_cache_base,  // packed INT4 pairs
    uint8_t* __restrict__ v_cache_base, half* __restrict__ k_scale_base, half* __restrict__ v_scale_base,
    int block_stride,        // kKVBlockSize * n_kv_heads * head_dim / 2 (bytes)
    int scale_block_stride,  // kKVBlockSize * n_kv_heads (half elems)
    int n_kv_heads, int head_dim, int block_size, int n_tokens, int max_blocks_per_seq, int n_sequences) {
    const int token_idx = blockIdx.x;
    if (token_idx >= n_tokens)
        return;

    const int pos = positions[token_idx];
    int slot_in_block;
    int block_id = kv_resolve_slot(block_tables, pos, block_size, token_idx, max_blocks_per_seq, n_sequences,
                                   slot_in_block);
    // Defense-in-depth (F-A12): a negative block_id — a freed StreamingLLM -1
    // sentinel, or a future block-table bug — would index the KV pool OOB.
    // block_id is uniform across the block (derived from blockIdx.x), so this
    // skips the whole write without divergence. Never fires today (host-side
    // admission guarantees every written position has a real block).
    if (block_id < 0)
        return;

    const half* src_base = (blockIdx.y == 0) ? k_in : v_in;
    uint8_t* cache_base = (blockIdx.y == 0) ? k_cache_base : v_cache_base;
    half* scale_base = (blockIdx.y == 0) ? k_scale_base : v_scale_base;

    const int row_elems = n_kv_heads * head_dim;
    const int row_bytes = row_elems / 2;  // 2 INT4 values per byte
    const half* src = src_base + static_cast<int64_t>(token_idx) * row_elems;
    uint8_t* dst = cache_base + static_cast<int64_t>(block_id) * block_stride +
                   static_cast<int64_t>(slot_in_block) * row_bytes;
    half* scale_dst = scale_base + static_cast<int64_t>(block_id) * scale_block_stride +
                      static_cast<int64_t>(slot_in_block) * n_kv_heads;

    const int warp_id = threadIdx.x / kWarpSize;
    const int lane_id = threadIdx.x % kWarpSize;
    const int num_warps = blockDim.x / kWarpSize;

    for (int h = warp_id; h < n_kv_heads; h += num_warps) {
        const int head_offset = h * head_dim;

        // Step 1: Per-head absmax
        float amax = 0.0f;
        for (int d = lane_id; d < head_dim; d += kWarpSize) {
            float val = __half2float(src[head_offset + d]);
            amax = fmaxf(amax, fabsf(val));
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset));

        // Step 2: Scale (symmetric INT4: [-8, 7], use 7 for range)
        float sc = amax / 7.0f;
        float inv_sc = (amax > 1e-8f) ? (7.0f / amax) : 0.0f;

        // Step 3: Quantize and pack pairs into bytes
        // Each lane handles 2 elements at a time (d, d+1) → 1 byte
        const int head_byte_offset = h * head_dim / 2;
        for (int d = lane_id * 2; d < head_dim; d += 2 * kWarpSize) {
            float v0 = __half2float(src[head_offset + d]);
            float v1 = (d + 1 < head_dim) ? __half2float(src[head_offset + d + 1]) : 0.0f;

            int q0 = __float2int_rn(v0 * inv_sc);
            int q1 = __float2int_rn(v1 * inv_sc);
            q0 = max(-8, min(7, q0));
            q1 = max(-8, min(7, q1));

            // Pack: low nibble = q0, high nibble = q1
            uint8_t packed = (static_cast<uint8_t>(q0 & 0xF)) | (static_cast<uint8_t>(q1 & 0xF) << 4);
            dst[head_byte_offset + d / 2] = packed;
        }

        // Step 4: Write scale
        if (lane_id == 0) {
            scale_dst[h] = __float2half(sc);
        }
    }
}

// ---------------------------------------------------------------------------
// NVFP4 KV cache write kernel
// Per (token, head, group of 16 elems along head_dim):
//   1. absmax over 16 elems
//   2. scale = absmax / 6.0  (FP4 E2M1 max = 6.0); store as UE4M3 byte
//   3. quant each elem to E2M1 nibble (nearest-magnitude + sign), pack 2/byte
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void write_kv_cache_nvfp4_kernel(
    const half* __restrict__ k_in, const half* __restrict__ v_in, const int* __restrict__ positions,
    const int* __restrict__ block_tables, uint8_t* __restrict__ k_cache_base,
    uint8_t* __restrict__ v_cache_base, uint8_t* __restrict__ k_scale_base,
    uint8_t* __restrict__ v_scale_base, int block_stride, int scale_block_stride, int n_kv_heads,
    int head_dim, int block_size, int n_tokens, int max_blocks_per_seq, int n_sequences) {
    constexpr int kGroup = 16;

    const int token_idx = blockIdx.x;
    if (token_idx >= n_tokens)
        return;

    const int pos = positions[token_idx];
    int slot_in_block;
    int block_id = kv_resolve_slot(block_tables, pos, block_size, token_idx, max_blocks_per_seq, n_sequences,
                                   slot_in_block);
    // Defense-in-depth (F-A12): a negative block_id — a freed StreamingLLM -1
    // sentinel, or a future block-table bug — would index the KV pool OOB.
    // block_id is uniform across the block (derived from blockIdx.x), so this
    // skips the whole write without divergence. Never fires today (host-side
    // admission guarantees every written position has a real block).
    if (block_id < 0)
        return;

    const half* src_base = (blockIdx.y == 0) ? k_in : v_in;
    uint8_t* cache_base = (blockIdx.y == 0) ? k_cache_base : v_cache_base;
    uint8_t* scale_base = (blockIdx.y == 0) ? k_scale_base : v_scale_base;

    const int row_elems = n_kv_heads * head_dim;
    const int row_bytes = row_elems / 2;
    const int row_scale_bytes = n_kv_heads * (head_dim / kGroup);
    const half* src = src_base + static_cast<int64_t>(token_idx) * row_elems;
    uint8_t* dst = cache_base + static_cast<int64_t>(block_id) * block_stride +
                   static_cast<int64_t>(slot_in_block) * row_bytes;
    uint8_t* scale_dst = scale_base + static_cast<int64_t>(block_id) * scale_block_stride +
                         static_cast<int64_t>(slot_in_block) * row_scale_bytes;

    const int n_groups_per_head = head_dim / kGroup;
    const int total_groups = n_kv_heads * n_groups_per_head;

    // One thread per group (each group = 16 elems = 8 bytes packed FP4 + 1 UE4M3 scale byte).
    for (int g = threadIdx.x; g < total_groups; g += blockDim.x) {
        int h = g / n_groups_per_head;
        int gh = g % n_groups_per_head;             // group within head
        int base_elem = h * head_dim + gh * kGroup;  // first elem in this group

        // absmax
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < kGroup; i++) {
            float v = __half2float(src[base_elem + i]);
            amax = fmaxf(amax, fabsf(v));
        }
        float sc = amax / 6.0f;
        float inv_sc = (sc > 1e-30f) ? (1.0f / sc) : 0.0f;

        // pack 16 nibbles → 8 bytes
        int dst_byte_off = h * (head_dim / 2) + gh * (kGroup / 2);
#pragma unroll
        for (int p = 0; p < kGroup / 2; p++) {
            float v0 = __half2float(src[base_elem + 2 * p]);
            float v1 = __half2float(src[base_elem + 2 * p + 1]);
            uint8_t q0 = e2m1_quantize(v0, inv_sc);
            uint8_t q1 = e2m1_quantize(v1, inv_sc);
            dst[dst_byte_off + p] = static_cast<uint8_t>(q0 | (q1 << 4));
        }

        // store UE4M3 scale (saturates oversize values, 0 if amax==0)
        __nv_fp8_e4m3 ue4m3(sc);
        scale_dst[h * n_groups_per_head + gh] = *reinterpret_cast<uint8_t*>(&ue4m3);
    }
}

// ---------------------------------------------------------------------------
// MXFP4-KV write kernel: same layout as NVFP4 but stores UE8M0 scale bytes.
//
// The only difference from write_kv_cache_nvfp4_kernel is the scale encoding:
//   NVFP4:    `__nv_fp8_e4m3 ue4m3(sc); scale_dst[...] = reinterpret_cast<uint8_t>(&ue4m3);`
//   MXFP4_KV: `scale_dst[...] = tq_float_to_ue8m0(sc);`   (pure-exponent 8-bit)
//
// All other fields (block_stride, scale_block_stride, FP4 packing, group size)
// are identical. The tq_float_to_ue8m0 alias is defined at line 1084.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void write_kv_cache_mxfp4_kv_kernel(
    const half* __restrict__ k_in, const half* __restrict__ v_in, const int* __restrict__ positions,
    const int* __restrict__ block_tables, uint8_t* __restrict__ k_cache_base,
    uint8_t* __restrict__ v_cache_base, uint8_t* __restrict__ k_scale_base,
    uint8_t* __restrict__ v_scale_base, int block_stride, int scale_block_stride, int n_kv_heads,
    int head_dim, int block_size, int n_tokens, int max_blocks_per_seq, int n_sequences) {
    constexpr int kGroup = 16;

    const int token_idx = blockIdx.x;
    if (token_idx >= n_tokens)
        return;

    const int pos = positions[token_idx];
    int slot_in_block;
    int block_id = kv_resolve_slot(block_tables, pos, block_size, token_idx, max_blocks_per_seq, n_sequences,
                                   slot_in_block);
    // Defense-in-depth (F-A12): a negative block_id — a freed StreamingLLM -1
    // sentinel, or a future block-table bug — would index the KV pool OOB.
    // block_id is uniform across the block (derived from blockIdx.x), so this
    // skips the whole write without divergence. Never fires today (host-side
    // admission guarantees every written position has a real block).
    if (block_id < 0)
        return;

    const half* src_base = (blockIdx.y == 0) ? k_in : v_in;
    uint8_t* cache_base = (blockIdx.y == 0) ? k_cache_base : v_cache_base;
    uint8_t* scale_base = (blockIdx.y == 0) ? k_scale_base : v_scale_base;

    const int row_elems = n_kv_heads * head_dim;
    const int row_bytes = row_elems / 2;
    const int row_scale_bytes = n_kv_heads * (head_dim / kGroup);
    const half* src = src_base + static_cast<int64_t>(token_idx) * row_elems;
    uint8_t* dst = cache_base + static_cast<int64_t>(block_id) * block_stride +
                   static_cast<int64_t>(slot_in_block) * row_bytes;
    uint8_t* scale_dst = scale_base + static_cast<int64_t>(block_id) * scale_block_stride +
                         static_cast<int64_t>(slot_in_block) * row_scale_bytes;

    const int n_groups_per_head = head_dim / kGroup;
    const int total_groups = n_kv_heads * n_groups_per_head;

    for (int g = threadIdx.x; g < total_groups; g += blockDim.x) {
        int h = g / n_groups_per_head;
        int gh = g % n_groups_per_head;
        int base_elem = h * head_dim + gh * kGroup;

        // absmax
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < kGroup; i++) {
            float v = __half2float(src[base_elem + i]);
            amax = fmaxf(amax, fabsf(v));
        }
        float sc_exact = amax / 6.0f;
        // Round-trip-consistent scale: quantize to UE8M0 first, then use the
        // ACTUAL decoded scale for nibble quantization. The NVFP4 write kernel
        // gets away with using sc_exact directly because E4M3's mantissa keeps
        // the rounding error ~1.5%, but UE8M0 is power-of-2 only (up to 2x
        // rounding error per group) — a mismatch between encoder/decoder
        // scales compounds catastrophically over 32 layers (degenerate output
        // observed in Phase 2 NIAH re-run, 0% retrieval even at 4K context).
        uint8_t sc_byte = tq_float_to_ue8m0(sc_exact);
        float sc_actual = tq_ue8m0_to_float(sc_byte);
        float inv_sc = (sc_actual > 1e-30f) ? (1.0f / sc_actual) : 0.0f;

        // pack 16 nibbles → 8 bytes
        int dst_byte_off = h * (head_dim / 2) + gh * (kGroup / 2);
#pragma unroll
        for (int p = 0; p < kGroup / 2; p++) {
            float v0 = __half2float(src[base_elem + 2 * p]);
            float v1 = __half2float(src[base_elem + 2 * p + 1]);
            uint8_t q0 = e2m1_quantize(v0, inv_sc);
            uint8_t q1 = e2m1_quantize(v1, inv_sc);
            dst[dst_byte_off + p] = static_cast<uint8_t>(q0 | (q1 << 4));
        }

        scale_dst[h * n_groups_per_head + gh] = sc_byte;
    }
}

// ---------------------------------------------------------------------------
// BitDecoding Phase 3c: FP16 residual ring write.
//
// blockIdx.x = token_idx (0..n_tokens-1); blockIdx.y selects K (0) or V (1).
// blockDim.x threads stripe across slot_elems = n_kv_heads * head_dim. The
// per-token destination pointer (already resolved on the host to the right
// (seq_slot, layer, K|V, ring_slot) location) is read from the per-token
// pointer array.
//
// Replaces a pair of `cudaMemcpyAsync(dst, src, slot_elems*sizeof(half),
// cudaMemcpyDeviceToDevice, stream)` calls per layer, which were observed
// to serialize on the copy engine and dominate decode tg/s when residual
// was enabled (-3× regression on Qwen3-4B Q8 NVFP4-KV bench at 4K ctx).
// ---------------------------------------------------------------------------
__global__ void residual_kv_write_multi_kernel(
    const half* __restrict__ k_in,
    const half* __restrict__ v_in,
    half* const* __restrict__ residual_k_dst_ptrs,
    half* const* __restrict__ residual_v_dst_ptrs,
    int slot_elems) {
    const int token_idx = blockIdx.x;
    const bool is_v = (blockIdx.y == 1);

    half* dst = is_v ? residual_v_dst_ptrs[token_idx] : residual_k_dst_ptrs[token_idx];
    if (dst == nullptr) return;
    const half* src = (is_v ? v_in : k_in) + static_cast<int64_t>(token_idx) * slot_elems;

    for (int i = threadIdx.x; i < slot_elems; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// Graph-safe single-seq variant: reads write_idx from a device pointer at
// kernel execution time, so the captured kernel sees the current ring state
// across graph replays. blockIdx.x ∈ {0, 1} selects K or V; threads stripe
// across slot_elems.
__global__ void residual_kv_write_indirect_kernel(
    const half* __restrict__ k_in,
    const half* __restrict__ v_in,
    half* __restrict__ residual_k_layer_seq_base,
    half* __restrict__ residual_v_layer_seq_base,
    const int* __restrict__ d_residual_widx_ptr,
    int seq_slot,
    int slot_elems) {
    const bool is_v = (blockIdx.x == 1);
    half* base = is_v ? residual_v_layer_seq_base : residual_k_layer_seq_base;
    if (base == nullptr) return;
    const half* src = is_v ? v_in : k_in;
    const int widx = d_residual_widx_ptr[seq_slot];
    half* dst = base + static_cast<int64_t>(widx) * slot_elems;

    const int i = threadIdx.x + blockIdx.y * blockDim.x;
    if (i < slot_elems) {
        dst[i] = src[i];
    }
}

__global__ void advance_residual_state_kernel(
    int* __restrict__ d_widx,
    int* __restrict__ d_fc,
    int slot,
    int residual_n_tokens) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int w = d_widx[slot];
        int f = d_fc[slot];
        d_widx[slot] = (w + 1) % residual_n_tokens;
        d_fc[slot] = (f < residual_n_tokens) ? (f + 1) : f;
    }
}

// Linear-mode RoPE cos/sin for one pair, shared by the two decode KV-write RoPE
// kernels below (write_kv_cache_rope_fused / rope_q_only). These fuse RoPE only
// when the model is linear-scaled (the can_fuse_rope_kv gate requires
// yarn_ext_factor <= 0), so this needs no YaRN blend and stays bit-exact with
// the previous inline copies — kept separate from the general rope_yarn() helper
// (rope_yarn.cuh), whose interpolate-then-scale grouping would shift the decode
// KV cache by an ULP for no functional gain.
static __device__ __forceinline__ void rope_linear_cos_sin(int pos, int pair_idx, float theta,
                                                           float inv_scaling, int rope_pairs,
                                                           const float* __restrict__ longrope_inv_freqs,
                                                           float& cos_val, float& sin_val) {
    float freq;
    if (longrope_inv_freqs) {
        // Pre-computed effective frequencies (see gguf_loader.cpp rope_freqs conversion)
        freq = longrope_inv_freqs[pair_idx];
    } else {
        freq = 1.0f / (powf(theta, (2.0f * pair_idx) / static_cast<float>(2 * rope_pairs)));
        freq *= inv_scaling;
    }
    float angle = static_cast<float>(pos) * freq;
    cos_val = __cosf(angle);
    sin_val = __sinf(angle);
}

// Fused KV cache write with RoPE on K: applies RoPE to K during write, copies V directly.
// blockIdx.x = token index, blockIdx.y = 0 (K+RoPE) or 1 (V copy).
// Eliminates the separate RoPE kernel launch for K in the decode path.
__global__ __launch_bounds__(256) void write_kv_cache_rope_fused_kernel(
    const half* __restrict__ k_in,  // [n_tokens, n_kv_heads * head_dim] raw K (no RoPE)
    const half* __restrict__ v_in,  // [n_tokens, n_kv_heads * head_dim]
    const int* __restrict__ positions, const int* __restrict__ block_tables, half* __restrict__ k_cache_base,
    half* __restrict__ v_cache_base, int block_stride, int row_elems, int block_size, int n_tokens,
    int max_blocks_per_seq, int n_sequences, int n_kv_heads, int head_dim, float theta, float inv_scaling,
    int rope_pairs,  // effective_rope_dim / 2
    bool neox, const float* __restrict__ longrope_inv_freqs) {
    int token_idx = blockIdx.x;
    if (token_idx >= n_tokens)
        return;

    int pos = positions[token_idx];
    int slot_in_block;
    int block_id = kv_resolve_slot(block_tables, pos, block_size, token_idx, max_blocks_per_seq, n_sequences,
                                   slot_in_block);
    // Defense-in-depth (F-A12): a negative block_id — a freed StreamingLLM -1
    // sentinel, or a future block-table bug — would index the KV pool OOB.
    // block_id is uniform across the block (derived from blockIdx.x), so this
    // skips the whole write without divergence. Never fires today (host-side
    // admission guarantees every written position has a real block).
    if (block_id < 0)
        return;

    if (blockIdx.y == 0) {
        // K path: apply RoPE during write
        const half* k_src = k_in + static_cast<int64_t>(token_idx) * row_elems;
        half* k_dst = k_cache_base + static_cast<int64_t>(block_id) * block_stride +
                      static_cast<int64_t>(slot_in_block) * row_elems;

        // Process RoPE pairs
        int total_pairs = n_kv_heads * rope_pairs;
        for (int p = threadIdx.x; p < total_pairs; p += blockDim.x) {
            int head = p / rope_pairs;
            int pair_idx = p % rope_pairs;
            int head_offset = head * head_dim;

            int idx0, idx1;
            if (neox) {
                idx0 = head_offset + pair_idx;
                idx1 = head_offset + pair_idx + rope_pairs;
            } else {
                idx0 = head_offset + 2 * pair_idx;
                idx1 = head_offset + 2 * pair_idx + 1;
            }

            float cos_val, sin_val;
            rope_linear_cos_sin(pos, pair_idx, theta, inv_scaling, rope_pairs, longrope_inv_freqs, cos_val,
                                sin_val);

            float k0 = __half2float(k_src[idx0]);
            float k1 = __half2float(k_src[idx1]);
            k_dst[idx0] = __float2half(k0 * cos_val - k1 * sin_val);
            k_dst[idx1] = __float2half(k0 * sin_val + k1 * cos_val);
        }

        // Copy non-rotated dimensions (partial RoPE: rope_dim < head_dim)
        int effective_rope_dim = rope_pairs * 2;
        if (effective_rope_dim < head_dim) {
            for (int h = 0; h < n_kv_heads; h++) {
                int base = h * head_dim;
                for (int d = effective_rope_dim + threadIdx.x; d < head_dim; d += blockDim.x) {
                    k_dst[base + d] = k_src[base + d];
                }
            }
        }
    } else {
        // V path: vectorized 128-bit copy (no RoPE)
        const half* v_src = v_in + static_cast<int64_t>(token_idx) * row_elems;
        half* v_dst = v_cache_base + static_cast<int64_t>(block_id) * block_stride +
                      static_cast<int64_t>(slot_in_block) * row_elems;
        const int vec_elems = row_elems / 8;
        const float4* vs4 = reinterpret_cast<const float4*>(v_src);
        float4* vd4 = reinterpret_cast<float4*>(v_dst);
        for (int i = threadIdx.x; i < vec_elems; i += blockDim.x) {
            vd4[i] = vs4[i];
        }
    }
}

// Fused K+V FP8 write: combines K and V quantize+write into one kernel launch.
// blockIdx.x = token index, blockIdx.y = 0 (K) or 1 (V).
__global__ __launch_bounds__(256) void write_kv_cache_fp8_fused_kernel(
    const half* __restrict__ k_in, const half* __restrict__ v_in, const int* __restrict__ positions,
    const int* __restrict__ block_tables, __nv_fp8_e4m3* __restrict__ k_cache_base,
    __nv_fp8_e4m3* __restrict__ v_cache_base, float inv_scale, int block_stride, int row_elems,
    int block_size, int n_tokens, int max_blocks_per_seq, int n_sequences) {
    int token_idx = blockIdx.x;
    if (token_idx >= n_tokens)
        return;

    int pos = positions[token_idx];
    int slot_in_block;
    int block_id = kv_resolve_slot(block_tables, pos, block_size, token_idx, max_blocks_per_seq, n_sequences,
                                   slot_in_block);
    // Defense-in-depth (F-A12): a negative block_id — a freed StreamingLLM -1
    // sentinel, or a future block-table bug — would index the KV pool OOB.
    // block_id is uniform across the block (derived from blockIdx.x), so this
    // skips the whole write without divergence. Never fires today (host-side
    // admission guarantees every written position has a real block).
    if (block_id < 0)
        return;

    const half* src;
    __nv_fp8_e4m3* dst;
    if (blockIdx.y == 0) {
        src = k_in + static_cast<int64_t>(token_idx) * row_elems;
        dst = k_cache_base + static_cast<int64_t>(block_id) * block_stride +
              static_cast<int64_t>(slot_in_block) * row_elems;
    } else {
        src = v_in + static_cast<int64_t>(token_idx) * row_elems;
        dst = v_cache_base + static_cast<int64_t>(block_id) * block_stride +
              static_cast<int64_t>(slot_in_block) * row_elems;
    }

    // Packed PTX cvt: 2 paired conversions per 4 elements (half→e4m3x2).
    const half inv_scale_h = __float2half(inv_scale);
    const half2 inv_scale_h2 = make_half2(inv_scale_h, inv_scale_h);
    const int vec_elems = row_elems / 4;
    const half2* src2 = reinterpret_cast<const half2*>(src);
    uint32_t* dst4 = reinterpret_cast<uint32_t*>(dst);
    for (int i = threadIdx.x; i < vec_elems; i += blockDim.x) {
        half2 lo = __hmul2(src2[2 * i], inv_scale_h2);
        half2 hi = __hmul2(src2[2 * i + 1], inv_scale_h2);
        uint16_t e4m3_lo = cvt_f16x2_to_e4m3x2(*reinterpret_cast<uint32_t*>(&lo));
        uint16_t e4m3_hi = cvt_f16x2_to_e4m3x2(*reinterpret_cast<uint32_t*>(&hi));
        dst4[i] = static_cast<uint32_t>(e4m3_lo) | (static_cast<uint32_t>(e4m3_hi) << 16);
    }
    // Scalar tail for non-aligned remainder
    for (int i = vec_elems * 4 + threadIdx.x; i < row_elems; i += blockDim.x) {
        dst[i] = __nv_fp8_e4m3(__half2float(src[i]) * inv_scale);
    }
}

// Q-only RoPE for decode (n=1): applies RoPE to Q in-place.
// Grid: (1, n_heads), Block: rope_pairs.
__global__ __launch_bounds__(256) void rope_q_only_fp16_kernel(half* __restrict__ Q,  // [n_heads * head_dim]
                                                               const int* __restrict__ positions, int n_heads,
                                                               int head_dim, float theta, float inv_scaling,
                                                               int rope_pairs, bool neox,
                                                               const float* __restrict__ longrope_inv_freqs) {
    int head_idx = blockIdx.y;
    int pair_idx = threadIdx.x;
    if (head_idx >= n_heads || pair_idx >= rope_pairs)
        return;

    int pos = positions[0];  // decode: single token

    float cos_val, sin_val;
    rope_linear_cos_sin(pos, pair_idx, theta, inv_scaling, rope_pairs, longrope_inv_freqs, cos_val, sin_val);

    int64_t base = static_cast<int64_t>(head_idx) * head_dim;
    int idx0 = neox ? pair_idx : (2 * pair_idx);
    int idx1 = neox ? (pair_idx + rope_pairs) : (2 * pair_idx + 1);

    float q0 = __half2float(Q[base + idx0]);
    float q1 = __half2float(Q[base + idx1]);
    Q[base + idx0] = __float2half(q0 * cos_val - q1 * sin_val);
    Q[base + idx1] = __float2half(q0 * sin_val + q1 * cos_val);
}

}  // namespace imp
