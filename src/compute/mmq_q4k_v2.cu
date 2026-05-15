// mmq_q4k v2 Phase 1a — precompute per-sub-block affine scales for Q4_K weights.
//
// See mmq_q4k_v2.h for the rationale. This file implements only the
// preprocessing kernel; the HMMA matmul kernel itself comes in Phase 2.

#include "mmq_q4k_v2.h"

#include <cstdint>
#include <cuda_fp16.h>

namespace imp {

namespace mmq_q4k_v2_detail {

// Q4_K super-block — fixed GGUF layout, duplicated from mmq_q4k.cu to keep
// this translation unit self-contained.
struct block_q4_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};
static_assert(sizeof(block_q4_K) == 144, "block_q4_K must be 144 bytes");

// Unpack the 8 (scale, min) 6-bit pairs from a Q4_K super-block's scales[12].
// Returns sc[0..7] and m[0..7] as uint8 (0..63).
//
// The packing scheme matches ggml's vec_dot_q4_K_q8_1: aux[0] / aux[1] pairs
// indexed by bo_step ∈ [0, 4) cover sub-blocks {2*bo_step, 2*bo_step+1}.
__device__ __forceinline__ void unpack_q4_K_scales_mins(
    const uint8_t* __restrict__ scales12, uint8_t sc_out[8], uint8_t m_out[8]) {
    const uint16_t* scales = reinterpret_cast<const uint16_t*>(scales12);

#pragma unroll
    for (int bo_step = 0; bo_step < 4; ++bo_step) {
        const int bq8_offset = 2 * bo_step;
        uint16_t aux[2];
        const int j = bo_step;
        if (j < 2) {
            aux[0] = scales[j + 0] & 0x3f3f;
            aux[1] = scales[j + 2] & 0x3f3f;
        } else {
            aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
            aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
        }
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
        const uint8_t* m = sc + 2;
        sc_out[bq8_offset + 0] = sc[0];
        sc_out[bq8_offset + 1] = sc[1];
        m_out [bq8_offset + 0] = m[0];
        m_out [bq8_offset + 1] = m[1];
    }
}

// One thread per Q4_K super-block: read 144 bytes, write 8 eff_scale + 8 eff_min.
__global__ void q4k_precompute_eff_scales_kernel(
    const block_q4_K* __restrict__ W, half* __restrict__ eff_scale,
    half* __restrict__ eff_min, int total_super_blocks, int K_blocks) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_super_blocks) return;

    const int n = tid / K_blocks;
    const int kbx = tid % K_blocks;

    const block_q4_K bq = W[n * K_blocks + kbx];

    uint8_t sc[8], m[8];
    unpack_q4_K_scales_mins(bq.scales, sc, m);

    const float d = __half2float(bq.d);
    const float dmin = __half2float(bq.dmin);

    half* es_row = &eff_scale[n * (K_blocks * 8) + kbx * 8];
    half* em_row = &eff_min  [n * (K_blocks * 8) + kbx * 8];

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        es_row[i] = __float2half(d    * static_cast<float>(sc[i]));
        em_row[i] = __float2half(dmin * static_cast<float>(m [i]));
    }
}

}  // namespace mmq_q4k_v2_detail

void q4k_precompute_eff_scales(const void* W, half* eff_scale_out,
                               half* eff_min_out, int N, int K,
                               cudaStream_t stream) {
    if (K % 256 != 0) return;
    using namespace mmq_q4k_v2_detail;

    const int K_blocks = K / 256;
    const int total = N * K_blocks;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    q4k_precompute_eff_scales_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const block_q4_K*>(W), eff_scale_out, eff_min_out, total,
        K_blocks);
}

namespace mmq_q4k_v2_detail {

// One thread per sub-block. Reads 32 nibbles from canonical qs[] (interleaved
// with the paired sub-block via low/high nibble) and writes 16 bytes of
// K-major packed nibbles.
__global__ void q4k_permute_kernel(const block_q4_K* __restrict__ W,
                                   uint8_t* __restrict__ eff_q4_out,
                                   int total_sub_blocks, int K_blocks) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_sub_blocks) return;

    // K_blocks * 8 sub-blocks per row of N.
    const int subs_per_row = K_blocks * 8;
    const int n = tid / subs_per_row;
    const int k_sub_in_row = tid % subs_per_row;
    const int k_super = k_sub_in_row / 8;
    const int s = k_sub_in_row % 8;

    const block_q4_K* bq = &W[n * K_blocks + k_super];
    const uint8_t* qs = bq->qs;

    // Canonical layout: bytes [(s/2)*32 .. (s/2)*32 + 32) contain the 32 nibbles
    // for sub-block s (low half if s is even, high half if s is odd) AND the 32
    // nibbles for its partner sub-block s^1 (in the other half).
    const int byte_base = (s >> 1) * 32;
    const bool use_high = (s & 1) != 0;

    uint8_t* out_row = &eff_q4_out[static_cast<size_t>(n) * subs_per_row * 16 +
                                   static_cast<size_t>(k_sub_in_row) * 16];

#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const uint8_t b1 = qs[byte_base + 2 * j + 0];
        const uint8_t b2 = qs[byte_base + 2 * j + 1];
        const uint8_t n1 = use_high ? ((b1 >> 4) & 0x0F) : (b1 & 0x0F);
        const uint8_t n2 = use_high ? ((b2 >> 4) & 0x0F) : (b2 & 0x0F);
        out_row[j] = static_cast<uint8_t>(n1 | (n2 << 4));
    }
}

}  // namespace mmq_q4k_v2_detail

void q4k_permute_to_v2_layout(const void* W, uint8_t* eff_q4_out, int N, int K,
                              cudaStream_t stream) {
    if (K % 256 != 0) return;
    using namespace mmq_q4k_v2_detail;

    const int K_blocks = K / 256;
    const int total_sub = N * K_blocks * 8;
    const int threads = 256;
    const int blocks = (total_sub + threads - 1) / threads;

    q4k_permute_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const block_q4_K*>(W), eff_q4_out, total_sub, K_blocks);
}

}  // namespace imp
