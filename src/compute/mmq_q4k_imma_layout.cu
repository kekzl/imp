// =============================================================================
// mmq_q4k_imma_layout.cu — Phase 2A reorder kernel
// =============================================================================
//
// Companion to design memo docs/plans/q4k_imma_design_2026_05_17.md §3 ("Q4_K →
// INT8 reordering") and findings memo 2026-05-18-q4k-imma-phase1-findings.md.
//
// Decodes one Q4_K super-block per CTA. For each sub-block j ∈ [0, 8):
//
//   sc[j]  — 6-bit sub-block scale  (from scales[12], unpacked via get_scale_min_k4)
//   m[j]   — 6-bit sub-block min    (same)
//   α[j]   = d_super * sc[j]
//   β[j]   = 8 * d_super * sc[j] - dmin_super * m[j]
//
// And for each element q in sub-block j:
//   q_sym  = q - 8         (∈ [-8, 7], fits cleanly in int8_t)
//   fp16   = α[j] * q_sym + β[j]
//
// The (q - 8) shift absorbs the unsigned-to-signed quantization into the GEMM
// epilogue. β couples to the activation row-sum via the standard
// (q_sym + 8) * a  =  q_sym * a + 8 * a  identity.
//
// Layout reference:
//   Q4_K block: [d:fp16, dmin:fp16, scales[12]:u6×16, qs[128]:nibbles].
//   Sub-block j ∈ [0,8) → 32 elements stored as:
//     j even (j=0,2,4,6): low nibbles of qs[(j/2)*32 + 0 .. +31]
//     j odd  (j=1,3,5,7): high nibbles of qs[(j/2)*32 + 0 .. +31]
//   (See ggml dequantize_row_q4_K — outer loop j∈{0,64,128,192} reads 32 bytes,
//    inner loop produces 32 low nibbles then 32 high nibbles.)

#include "compute/mmq_q4k_imma_layout.h"

#include <cuda_fp16.h>
#include <cstdint>

namespace imp {

namespace {

constexpr int kBlockBytes = 144;
constexpr int kQK4K = kQ4kSuperBlockSize;  // 256

// 6-bit scale + 6-bit min unpacker, matching ggml/llama.cpp get_scale_min_k4.
// q points at the 12-byte `scales` array; j ∈ [0, 8) selects the sub-block.
__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d_out,
                                                 uint8_t& m_out) {
    if (j < 4) {
        d_out = q[j] & 63u;
        m_out = q[j + 4] & 63u;
    } else {
        d_out = (q[j + 4] & 0xFu) | ((q[j - 4] >> 6) << 4);
        m_out = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

__global__ void mmq_q4k_imma_reorder_kernel(const uint8_t* __restrict__ q4k_blocks, int N, int K,
                                            int8_t* __restrict__ w_sym_s8,
                                            __half* __restrict__ eff_alpha,
                                            __half* __restrict__ eff_beta) {
    const int row = blockIdx.y;
    const int super_in_row = blockIdx.x;
    const int blocks_per_row = K / kQK4K;

    if (row >= N || super_in_row >= blocks_per_row) return;

    const int super_idx = row * blocks_per_row + super_in_row;
    const uint8_t* bp = q4k_blocks + static_cast<size_t>(super_idx) * kBlockBytes;

    // Read super-block header: d (FP16) + dmin (FP16).
    __half d_h, dmin_h;
    {
        uint16_t d_bits, dmin_bits;
        d_bits = static_cast<uint16_t>(bp[0]) | (static_cast<uint16_t>(bp[1]) << 8);
        dmin_bits = static_cast<uint16_t>(bp[2]) | (static_cast<uint16_t>(bp[3]) << 8);
        d_h = __ushort_as_half(d_bits);
        dmin_h = __ushort_as_half(dmin_bits);
    }
    const float d = __half2float(d_h);
    const float dmin = __half2float(dmin_h);

    const uint8_t* scales = bp + 4;
    const uint8_t* qs = bp + 4 + 12;  // 128 bytes of nibble-packed Q4 quants

    // Per-CTA shared scratch for α and β so all 8 sub-blocks are decoded once.
    __shared__ float s_alpha[kQ4kSubBlocksPerSuper];
    __shared__ float s_beta[kQ4kSubBlocksPerSuper];

    const int tid = threadIdx.x;
    if (tid < kQ4kSubBlocksPerSuper) {
        uint8_t sc_u, m_u;
        get_scale_min_k4(tid, scales, sc_u, m_u);
        const float sc_f = static_cast<float>(sc_u);
        const float m_f = static_cast<float>(m_u);
        s_alpha[tid] = d * sc_f;
        s_beta[tid] = 8.0f * d * sc_f - dmin * m_f;
    }
    __syncthreads();

    // Write α, β to output tensors. One thread per sub-block is enough.
    if (tid < kQ4kSubBlocksPerSuper) {
        const int alpha_idx =
            row * blocks_per_row * kQ4kSubBlocksPerSuper + super_in_row * kQ4kSubBlocksPerSuper + tid;
        eff_alpha[alpha_idx] = __float2half(s_alpha[tid]);
        eff_beta[alpha_idx] = __float2half(s_beta[tid]);
    }

    // Decode nibbles → symmetric s8. 256 elements per super-block ÷ 32 threads = 8 per thread.
    // The Q4_K layout interleaves sub-blocks as (j even = low nibbles, j odd = high nibbles)
    // over the 32-byte qs slabs of 64 elements each.
    //
    // For element k_in_super ∈ [0, 256):
    //   group   = k_in_super / 64       (0..3)
    //   in_grp  = k_in_super % 64       (0..63)
    //   is_high = in_grp >= 32          (1 if high nibble, 0 if low)
    //   byte_in_group = in_grp % 32     (0..31)
    //   byte_in_qs   = group * 32 + byte_in_group
    //   sub_block     = group * 2 + is_high
    const int8_t* w_row = reinterpret_cast<const int8_t*>(w_sym_s8);
    int8_t* w_super = const_cast<int8_t*>(w_row) +
                      static_cast<size_t>(row) * static_cast<size_t>(K) +
                      static_cast<size_t>(super_in_row) * kQK4K;
#pragma unroll
    for (int e = tid; e < kQK4K; e += 32) {
        const int group = e >> 6;            // /64
        const int in_grp = e & 63;           // %64
        const int is_high = (in_grp >> 5);   // /32 ∈ {0,1}
        const int byte_in_group = in_grp & 31;
        const int byte_in_qs = group * 32 + byte_in_group;
        const uint8_t packed = qs[byte_in_qs];
        const int nibble = is_high ? (packed >> 4) : (packed & 0xF);
        const int q_sym = nibble - 8;
        w_super[e] = static_cast<int8_t>(q_sym);
    }
}

}  // namespace

void mmq_q4k_imma_reorder(const void* q4k_blocks, int N, int K, int8_t* w_sym_s8,
                          __half* eff_alpha, __half* eff_beta, cudaStream_t stream) {
    if (K % kQ4kSuperBlockSize != 0) return;
    const int blocks_per_row = K / kQ4kSuperBlockSize;
    dim3 grid(blocks_per_row, N, 1);
    dim3 block(32, 1, 1);
    mmq_q4k_imma_reorder_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(q4k_blocks), N, K, w_sym_s8, eff_alpha, eff_beta);
}

}  // namespace imp
