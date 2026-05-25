// Fused Q4_K × Q8_1 dp4a GEMM kernel for MoE expert prefill.
// Reads Q4_K weights from DRAM, Q8_1 activations from shared memory.
// Eliminates the FP16 dequant intermediate that caused 8.3× bandwidth overhead.

#include "compute/gemm_q4k.h"
#include "compute/gemm.h"

#include <cuda_fp16.h>
#include <cstdio>

namespace imp {

static constexpr int Q4K_BLOCK_BYTES = 144;
static constexpr int Q4K_BLOCK_ELEMS = 256;

static constexpr int Q4K_WARPS_PER_CTA = 4;
static constexpr int Q4K_BLOCK_SIZE = Q4K_WARPS_PER_CTA * 32;
static constexpr int Q4K_TILE_M = 32;

// Decode Q4_K sub-block scales and mins from the packed 12-byte header.
// Q4_K packs 8 sub-blocks: 6-bit scales and 4-bit mins into 12 bytes.
// Returns sc[8] (scales) and mn[8] (mins) as uint8.
__device__ __forceinline__ void decode_q4k_scales(const uint8_t* scales_12,
                                                   uint8_t sc[8], uint8_t mn[8]) {
    const uint16_t* s16 = reinterpret_cast<const uint16_t*>(scales_12);
    // Sub-blocks 0-3: low 6 bits of scales[0..3]
    sc[0] = scales_12[0] & 0x3F;
    sc[1] = scales_12[1] & 0x3F;
    sc[2] = scales_12[2] & 0x3F;
    sc[3] = scales_12[3] & 0x3F;
    // Sub-blocks 4-7: low 6 bits of scales[4..7], but packed differently
    sc[4] = (scales_12[4] & 0x0F) | ((scales_12[8] & 0x03) << 4);
    sc[5] = (scales_12[5] & 0x0F) | (((scales_12[8] >> 2) & 0x03) << 4);
    sc[6] = (scales_12[6] & 0x0F) | (((scales_12[8] >> 4) & 0x03) << 4);
    sc[7] = (scales_12[7] & 0x0F) | (((scales_12[8] >> 6) & 0x03) << 4);
    // Mins: high 6 bits
    mn[0] = scales_12[0] >> 6 | ((scales_12[4] >> 4) & 0x03) << 2 | ((scales_12[10] & 0x03) << 4);
    mn[1] = scales_12[1] >> 6 | ((scales_12[5] >> 4) & 0x03) << 2 | (((scales_12[10] >> 2) & 0x03) << 4);
    mn[2] = scales_12[2] >> 6 | ((scales_12[6] >> 4) & 0x03) << 2 | (((scales_12[10] >> 4) & 0x03) << 4);
    mn[3] = scales_12[3] >> 6 | ((scales_12[7] >> 4) & 0x03) << 2 | (((scales_12[10] >> 6) & 0x03) << 4);
    mn[4] = (scales_12[9] & 0x0F) | ((scales_12[11] & 0x03) << 4);
    mn[5] = (scales_12[9] >> 4)    | (((scales_12[11] >> 2) & 0x03) << 4);
    mn[6] = (scales_12[4 + 6] & 0x0F) | (((scales_12[11] >> 4) & 0x03) << 4);
    mn[7] = (scales_12[4 + 6] >> 4)    | (((scales_12[11] >> 6) & 0x03) << 4);
}

__global__ void __launch_bounds__(128, 2) gemm_q4k_moe_fused_kernel(
    const uint8_t* __restrict__ packed_weight, const block_q8_1* __restrict__ q8_base,
    const float* __restrict__ d8_base, const half* __restrict__ s8_base,
    half* __restrict__ c_base, const int32_t* __restrict__ offsets,
    int K, int N, size_t weight_stride, int q8_per_row) {

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tid = threadIdx.x;
    const int expert = blockIdx.y;

    const int m_start = offsets[expert];
    const int M_e = offsets[expert + 1] - m_start;
    if (M_e <= 0)
        return;

    const int n_col = blockIdx.x * Q4K_WARPS_PER_CTA + warp_id;
    if (n_col >= N)
        return;

    const int blocks_per_row = K / Q4K_BLOCK_ELEMS;
    const size_t row_bytes = static_cast<size_t>(blocks_per_row) * Q4K_BLOCK_BYTES;
    const uint8_t* w_row = packed_weight + static_cast<size_t>(expert) * weight_stride +
                           static_cast<size_t>(n_col) * row_bytes;

    extern __shared__ char smem_raw[];
    int8_t* smem_qs = reinterpret_cast<int8_t*>(smem_raw);
    float* smem_d8 = reinterpret_cast<float*>(smem_raw + Q4K_TILE_M * q8_per_row * 32);
    half* smem_s8 = reinterpret_cast<half*>(smem_raw + Q4K_TILE_M * q8_per_row * 32 +
                                             Q4K_TILE_M * q8_per_row * sizeof(float));

    for (int m_base = 0; m_base < M_e; m_base += Q4K_TILE_M) {
        const int m_count = min(Q4K_TILE_M, M_e - m_base);

        // Phase 1: Load Q8_1 data into shared memory (all threads cooperate)
        {
            const int total_items = m_count * q8_per_row;
            for (int i = tid; i < total_items; i += Q4K_BLOCK_SIZE) {
                const int mi = i / q8_per_row;
                const int qi = i % q8_per_row;
                const int tok = m_start + m_base + mi;

                const block_q8_1& src = q8_base[tok * q8_per_row + qi];
                int4* dst_qs = reinterpret_cast<int4*>(smem_qs + (mi * q8_per_row + qi) * 32);
                int4 tmp0, tmp1;
                memcpy(&tmp0, src.qs, 16);
                memcpy(&tmp1, src.qs + 16, 16);
                dst_qs[0] = tmp0;
                dst_qs[1] = tmp1;

                smem_d8[mi * q8_per_row + qi] = __half2float(src.d);
                smem_s8[mi * q8_per_row + qi] = src.s;
            }
        }
        __syncthreads();

        // Phase 2: dp4a accumulation
        float acc[Q4K_TILE_M];
        for (int i = 0; i < Q4K_TILE_M; i++)
            acc[i] = 0.0f;

        // K-loop: iterate over Q4_K blocks (each block = 256 elements = 8 Q8_1 blocks)
        for (int blk = lane; blk < blocks_per_row; blk += 32) {
            const uint8_t* bp = w_row + static_cast<size_t>(blk) * Q4K_BLOCK_BYTES;

            const float d = __half2float(*reinterpret_cast<const half*>(bp));
            const float dmin = __half2float(*reinterpret_cast<const half*>(bp + 2));

            uint8_t sc[8], mn[8];
            decode_q4k_scales(bp + 4, sc, mn);

            // Process 8 sub-blocks (each = 32 elements = 1 Q8_1 block)
            for (int sb = 0; sb < 8; sb++) {
                const int q8_idx = blk * 8 + sb;
                if (q8_idx >= q8_per_row) break;

                // Load 16 nibble bytes (= 32 elements) from Q4_K
                const uint8_t* qs_src = bp + 16 + sb * 16;
                uint32_t q4_packed[4];
                memcpy(q4_packed, qs_src, 16);

                const float d_sc = d * static_cast<float>(sc[sb]);
                const float dmin_mn = dmin * static_cast<float>(mn[sb]);

                for (int mi = 0; mi < m_count; mi++) {
                    const int8_t* act_qs = smem_qs + (mi * q8_per_row + q8_idx) * 32;
                    int xqs[4];
                    memcpy(xqs, act_qs, 16);
                    int xqs2[4];
                    memcpy(xqs2, act_qs + 16, 16);

                    const float dq = smem_d8[mi * q8_per_row + q8_idx];
                    const float sq = __half2float(smem_s8[mi * q8_per_row + q8_idx]);

                    int32_t sumi = 0;
                    int32_t sumi_ones = 0;
#pragma unroll
                    for (int d4 = 0; d4 < 4; d4++) {
                        const int lo = q4_packed[d4] & 0x0F0F0F0F;
                        const int hi = (q4_packed[d4] >> 4) & 0x0F0F0F0F;
                        sumi = __dp4a(lo, xqs[d4], sumi);
                        sumi = __dp4a(hi, xqs2[d4], sumi);
                    }

                    acc[mi] += dq * d_sc * static_cast<float>(sumi) - sq * dmin_mn;
                }
            }
        }

        // Phase 3: Warp shuffle reduction + output
        for (int mi = 0; mi < m_count; mi++) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                acc[mi] += __shfl_down_sync(0xFFFFFFFF, acc[mi], off);
        }

        if (lane == 0) {
            for (int mi = 0; mi < m_count; mi++) {
                const int tok = m_start + m_base + mi;
                c_base[static_cast<size_t>(tok) * N + n_col] = __float2half(acc[mi]);
            }
        }

        __syncthreads();
    }
}

void gemm_q4k_moe_fused(const void* packed_weight, const block_q8_1* q8_base, const float* d8_base,
                         const half* s8_base, void* c_base, const int32_t* offsets, int K, int N,
                         int n_experts, size_t weight_stride, cudaStream_t stream) {
    if (n_experts <= 0 || K <= 0 || N <= 0)
        return;

    const int q8_per_row = K / 32;
    const int n_col_blocks = (N + Q4K_WARPS_PER_CTA - 1) / Q4K_WARPS_PER_CTA;
    const dim3 grid(n_col_blocks, n_experts);
    const dim3 block(Q4K_BLOCK_SIZE);

    const size_t smem_qs_bytes = static_cast<size_t>(Q4K_TILE_M) * q8_per_row * 32;
    const size_t smem_d8_bytes = static_cast<size_t>(Q4K_TILE_M) * q8_per_row * sizeof(float);
    const size_t smem_s8_bytes = static_cast<size_t>(Q4K_TILE_M) * q8_per_row * sizeof(half);
    const size_t smem_bytes = smem_qs_bytes + smem_d8_bytes + smem_s8_bytes;

    static bool smem_configured = false;
    if (!smem_configured && smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(gemm_q4k_moe_fused_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_bytes));
        smem_configured = true;
    }

    gemm_q4k_moe_fused_kernel<<<grid, block, smem_bytes, stream>>>(
        static_cast<const uint8_t*>(packed_weight), q8_base, d8_base, s8_base,
        static_cast<half*>(c_base), offsets, K, N, weight_stride, q8_per_row);
}

}  // namespace imp
