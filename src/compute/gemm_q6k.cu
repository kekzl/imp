#include "compute/gemm_q6k.h"
#include "compute/gemm.h"  // block_q8_1, quantize_fp16_to_q8_1

#include <cuda_fp16.h>
#include <cstdio>

namespace imp {

// Q6_K block: 210 bytes per 256 elements
// Layout: ql[128] | qh[64] | scales[16] | d[2]
static constexpr int Q6K_BLOCK_BYTES = 210;
static constexpr int Q6K_BLOCK_ELEMS = 256;

// ---------------------------------------------------------------------------
// Fused Q6_K × Q8_1 dp4a GEMM kernel for MoE prefill — v3 (smem-tiled).
//
// Weight-stationary, register-based Q6K dequant, dp4a integer accumulation.
// Each warp handles 1 output column, all 32 lanes split K.
//
// Key optimization: Q8_1 activation data is loaded into shared memory once
// per CTA, then shared across all warps. This eliminates the L2 bandwidth
// amplification (768x replay factor) that killed v1/v2 performance.
//
// The WMMA approach failed because it put DEQUANTIZED weights into smem
// (expensive per-element FP16 dequant + syncthreads). Here, smem holds
// already-quantized Q8_1 data (just a memcpy, no dequant), and Q6K
// dequant happens in registers per-warp.
//
// Grid:  (ceil(N / WARPS_PER_CTA), n_experts)
// Block: WARPS_PER_CTA * 32 threads
//
// Shared memory: TILE_M × q8_per_row × sizeof(block_q8_1) +
//                TILE_M × q8_per_row × sizeof(float)
// For K=2048, TILE_M=32: 32 × 64 × (36 + 4) = 80 KB
// ---------------------------------------------------------------------------

static constexpr int FUSED_WARPS_PER_CTA = 4;
static constexpr int FUSED_BLOCK_SIZE = FUSED_WARPS_PER_CTA * 32;  // 128 threads
static constexpr int TILE_M = 32;  // tokens per shared memory tile

__global__ void __launch_bounds__(128, 2)
gemm_q6k_moe_fused_kernel(
    const uint8_t* __restrict__ packed_weight,
    const block_q8_1* __restrict__ q8_base,
    const float* __restrict__ d8_base,
    half* __restrict__ c_base,
    const int32_t* __restrict__ offsets,
    int K, int N,
    size_t weight_stride,
    int q8_per_row  // K / 32
) {
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int tid     = threadIdx.x;
    const int expert  = blockIdx.y;

    // Read this expert's token range from device memory
    const int m_start = offsets[expert];
    const int M_e = offsets[expert + 1] - m_start;
    if (M_e <= 0) return;

    // My output column
    const int n_col = blockIdx.x * FUSED_WARPS_PER_CTA + warp_id;
    if (n_col >= N) return;

    // Weight row pointer for this expert and output column
    const int blocks_per_row = K / Q6K_BLOCK_ELEMS;
    const size_t row_bytes = static_cast<size_t>(blocks_per_row) * Q6K_BLOCK_BYTES;
    const uint8_t* w_row = packed_weight + static_cast<size_t>(expert) * weight_stride
                         + static_cast<size_t>(n_col) * row_bytes;

    // Shared memory layout:
    //   smem_qs: [TILE_M][q8_per_row][32] int8_t  — Q8_1 quantized values
    //   smem_d8: [TILE_M][q8_per_row] float       — Q8_1 block scales
    // We store qs as flat int8_t arrays (not full block_q8_1) to save smem.
    extern __shared__ char smem_raw[];
    int8_t* smem_qs = reinterpret_cast<int8_t*>(smem_raw);
    // smem_qs: TILE_M * q8_per_row * 32 bytes
    float* smem_d8 = reinterpret_cast<float*>(smem_raw + TILE_M * q8_per_row * 32);
    // smem_d8: TILE_M * q8_per_row * 4 bytes

    // Process M_e tokens in tiles of TILE_M
    for (int m_base = 0; m_base < M_e; m_base += TILE_M) {
        const int m_count = min(TILE_M, M_e - m_base);

        // Phase 1: Cooperatively load Q8_1 data into shared memory.
        // All 128 threads participate. Total items = m_count * q8_per_row.
        {
            const int total_items = m_count * q8_per_row;
            for (int i = tid; i < total_items; i += FUSED_BLOCK_SIZE) {
                const int mi = i / q8_per_row;
                const int qi = i % q8_per_row;
                const int tok = m_start + m_base + mi;

                // Load Q8_1 block: copy qs[32] and d8 scale
                const block_q8_1& src = q8_base[tok * q8_per_row + qi];
                // Copy 32 bytes of qs data — use memcpy for source because
                // block_q8_1::qs is at offset 4 in a 36-byte struct (not 16-byte aligned).
                // Destination in smem IS 32-byte aligned so int4 store is safe.
                int4* dst_qs = reinterpret_cast<int4*>(smem_qs + (mi * q8_per_row + qi) * 32);
                int4 tmp0, tmp1;
                memcpy(&tmp0, src.qs, 16);
                memcpy(&tmp1, src.qs + 16, 16);
                dst_qs[0] = tmp0;
                dst_qs[1] = tmp1;

                smem_d8[mi * q8_per_row + qi] = d8_base[tok * q8_per_row + qi];
            }
        }
        __syncthreads();

        // Phase 2: Each warp computes its output column using smem Q8_1 data.
        float acc[TILE_M];
        for (int i = 0; i < TILE_M; i++) acc[i] = 0.0f;

        // K-loop: weight loaded once per pass
        for (int q8_idx = lane; q8_idx < q8_per_row; q8_idx += 32) {
            const int q6k_blk = q8_idx / 8;
            const int g = q8_idx % 8;

            // Pre-load Q6K weight data into registers
            const uint8_t* bp = w_row + static_cast<size_t>(q6k_blk) * Q6K_BLOCK_BYTES;
            const float d_w = __half2float(*reinterpret_cast<const half*>(bp + 208));

            const int ql_base = (g / 4) * 64 + (g % 2) * 32;
            const int is_high = ((g % 4) >= 2);
            const int qh_base_off = (g < 4) ? 0 : 32;
            const int qh_shift = (g % 4) * 2;

            const int8_t* sc_ptr = reinterpret_cast<const int8_t*>(bp + 192);
            const int8_t sc0 = sc_ptr[2 * g + 0];
            const int8_t sc1 = sc_ptr[2 * g + 1];

            // Pre-load Q6K ql and qh values for this group
            uint32_t ql_reg[8], qh_reg[8];
            #pragma unroll
            for (int sb = 0; sb < 2; sb++) {
                #pragma unroll
                for (int d4 = 0; d4 < 4; d4++) {
                    const int k = sb * 16 + d4 * 4;
                    memcpy(&ql_reg[sb * 4 + d4], bp + ql_base + k, 4);
                    memcpy(&qh_reg[sb * 4 + d4], bp + 128 + qh_base_off + k, 4);
                }
            }

            // M-loop: process each token using Q8_1 from shared memory
            for (int mi = 0; mi < m_count; mi++) {
                // Read Q8_1 from shared memory (no L2 traffic!)
                const int8_t* qs_ptr = smem_qs + (mi * q8_per_row + q8_idx) * 32;
                int xqs[8];
                const int4* qs_v = reinterpret_cast<const int4*>(qs_ptr);
                int4 v0 = qs_v[0];
                int4 v1 = qs_v[1];
                memcpy(&xqs[0], &v0, 16);
                memcpy(&xqs[4], &v1, 16);

                const float dq = smem_d8[mi * q8_per_row + q8_idx];

                // dp4a with pre-loaded Q6K weights — sub-block 0
                {
                    int32_t sumi = 0;
                    #pragma unroll
                    for (int d4 = 0; d4 < 4; d4++) {
                        const uint32_t lo4 = is_high ? ((ql_reg[d4] >> 4) & 0x0F0F0F0FU)
                                                     : (ql_reg[d4] & 0x0F0F0F0FU);
                        const uint32_t hi4 = ((qh_reg[d4] >> qh_shift) & 0x03030303U) << 4;
                        const int vi = __vsubss4(lo4 | hi4, 0x20202020U);
                        sumi = __dp4a(vi, xqs[d4], sumi);
                    }
                    acc[mi] += d_w * dq * static_cast<float>(sc0) * static_cast<float>(sumi);
                }

                // dp4a with pre-loaded Q6K weights — sub-block 1
                {
                    int32_t sumi = 0;
                    #pragma unroll
                    for (int d4 = 0; d4 < 4; d4++) {
                        const uint32_t lo4 = is_high ? ((ql_reg[4 + d4] >> 4) & 0x0F0F0F0FU)
                                                     : (ql_reg[4 + d4] & 0x0F0F0F0FU);
                        const uint32_t hi4 = ((qh_reg[4 + d4] >> qh_shift) & 0x03030303U) << 4;
                        const int vi = __vsubss4(lo4 | hi4, 0x20202020U);
                        sumi = __dp4a(vi, xqs[4 + d4], sumi);
                    }
                    acc[mi] += d_w * dq * static_cast<float>(sc1) * static_cast<float>(sumi);
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

        __syncthreads();  // protect smem for next TILE_M iteration
    }
}

// Host launcher
void gemm_q6k_moe_fused(
    const void* packed_weight,
    const block_q8_1* q8_base,
    const float* d8_base,
    void* c_base,
    const int32_t* offsets,
    int K, int N,
    int n_experts,
    size_t weight_stride,
    cudaStream_t stream)
{
    if (n_experts <= 0 || K <= 0 || N <= 0) return;

    const int q8_per_row = K / 32;
    const int n_col_blocks = (N + FUSED_WARPS_PER_CTA - 1) / FUSED_WARPS_PER_CTA;
    const dim3 grid(n_col_blocks, n_experts);
    const dim3 block(FUSED_BLOCK_SIZE);

    // Shared memory: Q8_1 qs data + d8 scales for TILE_M tokens
    const size_t smem_qs_bytes = static_cast<size_t>(TILE_M) * q8_per_row * 32;
    const size_t smem_d8_bytes = static_cast<size_t>(TILE_M) * q8_per_row * sizeof(float);
    const size_t smem_bytes = smem_qs_bytes + smem_d8_bytes;

    // Request extended shared memory if needed
    static bool smem_configured = false;
    if (!smem_configured && smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(gemm_q6k_moe_fused_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_bytes));
        smem_configured = true;
    }

    gemm_q6k_moe_fused_kernel<<<grid, block, smem_bytes, stream>>>(
        static_cast<const uint8_t*>(packed_weight),
        q8_base,
        d8_base,
        static_cast<half*>(c_base),
        offsets,
        K, N, weight_stride, q8_per_row);
}

} // namespace imp
