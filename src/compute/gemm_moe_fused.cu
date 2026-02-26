#include "compute/gemm_moe_fused.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

namespace imp {

// ---------------------------------------------------------------------------
// Fused Q6_K × FP16 GEMM kernel for MoE expert prefill projections.
//
// Eliminates the intermediate FP16 dequantization buffer by reading Q6_K
// weights directly and multiplying with FP16 input activations.
// Reduces DRAM traffic by ~5x vs dequant-then-GEMM.
//
// Grid:  (ceil(N / FUSED_WARPS), n_experts)
// Block: 256 threads (8 warps)
//
// Each warp computes one output row for one expert across all assigned tokens.
// All 32 lanes cooperatively dequant the SAME Q6_K block (lane L handles
// elements L*8..L*8+7), giving coalesced reads within 2 cache lines.
// FP16 activations naturally cache in L1 (all rows of same expert share them).
//
// M_TILE=8 keeps register pressure low for high occupancy (~40 warps/SM).
// Weight data re-read per M_TILE hits L2 cache (row data = 1.7 KB fits easily).
// ---------------------------------------------------------------------------

constexpr int FUSED_WARPS = 8;
constexpr int FUSED_BLOCK = FUSED_WARPS * 32;
constexpr int FUSED_M_TILE = 8;

__global__ void __launch_bounds__(FUSED_BLOCK)
gemm_q6k_fused_moe_prefill_kernel(
    const uint8_t* __restrict__ packed_weights,
    const half*    __restrict__ activations,
    half*          __restrict__ output,
    const int32_t* __restrict__ offsets,
    int N, int K,
    size_t expert_stride_bytes,
    int n_experts)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int row = blockIdx.x * FUSED_WARPS + warp_id;
    const int expert_id = blockIdx.y;

    if (row >= N || expert_id >= n_experts) return;

    const int start = offsets[expert_id];
    const int M = offsets[expert_id + 1] - start;
    if (M == 0) return;

    // Weight pointer for this expert's output row
    const int blocks_per_row = K / 256;
    const size_t row_bytes = static_cast<size_t>(blocks_per_row) * 210;
    const uint8_t* W_row = packed_weights
                          + static_cast<size_t>(expert_id) * expert_stride_bytes
                          + static_cast<size_t>(row) * row_bytes;

    // Each lane handles 8 elements per Q6_K block (lane L → elements L*8..L*8+7)
    const int base_elem = lane * 8;

    // Pre-compute Q6_K indexing constants (fixed for all K blocks)
    const int group  = base_elem >> 7;
    const int within = base_elem & 127;
    const int quad   = within >> 5;
    const int l_base = within & 31;

    const int ql_off = (group << 6) + ((quad & 1) << 5) + l_base;
    const int qh_off = 128 + (group << 5) + l_base;
    const int is_high  = (quad >= 2) ? 1 : 0;
    const int qh_shift = quad * 2;
    const int scale_idx = base_elem >> 4;
    const int k_lane_offset = base_elem;

    // Process tokens in tiles of M_TILE
    for (int m_base = 0; m_base < M; m_base += FUSED_M_TILE) {
        const int M_cur = min(FUSED_M_TILE, M - m_base);

        float acc[FUSED_M_TILE];
        #pragma unroll
        for (int i = 0; i < FUSED_M_TILE; i++) acc[i] = 0.0f;

        // Walk K dimension in Q6_K blocks of 256 elements
        for (int blk = 0; blk < blocks_per_row; blk++) {
            const uint8_t* bp = W_row + blk * 210;

            // Load block-level and per-group scales
            float d_w = __half2float(*reinterpret_cast<const half*>(bp + 208));
            float sc = static_cast<float>(reinterpret_cast<const int8_t*>(bp + 192)[scale_idx]);
            float w_scale = d_w * sc;

            // Load 8 ql bytes and 8 qh bytes
            uint64_t ql8, qh8;
            memcpy(&ql8, bp + ql_off, 8);
            memcpy(&qh8, bp + qh_off, 8);

            // Dequantize 8 weight elements
            float w[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                uint32_t ql_byte = (static_cast<uint32_t>(ql8 >> (i * 8))) & 0xFFu;
                uint32_t qh_byte = (static_cast<uint32_t>(qh8 >> (i * 8))) & 0xFFu;
                uint32_t low4 = is_high ? ((ql_byte >> 4) & 0xFu) : (ql_byte & 0xFu);
                uint32_t high2 = (qh_byte >> qh_shift) & 0x3u;
                int q = static_cast<int>((high2 << 4) | low4) - 32;
                w[i] = w_scale * static_cast<float>(q);
            }

            // Multiply with each token's activation values
            const int k_offset = blk * 256 + k_lane_offset;

            #pragma unroll
            for (int m = 0; m < FUSED_M_TILE; m++) {
                if (m >= M_cur) break;
                const int64_t token = start + m_base + m;
                const half* a_ptr = activations + token * K + k_offset;

                // Vectorized 128-bit load (16-byte aligned)
                uint4 a_vec = *reinterpret_cast<const uint4*>(a_ptr);
                const half* ah = reinterpret_cast<const half*>(&a_vec);

                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < 8; i++)
                    dot += w[i] * __half2float(ah[i]);

                acc[m] += dot;
            }
        }

        // Warp reduction: sum partial products across 32 lanes
        #pragma unroll
        for (int m = 0; m < FUSED_M_TILE; m++) {
            if (m >= M_cur) break;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                acc[m] += __shfl_down_sync(0xFFFFFFFF, acc[m], off);
        }

        // Lane 0 writes output
        if (lane == 0) {
            #pragma unroll
            for (int m = 0; m < FUSED_M_TILE; m++) {
                if (m >= M_cur) break;
                output[static_cast<int64_t>(start + m_base + m) * N + row] =
                    __float2half(acc[m]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

void gemm_q6k_fused_moe_prefill(const void* packed_weights,
                                const void* activations,
                                void* output,
                                const int32_t* d_offsets,
                                int N, int K,
                                size_t expert_stride_bytes,
                                int n_experts,
                                cudaStream_t stream)
{
    if (n_experts == 0) return;

    const int blocks_x = (N + FUSED_WARPS - 1) / FUSED_WARPS;
    dim3 grid(blocks_x, n_experts);
    dim3 block(FUSED_BLOCK);

    gemm_q6k_fused_moe_prefill_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(packed_weights),
        static_cast<const half*>(activations),
        static_cast<half*>(output),
        d_offsets,
        N, K,
        expert_stride_bytes,
        n_experts);
}

} // namespace imp
