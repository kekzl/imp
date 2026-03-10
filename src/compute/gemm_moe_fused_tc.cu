#include "compute/gemm_moe_fused_tc.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

namespace imp {

using namespace nvcuda;

// Tile dimensions
constexpr int TC_TILE_M = 32;    // tokens per CTA per tile
constexpr int TC_TILE_N = 64;    // output rows per CTA
constexpr int TC_K_TILE = 256;   // matches Q6_K block size
constexpr int TC_PAD    = 8;     // smem padding to reduce bank conflicts
constexpr int TC_STRIDE = TC_K_TILE + TC_PAD; // 264

constexpr int TC_BLOCK = 256;    // 8 warps

// Dynamic shared memory layout:
//   B_smem:       [TC_TILE_N × TC_STRIDE] half  = 33792 bytes
//   A_smem:       [TC_TILE_M × TC_STRIDE] half  = 16896 bytes
//   tile_prefix:  [n_experts+1] int32            ≤ 528 bytes (up to 132 experts)
//   Total:        ~51216 bytes
constexpr int B_SIZE = TC_TILE_N * TC_STRIDE * sizeof(half); // 33792
constexpr int A_SIZE = TC_TILE_M * TC_STRIDE * sizeof(half); // 16896
constexpr int PREFIX_SIZE = 132 * sizeof(int32_t);           // 528
constexpr int SMEM_TOTAL  = B_SIZE + A_SIZE + PREFIX_SIZE;   // 51216

// ---------------------------------------------------------------------------
// Kernel: persistent work-queue fused Q6_K dequant + WMMA GEMM for MoE prefill
//
// Each CTA atomically grabs work tiles from a global counter.  A tile is
// (expert_id, m_base, n_base) representing one TC_TILE_M × TC_TILE_N output
// block.  This eliminates M-loop imbalance: all CTAs do exactly 1 tile per
// iteration, and heavy experts are spread across many CTAs.
//
// Tile mapping:  flat_idx → (n_tile, m_tile_flat) → binary-search expert_id
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(TC_BLOCK)
gemm_q6k_fused_moe_prefill_tc_kernel(
    const uint8_t* __restrict__ packed_weights,
    const half*    __restrict__ activations,
    half*          __restrict__ output,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ sorted_token_ids,
    int N, int K,
    size_t expert_stride_bytes,
    int n_experts,
    int* __restrict__ d_tile_counter)
{
    extern __shared__ char dyn_smem[];
    half* B_smem  = reinterpret_cast<half*>(dyn_smem);
    half* AC_smem = reinterpret_cast<half*>(dyn_smem + B_SIZE);
    int32_t* tile_prefix = reinterpret_cast<int32_t*>(dyn_smem + B_SIZE + A_SIZE);

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane    = tid % 32;
    const int warp_m  = warp_id / 4;  // 0 or 1
    const int warp_n  = warp_id % 4;  // 0..3

    const int blocks_per_row = K / 256;
    const int blocks_x = (N + TC_TILE_N - 1) / TC_TILE_N;
    const size_t row_bytes = static_cast<size_t>(blocks_per_row) * 210;

    // --- Build tile prefix sum in shared memory ---
    // tile_prefix[i] = cumulative M-tiles before expert i
    if (tid < n_experts) {
        int M_i = offsets[tid + 1] - offsets[tid];
        tile_prefix[tid] = (M_i + TC_TILE_M - 1) / TC_TILE_M;
    }
    __syncthreads();

    __shared__ int total_tiles_s;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < n_experts; i++) {
            int c = tile_prefix[i];
            tile_prefix[i] = sum;
            sum += c;
        }
        tile_prefix[n_experts] = sum;
        total_tiles_s = sum * blocks_x;
    }
    __syncthreads();

    const int total_tiles = total_tiles_s;

    // Pre-compute Q6_K indexing constants for this lane (8 elements per lane)
    const int base_elem  = lane * 8;
    const int group      = base_elem >> 7;
    const int within     = base_elem & 127;
    const int quad       = within >> 5;
    const int l_base     = within & 31;
    const int ql_off     = (group << 6) + ((quad & 1) << 5) + l_base;
    const int qh_off     = 128 + (group << 5) + l_base;
    const int low4_shift = (quad >= 2) ? 4 : 0;  // branchless: replaces is_high ternary
    const int qh_shift   = quad * 2;
    const int scale_idx  = base_elem >> 4;

    // --- Persistent work loop ---
    __shared__ int s_tile_idx;

    while (true) {
        if (tid == 0) s_tile_idx = atomicAdd(d_tile_counter, 1);
        __syncthreads();

        if (s_tile_idx >= total_tiles) return;

        // Decode tile → (expert_id, m_base, n_base)
        const int flat_idx    = s_tile_idx;
        const int n_tile      = flat_idx % blocks_x;
        const int m_tile_flat = flat_idx / blocks_x;
        const int n_base      = n_tile * TC_TILE_N;

        // Binary search for expert (7 iterations for 128 experts)
        int lo = 0, hi = n_experts;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (tile_prefix[mid + 1] <= m_tile_flat) lo = mid + 1;
            else hi = mid;
        }
        const int expert_id    = lo;
        const int m_tile_local = m_tile_flat - tile_prefix[expert_id];
        const int m_base       = m_tile_local * TC_TILE_M;
        const int start        = offsets[expert_id];
        const int M            = offsets[expert_id + 1] - start;
        const int M_cur        = min(TC_TILE_M, M - m_base);

        const uint8_t* W = packed_weights
                         + static_cast<size_t>(expert_id) * expert_stride_bytes;

        // --- GEMM for single M-tile ---
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        for (int k_block = 0; k_block < blocks_per_row; k_block++) {

            // --- Load B: warp-cooperative Q6_K dequant → B_smem ---
            #pragma unroll
            for (int iter = 0; iter < TC_TILE_N / 8; iter++) {
                const int row      = warp_id + iter * 8;
                const int n_global = n_base + row;

                half w_vals[8];

                if (n_global < N) {
                    const uint8_t* bp = W + static_cast<size_t>(n_global) * row_bytes
                                          + static_cast<size_t>(k_block) * 210;

                    float d_w    = __half2float(*reinterpret_cast<const half*>(bp + 208));
                    float sc     = static_cast<float>(
                                       reinterpret_cast<const int8_t*>(bp + 192)[scale_idx]);
                    float w_scale = d_w * sc;

                    uint64_t ql8, qh8;
                    memcpy(&ql8, bp + ql_off, 8);
                    memcpy(&qh8, bp + qh_off, 8);

                    // Dequant 8 elements (branchless low4 via pre-computed shift)
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        uint32_t ql_byte = (static_cast<uint32_t>(ql8 >> (i * 8))) & 0xFFu;
                        uint32_t qh_byte = (static_cast<uint32_t>(qh8 >> (i * 8))) & 0xFFu;
                        uint32_t low4  = (ql_byte >> low4_shift) & 0xFu;
                        uint32_t high2 = (qh_byte >> qh_shift) & 0x3u;
                        int q = static_cast<int>((high2 << 4) | low4) - 32;
                        w_vals[i] = __float2half(w_scale * static_cast<float>(q));
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < 8; i++) w_vals[i] = __float2half(0.0f);
                }

                *reinterpret_cast<uint4*>(&B_smem[row * TC_STRIDE + base_elem]) =
                    *reinterpret_cast<uint4*>(w_vals);
            }

            // --- Load A: activations → AC_smem ---
            #pragma unroll 4
            for (int i = 0; i < (TC_TILE_M * TC_K_TILE / 8) / TC_BLOCK; i++) {
                const int load_idx = tid + i * TC_BLOCK;
                const int flat = load_idx * 8;
                const int row  = flat / TC_K_TILE;
                const int col  = flat % TC_K_TILE;

                if (row < M_cur) {
                    const int expanded_idx = start + m_base + row;
                    const int64_t token = sorted_token_ids
                        ? static_cast<int64_t>(sorted_token_ids[expanded_idx])
                        : static_cast<int64_t>(expanded_idx);
                    *reinterpret_cast<uint4*>(&AC_smem[row * TC_STRIDE + col]) =
                        *reinterpret_cast<const uint4*>(
                            &activations[token * K + k_block * TC_K_TILE + col]);
                } else {
                    *reinterpret_cast<uint4*>(&AC_smem[row * TC_STRIDE + col]) =
                        make_uint4(0, 0, 0, 0);
                }
            }

            __syncthreads();

            // --- WMMA: 16 steps of 16×16×16 matmul ---
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;

            #pragma unroll
            for (int kk = 0; kk < TC_K_TILE / 16; kk++) {
                wmma::load_matrix_sync(a_frag,
                    &AC_smem[warp_m * 16 * TC_STRIDE + kk * 16], TC_STRIDE);
                wmma::load_matrix_sync(b_frag,
                    &B_smem[warp_n * 16 * TC_STRIDE + kk * 16], TC_STRIDE);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

            __syncthreads();
        }

        // --- Store: FP32 accumulators → FP16 output ---
        float* C_smem_f32 = reinterpret_cast<float*>(AC_smem);

        wmma::store_matrix_sync(
            &C_smem_f32[(warp_m * 16) * TC_TILE_N + (warp_n * 16)],
            c_frag, TC_TILE_N, wmma::mem_row_major);

        __syncthreads();

        for (int flat = tid; flat < TC_TILE_M * TC_TILE_N; flat += TC_BLOCK) {
            const int m        = flat / TC_TILE_N;
            const int n        = flat % TC_TILE_N;
            const int n_global = n_base + n;

            if (m < M_cur && n_global < N) {
                const int64_t token = start + m_base + m;
                output[token * N + n_global] =
                    __float2half(C_smem_f32[m * TC_TILE_N + n]);
            }
        }

        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
void gemm_q6k_fused_moe_prefill_tc(const void* packed_weights,
                                    const void* activations,
                                    void* output,
                                    const int32_t* d_offsets,
                                    int N, int K,
                                    size_t expert_stride_bytes,
                                    int n_experts,
                                    cudaStream_t stream,
                                    const int32_t* sorted_token_ids)
{
    if (n_experts == 0 || N == 0) return;

    static bool configured = false;
    static int grid_size = 0;
    static int* d_tile_counter = nullptr;

    if (!configured) {
        cudaFuncSetAttribute(gemm_q6k_fused_moe_prefill_tc_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             SMEM_TOTAL);

        // Query optimal grid size: CTAs/SM × num_SMs
        int max_blocks_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm,
            gemm_q6k_fused_moe_prefill_tc_kernel,
            TC_BLOCK,
            SMEM_TOTAL);
        int num_sms = 0;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
        grid_size = num_sms * max(max_blocks_per_sm, 1);

        cudaMalloc(&d_tile_counter, sizeof(int));
        configured = true;
    }

    cudaMemsetAsync(d_tile_counter, 0, sizeof(int), stream);

    gemm_q6k_fused_moe_prefill_tc_kernel<<<grid_size, TC_BLOCK, SMEM_TOTAL, stream>>>(
        static_cast<const uint8_t*>(packed_weights),
        static_cast<const half*>(activations),
        static_cast<half*>(output),
        d_offsets,
        sorted_token_ids,
        N, K,
        expert_stride_bytes,
        n_experts,
        d_tile_counter);
}

} // namespace imp
