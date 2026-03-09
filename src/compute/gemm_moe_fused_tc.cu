#include "compute/gemm_moe_fused_tc.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

namespace imp {

using namespace nvcuda;

// Tile dimensions
constexpr int TC_TILE_M = 32;    // tokens per CTA per M iteration
constexpr int TC_TILE_N = 64;    // output rows per CTA
constexpr int TC_K_TILE = 256;   // matches Q6_K block size
constexpr int TC_PAD    = 8;     // smem padding to reduce bank conflicts
constexpr int TC_STRIDE = TC_K_TILE + TC_PAD; // 264

constexpr int TC_BLOCK = 256;    // 8 warps

// Dynamic shared memory layout:
//   B_smem: [TC_TILE_N × TC_STRIDE] half  = 33792 bytes
//   A_smem: [TC_TILE_M × TC_STRIDE] half  = 16896 bytes
//   Total:  50688 bytes
constexpr int B_SIZE = TC_TILE_N * TC_STRIDE * sizeof(half); // 33792
constexpr int A_SIZE = TC_TILE_M * TC_STRIDE * sizeof(half); // 16896
constexpr int SMEM_TOTAL = B_SIZE + A_SIZE;                  // 50688

// ---------------------------------------------------------------------------
// Kernel: fused Q6_K dequant + WMMA GEMM for MoE prefill
//
// Grid:  (ceil(N / TC_TILE_N), n_experts)
// Block: TC_BLOCK (256 threads, 8 warps)
//
// Q6_K dequant uses warp-cooperative loading: each warp handles one Q6_K
// block (256 elements), with each lane processing 8 elements via vectorized
// 64-bit loads. This matches the proven scalar kernel's access pattern.
//
// Warp layout for WMMA: 2 M-tiles × 4 N-tiles = 8 warps
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(TC_BLOCK)
gemm_q6k_fused_moe_prefill_tc_kernel(
    const uint8_t* __restrict__ packed_weights,
    const half*    __restrict__ activations,
    half*          __restrict__ output,
    const int32_t* __restrict__ offsets,
    int N, int K,
    size_t expert_stride_bytes,
    int n_experts)
{
    extern __shared__ char dyn_smem[];
    half* B_smem  = reinterpret_cast<half*>(dyn_smem);
    half* AC_smem = reinterpret_cast<half*>(dyn_smem + B_SIZE);

    const int tid      = threadIdx.x;
    const int warp_id  = tid / 32;
    const int lane     = tid % 32;
    const int warp_m   = warp_id / 4;  // 0 or 1
    const int warp_n   = warp_id % 4;  // 0..3

    const int expert_id = blockIdx.y;
    const int n_base    = blockIdx.x * TC_TILE_N;

    if (expert_id >= n_experts) return;

    const int start = offsets[expert_id];
    const int M     = offsets[expert_id + 1] - start;
    if (M == 0) return;

    const int blocks_per_row = K / 256;
    const size_t row_bytes   = static_cast<size_t>(blocks_per_row) * 210;
    const uint8_t* W = packed_weights + static_cast<size_t>(expert_id) * expert_stride_bytes;

    // Pre-compute Q6_K indexing constants for this lane (8 elements per lane)
    const int base_elem = lane * 8;
    const int group     = base_elem >> 7;
    const int within    = base_elem & 127;
    const int quad      = within >> 5;
    const int l_base    = within & 31;
    const int ql_off    = (group << 6) + ((quad & 1) << 5) + l_base;
    const int qh_off    = 128 + (group << 5) + l_base;
    const int is_high   = (quad >= 2) ? 1 : 0;
    const int qh_shift  = quad * 2;
    const int scale_idx = base_elem >> 4;

    // Outer M loop: process TC_TILE_M tokens at a time
    for (int m_base = 0; m_base < M; m_base += TC_TILE_M) {
        const int M_cur = min(TC_TILE_M, M - m_base);

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        // Inner K loop: one Q6_K block column (256 elements) per iteration
        for (int k_block = 0; k_block < blocks_per_row; k_block++) {

            // --- Load B: warp-cooperative Q6_K dequant → B_smem ---
            // 8 warps handle 8 rows per iteration, 8 iterations for 64 rows
            #pragma unroll
            for (int iter = 0; iter < TC_TILE_N / 8; iter++) {
                const int row = warp_id + iter * 8;
                const int n_global = n_base + row;

                half w_vals[8];

                if (n_global < N) {
                    const uint8_t* bp = W + static_cast<size_t>(n_global) * row_bytes
                                          + static_cast<size_t>(k_block) * 210;

                    // Block-level and per-group scale
                    float d_w = __half2float(*reinterpret_cast<const half*>(bp + 208));
                    float sc = static_cast<float>(reinterpret_cast<const int8_t*>(bp + 192)[scale_idx]);
                    float w_scale = d_w * sc;

                    // Load 8 ql bytes and 8 qh bytes (vectorized 64-bit loads)
                    uint64_t ql8, qh8;
                    memcpy(&ql8, bp + ql_off, 8);
                    memcpy(&qh8, bp + qh_off, 8);

                    // Dequant 8 elements
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        uint32_t ql_byte = (static_cast<uint32_t>(ql8 >> (i * 8))) & 0xFFu;
                        uint32_t qh_byte = (static_cast<uint32_t>(qh8 >> (i * 8))) & 0xFFu;
                        uint32_t low4 = is_high ? ((ql_byte >> 4) & 0xFu) : (ql_byte & 0xFu);
                        uint32_t high2 = (qh_byte >> qh_shift) & 0x3u;
                        int q = static_cast<int>((high2 << 4) | low4) - 32;
                        w_vals[i] = __float2half(w_scale * static_cast<float>(q));
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < 8; i++) w_vals[i] = __float2half(0.0f);
                }

                // Store 8 dequanted values to shared memory
                *reinterpret_cast<uint4*>(&B_smem[row * TC_STRIDE + base_elem]) =
                    *reinterpret_cast<uint4*>(w_vals);
            }

            // --- Load A: activations → AC_smem ---
            // 256 threads load 32×256 = 8192 elements, 4 vectorized uint4 loads each
            #pragma unroll 4
            for (int i = 0; i < (TC_TILE_M * TC_K_TILE / 8) / TC_BLOCK; i++) {
                const int load_idx = tid + i * TC_BLOCK;
                const int flat = load_idx * 8;
                const int row  = flat / TC_K_TILE;
                const int col  = flat % TC_K_TILE;

                if (row < M_cur) {
                    const int64_t token = start + m_base + row;
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
        // Reuse AC_smem as float buffer for the C tile [TC_TILE_M × TC_TILE_N]
        float* C_smem_f32 = reinterpret_cast<float*>(AC_smem);

        wmma::store_matrix_sync(
            &C_smem_f32[(warp_m * 16) * TC_TILE_N + (warp_n * 16)],
            c_frag, TC_TILE_N, wmma::mem_row_major);

        __syncthreads();

        // Cooperative FP32→FP16 convert and write to global output
        for (int flat = tid; flat < TC_TILE_M * TC_TILE_N; flat += TC_BLOCK) {
            const int m = flat / TC_TILE_N;
            const int n = flat % TC_TILE_N;
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
                                    cudaStream_t stream)
{
    if (n_experts == 0 || N == 0) return;

    static bool smem_configured = false;
    if (!smem_configured) {
        cudaFuncSetAttribute(gemm_q6k_fused_moe_prefill_tc_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             SMEM_TOTAL);
        smem_configured = true;
    }

    const int blocks_x = (N + TC_TILE_N - 1) / TC_TILE_N;
    dim3 grid(blocks_x, n_experts);
    dim3 block(TC_BLOCK);

    gemm_q6k_fused_moe_prefill_tc_kernel<<<grid, block, SMEM_TOTAL, stream>>>(
        static_cast<const uint8_t*>(packed_weights),
        static_cast<const half*>(activations),
        static_cast<half*>(output),
        d_offsets,
        N, K,
        expert_stride_bytes,
        n_experts);
}

} // namespace imp
