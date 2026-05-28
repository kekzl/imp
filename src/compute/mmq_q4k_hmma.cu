// =============================================================================
// mmq_q4k_hmma.cu -- Phase 0 stub: Q4_K x FP16 tiled GEMM via HMMA m16n8k16
// =============================================================================
//
// Correctness baseline for the Q4_K HMMA GEMM project. The kernel
// dequantizes Q4_K weight super-blocks into FP16 in shared memory, then runs
// WMMA mma_sync (HMMA m16n8k16) on the dequantized tiles. The "in-SMEM
// nibble decode without full materialisation" optimisation comes in a later
// phase; this stub proves the dispatch wiring and correctness framework.
//
// Weight layout (Q4_K, 144 bytes per 256 elements):
//   d       : FP16 super-block scale
//   dmin    : FP16 super-block min
//   scales[12]: packed 6-bit sub-block scales + 6-bit mins (8 sub-blocks of 32)
//   qs[128] : 256 x 4-bit quants packed as 128 bytes
//
// Dequant formula per element e in sub-block j:
//   val = d * sc[j] * nibble - dmin * min[j]
//
// Tile sizes: TILE_M=64, TILE_N=64, TILE_K=256 (one Q4_K super-block per K-step).
// The 256-element K-chunk is processed as 16 consecutive WMMA m16n8k16 MMA
// operations (256 / 16 = 16 K-fragments).

#include "compute/mmq_q4k_hmma.h"
#include "core/logging.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>

namespace imp {
namespace {

using namespace nvcuda;

constexpr int kQ4kBlockBytes = 144;
constexpr int kQ4kSuperBlock = 256;  // elements per Q4_K block

constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 256;  // one Q4_K super-block

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int WARPS_M = 2;
constexpr int WARPS_N = 2;
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N;  // 4
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;  // 128

// Each warp handles (TILE_M/WARPS_M) x (TILE_N/WARPS_N) = 32 x 32 output.
// 32 / WMMA_M = 2 M-fragments, 32 / WMMA_N = 2 N-fragments per warp.
constexpr int FRAGS_M = TILE_M / (WARPS_M * WMMA_M);  // 2
constexpr int FRAGS_N = TILE_N / (WARPS_N * WMMA_N);  // 2
constexpr int K_FRAGS = TILE_K / WMMA_K;  // 16

// SMEM layout: two tiles for double-buffered dequant.
// A tile: [TILE_M, TILE_K] FP16 = 64 * 256 * 2 = 32 KiB
// B tile: [TILE_N, TILE_K] FP16 = 64 * 256 * 2 = 32 KiB
// Total: 64 KiB -- fits in sm_120 shared memory (up to 228 KiB per SM).
// No double-buffering needed for Phase 0 stub (single-stage).

// Unpack 6-bit scale and min from the 12-byte packed array.
// Matches ggml get_scale_min_k4.
__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* q,
                                                  uint8_t& sc_out, uint8_t& m_out) {
    if (j < 4) {
        sc_out = q[j] & 63u;
        m_out = q[j + 4] & 63u;
    } else {
        sc_out = (q[j + 4] & 0xFu) | ((q[j - 4] >> 6) << 4);
        m_out = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Dequantize one Q4_K super-block (256 elements) into FP16 in shared memory.
// Called by multiple threads cooperatively. `tid` in [0, THREADS_PER_BLOCK).
// `block_ptr` points to the 144-byte Q4_K block in global memory.
// `smem_out` points to the shared memory destination (256 halves).
__device__ void dequant_q4k_block_to_smem(const uint8_t* __restrict__ block_ptr,
                                           __half* __restrict__ smem_out,
                                           int tid, int num_threads) {
    // Parse block header
    float d    = __half2float(*reinterpret_cast<const __half*>(block_ptr));
    float dmin = __half2float(*reinterpret_cast<const __half*>(block_ptr + 2));
    const uint8_t* scales = block_ptr + 4;
    const uint8_t* qs     = block_ptr + 16;

    // Each thread dequantizes multiple elements (256 / num_threads).
    for (int e = tid; e < kQ4kSuperBlock; e += num_threads) {
        int group = e >> 6;           // 0..3  (64-element groups)
        int in_grp = e & 63;
        int is_high = (in_grp >> 5);  // 0 or 1 (low or high nibble)
        int byte_in_group = in_grp & 31;
        int byte_in_qs = group * 32 + byte_in_group;
        int sub_block = group * 2 + is_high;  // 0..7

        uint8_t sc_val, min_val;
        get_scale_min_k4(sub_block, scales, sc_val, min_val);

        int nibble = is_high ? (qs[byte_in_qs] >> 4) : (qs[byte_in_qs] & 0xF);
        float val = d * static_cast<float>(sc_val) * static_cast<float>(nibble)
                  - dmin * static_cast<float>(min_val);
        smem_out[e] = __float2half(val);
    }
}

// Main kernel: tiled GEMM with Q4_K dequant in SMEM + WMMA HMMA.
//
// Grid: (ceil(N/TILE_N), ceil(M/TILE_M))
// Block: THREADS_PER_BLOCK = 128
//
// A [M, K] FP16 row-major (activations)
// B [N, K] Q4_K packed (weights, N rows of K/256 super-blocks)
// C [M, K] @ B[N, K]^T -> C [M, N] FP16 row-major
__global__ void mmq_q4k_hmma_kernel(
    const __half* __restrict__ A,       // [M, K]
    const uint8_t* __restrict__ B_q4k,  // N * (K/256) * 144 bytes
    __half* __restrict__ C,             // [M, N]
    int M, int N, int K)
{
    // Block indices
    const int bx = blockIdx.x;  // N-tile index
    const int by = blockIdx.y;  // M-tile index
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_m = warp_id / WARPS_N;  // 0 or 1
    const int warp_n = warp_id % WARPS_N;  // 0 or 1

    const int m_start = by * TILE_M;
    const int n_start = bx * TILE_N;

    const int blocks_per_row = K / kQ4kSuperBlock;

    // Shared memory for dequantized tiles:
    //   A_s: [TILE_M][TILE_K] FP16 -- activation tile
    //   B_s: [TILE_N][TILE_K] FP16 -- weight tile (dequantized)
    extern __shared__ __half smem[];
    __half* A_s = smem;                          // [TILE_M * TILE_K]
    __half* B_s = smem + TILE_M * TILE_K;        // [TILE_N * TILE_K]

    // Accumulator fragments: each warp accumulates FRAGS_M x FRAGS_N 16x16 tiles.
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[FRAGS_M][FRAGS_N];
    for (int i = 0; i < FRAGS_M; ++i)
        for (int j = 0; j < FRAGS_N; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    // K-loop: one iteration per Q4_K super-block (256 elements).
    for (int kb = 0; kb < blocks_per_row; ++kb) {
        const int k_offset = kb * kQ4kSuperBlock;

        // --- Load activation tile A[m_start : m_start+TILE_M, k_offset : k_offset+TILE_K]
        //     directly into shared memory. Each thread loads multiple elements.
        for (int idx = tid; idx < TILE_M * TILE_K; idx += THREADS_PER_BLOCK) {
            int row = idx / TILE_K;
            int col = idx % TILE_K;
            int gm = m_start + row;
            int gk = k_offset + col;
            A_s[row * TILE_K + col] = (gm < M && gk < K) ? A[gm * K + gk] : __float2half(0.0f);
        }

        // --- Dequantize weight tile B[n_start : n_start+TILE_N, kb]
        //     Each row n has its own Q4_K super-block at offset (n * blocks_per_row + kb).
        //     All threads cooperate to dequant each row's 256 elements.
        //
        //     Strategy: assign rows round-robin across a group, each thread
        //     handles multiple elements within each row.
        for (int tn = 0; tn < TILE_N; ++tn) {
            int gn = n_start + tn;
            if (gn < N) {
                const uint8_t* block_ptr = B_q4k +
                    static_cast<int64_t>(gn * blocks_per_row + kb) * kQ4kBlockBytes;
                dequant_q4k_block_to_smem(block_ptr, B_s + tn * TILE_K, tid, THREADS_PER_BLOCK);
            } else {
                // Zero-fill for out-of-bounds rows.
                for (int e = tid; e < kQ4kSuperBlock; e += THREADS_PER_BLOCK)
                    B_s[tn * TILE_K + e] = __float2half(0.0f);
            }
        }

        __syncthreads();

        // --- WMMA MMA over the 256-element K-chunk, 16 elements at a time.
        const int warp_m_offset = warp_m * (TILE_M / WARPS_M);  // 0 or 32
        const int warp_n_offset = warp_n * (TILE_N / WARPS_N);  // 0 or 32

        for (int kk = 0; kk < K_FRAGS; ++kk) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag[FRAGS_M];
            for (int i = 0; i < FRAGS_M; ++i) {
                int row = warp_m_offset + i * WMMA_M;
                wmma::load_matrix_sync(a_frag[i], A_s + row * TILE_K + kk * WMMA_K, TILE_K);
            }

            // B is stored as [TILE_N, TILE_K] row-major but GEMM needs B^T,
            // so we load B as col_major (each row of B_s becomes a column).
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag[FRAGS_N];
            for (int j = 0; j < FRAGS_N; ++j) {
                int col = warp_n_offset + j * WMMA_N;
                wmma::load_matrix_sync(b_frag[j], B_s + col * TILE_K + kk * WMMA_K, TILE_K);
            }

            for (int i = 0; i < FRAGS_M; ++i)
                for (int j = 0; j < FRAGS_N; ++j)
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }

        __syncthreads();
    }

    // --- Store accumulated results to global memory via SMEM staging.
    // Each warp gets a dedicated SMEM slot for store_matrix_sync, avoiding
    // inter-warp races (same pattern as gemm_capture_fp16_sm120.cu).
    __shared__ float frag_smem[WARPS_PER_BLOCK * WMMA_M * WMMA_N];
    float* warp_frag = frag_smem + warp_id * (WMMA_M * WMMA_N);

    const int warp_m_out = warp_m * (TILE_M / WARPS_M);
    const int warp_n_out = warp_n * (TILE_N / WARPS_N);
    const int lane = tid % 32;

    for (int i = 0; i < FRAGS_M; ++i) {
        for (int j = 0; j < FRAGS_N; ++j) {
            wmma::store_matrix_sync(warp_frag, acc[i][j], WMMA_N, wmma::mem_row_major);
            __syncwarp();

            int out_m = m_start + warp_m_out + i * WMMA_M;
            int out_n = n_start + warp_n_out + j * WMMA_N;

            for (int t = 0; t < (WMMA_M * WMMA_N) / 32; ++t) {
                int idx = t * 32 + lane;
                int r = idx / WMMA_N;
                int c = idx % WMMA_N;
                int gm = out_m + r;
                int gn = out_n + c;
                if (gm < M && gn < N) {
                    C[gm * N + gn] = __float2half(warp_frag[idx]);
                }
            }
            __syncwarp();
        }
    }
}

}  // namespace

bool mmq_q4k_hmma_gemm(const void* A_fp16, const void* B_q4k, void* C_fp16,
                       int M, int N, int K, cudaStream_t stream) {
    // Validate constraints.
    if (K % kQ4kSuperBlock != 0) return false;
    if (M < 16 || N < 16) return false;
    // Pad M and N to tile boundaries internally via bounds checks in kernel.
    // No strict alignment requirement on M/N for correctness (kernel checks bounds).

    const int grid_x = (N + TILE_N - 1) / TILE_N;
    const int grid_y = (M + TILE_M - 1) / TILE_M;

    const size_t smem_bytes = static_cast<size_t>(TILE_M + TILE_N) * TILE_K * sizeof(__half);
    // (64 + 64) * 256 * 2 = 65536 bytes = 64 KiB

    // Request dynamic shared memory if > 48 KiB default.
    static bool smem_configured = false;
    if (!smem_configured && smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(mmq_q4k_hmma_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_bytes));
        smem_configured = true;
    }

    mmq_q4k_hmma_kernel<<<dim3(grid_x, grid_y), THREADS_PER_BLOCK, smem_bytes, stream>>>(
        static_cast<const __half*>(A_fp16),
        static_cast<const uint8_t*>(B_q4k),
        static_cast<__half*>(C_fp16),
        M, N, K);

    return true;
}

}  // namespace imp
