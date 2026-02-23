#include "quant/quant_gemm.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cassert>

namespace imp {

// ---------------------------------------------------------------------------
// Fused INT4 dequantization + GEMM kernel.
//
//   C[M,N] = A[M,K] @ dequant(B_quant[N,K/2], scales[N, K/group_size])
//
// B_quant is stored in [N, K/2] layout: for each output channel n, K weights
// are packed 2 per byte (low nibble first, GGML Q4_0 compatible).  This
// layout means consecutive bytes along the K dimension for a given n, which
// gives good coalescing when loading the B tile in the K direction.
//
// Algorithm: standard tiled GEMM with on-the-fly dequantization of B.
//   Tile sizes:  Tm x Tn x Tk  =  64 x 64 x 32
//   Block:       16 x 16 threads  (256 threads)
//   Each thread accumulates a 4x4 sub-tile of C in registers.
//
// For each K-tile iteration:
//   1. Cooperatively load Tm x Tk tile of A (FP16) into shared memory.
//   2. Cooperatively load Tn x (Tk/2) bytes of B_quant, dequantize to
//      Tn x Tk half values in shared memory, using scales[n][k/group_size].
//   3. Multiply A_smem[tm, tk] * B_smem[tn, tk] accumulated into C_reg.
//   4. After all K tiles, write C_reg back to global C[M,N].
// ---------------------------------------------------------------------------

static constexpr int TILE_M = 64;
static constexpr int TILE_N = 64;
static constexpr int TILE_K = 32;
static constexpr int BLOCK_DIM_X = 16;
static constexpr int BLOCK_DIM_Y = 16;
static constexpr int THREAD_TILE_M = 4;  // each thread computes 4 rows
static constexpr int THREAD_TILE_N = 4;  // each thread computes 4 cols

__global__ void quant_gemm_int4_kernel(
    const half*    __restrict__ A,         // [M, K]
    const uint8_t* __restrict__ B_quant,   // [N, K/2]
    const half*    __restrict__ scales,    // [N, num_groups] where num_groups = K / group_size
    half*          __restrict__ C,         // [M, N]
    int M, int N, int K,
    int group_size)
{
    // Block indices: each block covers a TILE_M x TILE_N tile of C.
    const int bm = blockIdx.x * TILE_M;
    const int bn = blockIdx.y * TILE_N;
    const int tx = threadIdx.x;  // 0..15
    const int ty = threadIdx.y;  // 0..15
    const int tid = ty * BLOCK_DIM_X + tx;  // flat thread id 0..255

    const int num_groups = (K + group_size - 1) / group_size;

    // Shared memory for A tile [TILE_M][TILE_K] and B tile [TILE_N][TILE_K].
    __shared__ half As[TILE_M][TILE_K];
    __shared__ half Bs[TILE_N][TILE_K];

    // Accumulator in registers: each thread owns a THREAD_TILE_M x THREAD_TILE_N block.
    float acc[THREAD_TILE_M][THREAD_TILE_N];
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; ++i)
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; ++j)
            acc[i][j] = 0.0f;

    // Number of K-tiles.
    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k_offset = kt * TILE_K;

        // --- Load A tile [TILE_M][TILE_K] into shared memory ----------------
        // 256 threads, TILE_M * TILE_K = 64 * 32 = 2048 elements -> 8 per thread.
        #pragma unroll
        for (int i = 0; i < (TILE_M * TILE_K) / (BLOCK_DIM_X * BLOCK_DIM_Y); ++i) {
            int flat = tid + i * (BLOCK_DIM_X * BLOCK_DIM_Y);
            int row = flat / TILE_K;
            int col = flat % TILE_K;
            int global_m = bm + row;
            int global_k = k_offset + col;
            if (global_m < M && global_k < K) {
                As[row][col] = A[global_m * K + global_k];
            } else {
                As[row][col] = __float2half(0.0f);
            }
        }

        // --- Load & dequantize B tile [TILE_N][TILE_K] ----------------------
        // B_quant is [N, K/2].  For a given (n, k), the packed byte is at
        // B_quant[n * (K/2) + k/2], with k%2 selecting low or high nibble.
        // We load TILE_N * TILE_K = 2048 half values; that's 2048 / 2 = 1024
        // bytes of packed data, but it's cleaner to just have each element
        // handled individually (2048 elements / 256 threads = 8 per thread).
        const int half_K = K / 2;

        #pragma unroll
        for (int i = 0; i < (TILE_N * TILE_K) / (BLOCK_DIM_X * BLOCK_DIM_Y); ++i) {
            int flat = tid + i * (BLOCK_DIM_X * BLOCK_DIM_Y);
            int n_local = flat / TILE_K;
            int k_local = flat % TILE_K;
            int global_n = bn + n_local;
            int global_k = k_offset + k_local;

            if (global_n < N && global_k < K) {
                int byte_idx = global_n * half_K + global_k / 2;
                uint8_t packed = B_quant[byte_idx];
                int nibble;
                if (global_k % 2 == 0) {
                    nibble = packed & 0x0F;         // low nibble
                } else {
                    nibble = (packed >> 4) & 0x0F;  // high nibble
                }
                int group_idx = global_k / group_size;
                float scale = __half2float(scales[global_n * num_groups + group_idx]);
                Bs[n_local][k_local] = __float2half((float)(nibble - 8) * scale);
            } else {
                Bs[n_local][k_local] = __float2half(0.0f);
            }
        }

        __syncthreads();

        // --- Compute: accumulate TILE_K products ----------------------------
        // Thread (tx, ty) owns sub-tile:
        //   rows: ty * THREAD_TILE_M  ..  ty * THREAD_TILE_M + 3
        //   cols: tx * THREAD_TILE_N  ..  tx * THREAD_TILE_N + 3
        #pragma unroll
        for (int tk = 0; tk < TILE_K; ++tk) {
            // Load A values for this thread's rows.
            float a_val[THREAD_TILE_M];
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                a_val[i] = __half2float(As[ty * THREAD_TILE_M + i][tk]);
            }
            // Load B values for this thread's cols.
            float b_val[THREAD_TILE_N];
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; ++j) {
                b_val[j] = __half2float(Bs[tx * THREAD_TILE_N + j][tk]);
            }
            // Outer product accumulation.
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i)
                #pragma unroll
                for (int j = 0; j < THREAD_TILE_N; ++j)
                    acc[i][j] += a_val[i] * b_val[j];
        }

        __syncthreads();
    }

    // --- Write C tile back to global memory ---------------------------------
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; ++i) {
        int global_m = bm + ty * THREAD_TILE_M + i;
        if (global_m >= M) continue;
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; ++j) {
            int global_n = bn + tx * THREAD_TILE_N + j;
            if (global_n < N) {
                C[global_m * N + global_n] = __float2half(acc[i][j]);
            }
        }
    }
}

void quant_gemm_int4(const Tensor& A, const Tensor& B_quant,
                     const Tensor& scales, Tensor& C,
                     cudaStream_t stream)
{
    // --- Validate dimensions ------------------------------------------------
    assert(A.ndim == 2 && "A must be 2D [M, K]");
    assert(B_quant.ndim == 2 && "B_quant must be 2D [N, K/2]");
    assert(scales.ndim == 2 && "scales must be 2D [N, num_groups]");
    assert(C.ndim == 2 && "C must be 2D [M, N]");

    const int M = static_cast<int>(A.shape[0]);
    const int K = static_cast<int>(A.shape[1]);
    const int N = static_cast<int>(B_quant.shape[0]);
    const int half_K = static_cast<int>(B_quant.shape[1]);

    assert(half_K == K / 2 && "B_quant.shape[1] must be K/2");
    assert(C.shape[0] == M && C.shape[1] == N);

    const int num_groups = static_cast<int>(scales.shape[1]);
    const int group_size = (K + num_groups - 1) / num_groups;

    // --- Launch kernel ------------------------------------------------------
    dim3 grid((M + TILE_M - 1) / TILE_M,
              (N + TILE_N - 1) / TILE_N);
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);

    quant_gemm_int4_kernel<<<grid, block, 0, stream>>>(
        static_cast<const half*>(A.data),
        static_cast<const uint8_t*>(B_quant.data),
        static_cast<const half*>(scales.data),
        static_cast<half*>(C.data),
        M, N, K,
        group_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "quant_gemm_int4 launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

} // namespace imp
