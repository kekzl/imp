// Capture-safe sm_120 FP16 dense GEMM via WMMA HMMA tensor cores.
//
// Drop-in replacement for cublasLtMatmul when the stream is in capture
// mode. cuBLASLt fails with CUBLAS_STATUS_INTERNAL_ERROR on the first
// GEMM under cudaStreamCapture on sm_120 — its algorithm heuristic and
// internal workspace allocation paths are not capture-safe. This
// hand-tuned WMMA kernel keeps all decisions on-device (no host
// heuristics, no cudaMalloc inside the captured region) so it composes
// cleanly with CUDA graph capture.
//
// Geometry (v1 — correctness first):
//   Block tile: BM × BN × BK = 128 × 128 × 32
//   Warps per block: 4 (laid out 2×2 for the M×N output)
//   Per-warp tile: 64 × 64
//   WMMA fragment: 16 × 16 × 16 (HMMA m16n8k16 via wmma::mma_sync)
//   Per-warp MMA loop: 4 × 4 × 2 (M frags × N frags × K frags)
//   Accumulator: FP32 (precision); downcast to FP16 on store.
//
// Layout: A row-major [M, K], B row-major [N, K] (semantically B^T in
// the GEMM, matching cuBLAS OP_T), D row-major [M, N]. Output:
//   D = alpha * A @ B^T + beta * D
//
// M and N must be multiples of BM=BN=128; K must be a multiple of 16.

#include "compute/gemm_capture_fp16_sm120.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

namespace imp {
namespace {

using namespace nvcuda;

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 2;
constexpr int WARP_M = BM / WARPS_M;  // 64
constexpr int WARP_N = BN / WARPS_N;  // 64

constexpr int FRAGS_M = WARP_M / WMMA_M;  // 4
constexpr int FRAGS_N = WARP_N / WMMA_N;  // 4
constexpr int FRAGS_K = BK / WMMA_K;      // 2

// Shared memory layout: A_smem [BM][BK], B_smem [BN][BK].
// B is laid out as row-major [BN][BK] so that within a tile the K dim
// is contiguous — matches cuBLAS's logical OP_T(B) view of [N, K] data.
// The WMMA matrix_b fragment uses col_major against B_smem so that the
// k-axis striding produces the correct B^T column gather.
struct alignas(16) SmemTile {
    __half A[BM * BK];
    __half B[BN * BK];
};

__launch_bounds__(THREADS_PER_BLOCK, 2) __global__
    void gemm_fp16_kernel(const __half* __restrict__ A, const __half* __restrict__ B,
                          __half* __restrict__ D, int M, int N, int K, float alpha, float beta) {
    extern __shared__ __align__(16) char smem_raw[];
    SmemTile* smem = reinterpret_cast<SmemTile*>(smem_raw);

    int block_m = blockIdx.y * BM;
    int block_n = blockIdx.x * BN;
    int tid     = threadIdx.x;
    int warp    = tid / 32;
    int lane    = tid % 32;
    int wm      = warp / WARPS_N;  // 0..WARPS_M-1
    int wn      = warp % WARPS_N;  // 0..WARPS_N-1

    // FP32 accumulators per fragment (FRAGS_M × FRAGS_N grid).
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[FRAGS_M][FRAGS_N];
#pragma unroll
    for (int i = 0; i < FRAGS_M; ++i)
#pragma unroll
        for (int j = 0; j < FRAGS_N; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        // Cooperative global → shared load.
        // A: BM × BK = 128 × 32 = 4096 halves; 128 threads → 32 halves each.
        // Use vectorized __half2 loads (16 halves per thread = 8 vector loads).
        constexpr int A_ELEMS = BM * BK;
        constexpr int B_ELEMS = BN * BK;

        // Each thread loads A_ELEMS / THREADS_PER_BLOCK = 32 halves from A.
        // We'll do 4 × float4 loads (16 halves per thread per loop) — actually
        // a simpler scheme: 32-half stride per thread covers 128 threads × 32 =
        // 4096 halves = exactly A_ELEMS. Pattern: thread t loads halves
        // [t * 32, (t+1) * 32) of the smem tile, which corresponds to row
        // (t * 32) / BK and starting col (t * 32) % BK.
#pragma unroll
        for (int i = 0; i < A_ELEMS / THREADS_PER_BLOCK; ++i) {
            int idx     = tid + i * THREADS_PER_BLOCK;
            int row     = idx / BK;
            int col     = idx % BK;
            int g_row   = block_m + row;
            int g_col   = k_tile + col;
            __half val  = __float2half(0.0f);
            if (g_row < M && g_col < K)
                val = A[(int64_t)g_row * K + g_col];
            smem->A[row * BK + col] = val;
        }
#pragma unroll
        for (int i = 0; i < B_ELEMS / THREADS_PER_BLOCK; ++i) {
            int idx     = tid + i * THREADS_PER_BLOCK;
            int row     = idx / BK;
            int col     = idx % BK;
            int g_row   = block_n + row;
            int g_col   = k_tile + col;
            __half val  = __float2half(0.0f);
            if (g_row < N && g_col < K)
                val = B[(int64_t)g_row * K + g_col];
            smem->B[row * BK + col] = val;
        }
        __syncthreads();

        // MMA loop. Each warp computes WARP_M × WARP_N from this BM × BN smem tile.
#pragma unroll
        for (int kk = 0; kk < FRAGS_K; ++kk) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag[FRAGS_M];
#pragma unroll
            for (int i = 0; i < FRAGS_M; ++i) {
                int a_row = wm * WARP_M + i * WMMA_M;
                wmma::load_matrix_sync(a_frag[i], smem->A + a_row * BK + kk * WMMA_K, BK);
            }
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag[FRAGS_N];
#pragma unroll
            for (int j = 0; j < FRAGS_N; ++j) {
                int b_row = wn * WARP_N + j * WMMA_N;
                // matrix_b col_major against [BN][BK] smem produces a [K][N] view,
                // so the b_frag k-axis aligns with B^T's k-axis. Stride is BK.
                wmma::load_matrix_sync(b_frag[j], smem->B + b_row * BK + kk * WMMA_K, BK);
            }
#pragma unroll
            for (int i = 0; i < FRAGS_M; ++i)
#pragma unroll
                for (int j = 0; j < FRAGS_N; ++j)
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
        __syncthreads();
    }

    // Epilogue: alpha * acc + beta * D, store as FP16. Process one fragment
    // at a time, staging through a per-warp slot of a small smem buffer
    // (WMMA_M × WMMA_N × WARPS_PER_BLOCK = 4 KiB total). Per-warp slots
    // keep concurrent warp writes race-free without an extra __syncthreads
    // around the wmma::store_matrix_sync.
    __shared__ float frag_smem[WARPS_PER_BLOCK * WMMA_M * WMMA_N];
    float* warp_frag = frag_smem + warp * (WMMA_M * WMMA_N);

    int warp_base_m = block_m + wm * WARP_M;
    int warp_base_n = block_n + wn * WARP_N;

#pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
#pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            wmma::store_matrix_sync(warp_frag, acc[i][j], WMMA_N, wmma::mem_row_major);
            __syncwarp();

            int frag_row0 = warp_base_m + i * WMMA_M;
            int frag_col0 = warp_base_n + j * WMMA_N;

            // 32 lanes × 8 elements = 256 = WMMA_M × WMMA_N.
            for (int t = 0; t < (WMMA_M * WMMA_N) / 32; ++t) {
                int idx   = t * 32 + lane;
                int r     = idx / WMMA_N;
                int c     = idx % WMMA_N;
                int g_row = frag_row0 + r;
                int g_col = frag_col0 + c;
                if (g_row >= M || g_col >= N) continue;
                float v = alpha * warp_frag[idx];
                if (beta != 0.0f) {
                    float prev = __half2float(D[(int64_t)g_row * N + g_col]);
                    v += beta * prev;
                }
                D[(int64_t)g_row * N + g_col] = __float2half(v);
            }
            __syncwarp();
        }
    }
}

}  // anonymous namespace

static int s_avail = -1;

bool capture_gemm_fp16_sm120_available() {
    if (s_avail >= 0) return s_avail;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    s_avail = (prop.major * 10 + prop.minor >= 120) ? 1 : 0;
    return s_avail;
}

bool gemm_capture_fp16_sm120(const void* A, const void* B, void* D, int M, int N, int K, float alpha,
                              float beta, cudaStream_t stream) {
    if (!capture_gemm_fp16_sm120_available()) return false;
    if (M <= 0 || N <= 0 || K <= 0) return false;
    // v1 requires tile-aligned shapes. cuBLAS shapes from prefill
    // (M=512, N=128/512/2048/4096, K=2048/4096) all satisfy this.
    if (M % BM != 0 || N % BN != 0 || K % WMMA_K != 0) return false;

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(THREADS_PER_BLOCK);
    size_t smem_bytes = sizeof(SmemTile);  // 16 KiB — well under default 48 KiB cap

    gemm_fp16_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const __half*>(A), reinterpret_cast<const __half*>(B),
        reinterpret_cast<__half*>(D), M, N, K, alpha, beta);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("gemm_capture_fp16_sm120: launch failed M=%d N=%d K=%d smem=%zu: %s", M, N, K,
                      smem_bytes, cudaGetErrorString(err));
        return false;
    }
    return true;
}

}  // namespace imp
