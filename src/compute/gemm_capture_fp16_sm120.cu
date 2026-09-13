// Capture-safe sm_120 FP16 dense GEMM via WMMA HMMA tensor cores: drop-in replacement
// for cublasLtMatmul when the stream is capturing (cuBLASLt's heuristic and internal
// workspace allocation are not capture-safe on sm_120). All decisions are on-device.
// Geometry (v3, cp.async pipeline + per-shape BM dispatch): block tile BM x BN x BK =
// (64 or 128) x 128 x 32, 4 warps in 2x2 layout. BM=64: per-warp 32x64 -> 2x4 frags x 2
// K-frags = 16 MMAs/iter/warp. BM=128: per-warp 64x64 -> 4x4 frags x 2 K-frags = 32
// MMAs/iter/warp. WMMA fragment 16x16x16 (HMMA m16n8k16), FP32 accumulator. 2 SMEM
// tiles (double-buffer), cp.async.cg with 16B chunks.
// Layout: A row-major [M,K], B row-major [N,K] (semantically B^T), D row-major [M,N];
// D = alpha*A@B^T + beta*D. M,N must be multiples of BM,BN; K must be a multiple of BK=32.

#include "compute/gemm_capture_fp16_sm120.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

namespace imp {
namespace {

using namespace nvcuda;

// Common (non-BM-dependent) constants.
constexpr int BN = 128;
constexpr int BK = 32;
// SMEM stride = BK (no padding): an 8-half pad to break the apparent ldmatrix.x4 bank
// conflict (BK=32 halves = 64B stride aligns lanes 0/2/4/6 on one bank) regressed
// 10-17% on every shape >= N=128. ldmatrix on sm_120 apparently absorbs this pattern
// via its own swizzle/broadcast unit; keep the simple unpadded layout.
constexpr int BK_SMEM = BK;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// 4 warps in 2x2 layout: 8 warps (2x4) regressed 8-22% despite halving register
// pressure (246->124). Root cause: 2x more total SMEM-fragment loads (each warp still
// reads its own A frags; 4 wn vs 2 wn means each A row loads 4x redundantly vs 2x),
// plus a costlier 8-warp __syncthreads. Kernel is compute/MMA-pipeline-bound, not
// register-bound.
constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 2;
constexpr int WARP_N = BN / WARPS_N;  // 64
constexpr int FRAGS_N = WARP_N / WMMA_N;  // 4
constexpr int FRAGS_K = BK / WMMA_K;      // 2

constexpr int CHUNK_HALVES = 8;  // 16B cp.async per thread per chunk

__device__ __forceinline__ void cp_async_cg16_zero(void* smem_ptr, const void* gmem_ptr, bool valid) {
    // cp.async predicated form: `cp-size=0` zero-fills the shared destination when
    // the source is out-of-bounds. We use the 4th operand (src-size) which when
    // smaller than cp-size causes the remainder to be zero-filled in smem.
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    int src_size = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem_ptr), "r"(src_size));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Issues all cp.async loads for one (A,B) tile. Templated on BM so the chunk-loop trip
// counts are compile-time constants and ptxas straightlines the cp.async issues (8 for
// BM=128, 6 for BM=64) into a back-to-back pipeline-friendly sequence.
template <int BM>
__device__ __forceinline__ void issue_tile_load(__half* a_smem, __half* b_smem, const __half* A,
                                                const __half* B, int block_m, int block_n,
                                                int k_tile, int M, int N, int K) {
    constexpr int A_CHUNKS_PER_THREAD = (BM * BK) / (CHUNK_HALVES * THREADS_PER_BLOCK);
    constexpr int B_CHUNKS_PER_THREAD = (BN * BK) / (CHUNK_HALVES * THREADS_PER_BLOCK);
    static_assert(BK == 32, "BK must be 32 for the bit-shift row/col split");
    static_assert(CHUNK_HALVES == 8, "CHUNK_HALVES must be 8 for cp.async 16B");

    int tid     = threadIdx.x;
    bool a_full = (block_m + BM <= M);
    bool b_full = (block_n + BN <= N);

#pragma unroll
    for (int c = 0; c < A_CHUNKS_PER_THREAD; ++c) {
        int chunk         = c * THREADS_PER_BLOCK + tid;
        int row           = chunk >> 2;       // chunk / (BK/CHUNK_HALVES) = chunk / 4
        int col           = (chunk & 3) << 3; // (chunk % 4) * CHUNK_HALVES
        int g_row         = block_m + row;
        int g_col         = k_tile + col;
        __half* dst       = a_smem + row * BK_SMEM + col;  // padded SMEM stride
        const __half* src = A + (int64_t)g_row * K + g_col;
        bool valid        = a_full || (g_row < M);
        cp_async_cg16_zero(dst, src, valid);
    }
#pragma unroll
    for (int c = 0; c < B_CHUNKS_PER_THREAD; ++c) {
        int chunk         = c * THREADS_PER_BLOCK + tid;
        int row           = chunk >> 2;
        int col           = (chunk & 3) << 3;
        int g_row         = block_n + row;
        int g_col         = k_tile + col;
        __half* dst       = b_smem + row * BK_SMEM + col;  // padded SMEM stride
        const __half* src = B + (int64_t)g_row * K + g_col;
        bool valid        = b_full || (g_row < N);
        cp_async_cg16_zero(dst, src, valid);
    }
}

template <int BM, int STAGES>
__launch_bounds__(THREADS_PER_BLOCK, 2) __global__
    void gemm_fp16_kernel(const __half* __restrict__ A, const __half* __restrict__ B,
                          __half* __restrict__ D, int M, int N, int K, float alpha, float beta) {
    constexpr int WARP_M = BM / WARPS_M;
    constexpr int FRAGS_M = WARP_M / WMMA_M;
    constexpr int A_HALVES_PER_STAGE = BM * BK_SMEM;
    constexpr int B_HALVES_PER_STAGE = BN * BK_SMEM;
    constexpr int STAGE_HALVES = A_HALVES_PER_STAGE + B_HALVES_PER_STAGE;
    static_assert(STAGES == 2 || STAGES == 3, "STAGES must be 2 or 3");

    extern __shared__ __align__(16) char smem_raw[];
    __half* smem_base = reinterpret_cast<__half*>(smem_raw);
    __half* A_stage[STAGES];
    __half* B_stage[STAGES];
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
        A_stage[s] = smem_base + s * STAGE_HALVES;
        B_stage[s] = smem_base + s * STAGE_HALVES + A_HALVES_PER_STAGE;
    }

    int block_m = blockIdx.y * BM;
    int block_n = blockIdx.x * BN;
    int tid     = threadIdx.x;
    int warp    = tid / 32;
    int lane    = tid % 32;
    int wm      = warp / WARPS_N;
    int wn      = warp % WARPS_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[FRAGS_M][FRAGS_N];
#pragma unroll
    for (int i = 0; i < FRAGS_M; ++i)
#pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    int n_tiles = K / BK;

    // Prologue: issue STAGES-1 commits ahead. The main loop maintains STAGES-1
    // in-flight commits so that wait_group<STAGES-1> in each iter unblocks
    // exactly the current stage's load.
    issue_tile_load<BM>(A_stage[0], B_stage[0], A, B, block_m, block_n, 0, M, N, K);
    cp_async_commit();
    if (STAGES == 3 && n_tiles > 1) {
        issue_tile_load<BM>(A_stage[1], B_stage[1], A, B, block_m, block_n, BK, M, N, K);
        cp_async_commit();
    }

    for (int k_idx = 0; k_idx < n_tiles; ++k_idx) {
        int cur            = k_idx % STAGES;
        int next_load_idx  = k_idx + STAGES - 1;
        int next_load_buf  = next_load_idx % STAGES;

        if (next_load_idx < n_tiles) {
            int next_k = next_load_idx * BK;
            issue_tile_load<BM>(A_stage[next_load_buf], B_stage[next_load_buf], A, B, block_m, block_n,
                                next_k, M, N, K);
            cp_async_commit();
            cp_async_wait_group<STAGES - 1>();
        } else if (STAGES == 3 && k_idx + 1 < n_tiles) {
            // Tail: one commit still in flight (for the next iter's MMA).
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();

        const __half* A_s = A_stage[cur];
        const __half* B_s = B_stage[cur];

#pragma unroll
        for (int kk = 0; kk < FRAGS_K; ++kk) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag[FRAGS_M];
#pragma unroll
            for (int i = 0; i < FRAGS_M; ++i) {
                int a_row = wm * WARP_M + i * WMMA_M;
                wmma::load_matrix_sync(a_frag[i], A_s + a_row * BK_SMEM + kk * WMMA_K, BK_SMEM);
            }
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag[FRAGS_N];
#pragma unroll
            for (int j = 0; j < FRAGS_N; ++j) {
                int b_row = wn * WARP_N + j * WMMA_N;
                wmma::load_matrix_sync(b_frag[j], B_s + b_row * BK_SMEM + kk * WMMA_K, BK_SMEM);
            }
#pragma unroll
            for (int i = 0; i < FRAGS_M; ++i)
#pragma unroll
                for (int j = 0; j < FRAGS_N; ++j)
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
        __syncthreads();
    }

    // Epilogue: alpha * acc + beta * D, store as FP16. Per-warp scratch slot
    // keeps concurrent warp writes race-free without an extra __syncthreads
    // around wmma::store_matrix_sync.
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

// BM=64 wins when SM saturation matters more than per-block compute amortization
// (small total block count); BM=128 wins when the grid already saturates the SM array
// (>= ~1 wave at 2 blocks/SM). Threshold (170 SMs, 2 blocks/SM at BM=128, 3 at BM=64):
// BM=64 clearly wins below ~128 BM=128-equivalent blocks, BM=128 above ~256;
// kBlockSaturationThreshold=200 splits the regime.
constexpr int kBlockSaturationThreshold = 200;

bool should_use_bm64(int M, int N) {
    int blocks_bm128 = ((M + 127) / 128) * ((N + 127) / 128);
    return blocks_bm128 < kBlockSaturationThreshold;
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
    // K must be a multiple of BK=32 (cp.async chunks fill full tiles). Any positive N is
    // safe: the load zero-fills past N/M (cp.async src-size=0) and the store masks
    // g_col>=N; a narrow N<BN just wastes part of the tile. This matters under capture:
    // cuBLASLt would fail with status 14 and abort the whole decode graph otherwise (#934,
    // GDN N=32 projections).
    if (K % BK != 0) return false;

    bool use_bm64 = should_use_bm64(M, N);
    int BM_v     = use_bm64 ? 64 : 128;
    if (M < BM_v) return false;  // need at least one M-block

    dim3 grid((N + BN - 1) / BN, (M + BM_v - 1) / BM_v);
    dim3 block(THREADS_PER_BLOCK);

    // Both variants use 2-stage cp.async pipelining. 3-stage on BM=64 regressed 2-5%: SMEM
    // growth (12->18 KiB/stage) dropped occupancy 3->2 blocks/SM, and the deeper pipeline
    // didn't recover the loss (kernel is compute-scheduling/register-bound, not
    // load-latency-bound).
    constexpr size_t smem_bytes_bm64  = 2 * ((64 + BN) * BK_SMEM) * sizeof(__half);
    constexpr size_t smem_bytes_bm128 = 2 * ((128 + BN) * BK_SMEM) * sizeof(__half);
    size_t smem_bytes                  = use_bm64 ? smem_bytes_bm64 : smem_bytes_bm128;

    if (use_bm64) {
        gemm_fp16_kernel<64, 2><<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const __half*>(A), reinterpret_cast<const __half*>(B),
            reinterpret_cast<__half*>(D), M, N, K, alpha, beta);
        IMP_CUDA_CHECK_LAUNCH();
    } else {
        gemm_fp16_kernel<128, 2><<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const __half*>(A), reinterpret_cast<const __half*>(B),
            reinterpret_cast<__half*>(D), M, N, K, alpha, beta);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("gemm_capture_fp16_sm120: launch failed M=%d N=%d K=%d BM=%d smem=%zu: %s", M, N, K,
                      BM_v, smem_bytes, cudaGetErrorString(err));
        return false;
    }
    return true;
}

}  // namespace imp
