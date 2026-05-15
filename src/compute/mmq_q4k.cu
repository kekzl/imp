// Tiled MMQ kernel for Q4_K — direct Q4_K @ Q8_1 GEMM without dequant pass.
//
// Phase A: simple SMEM-tiled kernel, FP32 register accumulation, no
// double-buffering, no warp-mma. Numerics match ggml_mmvq_q4k (uses the
// same vec_dot_q4_K_q8_1 dp4a sequence per super-block).
//
// Tile defaults: TILE_M=32, TILE_N=64, TILE_K=256 (one Q4_K super-block).
// Tuned in Phase B (see q4k_mmvq_crossover_2026_05_15 memo).

#include "mmq_q4k.h"

#include <cstdint>
#include <cstdlib>
#include <cuda_fp16.h>

namespace imp {

// -------------------------------------------------------------------------
// Block types — identical layout to ggml_mmvq.cu (duplicated here to keep
// this kernel self-contained; the structs are fixed by the GGUF spec).
// -------------------------------------------------------------------------

namespace mmq_q4k_detail {

struct block_q4_K {
    half d;              // super-block scale
    half dmin;           // super-block min
    uint8_t scales[12];  // sub-block scales + mins (6-bit packed)
    uint8_t qs[128];     // 4-bit quants (256 values packed)
};
static_assert(sizeof(block_q4_K) == 144, "block_q4_K must be 144 bytes");

struct block_q8_1 {
    half d;         // scale
    half s;         // delta * sum(qs)
    int8_t qs[32];  // quantized activations
};
static_assert(sizeof(block_q8_1) == 36, "block_q8_1 must be 36 bytes");

static constexpr int QK_K = 256;        // Q4_K super-block size
static constexpr int QK8_1 = 32;        // Q8_1 sub-block size
static constexpr int Q8_PER_SUPER = 8;  // Q8_1 sub-blocks per Q4_K super-block
static constexpr int QR4_K = 2;
static constexpr int QI8_1 = 8;

// -------------------------------------------------------------------------
// Q8_1 quantization (FP16 input). Identical to ggml_mmvq's variant.
// -------------------------------------------------------------------------
__global__ void quantize_fp16_to_q8_1_kernel(const half* __restrict__ x,
                                             block_q8_1* __restrict__ y, int total_elems) {
    const int block_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_blocks = total_elems / QK8_1;
    if (block_id >= num_blocks) return;

    const half* xb = x + block_id * QK8_1;
    block_q8_1* yb = y + block_id;

    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < QK8_1; ++i) {
        float v = __half2float(xb[i]);
        amax = fmaxf(amax, fabsf(v));
    }

    const float d = amax / 127.0f;
    const float id = (d != 0.0f) ? (127.0f / amax) : 0.0f;

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < QK8_1; ++i) {
        float v = __half2float(xb[i]);
        int8_t q = static_cast<int8_t>(roundf(v * id));
        yb->qs[i] = q;
        sum += static_cast<float>(q);
    }
    yb->d = __float2half(d);
    yb->s = __float2half(d * sum);
}

// -------------------------------------------------------------------------
// vec_dot core — slice-level dot product used both by the existing mmvq
// kernel and this tiled kernel. Processes 16 weight nibbles × 16 Q8_1
// quants per call (2 outer × 2 inner dp4a). Operates on a single super-
// block of W (bq4_K) and the 2 corresponding Q8_1 sub-blocks (bq8_pair).
// iqs ∈ {0, 2, ..., 30}.
// -------------------------------------------------------------------------

__device__ __forceinline__ int dp4a_(int a, int b, int c) { return __dp4a(a, b, c); }

// Full-super-block dot product for one (m, n) pair. Processes all 256
// elements of one Q4_K super-block against the matching 8 Q8_1 sub-blocks.
//
// Hoists out of the original 16× kqs loop:
//   - bq4_K.scales unpack into aux/sc/m (4 unique bq8_offset values, not 16)
//   - d4, dm4 conversion (once, not 16×)
//   - d8[i] reads (4× per bo, was 32× per super-block)
//
// Chains dp4a accumulators across the 4 (iqs/2)%4 positions instead of
// summing into FP32 each call. dp4a max value per chain ≈ 256 K, well
// within int32 — 16 chained dp4a remains safe.
__device__ __forceinline__ float vec_dot_q4_K_super(const block_q4_K& bq4_K,
                                                    const block_q8_1* bq8_super) {
    const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4_K.scales);
    const float d4 = __half2float(bq4_K.d);
    const float dm4 = __half2float(bq4_K.dmin);

    float sumf = 0.0f;

#pragma unroll
    for (int bo_step = 0; bo_step < 4; ++bo_step) {
        const int bq8_offset = 2 * bo_step;  // 0, 2, 4, 6

        // ---- Per-bo hoist: aux/sc/m ------------------------------------
        uint16_t aux[2];
        const int j = bo_step;  // == bq8_offset / 2
        if (j < 2) {
            aux[0] = scales[j + 0] & 0x3f3f;
            aux[1] = scales[j + 2] & 0x3f3f;
        } else {
            aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
            aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
        }
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
        const uint8_t* m = sc + 2;

        // ---- Per-bo hoist: d8 of the two paired Q8_1 sub-blocks --------
        const block_q8_1* bq8_0 = bq8_super + bq8_offset + 0;
        const block_q8_1* bq8_1 = bq8_super + bq8_offset + 1;
        const float d8_0 = __half2float(bq8_0->d);
        const float d8_1 = __half2float(bq8_1->d);

        const int* q4_base = reinterpret_cast<const int*>(bq4_K.qs + 16 * bq8_offset);
        const int* q8_0 = reinterpret_cast<const int*>(bq8_0->qs);
        const int* q8_1 = reinterpret_cast<const int*>(bq8_1->qs);

        // ---- Chain dp4a accumulators over (iqs/2)%4 = 0..3 -------------
        int s1_0 = 0, s1_1 = 0, s2_0 = 0, s2_1 = 0;
#pragma unroll
        for (int qbo = 0; qbo < 4; ++qbo) {
            const int v0 = q4_base[qbo];          // bytes 16*bo + 4*qbo
            const int v1 = q4_base[qbo + 4];      // bytes 16*bo + 4*qbo + 16
            const int u00 = q8_0[qbo];
            const int u01 = q8_0[qbo + 4];
            const int u10 = q8_1[qbo];
            const int u11 = q8_1[qbo + 4];

            const int v0_lo = v0 & 0x0F0F0F0F;
            const int v1_lo = v1 & 0x0F0F0F0F;
            const int v0_hi = (v0 >> 4) & 0x0F0F0F0F;
            const int v1_hi = (v1 >> 4) & 0x0F0F0F0F;

            // i=0 contributions (low nibbles, sub-block bq8_offset+0)
            s1_0 = dp4a_(v1_lo, u01, dp4a_(v0_lo, u00, s1_0));
            s2_0 = dp4a_(0x01010101, u01, dp4a_(0x01010101, u00, s2_0));
            // i=1 contributions (high nibbles, sub-block bq8_offset+1)
            s1_1 = dp4a_(v1_hi, u11, dp4a_(v0_hi, u10, s1_1));
            s2_1 = dp4a_(0x01010101, u11, dp4a_(0x01010101, u10, s2_1));
        }

        // ---- Per-bo FP32 combine ---------------------------------------
        const float sd0 = static_cast<float>(s1_0) * static_cast<float>(sc[0]);
        const float sd1 = static_cast<float>(s1_1) * static_cast<float>(sc[1]);
        const float sm0 = static_cast<float>(s2_0) * static_cast<float>(m[0]);
        const float sm1 = static_cast<float>(s2_1) * static_cast<float>(m[1]);
        sumf += d4 * (d8_0 * sd0 + d8_1 * sd1) - dm4 * (d8_0 * sm0 + d8_1 * sm1);
    }

    return sumf;
}

// -------------------------------------------------------------------------
// Tiled kernel.
//
// Each block produces a TILE_M × TILE_N output tile. Iterates K in
// TILE_K=256 steps (one Q4_K super-block per step). For each K-step:
//   - cooperatively loads TILE_N super-blocks of W into shared memory
//   - cooperatively loads TILE_M × 8 sub-blocks of Q8_1 activations
//   - each thread accumulates REG_M × REG_N FP32 outputs by calling
//     vec_dot_q4_K_slice 16 times per output (iqs = 0, 2, ..., 30)
//
// Thread layout: (TILE_M / REG_M) × (TILE_N / REG_N) threads in a 1-D
// blockDim; M-major (m_thread = tid / N_THREADS).
// -------------------------------------------------------------------------

template <int TILE_M_, int TILE_N_, int THREADS_, int REG_M_, int REG_N_>
__global__ __launch_bounds__(THREADS_, 2)
void mmq_q4k_kernel(const block_q4_K* __restrict__ W,
                    const block_q8_1* __restrict__ x_q8,
                    half* __restrict__ y, int M, int N, int K) {
    constexpr int TILE_M = TILE_M_;
    constexpr int TILE_N = TILE_N_;
    constexpr int THREADS = THREADS_;
    constexpr int REG_M = REG_M_;
    constexpr int REG_N = REG_N_;
    constexpr int M_THREADS = TILE_M / REG_M;
    constexpr int N_THREADS = TILE_N / REG_N;
    static_assert(M_THREADS * N_THREADS == THREADS, "thread layout mismatch");

    const int tid = threadIdx.x;
    const int m_thread = tid / N_THREADS;
    const int n_thread = tid % N_THREADS;

    const int m_block = blockIdx.y * TILE_M;
    const int n_block = blockIdx.x * TILE_N;

    const int K_blocks = K / QK_K;
    const int x_sub_per_row = K / QK8_1;  // = K_blocks * 8

    // SMEM bank-conflict mitigation: sizeof(block_q4_K) = 144 B = 36 ints.
    // Stride 36 % 32 = 4 banks → 16 threads at varying n_local hit only
    // 8 unique banks (2-way conflict). Padding to 148 B (37 ints, gcd(37,32)=1)
    // gives 16 distinct banks for any 16-thread sweep across n_local.
    // ggml_block_q8_1 = 36 B = 9 ints already coprime-mod-32 → no padding needed.
    constexpr int W_STRIDE_INTS = (sizeof(block_q4_K) / 4) + 1;  // 37
    __shared__ int sW_raw[TILE_N * W_STRIDE_INTS];
    __shared__ block_q8_1 sX[TILE_M][Q8_PER_SUPER];

    auto sW_ptr = [&](int n_local) -> const block_q4_K* {
        return reinterpret_cast<const block_q4_K*>(&sW_raw[n_local * W_STRIDE_INTS]);
    };

    float acc[REG_M][REG_N];
#pragma unroll
    for (int i = 0; i < REG_M; ++i)
#pragma unroll
        for (int j = 0; j < REG_N; ++j) acc[i][j] = 0.0f;

    // Constants for cooperative copy in 4-byte words.
    constexpr int W_INTS_PER_BLOCK = sizeof(block_q4_K) / 4;        // 36
    constexpr int X_INTS_PER_SUB = sizeof(block_q8_1) / 4;          // 9
    constexpr int W_TOTAL_INTS = TILE_N * W_INTS_PER_BLOCK;         // 64*36 = 2304 (source side)
    constexpr int X_TOTAL_INTS = TILE_M * Q8_PER_SUPER * X_INTS_PER_SUB;  // 32*8*9 = 2304

    for (int kbx = 0; kbx < K_blocks; ++kbx) {
        // ---- Load TILE_N weight super-blocks into padded SMEM ----------
        // dst index = n_row * W_STRIDE_INTS + word (37-int stride, 36-int payload,
        // 1 int padding skipped per super-block).
        {
#pragma unroll
            for (int i = tid; i < W_TOTAL_INTS; i += THREADS) {
                const int n_row = i / W_INTS_PER_BLOCK;
                const int word = i % W_INTS_PER_BLOCK;
                const int n_global = n_block + n_row;
                int v = 0;
                if (n_global < N) {
                    const int* src = reinterpret_cast<const int*>(
                        &W[n_global * K_blocks + kbx]);
                    v = src[word];
                }
                sW_raw[n_row * W_STRIDE_INTS + word] = v;
            }
        }
        // ---- Load TILE_M × 8 Q8_1 sub-blocks ---------------------------
        {
            int* dst = reinterpret_cast<int*>(&sX[0][0]);
#pragma unroll
            for (int i = tid; i < X_TOTAL_INTS; i += THREADS) {
                const int m_row = i / (Q8_PER_SUPER * X_INTS_PER_SUB);
                const int rem = i % (Q8_PER_SUPER * X_INTS_PER_SUB);
                const int sub = rem / X_INTS_PER_SUB;
                const int word = rem % X_INTS_PER_SUB;
                const int m_global = m_block + m_row;
                if (m_global < M) {
                    const int* src = reinterpret_cast<const int*>(
                        &x_q8[m_global * x_sub_per_row + kbx * Q8_PER_SUPER + sub]);
                    dst[i] = src[word];
                } else {
                    dst[i] = 0;
                }
            }
        }
        __syncthreads();

        // ---- Compute REG_M × REG_N output sub-tile ---------------------
#pragma unroll
        for (int rm = 0; rm < REG_M; ++rm) {
            const int m_local = m_thread * REG_M + rm;
            const block_q8_1* bq8_super = &sX[m_local][0];
#pragma unroll
            for (int rn = 0; rn < REG_N; ++rn) {
                const int n_local = n_thread * REG_N + rn;
                acc[rm][rn] += vec_dot_q4_K_super(*sW_ptr(n_local), bq8_super);
            }
        }
        __syncthreads();
    }

    // ---- Store FP16 output --------------------------------------------
#pragma unroll
    for (int rm = 0; rm < REG_M; ++rm) {
        const int m_global = m_block + m_thread * REG_M + rm;
        if (m_global >= M) continue;
#pragma unroll
        for (int rn = 0; rn < REG_N; ++rn) {
            const int n_global = n_block + n_thread * REG_N + rn;
            if (n_global >= N) continue;
            y[m_global * N + n_global] = __float2half(acc[rm][rn]);
        }
    }
}

}  // namespace mmq_q4k_detail

// -------------------------------------------------------------------------
// Public dispatch
// -------------------------------------------------------------------------

// Template dispatch helper — bench can flip configs via env var.
namespace mmq_q4k_detail {

template <int TILE_M, int TILE_N, int REG_M, int REG_N>
static void launch_mmq_q4k(const void* W, const block_q8_1* x_q8, half* y,
                           int M, int N, int K, cudaStream_t stream) {
    constexpr int THREADS = (TILE_M / REG_M) * (TILE_N / REG_N);
    static_assert(THREADS == (TILE_M / REG_M) * (TILE_N / REG_N), "");
    dim3 block(THREADS);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    mmq_q4k_kernel<TILE_M, TILE_N, THREADS, REG_M, REG_N>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const block_q4_K*>(W), x_q8, y, M, N, K);
}

}  // namespace mmq_q4k_detail

void mmq_q4k(const void* W, const half* x, half* y, int M, int N, int K,
             void* scratch, size_t scratch_size, cudaStream_t stream) {
    using namespace mmq_q4k_detail;

    const int x_q8_blocks = M * (K / QK8_1);
    const size_t need = static_cast<size_t>(x_q8_blocks) * sizeof(block_q8_1);
    if (need > scratch_size || K % QK_K != 0) return;

    block_q8_1* x_q8 = reinterpret_cast<block_q8_1*>(scratch);
    {
        const int threads = 256;
        const int blocks = (x_q8_blocks + threads - 1) / threads;
        quantize_fp16_to_q8_1_kernel<<<blocks, threads, 0, stream>>>(x, x_q8, M * K);
    }

    // Tile sweep on Qwen3-32B Q4_K_M shapes picked <16, 32, 1, 1> as the
    // winner across the full M=32..512 range (2.0-2.4× mmvq throughout).
    // Smaller tiles win because the kernel is launch/parallelism-bound, not
    // SMEM-reuse-bound — more blocks → more wave concurrency on the 170-SM
    // GB202. IMP_MMQ_Q4K_TILE env var overrides for re-sweeping.
    const char* tile_env = std::getenv("IMP_MMQ_Q4K_TILE");
    const int tile = tile_env ? std::atoi(tile_env) : 1;
    switch (tile) {
        case 0:  // original Phase A default, large per-thread workload
            launch_mmq_q4k<32, 64, 2, 4>(W, x_q8, y, M, N, K, stream);  // 256 thr
            break;
        case 2:  // medium-wide
            launch_mmq_q4k<16, 64, 1, 2>(W, x_q8, y, M, N, K, stream);  // 512 thr
            break;
        case 3:  // big tile, more reuse
            launch_mmq_q4k<64, 128, 4, 4>(W, x_q8, y, M, N, K, stream);  // 512 thr
            break;
        case 4:  // square big
            launch_mmq_q4k<64, 64, 4, 4>(W, x_q8, y, M, N, K, stream);  // 256 thr
            break;
        case 1:
        default:
            launch_mmq_q4k<16, 32, 1, 1>(W, x_q8, y, M, N, K, stream);  // 512 thr
            break;
    }
}

}  // namespace imp
