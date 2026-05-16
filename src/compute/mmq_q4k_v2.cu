// mmq_q4k v2 Phase 1a — precompute per-sub-block affine scales for Q4_K weights.
//
// See mmq_q4k_v2.h for the rationale. This file implements only the
// preprocessing kernel; the HMMA matmul kernel itself comes in Phase 2.

#include "mmq_q4k_v2.h"

#include <cstdint>
#include <cstdlib>
#include <cuda_fp16.h>
#include <mma.h>

namespace imp {

namespace mmq_q4k_v2_detail {

// Q4_K super-block — fixed GGUF layout, duplicated from mmq_q4k.cu to keep
// this translation unit self-contained.
struct block_q4_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};
static_assert(sizeof(block_q4_K) == 144, "block_q4_K must be 144 bytes");

// Unpack the 8 (scale, min) 6-bit pairs from a Q4_K super-block's scales[12].
// Returns sc[0..7] and m[0..7] as uint8 (0..63).
//
// The packing scheme matches ggml's vec_dot_q4_K_q8_1: aux[0] / aux[1] pairs
// indexed by bo_step ∈ [0, 4) cover sub-blocks {2*bo_step, 2*bo_step+1}.
__device__ __forceinline__ void unpack_q4_K_scales_mins(
    const uint8_t* __restrict__ scales12, uint8_t sc_out[8], uint8_t m_out[8]) {
    const uint16_t* scales = reinterpret_cast<const uint16_t*>(scales12);

#pragma unroll
    for (int bo_step = 0; bo_step < 4; ++bo_step) {
        const int bq8_offset = 2 * bo_step;
        uint16_t aux[2];
        const int j = bo_step;
        if (j < 2) {
            aux[0] = scales[j + 0] & 0x3f3f;
            aux[1] = scales[j + 2] & 0x3f3f;
        } else {
            aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
            aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
        }
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
        const uint8_t* m = sc + 2;
        sc_out[bq8_offset + 0] = sc[0];
        sc_out[bq8_offset + 1] = sc[1];
        m_out [bq8_offset + 0] = m[0];
        m_out [bq8_offset + 1] = m[1];
    }
}

// One thread per Q4_K super-block: read 144 bytes, write 8 eff_scale + 8 eff_min.
__global__ void q4k_precompute_eff_scales_kernel(
    const block_q4_K* __restrict__ W, half* __restrict__ eff_scale,
    half* __restrict__ eff_min, int total_super_blocks, int K_blocks) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_super_blocks) return;

    const int n = tid / K_blocks;
    const int kbx = tid % K_blocks;

    const block_q4_K bq = W[n * K_blocks + kbx];

    uint8_t sc[8], m[8];
    unpack_q4_K_scales_mins(bq.scales, sc, m);

    const float d = __half2float(bq.d);
    const float dmin = __half2float(bq.dmin);

    half* es_row = &eff_scale[n * (K_blocks * 8) + kbx * 8];
    half* em_row = &eff_min  [n * (K_blocks * 8) + kbx * 8];

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        es_row[i] = __float2half(d    * static_cast<float>(sc[i]));
        em_row[i] = __float2half(dmin * static_cast<float>(m [i]));
    }
}

}  // namespace mmq_q4k_v2_detail

void q4k_precompute_eff_scales(const void* W, half* eff_scale_out,
                               half* eff_min_out, int N, int K,
                               cudaStream_t stream) {
    if (K % 256 != 0) return;
    using namespace mmq_q4k_v2_detail;

    const int K_blocks = K / 256;
    const int total = N * K_blocks;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    q4k_precompute_eff_scales_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const block_q4_K*>(W), eff_scale_out, eff_min_out, total,
        K_blocks);
}

namespace mmq_q4k_v2_detail {

// One thread per sub-block. Reads 32 nibbles from canonical qs[] (interleaved
// with the paired sub-block via low/high nibble) and writes 16 bytes of
// K-major packed nibbles.
__global__ void q4k_permute_kernel(const block_q4_K* __restrict__ W,
                                   uint8_t* __restrict__ eff_q4_out,
                                   int total_sub_blocks, int K_blocks) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_sub_blocks) return;

    // K_blocks * 8 sub-blocks per row of N.
    const int subs_per_row = K_blocks * 8;
    const int n = tid / subs_per_row;
    const int k_sub_in_row = tid % subs_per_row;
    const int k_super = k_sub_in_row / 8;
    const int s = k_sub_in_row % 8;

    const block_q4_K* bq = &W[n * K_blocks + k_super];
    const uint8_t* qs = bq->qs;

    // Canonical layout: bytes [(s/2)*32 .. (s/2)*32 + 32) contain the 32 nibbles
    // for sub-block s (low half if s is even, high half if s is odd) AND the 32
    // nibbles for its partner sub-block s^1 (in the other half).
    const int byte_base = (s >> 1) * 32;
    const bool use_high = (s & 1) != 0;

    uint8_t* out_row = &eff_q4_out[static_cast<size_t>(n) * subs_per_row * 16 +
                                   static_cast<size_t>(k_sub_in_row) * 16];

#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const uint8_t b1 = qs[byte_base + 2 * j + 0];
        const uint8_t b2 = qs[byte_base + 2 * j + 1];
        const uint8_t n1 = use_high ? ((b1 >> 4) & 0x0F) : (b1 & 0x0F);
        const uint8_t n2 = use_high ? ((b2 >> 4) & 0x0F) : (b2 & 0x0F);
        out_row[j] = static_cast<uint8_t>(n1 | (n2 << 4));
    }
}

}  // namespace mmq_q4k_v2_detail

void q4k_permute_to_v2_layout(const void* W, uint8_t* eff_q4_out, int N, int K,
                              cudaStream_t stream) {
    if (K % 256 != 0) return;
    using namespace mmq_q4k_v2_detail;

    const int K_blocks = K / 256;
    const int total_sub = N * K_blocks * 8;
    const int threads = 256;
    const int blocks = (total_sub + threads - 1) / threads;

    q4k_permute_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const block_q4_K*>(W), eff_q4_out, total_sub, K_blocks);
}

// ---------------------------------------------------------------------------
// Phase 2: HMMA GEMM kernel
//
// Single-buffered scaffold — Phase 3 will replace the per-K-step
// __syncthreads-bracketed load+dequant with a cp.async triple-buffer pipeline
// and move dequant into registers (saves the round-trip through sB).
//
// Per K-iteration (one Q4_K sub-block = 32 elements):
//   1. Cooperative cp-style global → SMEM load of FP16 activations.
//   2. Cooperative load of (eff_scale, eff_min, eff_q4) and dequant Q4→FP16
//      into sB. Each thread handles one n_local row × 16 K-positions
//      (8 packed Q4 bytes) — a single uint64 load per thread.
//   3. WMMA mma.sync.m16n8k16: 4 acc frags × 2 K-inner-frags = 8 MMAs per warp.
//   4. Epilogue: store FP32 accumulators as FP16 to y[M, N].
// ---------------------------------------------------------------------------

namespace mmq_q4k_v2_detail {

using namespace nvcuda;

constexpr int kBM = 64;
constexpr int kBN = 64;
constexpr int kBK = 32;
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr int kWarpsM = 2;
constexpr int kWarpsN = 2;
constexpr int kWarpsPerBlock = kWarpsM * kWarpsN;        // 4
constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;    // 128
constexpr int kWarpM = kBM / kWarpsM;                    // 32
constexpr int kWarpN = kBN / kWarpsN;                    // 32
constexpr int kFragsM = kWarpM / kWmmaM;                 // 2
constexpr int kFragsN = kWarpN / kWmmaN;                 // 2
constexpr int kFragsK = kBK / kWmmaK;                    // 2

__global__ void mmq_q4k_v2_kernel_scaffold(
    const half* __restrict__ A,           // [M, K]
    const uint8_t* __restrict__ eff_q4,   // [N, K/32, 16]
    const half* __restrict__ eff_scale,   // [N, K/32]
    const half* __restrict__ eff_min,     // [N, K/32]
    half* __restrict__ y,                 // [M, N]
    int M, int N, int K) {
    const int block_m = blockIdx.y * kBM;
    const int block_n = blockIdx.x * kBN;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int wm = warp / kWarpsN;
    const int wn = warp % kWarpsN;

    __shared__ half sA[kBM][kBK];       // 4 KB
    __shared__ half sB[kBN][kBK];       // 4 KB
    __shared__ half sScale[kBN];        // 128 B
    __shared__ half sMin[kBN];          // 128 B

    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> acc[kFragsM][kFragsN];
#pragma unroll
    for (int i = 0; i < kFragsM; ++i)
#pragma unroll
        for (int j = 0; j < kFragsN; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    const int K_subs = K / kBK;

    for (int kbx = 0; kbx < K_subs; ++kbx) {
        // ---- Load activations sA[BM][BK] (uint4 = 8 halves per chunk). -----
        // BM*BK = 2048 halves / 128 threads / 8 halves per chunk = 2 chunks/thread.
        {
            constexpr int kChunksPerThread = (kBM * kBK) / (8 * kThreadsPerBlock);  // 2
#pragma unroll
            for (int c = 0; c < kChunksPerThread; ++c) {
                int chunk = c * kThreadsPerBlock + tid;  // 0..255
                int row = chunk >> 2;                    // BK=32 → 4 chunks per row
                int col = (chunk & 3) << 3;              // 0, 8, 16, 24
                int g_row = block_m + row;
                int g_col = kbx * kBK + col;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (g_row < M) {
                    v = *reinterpret_cast<const uint4*>(&A[(int64_t)g_row * K + g_col]);
                }
                *reinterpret_cast<uint4*>(&sA[row][col]) = v;
            }
        }

        // ---- Load (eff_scale, eff_min) for current sub-block kbx ------------
        if (tid < kBN) {
            int n_global = block_n + tid;
            if (n_global < N) {
                int64_t off = (int64_t)n_global * K_subs + kbx;
                sScale[tid] = eff_scale[off];
                sMin[tid] = eff_min[off];
            } else {
                sScale[tid] = __float2half(0.0f);
                sMin[tid] = __float2half(0.0f);
            }
        }

        // ---- Load + dequant Q4 → FP16 into sB[BN][BK] ----------------------
        // 2 threads cover one (n_local) row × 16 K-positions each (8 packed
        // bytes = uint64 load). Layout matches Phase 1b output: byte j holds
        // K=2j (low nibble) and K=2j+1 (high nibble).
        {
            int n_local = tid >> 1;                  // 0..63
            int k_half = tid & 1;                    // 0 or 1
            int n_global = block_n + n_local;
            uint64_t packed = 0;
            if (n_global < N) {
                int64_t off = ((int64_t)n_global * K_subs + kbx) * 16 + k_half * 8;
                packed = *reinterpret_cast<const uint64_t*>(eff_q4 + off);
            }
            __syncthreads();  // wait for sScale/sMin writes above to be visible
            half scale = sScale[n_local];
            half neg_min = __hneg(sMin[n_local]);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint8_t b = (uint8_t)(packed >> (j * 8));
                half qlo = __float2half((float)(b & 0xF));
                half qhi = __float2half((float)((b >> 4) & 0xF));
                int k_pos = k_half * 16 + j * 2;
                sB[n_local][k_pos + 0] = __hfma(qlo, scale, neg_min);
                sB[n_local][k_pos + 1] = __hfma(qhi, scale, neg_min);
            }
        }
        __syncthreads();

        // ---- WMMA: 2 K-inner steps, 4 acc frags per warp -------------------
#pragma unroll
        for (int kk = 0; kk < kFragsK; ++kk) {
            wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> a_frag[kFragsM];
#pragma unroll
            for (int i = 0; i < kFragsM; ++i) {
                int a_row = wm * kWarpM + i * kWmmaM;
                wmma::load_matrix_sync(a_frag[i], &sA[a_row][kk * kWmmaK], kBK);
            }
            wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major> b_frag[kFragsN];
#pragma unroll
            for (int j = 0; j < kFragsN; ++j) {
                int b_row = wn * kWarpN + j * kWmmaN;
                wmma::load_matrix_sync(b_frag[j], &sB[b_row][kk * kWmmaK], kBK);
            }
#pragma unroll
            for (int i = 0; i < kFragsM; ++i)
#pragma unroll
                for (int j = 0; j < kFragsN; ++j)
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
        __syncthreads();
    }

    // ---- Epilogue: per-warp SMEM scratch, FP32 → FP16 store ----------------
    __shared__ float frag_smem[kWarpsPerBlock * kWmmaM * kWmmaN];
    float* warp_frag = frag_smem + warp * (kWmmaM * kWmmaN);
    int warp_base_m = block_m + wm * kWarpM;
    int warp_base_n = block_n + wn * kWarpN;

#pragma unroll
    for (int i = 0; i < kFragsM; ++i) {
#pragma unroll
        for (int j = 0; j < kFragsN; ++j) {
            wmma::store_matrix_sync(warp_frag, acc[i][j], kWmmaN, wmma::mem_row_major);
            __syncwarp();
            int frag_row0 = warp_base_m + i * kWmmaM;
            int frag_col0 = warp_base_n + j * kWmmaN;
            for (int t = 0; t < (kWmmaM * kWmmaN) / 32; ++t) {
                int idx = t * 32 + lane;
                int r = idx / kWmmaN;
                int c = idx % kWmmaN;
                int g_row = frag_row0 + r;
                int g_col = frag_col0 + c;
                if (g_row >= M || g_col >= N) continue;
                y[(int64_t)g_row * N + g_col] = __float2half(warp_frag[idx]);
            }
            __syncwarp();
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 3: register-only dequant via mma.sync.aligned.m16n8k16 + ldmatrix.x4
//
// Eliminates the sB SMEM round-trip from the scaffold. Each thread directly
// computes its m16n8k16 B-fragment from sQ4-packed nibbles + sScale/sMin
// inside its registers, then feeds the fragment to inline mma.sync PTX.
//
// Per warp per K-iter: 16 mma.sync calls (WRM=2 × WRN=4 × WRK=2).
//   - Phase 2's WMMA m16n16k16 unfolds to the same 16 m16n8k16 underneath,
//     so the compute is equivalent — Phase 3 just avoids materializing the
//     dequantized weights into SMEM at all.
//
// SMEM footprint vs scaffold: 4 KB sA + 1 KB sQ4 + 256 B sScale/sMin =
// ~5.3 KB (vs ~12 KB scaffold) → 6+ blocks/SM occupancy.
// ---------------------------------------------------------------------------

__device__ __forceinline__ half2 q4_pair_to_half2(uint8_t byte, half scale,
                                                  half neg_min) {
    int lo = byte & 0xF;
    int hi = (byte >> 4) & 0xF;
    half hlo = __hfma(__int2half_rn(lo), scale, neg_min);
    half hhi = __hfma(__int2half_rn(hi), scale, neg_min);
    return __halves2half2(hlo, hhi);
}

// ---------------------------------------------------------------------------
// cp.async helpers (Phase 7): GMEM → SMEM transfers that issue async then let
// the warp continue. Predicated form zero-fills SMEM when src_size=0.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void cp_async_cg_16(void* smem_ptr,
                                               const void* gmem_ptr,
                                               bool valid) {
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    int src_size = valid ? 16 : 0;
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s),
        "l"(gmem_ptr), "r"(src_size));
}
__device__ __forceinline__ void cp_async_ca_8(void* smem_ptr,
                                              const void* gmem_ptr,
                                              bool valid) {
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    int src_size = valid ? 8 : 0;
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(s),
        "l"(gmem_ptr), "r"(src_size));
}
__device__ __forceinline__ void cp_async_ca_4(void* smem_ptr,
                                              const void* gmem_ptr,
                                              bool valid) {
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    int src_size = valid ? 4 : 0;
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(s),
        "l"(gmem_ptr), "r"(src_size));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

template <int kP3BN>
__global__ void mmq_q4k_v2_kernel_p3_t(
    const half* __restrict__ A,           // [M, K]
    const uint8_t* __restrict__ eff_q4,   // [N, K/32, 16]
    const half* __restrict__ eff_scale,   // [N, K/32]
    const half* __restrict__ eff_min,     // [N, K/32]
    half* __restrict__ y,                 // [M, N]
    int M, int N, int K) {
    constexpr int kP3WarpN = kP3BN / kWarpsN;
    constexpr int kP3WRM = kWarpM / 16;        // 2
    constexpr int kP3WRN = kP3WarpN / 8;       // 4 (BN=64) or 8 (BN=128)
    constexpr int kP3WRK = kBK / 16;           // 2

    const int block_m = blockIdx.y * kBM;
    const int block_n = blockIdx.x * kP3BN;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int wm = warp / kWarpsN;
    const int wn = warp % kWarpsN;

    __shared__ __align__(16) half sA[kBM * kBK];                  // 4 KB
    __shared__ __align__(16) uint8_t sQ4[kP3BN * (kBK / 2)];      // BN=64→1KB, BN=128→2KB
    __shared__ half sScale[kP3BN];                                 // BN=64→128B, BN=128→256B
    __shared__ half sMin[kP3BN];                                   // same

    // FP32 accumulators — m16n8k16 places 4 fp32 per thread per (rm, rn).
    float acc[kP3WRM][kP3WRN][4];
#pragma unroll
    for (int i = 0; i < kP3WRM; ++i)
#pragma unroll
        for (int j = 0; j < kP3WRN; ++j)
#pragma unroll
            for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.0f;

    const int K_subs = K / kBK;

    for (int kbx = 0; kbx < K_subs; ++kbx) {
        // ---- Load sA (FP16 activations, uint4 chunks) -------------------
        {
            constexpr int kChunks = 2;  // 64*32/(8*128)
#pragma unroll
            for (int c = 0; c < kChunks; ++c) {
                int chunk = c * kThreadsPerBlock + tid;
                int row = chunk >> 2;
                int col = (chunk & 3) << 3;
                int g_row = block_m + row;
                int g_col = kbx * kBK + col;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (g_row < M) {
                    v = *reinterpret_cast<const uint4*>(&A[(int64_t)g_row * K + g_col]);
                }
                *reinterpret_cast<uint4*>(&sA[row * kBK + col]) = v;
            }
        }
        // ---- Load sQ4 (packed Q4 bytes) ---------------------------------
        // sQ4 = BN × 16 bytes. 128 threads × 8 B = 1 KB per pass; for BN=128
        // we run 2 passes.
        {
            constexpr int kSQ4Bytes = kP3BN * (kBK / 2);
            constexpr int kSQ4Chunks = kSQ4Bytes / (kThreadsPerBlock * 8);
#pragma unroll
            for (int c = 0; c < kSQ4Chunks; ++c) {
                int byte_idx = c * (kThreadsPerBlock * 8) + tid * 8;
                int n_local = byte_idx >> 4;           // / 16
                int byte_within_n = byte_idx & 0xF;    // % 16  → 0 or 8
                int n_global = block_n + n_local;
                uint64_t packed = 0;
                if (n_global < N) {
                    int64_t off = ((int64_t)n_global * K_subs + kbx) * 16 + byte_within_n;
                    packed = *reinterpret_cast<const uint64_t*>(eff_q4 + off);
                }
                *reinterpret_cast<uint64_t*>(&sQ4[n_local * 16 + byte_within_n]) = packed;
            }
        }
        // ---- Load sScale, sMin -----------------------------------------
        // BN ≤ kThreadsPerBlock (128); when BN=128, all threads participate.
        if (tid < kP3BN) {
            int n_global = block_n + tid;
            if (n_global < N) {
                int64_t off = (int64_t)n_global * K_subs + kbx;
                sScale[tid] = eff_scale[off];
                sMin[tid] = eff_min[off];
            } else {
                sScale[tid] = __float2half(0.0f);
                sMin[tid] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // ---- MMA loop ---------------------------------------------------
#pragma unroll
        for (int kk = 0; kk < kP3WRK; ++kk) {
            // ldmatrix.x4 — A frag per rm; cached across all rn iterations
            uint32_t a_frag[kP3WRM][4];
#pragma unroll
            for (int rm = 0; rm < kP3WRM; ++rm) {
                int m_start = wm * kWarpM + rm * 16;
                int k_start = kk * 16;
                int row = m_start + (lane & 0xF);          // L % 16
                int col = k_start + ((lane >> 4) << 3);    // (L / 16) * 8
                unsigned smem_addr =
                    __cvta_generic_to_shared(&sA[row * kBK + col]);
                asm volatile(
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                    "{%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag[rm][0]), "=r"(a_frag[rm][1]),
                      "=r"(a_frag[rm][2]), "=r"(a_frag[rm][3])
                    : "r"(smem_addr));
            }

            // B-frag layout (m16n8k16 col, fp16):
            //   groupID     = lane / 4    ∈ [0, 8)  → identifies N column
            //   lane_in_grp = lane % 4    ∈ [0, 4)  → identifies K row pair
            //   b0 half2 = B[(lane%4)*2 + {0, 1}, lane/4]
            //   b1 half2 = B[(lane%4)*2 + {8, 9}, lane/4]
            // For our W^T(K, N): B[k, n] = W_dequant[n, k] where
            //   n_block_local = wn*WARP_N + rn*8 + lane/4
            //   k_in_sub      = kk*16 + (lane%4)*2 + {0, 1, 8, 9}
            // The 4 nibbles come from sQ4[n_block_local][...] at byte indices
            //   byte_lo_idx = kk*8 + (lane%4)        — K offsets 0..1
            //   byte_hi_idx = kk*8 + (lane%4) + 4    — K offsets 8..9
#pragma unroll
            for (int rn = 0; rn < kP3WRN; ++rn) {
                int n_block_local = wn * kP3WarpN + rn * 8 + (lane >> 2);
                half scale = sScale[n_block_local];
                half neg_min = __hneg(sMin[n_block_local]);
                int byte_lo_idx = kk * 8 + (lane & 3);
                int byte_hi_idx = byte_lo_idx + 4;
                uint8_t byte_lo = sQ4[n_block_local * 16 + byte_lo_idx];
                uint8_t byte_hi = sQ4[n_block_local * 16 + byte_hi_idx];
                half2 b0 = q4_pair_to_half2(byte_lo, scale, neg_min);
                half2 b1 = q4_pair_to_half2(byte_hi, scale, neg_min);
                uint32_t b0_reg = *reinterpret_cast<uint32_t*>(&b0);
                uint32_t b1_reg = *reinterpret_cast<uint32_t*>(&b1);
#pragma unroll
                for (int rm = 0; rm < kP3WRM; ++rm) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%0, %1, %2, %3};\n"
                        : "+f"(acc[rm][rn][0]), "+f"(acc[rm][rn][1]),
                          "+f"(acc[rm][rn][2]), "+f"(acc[rm][rn][3])
                        : "r"(a_frag[rm][0]), "r"(a_frag[rm][1]),
                          "r"(a_frag[rm][2]), "r"(a_frag[rm][3]),
                          "r"(b0_reg), "r"(b1_reg));
                }
            }
        }
        __syncthreads();
    }

    // ---- Epilogue: store FP32 accumulators as FP16 -----------------------
    // m16n8 acc per thread holds 4 fp32 at:
    //   (groupID,     lane_in_grp*2 + 0..1)  and
    //   (groupID + 8, lane_in_grp*2 + 0..1)
    const int groupID = lane >> 2;
    const int lig = lane & 3;
#pragma unroll
    for (int rm = 0; rm < kP3WRM; ++rm) {
        int m_base = block_m + wm * kWarpM + rm * 16;
#pragma unroll
        for (int rn = 0; rn < kP3WRN; ++rn) {
            int n_base = block_n + wn * kP3WarpN + rn * 8;
            int rows[4] = {groupID, groupID, groupID + 8, groupID + 8};
            int cols[4] = {lig * 2 + 0, lig * 2 + 1, lig * 2 + 0, lig * 2 + 1};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int g_row = m_base + rows[i];
                int g_col = n_base + cols[i];
                if (g_row < M && g_col < N) {
                    y[(int64_t)g_row * N + g_col] = __float2half(acc[rm][rn][i]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 7a: cp.async-pipelined Q4_K v2 (2-stage double-buffer)
//
// Issues cp.async.cg / cp.async.ca for sA / sQ4 / sScale / sMin into a
// 2-stage SMEM buffer. Each main-loop iteration:
//   1. Issue cp.async for stage (kbx+1) % 2 (next K-iter's data).
//   2. cp.async.commit_group; cp.async.wait_group<1>  (wait for current stage).
//   3. __syncthreads; MMA compute on stage (kbx % 2).
//   4. __syncthreads; next iteration.
//
// The GMEM load for the NEXT K-iter executes in parallel with the compute on
// the CURRENT K-iter — eliminates the sync-then-load stall that bounded p3.
// ---------------------------------------------------------------------------
template <int kP3BN>
__global__ void mmq_q4k_v2_kernel_p4_t(
    const half* __restrict__ A, const uint8_t* __restrict__ eff_q4,
    const half* __restrict__ eff_scale, const half* __restrict__ eff_min,
    half* __restrict__ y, int M, int N, int K) {
    constexpr int kP3WarpN = kP3BN / kWarpsN;
    constexpr int kP3WRM = kWarpM / 16;
    constexpr int kP3WRN = kP3WarpN / 8;
    constexpr int kP3WRK = kBK / 16;
    constexpr int kStages = 2;

    const int block_m = blockIdx.y * kBM;
    const int block_n = blockIdx.x * kP3BN;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int wm = warp / kWarpsN;
    const int wn = warp % kWarpsN;

    __shared__ __align__(16) half sA[kStages][kBM * kBK];
    __shared__ __align__(16) uint8_t sQ4[kStages][kP3BN * (kBK / 2)];
    __shared__ __align__(16) half sScale[kStages][kP3BN];
    __shared__ __align__(16) half sMin[kStages][kP3BN];

    float acc[kP3WRM][kP3WRN][4];
#pragma unroll
    for (int i = 0; i < kP3WRM; ++i)
#pragma unroll
        for (int j = 0; j < kP3WRN; ++j)
#pragma unroll
            for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.0f;

    const int K_subs = K / kBK;

#define ISSUE_STAGE_LOAD(buf, kbx_val)                                          \
    do {                                                                        \
        constexpr int kAChunks = (kBM * kBK) / (8 * kThreadsPerBlock);          \
        _Pragma("unroll")                                                       \
        for (int c = 0; c < kAChunks; ++c) {                                    \
            int chunk = c * kThreadsPerBlock + tid;                             \
            int row = chunk >> 2;                                               \
            int col = (chunk & 3) << 3;                                         \
            int g_row = block_m + row;                                          \
            int g_col = (kbx_val) * kBK + col;                                  \
            const half* gptr = &A[(int64_t)g_row * K + g_col];                  \
            half* sptr = &sA[buf][row * kBK + col];                             \
            cp_async_cg_16(sptr, gptr, g_row < M);                              \
        }                                                                       \
        constexpr int kSQ4Bytes = kP3BN * (kBK / 2);                            \
        constexpr int kSQ4Chunks = kSQ4Bytes / (kThreadsPerBlock * 8);          \
        _Pragma("unroll")                                                       \
        for (int c = 0; c < kSQ4Chunks; ++c) {                                  \
            int byte_idx = c * (kThreadsPerBlock * 8) + tid * 8;                \
            int n_local = byte_idx >> 4;                                        \
            int byte_within_n = byte_idx & 0xF;                                 \
            int n_global = block_n + n_local;                                   \
            int64_t off = ((int64_t)n_global * K_subs + (kbx_val)) * 16 +       \
                          byte_within_n;                                        \
            const uint8_t* gptr = eff_q4 + off;                                 \
            uint8_t* sptr = &sQ4[buf][n_local * 16 + byte_within_n];            \
            cp_async_ca_8(sptr, gptr, n_global < N);                            \
        }                                                                       \
        /* sScale/sMin: strided in GMEM (different N's are K_subs apart), so   \
         * load synchronously per-thread; cp.async.ca-4 would read into the    \
         * next N's slot. These are 256-512 bytes total — negligible.          */ \
        if (tid < kP3BN) {                                                      \
            int n_global = block_n + tid;                                       \
            if (n_global < N) {                                                 \
                int64_t off = (int64_t)n_global * K_subs + (kbx_val);           \
                sScale[buf][tid] = eff_scale[off];                              \
                sMin[buf][tid] = eff_min[off];                                  \
            } else {                                                            \
                sScale[buf][tid] = __float2half(0.0f);                          \
                sMin[buf][tid] = __float2half(0.0f);                            \
            }                                                                   \
        }                                                                       \
    } while (0)

    // Prologue: issue stage 0.
    ISSUE_STAGE_LOAD(0, 0);
    cp_async_commit();

    for (int kbx = 0; kbx < K_subs; ++kbx) {
        const int cur = kbx & 1;  // kStages=2
        if (kbx + 1 < K_subs) {
            const int next_buf = (kbx + 1) & 1;
            ISSUE_STAGE_LOAD(next_buf, kbx + 1);
            cp_async_commit();
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();

        // ---- MMA on stage `cur` ---------------------------------------
#pragma unroll
        for (int kk = 0; kk < kP3WRK; ++kk) {
            uint32_t a_frag[kP3WRM][4];
#pragma unroll
            for (int rm = 0; rm < kP3WRM; ++rm) {
                int m_start = wm * kWarpM + rm * 16;
                int k_start = kk * 16;
                int row = m_start + (lane & 0xF);
                int col = k_start + ((lane >> 4) << 3);
                unsigned smem_addr =
                    __cvta_generic_to_shared(&sA[cur][row * kBK + col]);
                asm volatile(
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                    "{%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag[rm][0]), "=r"(a_frag[rm][1]),
                      "=r"(a_frag[rm][2]), "=r"(a_frag[rm][3])
                    : "r"(smem_addr));
            }
#pragma unroll
            for (int rn = 0; rn < kP3WRN; ++rn) {
                int n_block_local = wn * kP3WarpN + rn * 8 + (lane >> 2);
                half scale = sScale[cur][n_block_local];
                half neg_min = __hneg(sMin[cur][n_block_local]);
                int byte_lo_idx = kk * 8 + (lane & 3);
                int byte_hi_idx = byte_lo_idx + 4;
                uint8_t byte_lo = sQ4[cur][n_block_local * 16 + byte_lo_idx];
                uint8_t byte_hi = sQ4[cur][n_block_local * 16 + byte_hi_idx];
                half2 b0 = q4_pair_to_half2(byte_lo, scale, neg_min);
                half2 b1 = q4_pair_to_half2(byte_hi, scale, neg_min);
                uint32_t b0_reg = *reinterpret_cast<uint32_t*>(&b0);
                uint32_t b1_reg = *reinterpret_cast<uint32_t*>(&b1);
#pragma unroll
                for (int rm = 0; rm < kP3WRM; ++rm) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%0, %1, %2, %3};\n"
                        : "+f"(acc[rm][rn][0]), "+f"(acc[rm][rn][1]),
                          "+f"(acc[rm][rn][2]), "+f"(acc[rm][rn][3])
                        : "r"(a_frag[rm][0]), "r"(a_frag[rm][1]),
                          "r"(a_frag[rm][2]), "r"(a_frag[rm][3]),
                          "r"(b0_reg), "r"(b1_reg));
                }
            }
        }
        __syncthreads();
    }

    // Epilogue (same as p3).
    const int groupID = lane >> 2;
    const int lig = lane & 3;
#pragma unroll
    for (int rm = 0; rm < kP3WRM; ++rm) {
        int m_base = block_m + wm * kWarpM + rm * 16;
#pragma unroll
        for (int rn = 0; rn < kP3WRN; ++rn) {
            int n_base = block_n + wn * kP3WarpN + rn * 8;
            int rows[4] = {groupID, groupID, groupID + 8, groupID + 8};
            int cols[4] = {lig * 2 + 0, lig * 2 + 1, lig * 2 + 0, lig * 2 + 1};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int g_row = m_base + rows[i];
                int g_col = n_base + cols[i];
                if (g_row < M && g_col < N) {
                    y[(int64_t)g_row * N + g_col] = __float2half(acc[rm][rn][i]);
                }
            }
        }
    }
}

}  // namespace mmq_q4k_v2_detail

// Tile / pipeline crossover (Phase 4 + Phase 7a sweep, RTX 5090 / 170 SMs):
//
//   BN=64 + cp.async (p4)   wins at moderate M and grids ≤ ~250 blocks.
//                            cp.async hides GMEM load latency that the smaller
//                            block can't hide otherwise.
//   BN=128 + sync (p3)      wins at large M (≥ 512) and large N (FFN-up).
//                            Extra A-reuse amortizes per-block setup once the
//                            SM array is fully saturated; cp.async adds
//                            overhead without offsetting gain at this scale.
//
// Threshold bumped from 150 → 250 after the Phase 7 sweep: at M=256 N=5120
// (blocks_bn128=160) the BN=64-p4 path wins by ~12% over BN=128-p3 even
// though the BN=128 grid is "saturated" — A-reuse doesn't help enough.
static constexpr int kP3BlockSaturationThreshold = 250;

void mmq_q4k_v2(const half* x, const uint8_t* eff_q4, const half* eff_scale,
                const half* eff_min, half* y, int M, int N, int K,
                cudaStream_t stream) {
    using namespace mmq_q4k_v2_detail;
    if (K % kBK != 0 || M <= 0 || N <= 0) return;
    dim3 block(kThreadsPerBlock);

    // IMP_MMQ_Q4K_V2_SCAFFOLD=1 forces the WMMA scaffold (debug fallback).
    const char* scaffold_env = std::getenv("IMP_MMQ_Q4K_V2_SCAFFOLD");
    if (scaffold_env && std::atoi(scaffold_env) != 0) {
        dim3 grid((N + kBN - 1) / kBN, (M + kBM - 1) / kBM);
        mmq_q4k_v2_kernel_scaffold<<<grid, block, 0, stream>>>(
            x, eff_q4, eff_scale, eff_min, y, M, N, K);
        return;
    }

    // IMP_MMQ_Q4K_V2_BN forces a specific BN (64 or 128) for benching.
    const char* bn_env = std::getenv("IMP_MMQ_Q4K_V2_BN");
    int forced_bn = bn_env ? std::atoi(bn_env) : 0;

    // Phase 7a hybrid dispatch:
    //   BN=64  → p4 (cp.async double-buffer): wins +24-37% at M=32..256 when
    //            the SM array is under-saturated — overlap with compute hides
    //            GMEM load latency that p3 had to sync on.
    //   BN=128 → p3 (sync): wins by +5-10% — at BN=128 the grid is already
    //            SM-saturated, and the extra 2-stage SMEM + cp.async overhead
    //            outweigh the (now small) load-stall benefit.
    // Override the auto choice via IMP_MMQ_Q4K_V2_PIPELINE={0,1} (0 forces p3,
    // 1 forces p4 — useful for A/B benching).
    const char* pipeline_env = std::getenv("IMP_MMQ_Q4K_V2_PIPELINE");
    int forced_pipeline = pipeline_env ? std::atoi(pipeline_env) : -1;

    const int blocks_bn128 = ((M + kBM - 1) / kBM) * ((N + 127) / 128);
    const bool use_bn128 = (forced_bn == 128) ||
                           (forced_bn == 0 && N >= 128 &&
                            blocks_bn128 >= kP3BlockSaturationThreshold);
    if (use_bn128) {
        dim3 grid((N + 127) / 128, (M + kBM - 1) / kBM);
        const bool use_p4 = (forced_pipeline == 1);  // default p3 at BN=128
        if (use_p4) {
            mmq_q4k_v2_kernel_p4_t<128><<<grid, block, 0, stream>>>(
                x, eff_q4, eff_scale, eff_min, y, M, N, K);
        } else {
            mmq_q4k_v2_kernel_p3_t<128><<<grid, block, 0, stream>>>(
                x, eff_q4, eff_scale, eff_min, y, M, N, K);
        }
    } else {
        dim3 grid((N + 63) / 64, (M + kBM - 1) / kBM);
        const bool use_p4 = (forced_pipeline != 0);  // default p4 at BN=64
        if (use_p4) {
            mmq_q4k_v2_kernel_p4_t<64><<<grid, block, 0, stream>>>(
                x, eff_q4, eff_scale, eff_min, y, M, N, K);
        } else {
            mmq_q4k_v2_kernel_p3_t<64><<<grid, block, 0, stream>>>(
                x, eff_q4, eff_scale, eff_min, y, M, N, K);
        }
    }
}

// ===========================================================================
// Phase 6: Q5_K v2 (= Q4_K + 1-bit overlay per quant)
// ===========================================================================

namespace mmq_q4k_v2_detail {

struct block_q5_K {
    half d;              // super-block scale
    half dmin;           // super-block min
    uint8_t scales[12];  // SAME 6-bit packed layout as Q4_K
    uint8_t qh[32];      // high bits, byte b covers K=8b..8b+7
    uint8_t qs[128];     // low 4 bits, SAME packing as Q4_K
};
static_assert(sizeof(block_q5_K) == 176, "block_q5_K must be 176 bytes");

// Phase 1a for Q5_K: scales[12] layout is identical to Q4_K, so the unpack
// + affine math is reusable. Different block stride means we can't share the
// kernel directly — write a thin variant.
__global__ void q5k_precompute_eff_scales_kernel(
    const block_q5_K* __restrict__ W, half* __restrict__ eff_scale,
    half* __restrict__ eff_min, int total_super_blocks, int K_blocks) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_super_blocks) return;
    const int n = tid / K_blocks;
    const int kbx = tid % K_blocks;
    const block_q5_K bq = W[n * K_blocks + kbx];
    uint8_t sc[8], m[8];
    unpack_q4_K_scales_mins(bq.scales, sc, m);  // shared with Q4_K
    const float d = __half2float(bq.d);
    const float dmin = __half2float(bq.dmin);
    half* es_row = &eff_scale[n * (K_blocks * 8) + kbx * 8];
    half* em_row = &eff_min[n * (K_blocks * 8) + kbx * 8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        es_row[i] = __float2half(d * static_cast<float>(sc[i]));
        em_row[i] = __float2half(dmin * static_cast<float>(m[i]));
    }
}

// Phase 1b for Q5_K: same nibble permutation as Q4_K plus a straight copy of
// the 4 qh bytes that cover this sub-block's 32 K positions.
__global__ void q5k_permute_kernel(const block_q5_K* __restrict__ W,
                                   uint8_t* __restrict__ eff_ql,
                                   uint8_t* __restrict__ eff_qh,
                                   int total_sub_blocks, int K_blocks) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_sub_blocks) return;
    const int subs_per_row = K_blocks * 8;
    const int n = tid / subs_per_row;
    const int k_sub_in_row = tid % subs_per_row;
    const int k_super = k_sub_in_row / 8;
    const int s = k_sub_in_row % 8;
    const block_q5_K* bq = &W[n * K_blocks + k_super];

    // --- ql nibble permutation (identical to Q4_K Phase 1b) ---
    const uint8_t* qs = bq->qs;
    const int byte_base = (s >> 1) * 32;
    const bool use_high = (s & 1) != 0;
    uint8_t* ql_out = &eff_ql[(int64_t)n * subs_per_row * 16 +
                              (int64_t)k_sub_in_row * 16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const uint8_t b1 = qs[byte_base + 2 * j + 0];
        const uint8_t b2 = qs[byte_base + 2 * j + 1];
        const uint8_t n1 = use_high ? ((b1 >> 4) & 0x0F) : (b1 & 0x0F);
        const uint8_t n2 = use_high ? ((b2 >> 4) & 0x0F) : (b2 & 0x0F);
        ql_out[j] = static_cast<uint8_t>(n1 | (n2 << 4));
    }
    // --- qh: 4 bytes per sub-block, byte b → K_local=8b..8b+7 ---
    // qh in canonical layout: bit b of qh[i] = K = 8i + b (global within super).
    // For sub-block s, K_local ∈ [0, 32) maps to qh bytes [4s, 4s+4).
    uint8_t* qh_out = &eff_qh[(int64_t)n * subs_per_row * 4 +
                              (int64_t)k_sub_in_row * 4];
    *reinterpret_cast<uint32_t*>(qh_out) =
        *reinterpret_cast<const uint32_t*>(&bq->qh[s * 4]);
}

}  // namespace mmq_q4k_v2_detail

void q5k_precompute_eff_scales(const void* W, half* eff_scale_out,
                               half* eff_min_out, int N, int K,
                               cudaStream_t stream) {
    if (K % 256 != 0) return;
    using namespace mmq_q4k_v2_detail;
    const int K_blocks = K / 256;
    const int total = N * K_blocks;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    q5k_precompute_eff_scales_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const block_q5_K*>(W), eff_scale_out, eff_min_out,
        total, K_blocks);
}

void q5k_permute_to_v2_layout(const void* W, uint8_t* eff_ql_out,
                              uint8_t* eff_qh_out, int N, int K,
                              cudaStream_t stream) {
    if (K % 256 != 0) return;
    using namespace mmq_q4k_v2_detail;
    const int K_blocks = K / 256;
    const int total_sub = N * K_blocks * 8;
    const int threads = 256;
    const int blocks = (total_sub + threads - 1) / threads;
    q5k_permute_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const block_q5_K*>(W), eff_ql_out, eff_qh_out,
        total_sub, K_blocks);
}

namespace mmq_q4k_v2_detail {

// Q5_K HMMA kernel — clone of Q4_K Phase 3 with an extra qh path.
// Per thread per MMA: 2 ql bytes (4 low nibbles) + 1 qh byte (4 high bits at
// positions (lane%4)*2 + {0, 1} of that byte). The high bits OR into bit 4 of
// the corresponding nibble to form the 5-bit quant in [0, 31].
template <int kP3BN>
__global__ void mmq_q5k_v2_kernel_p3_t(
    const half* __restrict__ A,
    const uint8_t* __restrict__ eff_ql,
    const uint8_t* __restrict__ eff_qh,
    const half* __restrict__ eff_scale,
    const half* __restrict__ eff_min,
    half* __restrict__ y, int M, int N, int K) {
    constexpr int kP3WarpN = kP3BN / kWarpsN;
    constexpr int kP3WRM = kWarpM / 16;
    constexpr int kP3WRN = kP3WarpN / 8;
    constexpr int kP3WRK = kBK / 16;

    const int block_m = blockIdx.y * kBM;
    const int block_n = blockIdx.x * kP3BN;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int wm = warp / kWarpsN;
    const int wn = warp % kWarpsN;

    __shared__ __align__(16) half sA[kBM * kBK];
    __shared__ __align__(16) uint8_t sQL[kP3BN * (kBK / 2)];   // 16 bytes/sub-block
    __shared__ __align__(16) uint8_t sQH[kP3BN * 4];           // 4  bytes/sub-block
    __shared__ half sScale[kP3BN];
    __shared__ half sMin[kP3BN];

    float acc[kP3WRM][kP3WRN][4];
#pragma unroll
    for (int i = 0; i < kP3WRM; ++i)
#pragma unroll
        for (int j = 0; j < kP3WRN; ++j)
#pragma unroll
            for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.0f;

    const int K_subs = K / kBK;

    for (int kbx = 0; kbx < K_subs; ++kbx) {
        // ---- Load sA -----------------------------------------------------
        {
            constexpr int kChunks = 2;
#pragma unroll
            for (int c = 0; c < kChunks; ++c) {
                int chunk = c * kThreadsPerBlock + tid;
                int row = chunk >> 2;
                int col = (chunk & 3) << 3;
                int g_row = block_m + row;
                int g_col = kbx * kBK + col;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (g_row < M) {
                    v = *reinterpret_cast<const uint4*>(&A[(int64_t)g_row * K + g_col]);
                }
                *reinterpret_cast<uint4*>(&sA[row * kBK + col]) = v;
            }
        }
        // ---- Load sQL (16 bytes/row, total BN*16 bytes) -----------------
        {
            constexpr int kSqlBytes = kP3BN * (kBK / 2);
            constexpr int kSqlChunks = kSqlBytes / (kThreadsPerBlock * 8);
#pragma unroll
            for (int c = 0; c < kSqlChunks; ++c) {
                int byte_idx = c * (kThreadsPerBlock * 8) + tid * 8;
                int n_local = byte_idx >> 4;
                int byte_within_n = byte_idx & 0xF;
                int n_global = block_n + n_local;
                uint64_t packed = 0;
                if (n_global < N) {
                    int64_t off = ((int64_t)n_global * K_subs + kbx) * 16 + byte_within_n;
                    packed = *reinterpret_cast<const uint64_t*>(eff_ql + off);
                }
                *reinterpret_cast<uint64_t*>(&sQL[n_local * 16 + byte_within_n]) = packed;
            }
        }
        // ---- Load sQH (4 bytes/row, total BN*4 bytes) -------------------
        {
            // BN=64: 256 bytes / 128 threads / 4 = 2 threads per row.
            // BN=128: 512 bytes / 128 threads / 4 = 1 thread per row.
            if (tid < kP3BN) {
                int n_global = block_n + tid;
                uint32_t v = 0;
                if (n_global < N) {
                    int64_t off = ((int64_t)n_global * K_subs + kbx) * 4;
                    v = *reinterpret_cast<const uint32_t*>(eff_qh + off);
                }
                *reinterpret_cast<uint32_t*>(&sQH[tid * 4]) = v;
            }
        }
        // ---- Load sScale, sMin ------------------------------------------
        if (tid < kP3BN) {
            int n_global = block_n + tid;
            if (n_global < N) {
                int64_t off = (int64_t)n_global * K_subs + kbx;
                sScale[tid] = eff_scale[off];
                sMin[tid] = eff_min[off];
            } else {
                sScale[tid] = __float2half(0.0f);
                sMin[tid] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // ---- MMA inner loop ---------------------------------------------
#pragma unroll
        for (int kk = 0; kk < kP3WRK; ++kk) {
            uint32_t a_frag[kP3WRM][4];
#pragma unroll
            for (int rm = 0; rm < kP3WRM; ++rm) {
                int m_start = wm * kWarpM + rm * 16;
                int k_start = kk * 16;
                int row = m_start + (lane & 0xF);
                int col = k_start + ((lane >> 4) << 3);
                unsigned smem_addr =
                    __cvta_generic_to_shared(&sA[row * kBK + col]);
                asm volatile(
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                    "{%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag[rm][0]), "=r"(a_frag[rm][1]),
                      "=r"(a_frag[rm][2]), "=r"(a_frag[rm][3])
                    : "r"(smem_addr));
            }
#pragma unroll
            for (int rn = 0; rn < kP3WRN; ++rn) {
                int n_block_local = wn * kP3WarpN + rn * 8 + (lane >> 2);
                half scale = sScale[n_block_local];
                half neg_min = __hneg(sMin[n_block_local]);
                int byte_lo_idx = kk * 8 + (lane & 3);
                int byte_hi_idx = byte_lo_idx + 4;
                uint8_t byte_lo = sQL[n_block_local * 16 + byte_lo_idx];
                uint8_t byte_hi = sQL[n_block_local * 16 + byte_hi_idx];
                // qh: byte (2*kk + 0) holds 8 bits for K_local=8(2kk)..8(2kk)+7,
                // byte (2*kk + 1) holds bits for K_local 8..15 above that.
                // Thread holds K_local = (kk*16) + (lane%4)*2 + {0,1,8,9} →
                //   bit positions in qh bytes (2*kk + 0) and (2*kk + 1):
                //     (lane%4)*2 + 0  and  (lane%4)*2 + 1
                uint8_t qh_byte_lo = sQH[n_block_local * 4 + 2 * kk + 0];
                uint8_t qh_byte_hi = sQH[n_block_local * 4 + 2 * kk + 1];
                int bit_shift = (lane & 3) * 2;
                int hi_lo_a = (qh_byte_lo >> (bit_shift + 0)) & 1;  // K offset 0
                int hi_lo_b = (qh_byte_lo >> (bit_shift + 1)) & 1;  // K offset 1
                int hi_hi_a = (qh_byte_hi >> (bit_shift + 0)) & 1;  // K offset 8
                int hi_hi_b = (qh_byte_hi >> (bit_shift + 1)) & 1;  // K offset 9
                // Reconstruct 5-bit quants for the four K positions
                int q0 = (byte_lo & 0xF)        | (hi_lo_a << 4);
                int q1 = ((byte_lo >> 4) & 0xF) | (hi_lo_b << 4);
                int q2 = (byte_hi & 0xF)        | (hi_hi_a << 4);
                int q3 = ((byte_hi >> 4) & 0xF) | (hi_hi_b << 4);
                half h0 = __hfma(__int2half_rn(q0), scale, neg_min);
                half h1 = __hfma(__int2half_rn(q1), scale, neg_min);
                half h2 = __hfma(__int2half_rn(q2), scale, neg_min);
                half h3 = __hfma(__int2half_rn(q3), scale, neg_min);
                half2 b0 = __halves2half2(h0, h1);
                half2 b1 = __halves2half2(h2, h3);
                uint32_t b0_reg = *reinterpret_cast<uint32_t*>(&b0);
                uint32_t b1_reg = *reinterpret_cast<uint32_t*>(&b1);
#pragma unroll
                for (int rm = 0; rm < kP3WRM; ++rm) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%0, %1, %2, %3};\n"
                        : "+f"(acc[rm][rn][0]), "+f"(acc[rm][rn][1]),
                          "+f"(acc[rm][rn][2]), "+f"(acc[rm][rn][3])
                        : "r"(a_frag[rm][0]), "r"(a_frag[rm][1]),
                          "r"(a_frag[rm][2]), "r"(a_frag[rm][3]),
                          "r"(b0_reg), "r"(b1_reg));
                }
            }
        }
        __syncthreads();
    }

    // ---- Epilogue ----------------------------------------------------------
    const int groupID = lane >> 2;
    const int lig = lane & 3;
#pragma unroll
    for (int rm = 0; rm < kP3WRM; ++rm) {
        int m_base = block_m + wm * kWarpM + rm * 16;
#pragma unroll
        for (int rn = 0; rn < kP3WRN; ++rn) {
            int n_base = block_n + wn * kP3WarpN + rn * 8;
            int rows[4] = {groupID, groupID, groupID + 8, groupID + 8};
            int cols[4] = {lig * 2 + 0, lig * 2 + 1, lig * 2 + 0, lig * 2 + 1};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int g_row = m_base + rows[i];
                int g_col = n_base + cols[i];
                if (g_row < M && g_col < N) {
                    y[(int64_t)g_row * N + g_col] = __float2half(acc[rm][rn][i]);
                }
            }
        }
    }
}

// ---- Q5_K p4: cp.async pipelined variant (Phase 7b) ---------------------
// Same hybrid dispatch rule as Q4_K: p4 at BN=64, p3 at BN=128.
template <int kP3BN>
__global__ void mmq_q5k_v2_kernel_p4_t(
    const half* __restrict__ A, const uint8_t* __restrict__ eff_ql,
    const uint8_t* __restrict__ eff_qh, const half* __restrict__ eff_scale,
    const half* __restrict__ eff_min, half* __restrict__ y, int M, int N,
    int K) {
    constexpr int kP3WarpN = kP3BN / kWarpsN;
    constexpr int kP3WRM = kWarpM / 16;
    constexpr int kP3WRN = kP3WarpN / 8;
    constexpr int kP3WRK = kBK / 16;
    constexpr int kStages = 2;

    const int block_m = blockIdx.y * kBM;
    const int block_n = blockIdx.x * kP3BN;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int wm = warp / kWarpsN;
    const int wn = warp % kWarpsN;

    __shared__ __align__(16) half sA[kStages][kBM * kBK];
    __shared__ __align__(16) uint8_t sQL[kStages][kP3BN * (kBK / 2)];
    __shared__ __align__(16) uint8_t sQH[kStages][kP3BN * 4];
    __shared__ half sScale[kStages][kP3BN];
    __shared__ half sMin[kStages][kP3BN];

    float acc[kP3WRM][kP3WRN][4];
#pragma unroll
    for (int i = 0; i < kP3WRM; ++i)
#pragma unroll
        for (int j = 0; j < kP3WRN; ++j)
#pragma unroll
            for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.0f;

    const int K_subs = K / kBK;

#define ISSUE_STAGE_LOAD_Q5(buf, kbx_val)                                       \
    do {                                                                        \
        constexpr int kAChunks = (kBM * kBK) / (8 * kThreadsPerBlock);          \
        _Pragma("unroll")                                                       \
        for (int c = 0; c < kAChunks; ++c) {                                    \
            int chunk = c * kThreadsPerBlock + tid;                             \
            int row = chunk >> 2;                                               \
            int col = (chunk & 3) << 3;                                         \
            int g_row = block_m + row;                                          \
            int g_col = (kbx_val) * kBK + col;                                  \
            const half* gptr = &A[(int64_t)g_row * K + g_col];                  \
            half* sptr = &sA[buf][row * kBK + col];                             \
            cp_async_cg_16(sptr, gptr, g_row < M);                              \
        }                                                                       \
        constexpr int kSQLBytes = kP3BN * (kBK / 2);                            \
        constexpr int kSQLChunks = kSQLBytes / (kThreadsPerBlock * 8);          \
        _Pragma("unroll")                                                       \
        for (int c = 0; c < kSQLChunks; ++c) {                                  \
            int byte_idx = c * (kThreadsPerBlock * 8) + tid * 8;                \
            int n_local = byte_idx >> 4;                                        \
            int byte_within_n = byte_idx & 0xF;                                 \
            int n_global = block_n + n_local;                                   \
            int64_t off = ((int64_t)n_global * K_subs + (kbx_val)) * 16 +       \
                          byte_within_n;                                        \
            cp_async_ca_8(&sQL[buf][n_local * 16 + byte_within_n],              \
                          eff_ql + off, n_global < N);                          \
        }                                                                       \
        /* sQH (4 bytes/row × BN) — strided in GMEM along N, sync load.        */ \
        if (tid < kP3BN) {                                                      \
            int n_global = block_n + tid;                                       \
            uint32_t qh = 0;                                                    \
            if (n_global < N) {                                                 \
                int64_t off = ((int64_t)n_global * K_subs + (kbx_val)) * 4;     \
                qh = *reinterpret_cast<const uint32_t*>(eff_qh + off);          \
            }                                                                   \
            *reinterpret_cast<uint32_t*>(&sQH[buf][tid * 4]) = qh;              \
        }                                                                       \
        if (tid < kP3BN) {                                                      \
            int n_global = block_n + tid;                                       \
            if (n_global < N) {                                                 \
                int64_t off = (int64_t)n_global * K_subs + (kbx_val);           \
                sScale[buf][tid] = eff_scale[off];                              \
                sMin[buf][tid] = eff_min[off];                                  \
            } else {                                                            \
                sScale[buf][tid] = __float2half(0.0f);                          \
                sMin[buf][tid] = __float2half(0.0f);                            \
            }                                                                   \
        }                                                                       \
    } while (0)

    ISSUE_STAGE_LOAD_Q5(0, 0);
    cp_async_commit();

    for (int kbx = 0; kbx < K_subs; ++kbx) {
        const int cur = kbx & 1;
        if (kbx + 1 < K_subs) {
            const int next_buf = (kbx + 1) & 1;
            ISSUE_STAGE_LOAD_Q5(next_buf, kbx + 1);
            cp_async_commit();
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < kP3WRK; ++kk) {
            uint32_t a_frag[kP3WRM][4];
#pragma unroll
            for (int rm = 0; rm < kP3WRM; ++rm) {
                int m_start = wm * kWarpM + rm * 16;
                int k_start = kk * 16;
                int row = m_start + (lane & 0xF);
                int col = k_start + ((lane >> 4) << 3);
                unsigned smem_addr =
                    __cvta_generic_to_shared(&sA[cur][row * kBK + col]);
                asm volatile(
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                    "{%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag[rm][0]), "=r"(a_frag[rm][1]),
                      "=r"(a_frag[rm][2]), "=r"(a_frag[rm][3])
                    : "r"(smem_addr));
            }
#pragma unroll
            for (int rn = 0; rn < kP3WRN; ++rn) {
                int n_block_local = wn * kP3WarpN + rn * 8 + (lane >> 2);
                half scale = sScale[cur][n_block_local];
                half neg_min = __hneg(sMin[cur][n_block_local]);
                int byte_lo_idx = kk * 8 + (lane & 3);
                int byte_hi_idx = byte_lo_idx + 4;
                uint8_t byte_lo = sQL[cur][n_block_local * 16 + byte_lo_idx];
                uint8_t byte_hi = sQL[cur][n_block_local * 16 + byte_hi_idx];
                uint8_t qh_byte_lo = sQH[cur][n_block_local * 4 + 2 * kk + 0];
                uint8_t qh_byte_hi = sQH[cur][n_block_local * 4 + 2 * kk + 1];
                int bit_shift = (lane & 3) * 2;
                int hi_lo_a = (qh_byte_lo >> (bit_shift + 0)) & 1;
                int hi_lo_b = (qh_byte_lo >> (bit_shift + 1)) & 1;
                int hi_hi_a = (qh_byte_hi >> (bit_shift + 0)) & 1;
                int hi_hi_b = (qh_byte_hi >> (bit_shift + 1)) & 1;
                int q0 = (byte_lo & 0xF) | (hi_lo_a << 4);
                int q1 = ((byte_lo >> 4) & 0xF) | (hi_lo_b << 4);
                int q2 = (byte_hi & 0xF) | (hi_hi_a << 4);
                int q3 = ((byte_hi >> 4) & 0xF) | (hi_hi_b << 4);
                half h0 = __hfma(__int2half_rn(q0), scale, neg_min);
                half h1 = __hfma(__int2half_rn(q1), scale, neg_min);
                half h2 = __hfma(__int2half_rn(q2), scale, neg_min);
                half h3 = __hfma(__int2half_rn(q3), scale, neg_min);
                half2 b0 = __halves2half2(h0, h1);
                half2 b1 = __halves2half2(h2, h3);
                uint32_t b0_reg = *reinterpret_cast<uint32_t*>(&b0);
                uint32_t b1_reg = *reinterpret_cast<uint32_t*>(&b1);
#pragma unroll
                for (int rm = 0; rm < kP3WRM; ++rm) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%0, %1, %2, %3};\n"
                        : "+f"(acc[rm][rn][0]), "+f"(acc[rm][rn][1]),
                          "+f"(acc[rm][rn][2]), "+f"(acc[rm][rn][3])
                        : "r"(a_frag[rm][0]), "r"(a_frag[rm][1]),
                          "r"(a_frag[rm][2]), "r"(a_frag[rm][3]),
                          "r"(b0_reg), "r"(b1_reg));
                }
            }
        }
        __syncthreads();
    }

#undef ISSUE_STAGE_LOAD_Q5

    const int groupID = lane >> 2;
    const int lig = lane & 3;
#pragma unroll
    for (int rm = 0; rm < kP3WRM; ++rm) {
        int m_base = block_m + wm * kWarpM + rm * 16;
#pragma unroll
        for (int rn = 0; rn < kP3WRN; ++rn) {
            int n_base = block_n + wn * kP3WarpN + rn * 8;
            int rows[4] = {groupID, groupID, groupID + 8, groupID + 8};
            int cols[4] = {lig * 2 + 0, lig * 2 + 1, lig * 2 + 0, lig * 2 + 1};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int g_row = m_base + rows[i];
                int g_col = n_base + cols[i];
                if (g_row < M && g_col < N) {
                    y[(int64_t)g_row * N + g_col] = __float2half(acc[rm][rn][i]);
                }
            }
        }
    }
}

}  // namespace mmq_q4k_v2_detail

void mmq_q5k_v2(const half* x, const uint8_t* eff_ql, const uint8_t* eff_qh,
                const half* eff_scale, const half* eff_min, half* y, int M,
                int N, int K, cudaStream_t stream) {
    using namespace mmq_q4k_v2_detail;
    if (K % kBK != 0 || M <= 0 || N <= 0) return;
    dim3 block(kThreadsPerBlock);
    const char* pipeline_env = std::getenv("IMP_MMQ_Q4K_V2_PIPELINE");
    int forced_pipeline = pipeline_env ? std::atoi(pipeline_env) : -1;

    const int blocks_bn128 = ((M + kBM - 1) / kBM) * ((N + 127) / 128);
    const bool use_bn128 =
        (N >= 128) && (blocks_bn128 >= kP3BlockSaturationThreshold);
    if (use_bn128) {
        dim3 grid((N + 127) / 128, (M + kBM - 1) / kBM);
        const bool use_p4 = (forced_pipeline == 1);  // default p3 at BN=128
        if (use_p4) {
            mmq_q5k_v2_kernel_p4_t<128><<<grid, block, 0, stream>>>(
                x, eff_ql, eff_qh, eff_scale, eff_min, y, M, N, K);
        } else {
            mmq_q5k_v2_kernel_p3_t<128><<<grid, block, 0, stream>>>(
                x, eff_ql, eff_qh, eff_scale, eff_min, y, M, N, K);
        }
    } else {
        dim3 grid((N + 63) / 64, (M + kBM - 1) / kBM);
        const bool use_p4 = (forced_pipeline != 0);  // default p4 at BN=64
        if (use_p4) {
            mmq_q5k_v2_kernel_p4_t<64><<<grid, block, 0, stream>>>(
                x, eff_ql, eff_qh, eff_scale, eff_min, y, M, N, K);
        } else {
            mmq_q5k_v2_kernel_p3_t<64><<<grid, block, 0, stream>>>(
                x, eff_ql, eff_qh, eff_scale, eff_min, y, M, N, K);
        }
    }
}

// ===========================================================================
// Phase 6b: Q6_K v2 (BK=16 sub-blocks, signed quants, int8 scales)
// ===========================================================================

namespace mmq_q4k_v2_detail {

struct block_q6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};
static_assert(sizeof(block_q6_K) == 210, "block_q6_K must be 210 bytes");

// GGML Q6_K bit decoding — for element i ∈ [0, 256) of a super-block,
// returns q_unsigned ∈ [0, 63]. Mirrors `dequant_q6k_element` in
// dequant_gpu.cu (kept here so the kernel doesn't need that include).
__device__ __forceinline__ int q6k_decode_unsigned(const block_q6_K* bq,
                                                   int i) {
    int group = i >> 7;            // 0 or 1
    int within = i & 127;          // 0..127
    int quad = within >> 5;        // 0..3
    int l = within & 31;           // 0..31
    int ql_idx = (group << 6) + ((quad & 1) << 5) + l;
    int qh_idx = (group << 5) + l;
    uint8_t ql_byte = bq->ql[ql_idx];
    uint8_t low4 =
        (quad >= 2) ? ((ql_byte >> 4) & 0xFu) : (ql_byte & 0xFu);
    uint8_t high2 = (bq->qh[qh_idx] >> (quad * 2)) & 0x3u;
    return static_cast<int>((high2 << 4) | low4);  // [0, 63]
}

// One CTA per super-block: 256 threads dequant + write the byte stream,
// plus 16 threads compute eff_scale/eff_min for the 16 sub-blocks.
__global__ void q6k_prepare_kernel(const block_q6_K* __restrict__ W,
                                   uint8_t* __restrict__ eff_q6,
                                   half* __restrict__ eff_scale,
                                   half* __restrict__ eff_min, int N,
                                   int K_blocks) {
    int n = blockIdx.y;
    int super = blockIdx.x;
    int tid = threadIdx.x;
    const block_q6_K* bq = &W[n * K_blocks + super];

    // Per-element byte expansion.
    int q = q6k_decode_unsigned(bq, tid);
    int K_global = super * 256 + tid;
    eff_q6[(int64_t)n * (K_blocks * 256) + K_global] =
        static_cast<uint8_t>(q);

    // Per-sub-block scale (16 entries per super-block).
    if (tid < 16) {
        float scale = __half2float(bq->d) * static_cast<float>(bq->scales[tid]);
        half es = __float2half(scale);
        half em = __float2half(32.0f * scale);
        int eff_idx = n * (K_blocks * 16) + super * 16 + tid;
        eff_scale[eff_idx] = es;
        eff_min[eff_idx] = em;
    }
}

constexpr int kQ6BK = 16;   // one Q6_K sub-block per K-iter
constexpr int kQ6WRK = 1;   // one m16n8k16 per K-step (BK = MMA_K)

template <int kP3BN>
__global__ void mmq_q6k_v2_kernel_p3_t(
    const half* __restrict__ A, const uint8_t* __restrict__ eff_q6,
    const half* __restrict__ eff_scale, const half* __restrict__ eff_min,
    half* __restrict__ y, int M, int N, int K) {
    constexpr int kP3WarpN = kP3BN / kWarpsN;
    constexpr int kP3WRM = kWarpM / 16;     // 2
    constexpr int kP3WRN = kP3WarpN / 8;    // 4 (BN=64) or 8 (BN=128)

    const int block_m = blockIdx.y * kBM;
    const int block_n = blockIdx.x * kP3BN;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int wm = warp / kWarpsN;
    const int wn = warp % kWarpsN;

    __shared__ __align__(16) half sA[kBM * kQ6BK];         // 64 × 16 = 2 KB
    __shared__ __align__(16) uint8_t sQ6[kP3BN * kQ6BK];   // BN=64 → 1 KB
    __shared__ half sScale[kP3BN];
    __shared__ half sMin[kP3BN];

    float acc[kP3WRM][kP3WRN][4];
#pragma unroll
    for (int i = 0; i < kP3WRM; ++i)
#pragma unroll
        for (int j = 0; j < kP3WRN; ++j)
#pragma unroll
            for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.0f;

    const int K_subs = K / kQ6BK;

    for (int kbx = 0; kbx < K_subs; ++kbx) {
        // ---- Load sA (64×16 halves = 1024 halves = 128 uint4 chunks) ----
        {
            // 1 chunk/thread.
            int chunk = tid;
            int row = chunk >> 1;             // BK=16 → 2 chunks per row
            int col = (chunk & 1) << 3;       // 0 or 8
            int g_row = block_m + row;
            int g_col = kbx * kQ6BK + col;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (g_row < M) {
                v = *reinterpret_cast<const uint4*>(&A[(int64_t)g_row * K + g_col]);
            }
            *reinterpret_cast<uint4*>(&sA[row * kQ6BK + col]) = v;
        }
        // ---- Load sQ6 (BN × 16 bytes) ----
        {
            constexpr int kSQ6Bytes = kP3BN * kQ6BK;
            constexpr int kSQ6Chunks = kSQ6Bytes / (kThreadsPerBlock * 8);
#pragma unroll
            for (int c = 0; c < kSQ6Chunks; ++c) {
                int byte_idx = c * (kThreadsPerBlock * 8) + tid * 8;
                int n_local = byte_idx >> 4;
                int byte_within_n = byte_idx & 0xF;   // 0 or 8
                int n_global = block_n + n_local;
                uint64_t packed = 0;
                if (n_global < N) {
                    int64_t off =
                        (int64_t)n_global * K + kbx * kQ6BK + byte_within_n;
                    packed = *reinterpret_cast<const uint64_t*>(eff_q6 + off);
                }
                *reinterpret_cast<uint64_t*>(&sQ6[n_local * kQ6BK + byte_within_n]) =
                    packed;
            }
        }
        // ---- Load sScale, sMin ------------------------------------------
        if (tid < kP3BN) {
            int n_global = block_n + tid;
            if (n_global < N) {
                int64_t off = (int64_t)n_global * K_subs + kbx;
                sScale[tid] = eff_scale[off];
                sMin[tid] = eff_min[off];
            } else {
                sScale[tid] = __float2half(0.0f);
                sMin[tid] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // ---- One m16n8k16 MMA per K-iter (WRK=1) ------------------------
        uint32_t a_frag[kP3WRM][4];
#pragma unroll
        for (int rm = 0; rm < kP3WRM; ++rm) {
            int m_start = wm * kWarpM + rm * 16;
            int row = m_start + (lane & 0xF);
            int col = ((lane >> 4) << 3);  // 0 or 8
            unsigned smem_addr =
                __cvta_generic_to_shared(&sA[row * kQ6BK + col]);
            asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                "{%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag[rm][0]), "=r"(a_frag[rm][1]),
                  "=r"(a_frag[rm][2]), "=r"(a_frag[rm][3])
                : "r"(smem_addr));
        }
#pragma unroll
        for (int rn = 0; rn < kP3WRN; ++rn) {
            int n_block_local = wn * kP3WarpN + rn * 8 + (lane >> 2);
            half scale = sScale[n_block_local];
            half neg_min = __hneg(sMin[n_block_local]);
            // Thread's 4 K positions within this MMA's K-range [0, 16):
            //   (lane%4)*2 + {0, 1, 8, 9}
            int b_off_lo = (lane & 3) * 2;          // K offsets 0, 1
            int b_off_hi = b_off_lo + 8;            // K offsets 8, 9
            uint16_t bytes_lo = *reinterpret_cast<uint16_t*>(
                &sQ6[n_block_local * kQ6BK + b_off_lo]);
            uint16_t bytes_hi = *reinterpret_cast<uint16_t*>(
                &sQ6[n_block_local * kQ6BK + b_off_hi]);
            int q0 = bytes_lo & 0xFF;          // q_unsigned for K offset 0
            int q1 = (bytes_lo >> 8) & 0xFF;   //                   1
            int q2 = bytes_hi & 0xFF;          //                   8
            int q3 = (bytes_hi >> 8) & 0xFF;   //                   9
            // w = eff_scale · q - eff_min  (eff_min = 32 · eff_scale, so
            // this absorbs the (q - 32) shift)
            half h0 = __hfma(__int2half_rn(q0), scale, neg_min);
            half h1 = __hfma(__int2half_rn(q1), scale, neg_min);
            half h2 = __hfma(__int2half_rn(q2), scale, neg_min);
            half h3 = __hfma(__int2half_rn(q3), scale, neg_min);
            half2 b0 = __halves2half2(h0, h1);
            half2 b1 = __halves2half2(h2, h3);
            uint32_t b0_reg = *reinterpret_cast<uint32_t*>(&b0);
            uint32_t b1_reg = *reinterpret_cast<uint32_t*>(&b1);
#pragma unroll
            for (int rm = 0; rm < kP3WRM; ++rm) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3};\n"
                    : "+f"(acc[rm][rn][0]), "+f"(acc[rm][rn][1]),
                      "+f"(acc[rm][rn][2]), "+f"(acc[rm][rn][3])
                    : "r"(a_frag[rm][0]), "r"(a_frag[rm][1]),
                      "r"(a_frag[rm][2]), "r"(a_frag[rm][3]),
                      "r"(b0_reg), "r"(b1_reg));
            }
        }
        __syncthreads();
    }

    // ---- Epilogue ----------------------------------------------------------
    const int groupID = lane >> 2;
    const int lig = lane & 3;
#pragma unroll
    for (int rm = 0; rm < kP3WRM; ++rm) {
        int m_base = block_m + wm * kWarpM + rm * 16;
#pragma unroll
        for (int rn = 0; rn < kP3WRN; ++rn) {
            int n_base = block_n + wn * kP3WarpN + rn * 8;
            int rows[4] = {groupID, groupID, groupID + 8, groupID + 8};
            int cols[4] = {lig * 2 + 0, lig * 2 + 1, lig * 2 + 0, lig * 2 + 1};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int g_row = m_base + rows[i];
                int g_col = n_base + cols[i];
                if (g_row < M && g_col < N) {
                    y[(int64_t)g_row * N + g_col] = __float2half(acc[rm][rn][i]);
                }
            }
        }
    }
}

// ---- Q6_K p4: cp.async pipelined variant (Phase 7b) ---------------------
// BK=16, WRK=1 — one m16n8k16 MMA per K-iter. Sub-block size 16 elements,
// 1 byte/element from eff_q6 (q_unsigned ∈ [0, 63]).
template <int kP3BN>
__global__ void mmq_q6k_v2_kernel_p4_t(
    const half* __restrict__ A, const uint8_t* __restrict__ eff_q6,
    const half* __restrict__ eff_scale, const half* __restrict__ eff_min,
    half* __restrict__ y, int M, int N, int K) {
    constexpr int kP3WarpN = kP3BN / kWarpsN;
    constexpr int kP3WRM = kWarpM / 16;
    constexpr int kP3WRN = kP3WarpN / 8;
    constexpr int kStages = 2;

    const int block_m = blockIdx.y * kBM;
    const int block_n = blockIdx.x * kP3BN;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int wm = warp / kWarpsN;
    const int wn = warp % kWarpsN;

    __shared__ __align__(16) half sA[kStages][kBM * kQ6BK];        // 64×16 = 2 KB/stage
    __shared__ __align__(16) uint8_t sQ6[kStages][kP3BN * kQ6BK];  // BN=64 → 1 KB/stage
    __shared__ half sScale[kStages][kP3BN];
    __shared__ half sMin[kStages][kP3BN];

    float acc[kP3WRM][kP3WRN][4];
#pragma unroll
    for (int i = 0; i < kP3WRM; ++i)
#pragma unroll
        for (int j = 0; j < kP3WRN; ++j)
#pragma unroll
            for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.0f;

    const int K_subs = K / kQ6BK;

#define ISSUE_STAGE_LOAD_Q6(buf, kbx_val)                                       \
    do {                                                                        \
        /* sA: 64×16 = 1024 halves = 128 uint4 chunks = 1/thread.              */ \
        {                                                                       \
            int chunk = tid;                                                    \
            int row = chunk >> 1;                                               \
            int col = (chunk & 1) << 3;                                         \
            int g_row = block_m + row;                                          \
            int g_col = (kbx_val) * kQ6BK + col;                                \
            const half* gptr = &A[(int64_t)g_row * K + g_col];                  \
            half* sptr = &sA[buf][row * kQ6BK + col];                           \
            cp_async_cg_16(sptr, gptr, g_row < M);                              \
        }                                                                       \
        /* sQ6: BN × 16 bytes. 8-byte chunks = 1 per thread (BN=64) or 2 (128).*/ \
        constexpr int kSQ6Bytes = kP3BN * kQ6BK;                                \
        constexpr int kSQ6Chunks = kSQ6Bytes / (kThreadsPerBlock * 8);          \
        _Pragma("unroll")                                                       \
        for (int c = 0; c < kSQ6Chunks; ++c) {                                  \
            int byte_idx = c * (kThreadsPerBlock * 8) + tid * 8;                \
            int n_local = byte_idx >> 4;                                        \
            int byte_within_n = byte_idx & 0xF;                                 \
            int n_global = block_n + n_local;                                   \
            int64_t off =                                                       \
                (int64_t)n_global * K + (kbx_val)*kQ6BK + byte_within_n;        \
            cp_async_ca_8(&sQ6[buf][n_local * kQ6BK + byte_within_n],           \
                          eff_q6 + off, n_global < N);                          \
        }                                                                       \
        if (tid < kP3BN) {                                                      \
            int n_global = block_n + tid;                                       \
            if (n_global < N) {                                                 \
                int64_t off = (int64_t)n_global * K_subs + (kbx_val);           \
                sScale[buf][tid] = eff_scale[off];                              \
                sMin[buf][tid] = eff_min[off];                                  \
            } else {                                                            \
                sScale[buf][tid] = __float2half(0.0f);                          \
                sMin[buf][tid] = __float2half(0.0f);                            \
            }                                                                   \
        }                                                                       \
    } while (0)

    ISSUE_STAGE_LOAD_Q6(0, 0);
    cp_async_commit();

    for (int kbx = 0; kbx < K_subs; ++kbx) {
        const int cur = kbx & 1;
        if (kbx + 1 < K_subs) {
            const int next_buf = (kbx + 1) & 1;
            ISSUE_STAGE_LOAD_Q6(next_buf, kbx + 1);
            cp_async_commit();
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();

        uint32_t a_frag[kP3WRM][4];
#pragma unroll
        for (int rm = 0; rm < kP3WRM; ++rm) {
            int m_start = wm * kWarpM + rm * 16;
            int row = m_start + (lane & 0xF);
            int col = ((lane >> 4) << 3);
            unsigned smem_addr =
                __cvta_generic_to_shared(&sA[cur][row * kQ6BK + col]);
            asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                "{%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag[rm][0]), "=r"(a_frag[rm][1]),
                  "=r"(a_frag[rm][2]), "=r"(a_frag[rm][3])
                : "r"(smem_addr));
        }
#pragma unroll
        for (int rn = 0; rn < kP3WRN; ++rn) {
            int n_block_local = wn * kP3WarpN + rn * 8 + (lane >> 2);
            half scale = sScale[cur][n_block_local];
            half neg_min = __hneg(sMin[cur][n_block_local]);
            int b_off_lo = (lane & 3) * 2;
            int b_off_hi = b_off_lo + 8;
            uint16_t bytes_lo = *reinterpret_cast<uint16_t*>(
                &sQ6[cur][n_block_local * kQ6BK + b_off_lo]);
            uint16_t bytes_hi = *reinterpret_cast<uint16_t*>(
                &sQ6[cur][n_block_local * kQ6BK + b_off_hi]);
            int q0 = bytes_lo & 0xFF;
            int q1 = (bytes_lo >> 8) & 0xFF;
            int q2 = bytes_hi & 0xFF;
            int q3 = (bytes_hi >> 8) & 0xFF;
            half h0 = __hfma(__int2half_rn(q0), scale, neg_min);
            half h1 = __hfma(__int2half_rn(q1), scale, neg_min);
            half h2 = __hfma(__int2half_rn(q2), scale, neg_min);
            half h3 = __hfma(__int2half_rn(q3), scale, neg_min);
            half2 b0 = __halves2half2(h0, h1);
            half2 b1 = __halves2half2(h2, h3);
            uint32_t b0_reg = *reinterpret_cast<uint32_t*>(&b0);
            uint32_t b1_reg = *reinterpret_cast<uint32_t*>(&b1);
#pragma unroll
            for (int rm = 0; rm < kP3WRM; ++rm) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3};\n"
                    : "+f"(acc[rm][rn][0]), "+f"(acc[rm][rn][1]),
                      "+f"(acc[rm][rn][2]), "+f"(acc[rm][rn][3])
                    : "r"(a_frag[rm][0]), "r"(a_frag[rm][1]),
                      "r"(a_frag[rm][2]), "r"(a_frag[rm][3]),
                      "r"(b0_reg), "r"(b1_reg));
            }
        }
        __syncthreads();
    }

#undef ISSUE_STAGE_LOAD_Q6

    const int groupID = lane >> 2;
    const int lig = lane & 3;
#pragma unroll
    for (int rm = 0; rm < kP3WRM; ++rm) {
        int m_base = block_m + wm * kWarpM + rm * 16;
#pragma unroll
        for (int rn = 0; rn < kP3WRN; ++rn) {
            int n_base = block_n + wn * kP3WarpN + rn * 8;
            int rows[4] = {groupID, groupID, groupID + 8, groupID + 8};
            int cols[4] = {lig * 2 + 0, lig * 2 + 1, lig * 2 + 0, lig * 2 + 1};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int g_row = m_base + rows[i];
                int g_col = n_base + cols[i];
                if (g_row < M && g_col < N) {
                    y[(int64_t)g_row * N + g_col] = __float2half(acc[rm][rn][i]);
                }
            }
        }
    }
}

}  // namespace mmq_q4k_v2_detail

void q6k_prepare_v2_layout(const void* W, uint8_t* eff_q6_out,
                           half* eff_scale_out, half* eff_min_out, int N,
                           int K, cudaStream_t stream) {
    if (K % 256 != 0) return;
    using namespace mmq_q4k_v2_detail;
    const int K_blocks = K / 256;
    dim3 grid(K_blocks, N);
    dim3 block(256);   // 256 threads per super-block (one per element)
    q6k_prepare_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const block_q6_K*>(W), eff_q6_out, eff_scale_out,
        eff_min_out, N, K_blocks);
}

void mmq_q6k_v2(const half* x, const uint8_t* eff_q6, const half* eff_scale,
                const half* eff_min, half* y, int M, int N, int K,
                cudaStream_t stream) {
    using namespace mmq_q4k_v2_detail;
    if (K % kQ6BK != 0 || M <= 0 || N <= 0) return;
    dim3 block(kThreadsPerBlock);
    const char* pipeline_env = std::getenv("IMP_MMQ_Q4K_V2_PIPELINE");
    int forced_pipeline = pipeline_env ? std::atoi(pipeline_env) : -1;

    const int blocks_bn128 = ((M + kBM - 1) / kBM) * ((N + 127) / 128);
    const bool use_bn128 =
        (N >= 128) && (blocks_bn128 >= kP3BlockSaturationThreshold);
    if (use_bn128) {
        dim3 grid((N + 127) / 128, (M + kBM - 1) / kBM);
        const bool use_p4 = (forced_pipeline == 1);  // default p3 at BN=128
        if (use_p4) {
            mmq_q6k_v2_kernel_p4_t<128><<<grid, block, 0, stream>>>(
                x, eff_q6, eff_scale, eff_min, y, M, N, K);
        } else {
            mmq_q6k_v2_kernel_p3_t<128><<<grid, block, 0, stream>>>(
                x, eff_q6, eff_scale, eff_min, y, M, N, K);
        }
    } else {
        dim3 grid((N + 63) / 64, (M + kBM - 1) / kBM);
        const bool use_p4 = (forced_pipeline != 0);  // default p4 at BN=64
        if (use_p4) {
            mmq_q6k_v2_kernel_p4_t<64><<<grid, block, 0, stream>>>(
                x, eff_q6, eff_scale, eff_min, y, M, N, K);
        } else {
            mmq_q6k_v2_kernel_p3_t<64><<<grid, block, 0, stream>>>(
                x, eff_q6, eff_scale, eff_min, y, M, N, K);
        }
    }
}

}  // namespace imp
