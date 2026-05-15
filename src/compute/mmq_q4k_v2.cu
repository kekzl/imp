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

constexpr int kP3WRM = 2;  // m16n8k16: WARP_M / 16 = 32 / 16
constexpr int kP3WRN = 4;  // WARP_N /  8 = 32 /  8
constexpr int kP3WRK = 2;  // BK     / 16 = 32 / 16

__device__ __forceinline__ half2 q4_pair_to_half2(uint8_t byte, half scale,
                                                  half neg_min) {
    int lo = byte & 0xF;
    int hi = (byte >> 4) & 0xF;
    half hlo = __hfma(__int2half_rn(lo), scale, neg_min);
    half hhi = __hfma(__int2half_rn(hi), scale, neg_min);
    return __halves2half2(hlo, hhi);
}

__global__ void mmq_q4k_v2_kernel_p3(
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

    __shared__ __align__(16) half sA[kBM * kBK];               // 4 KB
    __shared__ __align__(16) uint8_t sQ4[kBN * (kBK / 2)];     // 1 KB
    __shared__ half sScale[kBN];                               // 128 B
    __shared__ half sMin[kBN];                                 // 128 B

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
        // 1024 bytes / 128 threads / 8 bytes per chunk = 1 chunk/thread.
        {
            int n_local = tid >> 1;          // 0..63
            int half_byte = tid & 1;         // 0 or 1
            int n_global = block_n + n_local;
            uint64_t packed = 0;
            if (n_global < N) {
                int64_t off = ((int64_t)n_global * K_subs + kbx) * 16 + half_byte * 8;
                packed = *reinterpret_cast<const uint64_t*>(eff_q4 + off);
            }
            *reinterpret_cast<uint64_t*>(&sQ4[n_local * 16 + half_byte * 8]) = packed;
        }
        // ---- Load sScale, sMin -----------------------------------------
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
                int n_block_local = wn * kWarpN + rn * 8 + (lane >> 2);
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
            int n_base = block_n + wn * kWarpN + rn * 8;
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

void mmq_q4k_v2(const half* x, const uint8_t* eff_q4, const half* eff_scale,
                const half* eff_min, half* y, int M, int N, int K,
                cudaStream_t stream) {
    using namespace mmq_q4k_v2_detail;
    if (K % kBK != 0 || M <= 0 || N <= 0) return;
    dim3 grid((N + kBN - 1) / kBN, (M + kBM - 1) / kBM);
    dim3 block(kThreadsPerBlock);
    // IMP_MMQ_Q4K_V2_SCAFFOLD=1 forces the WMMA scaffold (kept as A/B fallback
    // and as a reference implementation). Default is the Phase-3 register-only
    // path.
    const char* scaffold_env = std::getenv("IMP_MMQ_Q4K_V2_SCAFFOLD");
    if (scaffold_env && std::atoi(scaffold_env) != 0) {
        mmq_q4k_v2_kernel_scaffold<<<grid, block, 0, stream>>>(
            x, eff_q4, eff_scale, eff_min, y, M, N, K);
    } else {
        mmq_q4k_v2_kernel_p3<<<grid, block, 0, stream>>>(
            x, eff_q4, eff_scale, eff_min, y, M, N, K);
    }
}

}  // namespace imp
