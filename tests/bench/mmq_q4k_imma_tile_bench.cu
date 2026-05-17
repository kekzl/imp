// =============================================================================
// mmq_q4k_imma_tile_bench.cu — Phase 2B INT8 IMMA tile kernel (minimum viable)
// =============================================================================
//
// One warp per CTA, BLOCK_M=16, BLOCK_N=8, BLOCK_K=32 (one m16n8k32 MMA tile).
// Synchronous SMEM staging. Phase 2B.1 will add cp.async + ldmatrix.x4 for
// real perf. This version is for correctness verification only.
//
// Math identity (per design memo §3.2):
//   out_ref[m, n] = Σ_k x_fp16[m, k] · w_fp16[n, k]
//
//   x_fp16[m, k] = x_scale[m, sub_k] · X_s8[m, k]
//   w_fp16[n, k] = d_n · sc[n, sub_k] · q_w[n, k] − dmin_n · m[n, sub_k]
//                = α[n, sub_k] · q_w[n, k] − dmin_n · m[n, sub_k]
//                = α[n, sub_k] · (W_s8[n, k] + 8) − dmin_n · m[n, sub_k]
//
//   Substituting and grouping by sub-block:
//     out[m, n] = Σ_sub x_scale[m, sub] · {
//                     α[n, sub] · Σ_{k in sub} X_s8[m, k] · W_s8[n, k]
//                   + β[n, sub] · Σ_{k in sub} X_s8[m, k]                }
//
//   where β[n, sub] = 8·α[n, sub] − dmin_n · m[n, sub].
//
// The IMMA's role is computing Σ_{k in sub} X_s8 · W_s8 (the s32 accumulator
// over a 32-wide K slab). Each sub-block is exactly one MMA's worth of K.

#include "bench/mmq_q4k_imma_tile_bench.h"

#include <cstdint>
#include <cuda_fp16.h>

namespace imp {

namespace {

constexpr int kBlockM = 16;
constexpr int kBlockN = 8;
constexpr int kBlockK = 32;

// One warp per CTA. Each thread holds 4 s32 accumulators (its share of the 16×8
// output tile per m16n8k32 fragment layout). We accumulate float (after scale
// application) across all sub-blocks of K.
__global__ void mmq_q4k_imma_tile_kernel(const int8_t* __restrict__ X_s8,
                                         const __half* __restrict__ x_scale,
                                         const float* __restrict__ x_rowsum,
                                         const int8_t* __restrict__ W_s8,
                                         const __half* __restrict__ eff_alpha,
                                         const __half* __restrict__ eff_beta,
                                         __half* __restrict__ out, int M, int N, int K) {
    const int n_block = blockIdx.x;
    const int m_block = blockIdx.y;
    if (m_block * kBlockM >= M || n_block * kBlockN >= N) return;

    const int lane = threadIdx.x;  // 0..31

    // SMEM stage for the A and B tiles. 16×32 + 8×32 = 768 bytes per stage.
    __shared__ int8_t sA[kBlockM][kBlockK];  // [16][32]
    __shared__ int8_t sB[kBlockN][kBlockK];  // [8][32]

    // FP32 accumulator (post-MMA, post-scale-apply) for the 4 output values
    // this thread owns within the 16×8 tile.
    float c_f32_0 = 0.0f, c_f32_1 = 0.0f, c_f32_2 = 0.0f, c_f32_3 = 0.0f;

    const int subs_per_K = K / kBlockK;
    const int subs_per_row = subs_per_K;  // alias for clarity

    const int base_m = m_block * kBlockM;
    const int base_n = n_block * kBlockN;

    for (int kb = 0; kb < subs_per_K; ++kb) {
        const int k_base = kb * kBlockK;

        // -------- Load A (16×32) into SMEM --------
        // 32 lanes × 16 bytes = 512 bytes = 16×32 ✓
        // Each lane brings 16 bytes from X_s8[base_m + (lane/2), k_base + (lane%2)*16 .. +15]
        {
            const int row_in_tile = lane >> 1;            // 0..15
            const int col_off = (lane & 1) * 16;          // 0 or 16
            const int8_t* src = X_s8 + (base_m + row_in_tile) * K + k_base + col_off;
            int8_t* dst = &sA[row_in_tile][col_off];
            // 16 bytes via 2× int4 reads.
            int4 v0 = *reinterpret_cast<const int4*>(src);
            *reinterpret_cast<int4*>(dst) = v0;
        }
        // -------- Load B (8×32) into SMEM --------
        // 32 lanes × 8 bytes = 256 bytes = 8×32 ✓
        // 4 lanes per row, each lane brings 8 bytes.
        {
            const int row_in_tile = lane >> 2;            // 0..7
            const int col_off = (lane & 3) * 8;           // 0, 8, 16, 24
            const int8_t* src = W_s8 + (base_n + row_in_tile) * K + k_base + col_off;
            int8_t* dst = &sB[row_in_tile][col_off];
            // 8 bytes via 1× int2 read.
            int2 v0 = *reinterpret_cast<const int2*>(src);
            *reinterpret_cast<int2*>(dst) = v0;
        }
        __syncthreads();

        // -------- Build A fragment (4 × b32 = 16 bytes per thread) --------
        // m16n8k32 A operand layout (PTX ISA 8.5 §9.7.13.4.5 Figure 41,
        // .s8.s8.s32 row.col case):
        //   K is split into two 16-wide blocks. Per thread:
        //     a0 = A[lane/4    ][k_base..k_base+3]    (K-block 0)
        //     a1 = A[lane/4 + 8][k_base..k_base+3]    (K-block 0, row+8)
        //     a2 = A[lane/4    ][k_base+16..+19]      (K-block 1)
        //     a3 = A[lane/4 + 8][k_base+16..+19]      (K-block 1, row+8)
        //   where k_base = (lane%4) * 4.
        const int a_row_lo = lane >> 2;
        const int a_row_hi = a_row_lo + 8;
        const int a_col_base = (lane & 3) * 4;
        uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sA[a_row_lo][a_col_base]);
        uint32_t a1 = *reinterpret_cast<const uint32_t*>(&sA[a_row_hi][a_col_base]);
        uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sA[a_row_lo][a_col_base + 16]);
        uint32_t a3 = *reinterpret_cast<const uint32_t*>(&sA[a_row_hi][a_col_base + 16]);

        // -------- Build B fragment (2 × b32 = 8 bytes per thread) --------
        // m16n8k32 B operand layout: cols 0..7, each 4 lanes share a col.
        //   lane → col (lane/4) ∈ [0, 8)
        //   lane → k base (lane%4)*4
        //   2 b32 registers: b0 = col[lane/4][k_base..k_base+3]
        //                    b1 = col[lane/4][k_base+16..k_base+19]
        const int b_col = lane >> 2;
        const int b_k_base = (lane & 3) * 4;
        uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sB[b_col][b_k_base]);
        uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sB[b_col][b_k_base + 16]);

        // -------- IMMA m16n8k32.row.col.s32.s8.s8.s32 --------
        int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#if __CUDA_ARCH__ >= 750
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
              "r"(0), "r"(0), "r"(0), "r"(0));
#endif

        // -------- Per-sub-block scale apply --------
        // Output mapping for m16n8k32 (PTX ISA 8.5 §9.7.13.4 Table 26):
        //   thread t owns 4 outputs at (row, col):
        //     d0 → ((t/4) % 8,       (t%4)*2)
        //     d1 → ((t/4) % 8,       (t%4)*2 + 1)
        //     d2 → ((t/4) % 8 + 8,   (t%4)*2)
        //     d3 → ((t/4) % 8 + 8,   (t%4)*2 + 1)
        const int row_lo = (lane >> 2) & 7;    // 0..7
        const int row_hi = row_lo + 8;          // 8..15
        const int col_lo = (lane & 3) * 2;     // 0..6 (step 2)
        const int col_hi = col_lo + 1;

        const int m_lo = base_m + row_lo;
        const int m_hi = base_m + row_hi;
        const int n_lo = base_n + col_lo;
        const int n_hi = base_n + col_hi;

        const float xs_lo = __half2float(x_scale[m_lo * subs_per_row + kb]);
        const float xs_hi = __half2float(x_scale[m_hi * subs_per_row + kb]);
        const float xrs_lo = x_rowsum[m_lo * subs_per_row + kb];
        const float xrs_hi = x_rowsum[m_hi * subs_per_row + kb];

        const float a_lo = __half2float(eff_alpha[n_lo * subs_per_row + kb]);
        const float a_hi = __half2float(eff_alpha[n_hi * subs_per_row + kb]);
        const float b_lo = __half2float(eff_beta[n_lo * subs_per_row + kb]);
        const float b_hi = __half2float(eff_beta[n_hi * subs_per_row + kb]);

        c_f32_0 += xs_lo * (a_lo * static_cast<float>(c0) + b_lo * xrs_lo);
        c_f32_1 += xs_lo * (a_hi * static_cast<float>(c1) + b_hi * xrs_lo);
        c_f32_2 += xs_hi * (a_lo * static_cast<float>(c2) + b_lo * xrs_hi);
        c_f32_3 += xs_hi * (a_hi * static_cast<float>(c3) + b_hi * xrs_hi);

        __syncthreads();
    }

    // -------- Epilogue: FP16 write --------
    const int row_lo = (lane >> 2) & 7;
    const int row_hi = row_lo + 8;
    const int col_lo = (lane & 3) * 2;
    const int col_hi = col_lo + 1;
    const int m_lo = base_m + row_lo;
    const int m_hi = base_m + row_hi;
    const int n_lo = base_n + col_lo;
    const int n_hi = base_n + col_hi;

    if (m_lo < M && n_lo < N) out[m_lo * N + n_lo] = __float2half(c_f32_0);
    if (m_lo < M && n_hi < N) out[m_lo * N + n_hi] = __float2half(c_f32_1);
    if (m_hi < M && n_lo < N) out[m_hi * N + n_lo] = __float2half(c_f32_2);
    if (m_hi < M && n_hi < N) out[m_hi * N + n_hi] = __float2half(c_f32_3);
}

}  // namespace

void mmq_q4k_imma_tile(const int8_t* X_s8, const __half* x_scale, const float* x_rowsum,
                       const int8_t* W_s8, const __half* eff_alpha, const __half* eff_beta,
                       __half* out, int M, int N, int K, cudaStream_t stream) {
    if (M % kBlockM != 0 || N % kBlockN != 0 || K % kBlockK != 0) return;
    dim3 grid(N / kBlockN, M / kBlockM, 1);
    dim3 block(32, 1, 1);
    mmq_q4k_imma_tile_kernel<<<grid, block, 0, stream>>>(
        X_s8, x_scale, x_rowsum, W_s8, eff_alpha, eff_beta, out, M, N, K);
}

}  // namespace imp
