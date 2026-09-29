#ifndef IMP_TESTS_MMQ_Q8_IMMA_LEGACY_CUH
#define IMP_TESTS_MMQ_Q8_IMMA_LEGACY_CUH

// Pre-#2267 Q8_0 IMMA kernel (2-stage cp.async, 32-bit LDS fragments, 2 barriers per K-step),
// verbatim from src/compute/mmq_q8_imma.cu at 287c25a5~1 (BM=128 path unchanged to 6be82977).
// Test-binary only: bit-exact reference and timing baseline for the BM=128 pipeline kernel.

#include <cuda_fp16.h>
#include <cstdint>

#include "compute/mmq_q8_imma_internal.cuh"

namespace imp {
namespace legacy2267 {

// One K-step tile load, all 256 threads cooperating, loops unrolled for both BM variants:
//   A[BM][kBK] s8, M-tail zero-filled; B[kBN][kBK] s8, weight rows always full (N % kBN == 0)
//   Asc[BM][2] half / Ars[BM][2] float: activation scale/rowsum, d-plane cols (kb0, kb0+1)
//   Bsc[2][kBN][2] half: weight (α, β) per kb col, kb-major
template <int BM, bool WB>
__device__ __forceinline__ void load_kstep(int tid, const int8_t* __restrict__ A,
                                           const __half* __restrict__ Asc,
                                           const float* __restrict__ Ars,
                                           const int8_t* __restrict__ B,
                                           const __half* __restrict__ Bsc, int8_t (*sA)[kRow],
                                           int8_t (*sB)[kRow], __half (*sAsc)[2],
                                           float (*sArs)[2], __half (*sBsc)[kBN][2], int base_m,
                                           int M, int K, int subs, int k_base,
                                           int base_n_rows) {
#pragma unroll
    for (int i = tid; i < BM * 4; i += kThreads) {
        const int row = i >> 2;
        const int col = (i & 3) * 16;
        const bool valid = (base_m + row) < M;
        cp_async_cg_16(&sA[row][col],
                       A + static_cast<size_t>(base_m + row) * K + k_base + col, valid);
    }
#pragma unroll
    for (int i = tid; i < kBN * 4; i += kThreads) {
        const int row = i >> 2;
        const int col = (i & 3) * 16;
        cp_async_cg_16(&sB[row][col], B + static_cast<size_t>(row) * K + k_base + col,
                       (base_n_rows < 0) || (row < base_n_rows));
    }
    const int kb0 = k_base / 32;
#pragma unroll
    for (int i = tid; i < BM; i += kThreads) {
        const bool valid = (base_m + i) < M;
        cp_async_ca_4(&sAsc[i][0], Asc + static_cast<size_t>(base_m + i) * subs + kb0, valid);
        if (WB)
            cp_async_ca_8(&sArs[i][0], Ars + static_cast<size_t>(base_m + i) * subs + kb0, valid);
    }
#pragma unroll
    for (int i = tid; i < kBN * 2; i += kThreads) {
        // kb-major [kb][n][2]: one fragment's two columns read as one 8-B word
        const int n = i >> 1, kb = i & 1;
        cp_async_ca_4(&sBsc[kb][n][0], Bsc + (static_cast<size_t>(n) * subs + kb0 + kb) * 2,
                      (base_n_rows < 0) || (n < base_n_rows));
    }
}

// out = alpha/beta-scaled IMMA over [BM,kBN] tiles, += when BETA1.
// MoE: gridDim.z = ne, indexed via expert_offsets; dense passes offsets = nullptr (z=1).
// SPLITK (dense-only): gridDim.z = K-split index; a single M-tile grid (N/kBN blocks) is too small to hide
// K-loop latency (spec-decode verify bottleneck, issue #667).
// Each split writes ks_per_split K-steps to split_out[z][M][N]; mmq_splitk_finalize_kernel reduces in fixed
// order (bit-reproducible) and applies beta/residual.
template <int BM, bool BETA1, bool WB /* weight beta term (Q4_K); false = pure alpha (Q8_0) */,
          bool SPLITK = false>
__global__ void __launch_bounds__(kThreads)
    mmq_imma_kernel(const int8_t* __restrict__ X_s8, const __half* __restrict__ x_scale,
                    const float* __restrict__ x_rowsum, const int8_t* __restrict__ W_s8,
                    const __half* __restrict__ w_sc, __half* __restrict__ out, int M, int N,
                    int K, const int32_t* __restrict__ expert_offsets, size_t w_stride,
                    size_t wsc_stride, float* __restrict__ split_out = nullptr,
                    int ks_per_split = 0) {
    constexpr int kWM = (BM == 128) ? 4 : 2;  // warp grid
    constexpr int kWN = (BM == 128) ? 2 : 4;
    constexpr int kTileM = BM / kWM;       // 32 / 16
    constexpr int kTileN = kBN / kWN;      // 64 / 32
    constexpr int kMF = kTileM / 16;       // 2 / 1
    constexpr int kNF = kTileN / 8;        // 8 / 4
    static_assert(kWM * kWN * 32 == kThreads, "warp grid must fill the CTA");

    int rows = M;
    size_t row_off = 0;
    const int e = SPLITK ? 0 : blockIdx.z;  // SPLITK reuses gridDim.z for the K-split
    if (expert_offsets != nullptr) {
        const int32_t o0 = __ldg(&expert_offsets[e]);
        const int32_t o1 = __ldg(&expert_offsets[e + 1]);
        row_off = static_cast<size_t>(o0);
        rows = o1 - o0;
    }
    const int base_m = blockIdx.y * BM;
    const int base_n = blockIdx.x * kBN;
    if (base_m >= rows || rows == 0) return;
    const int n_rem = (N - base_n >= kBN) ? -1 : (N - base_n);  // -1 = full tile

    const int subs = K / 32;
    const int8_t* A = X_s8 + row_off * K;
    const __half* Asc = x_scale + row_off * subs;
    const float* Ars = x_rowsum + row_off * subs;
    const int8_t* B = W_s8 + static_cast<size_t>(e) * w_stride + static_cast<size_t>(base_n) * K;
    const __half* Bsc = w_sc + static_cast<size_t>(e) * wsc_stride +
                        static_cast<size_t>(base_n) * subs * 2;
    __half* C = out + row_off * N;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = warp_id / kWN;
    const int warp_n = warp_id % kWN;
    const int rl = lane >> 2;  // 0..7
    const int cl = lane & 3;   // 0..3

    __shared__ int8_t sA[kStages][BM][kRow];
    __shared__ int8_t sB[kStages][kBN][kRow];
    __shared__ __half sAsc[kStages][BM][2];
    __shared__ float sArs[kStages][BM][2];
    __shared__ __half sBsc[kStages][2][kBN][2];

    float acc[kMF][kNF][4];
#pragma unroll
    for (int i = 0; i < kMF; ++i)
#pragma unroll
        for (int j = 0; j < kNF; ++j)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

    const int ksteps = K / kBK;
    int ks_begin = 0;
    int ks_end = ksteps;
    if (SPLITK) {
        ks_begin = blockIdx.z * ks_per_split;
        ks_end = min(ksteps, ks_begin + ks_per_split);
        if (ks_begin >= ks_end)
            return;
    }
    load_kstep<BM, WB>(tid, A, Asc, Ars, B, Bsc, sA[0], sB[0], sAsc[0], sArs[0], sBsc[0],
                       base_m, rows, K, subs, ks_begin * kBK, n_rem);
    cp_async_commit();

    for (int ks = ks_begin; ks < ks_end; ++ks) {
        const int stage = (ks - ks_begin) & 1;
        if (ks + 1 < ks_end) {
            const int nstage = (ks + 1 - ks_begin) & 1;
            load_kstep<BM, WB>(tid, A, Asc, Ars, B, Bsc, sA[nstage], sB[nstage], sAsc[nstage],
                               sArs[nstage], sBsc[nstage], base_m, rows, K, subs, (ks + 1) * kBK,
                               n_rem);
            cp_async_commit();
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();

        // nf outer, mf inner: each B fragment and its (alpha, beta) pair is read from smem once
        // per nf instead of once per (mf, nf); per-element accumulation order is unchanged.
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
            const int kc = kb * 32;
            uint32_t a[kMF][4];
            float da_lo[kMF], da_hi[kMF], rs_lo[kMF], rs_hi[kMF];
#pragma unroll
            for (int mf = 0; mf < kMF; ++mf) {
                const int arow_lo = warp_m * kTileM + mf * 16 + rl;
                const int arow_hi = arow_lo + 8;
                const int acol = kc + cl * 4;
                a[mf][0] = *reinterpret_cast<const uint32_t*>(&sA[stage][arow_lo][acol]);
                a[mf][1] = *reinterpret_cast<const uint32_t*>(&sA[stage][arow_hi][acol]);
                a[mf][2] = *reinterpret_cast<const uint32_t*>(&sA[stage][arow_lo][acol + 16]);
                a[mf][3] = *reinterpret_cast<const uint32_t*>(&sA[stage][arow_hi][acol + 16]);
                da_lo[mf] = __half2float(sAsc[stage][arow_lo][kb]);
                da_hi[mf] = __half2float(sAsc[stage][arow_hi][kb]);
                rs_lo[mf] = WB ? sArs[stage][arow_lo][kb] : 0.0f;
                rs_hi[mf] = WB ? sArs[stage][arow_hi][kb] : 0.0f;
            }

#pragma unroll
            for (int nf = 0; nf < kNF; ++nf) {
                const int bcol = warp_n * kTileN + nf * 8 + rl;
                const int bk = kc + cl * 4;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sB[stage][bcol][bk]);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sB[stage][bcol][bk + 16]);
                // c0,c1 = rows (rl) cols (cl*2, cl*2+1); c2,c3 = rows (rl+8)
                const int ncol_lo = warp_n * kTileN + nf * 8 + cl * 2;
                const uint2 ab = *reinterpret_cast<const uint2*>(&sBsc[stage][kb][ncol_lo][0]);
                const __half2 ab_lo = *reinterpret_cast<const __half2*>(&ab.x);
                const __half2 ab_hi = *reinterpret_cast<const __half2*>(&ab.y);
                const float al = __half2float(__low2half(ab_lo));
                const float bl = __half2float(__high2half(ab_lo));
                const float ah = __half2float(__low2half(ab_hi));
                const float bh = __half2float(__high2half(ab_hi));
#pragma unroll
                for (int mf = 0; mf < kMF; ++mf) {
                    int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#if __CUDA_ARCH__ >= 800
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
                        : "r"(a[mf][0]), "r"(a[mf][1]), "r"(a[mf][2]), "r"(a[mf][3]), "r"(b0), "r"(b1),
                          "r"(0), "r"(0), "r"(0), "r"(0));
#endif
                    if (WB) {
                        acc[mf][nf][0] += da_lo[mf] * fmaf(al, static_cast<float>(c0), bl * rs_lo[mf]);
                        acc[mf][nf][1] += da_lo[mf] * fmaf(ah, static_cast<float>(c1), bh * rs_lo[mf]);
                        acc[mf][nf][2] += da_hi[mf] * fmaf(al, static_cast<float>(c2), bl * rs_hi[mf]);
                        acc[mf][nf][3] += da_hi[mf] * fmaf(ah, static_cast<float>(c3), bh * rs_hi[mf]);
                    } else {
                        // pure-alpha fast path: the unified beta form costs Q8_0 throughput
                        acc[mf][nf][0] += (da_lo[mf] * al) * static_cast<float>(c0);
                        acc[mf][nf][1] += (da_lo[mf] * ah) * static_cast<float>(c1);
                        acc[mf][nf][2] += (da_hi[mf] * al) * static_cast<float>(c2);
                        acc[mf][nf][3] += (da_hi[mf] * ah) * static_cast<float>(c3);
                    }
                }
            }
        }
        __syncthreads();
    }

    // SPLITK epilogue: store the fp32 partial tile to this split's slice;
    // mmq_splitk_finalize_kernel reduces slices and applies beta/residual.
    if constexpr (SPLITK) {
        float* Cs = split_out + static_cast<size_t>(blockIdx.z) * (static_cast<size_t>(M) * N);
#pragma unroll
        for (int mf = 0; mf < kMF; ++mf) {
            const int row_lo = base_m + warp_m * kTileM + mf * 16 + rl;
            const int row_hi = row_lo + 8;
#pragma unroll
            for (int nf = 0; nf < kNF; ++nf) {
                const int col = base_n + warp_n * kTileN + nf * 8 + cl * 2;
                if (col + 1 >= N) continue;
                if (row_lo < rows) {
                    Cs[static_cast<size_t>(row_lo) * N + col] = acc[mf][nf][0];
                    Cs[static_cast<size_t>(row_lo) * N + col + 1] = acc[mf][nf][1];
                }
                if (row_hi < rows) {
                    Cs[static_cast<size_t>(row_hi) * N + col] = acc[mf][nf][2];
                    Cs[static_cast<size_t>(row_hi) * N + col + 1] = acc[mf][nf][3];
                }
            }
        }
    } else {
        // Epilogue: FP16 store, M-tail predicated (N is a multiple of kBN).
#pragma unroll
        for (int mf = 0; mf < kMF; ++mf) {
            const int row_lo = base_m + warp_m * kTileM + mf * 16 + rl;
            const int row_hi = row_lo + 8;
#pragma unroll
            for (int nf = 0; nf < kNF; ++nf) {
                const int col = base_n + warp_n * kTileN + nf * 8 + cl * 2;
                if (col + 1 >= N) continue;  // N-tail: OOB columns are never stored
                if (row_lo < rows) {
                    __half2* p = reinterpret_cast<__half2*>(&C[static_cast<size_t>(row_lo) * N + col]);
                    __half2 v = __floats2half2_rn(acc[mf][nf][0], acc[mf][nf][1]);
                    *p = BETA1 ? __hadd2(*p, v) : v;
                }
                if (row_hi < rows) {
                    __half2* p = reinterpret_cast<__half2*>(&C[static_cast<size_t>(row_hi) * N + col]);
                    __half2 v = __floats2half2_rn(acc[mf][nf][2], acc[mf][nf][3]);
                    *p = BETA1 ? __hadd2(*p, v) : v;
                }
            }
        }
    }
}

}  // namespace legacy2267
}  // namespace imp

#endif  // IMP_TESTS_MMQ_Q8_IMMA_LEGACY_CUH
