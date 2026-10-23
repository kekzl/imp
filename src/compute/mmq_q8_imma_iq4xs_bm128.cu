// Dense IQ4_XS IMMA kernel for prompt chunks above 32 rows (sm_120a): BM=128 x BN=128 tiles, 64-K steps
// (2 sub-blocks of a 136-B super-block), cp.async staging. Each warp decodes its B fragments once per
// sub-block (iq4nl_lut8) and reuses them over 2 m-fragments; the 32-row kernel decoded them per 32 rows.

#include "compute/iq4_lut.cuh"
#include "compute/mmq_q8_imma_internal.cuh"
#include "core/logging.h"

namespace imp {

namespace {

constexpr int kXsBlock = 136;     // fp16 d, scales_h u16, scales_l u32, qs[128]
constexpr int kXsQRow = 32 + 16;  // 2 sub-blocks of qs + pad: 12 words, rl rows on distinct banks
constexpr int kBM = 128;

// One 64-K step ks: A [128][kRow] s8 + scales, B qs of sub-blocks 2 (ks & 3), +1 and the super-block header.
__device__ __forceinline__ void load_step_iq4xs(int tid, const int8_t* __restrict__ A,
                                                const __half* __restrict__ Asc, const uint8_t* __restrict__ W,
                                                int8_t (*sA)[kRow], __half (*sAsc)[2],
                                                uint8_t (*sBq)[kXsQRow], uint2* sBh, int base_m, int M, int K,
                                                int base_n, int n_valid, int ks) {
    const int subs = K / 32;
    const int k_base = ks * kBK;
#pragma unroll
    for (int i = tid; i < kBM * 4; i += kThreads) {
        const int row = i >> 2;
        const int col = (i & 3) * 16;
        const bool valid = (base_m + row) < M;
        cp_async_cg_16(&sA[row][col], A + (static_cast<size_t>(valid ? base_m + row : 0) * K) + k_base + col,
                       valid);
    }
    for (int i = tid; i < kBM; i += kThreads) {
        const bool valid = (base_m + i) < M;
        cp_async_ca_4(&sAsc[i][0], Asc + (static_cast<size_t>(valid ? base_m + i : 0) * subs) + (k_base / 32),
                      valid);
    }
    const size_t row_bytes = static_cast<size_t>(K / 256) * kXsBlock;
    const int sblk = ks >> 2;
    const int qs_off = 8 + ((ks & 3) * 32);
#pragma unroll
    for (int i = tid; i < kBN * 5; i += kThreads) {
        const int row = i / 5;
        const int part = i - (row * 5);
        const bool live = row < n_valid;
        const uint8_t* blk = W + ((live ? static_cast<size_t>(base_n + row) : 0) * row_bytes) +
                             (static_cast<size_t>(sblk) * kXsBlock);
        if (part == 0) {
            cp_async_ca_8(&sBh[row], blk, live);
        } else {
            const ptrdiff_t q = static_cast<ptrdiff_t>(part - 1) * 8;
            cp_async_ca_8(&sBq[row][q], blk + qs_off + q, live);
        }
    }
}

template <bool BETA1>
__global__ void __launch_bounds__(kThreads) mmq_imma_iq4xs_bm128_kernel(const int8_t* __restrict__ X_s8,
                                                                        const __half* __restrict__ x_scale,
                                                                        const uint8_t* __restrict__ W,
                                                                        __half* __restrict__ out, int M,
                                                                        int N, int K) {
    constexpr int kWM = 4, kWN = 2;
    constexpr int kTileM = kBM / kWM;  // 32
    constexpr int kTileN = kBN / kWN;  // 64
    constexpr int kMF = kTileM / 16;
    constexpr int kNF = kTileN / 8;

    const int base_n = static_cast<int>(blockIdx.x) * kBN;
    const int base_m = static_cast<int>(blockIdx.y) * kBM;
    const int n_valid = min(kBN, N - base_n);
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = warp_id / kWN;
    const int warp_n = warp_id % kWN;
    const int rl = lane >> 2;
    const int cl = lane & 3;

    __shared__ __align__(16) int8_t sA[kStages][kBM][kRow];
    __shared__ __align__(16) uint8_t sBq[kStages][kBN][kXsQRow];
    __shared__ __align__(16) uint2 sBh[kStages][kBN];
    __shared__ __align__(16) __half sAsc[kStages][kBM][2];
    __shared__ float sBs[kStages][kBN][2];  // d * (ls - 32) per (row, sub-block of the step)

    float acc[kMF][kNF][4];
#pragma unroll
    for (int i = 0; i < kMF; ++i)
#pragma unroll
        for (int j = 0; j < kNF; ++j)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

    const int ksteps = K / kBK;
    load_step_iq4xs(tid, X_s8, x_scale, W, sA[0], sAsc[0], sBq[0], sBh[0], base_m, M, K, base_n, n_valid, 0);
    cp_async_commit();
    for (int ks = 0; ks < ksteps; ++ks) {
        const int stage = ks & 1;
        if (ks + 1 < ksteps) {
            const int ns = stage ^ 1;
            load_step_iq4xs(tid, X_s8, x_scale, W, sA[ns], sAsc[ns], sBq[ns], sBh[ns], base_m, M, K, base_n,
                            n_valid, ks + 1);
            cp_async_commit();
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();
        const int j0 = (ks & 3) * 2;
        for (int i = tid; i < kBN * 2; i += kThreads) {
            const int row = i >> 1;
            const int j = j0 + (i & 1);
            const uint2 h = sBh[stage][row];
            const uint32_t ls = ((h.y >> (4 * j)) & 0xFu) | (((h.x >> (16 + (2 * j))) & 3u) << 4);
            const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(h.x & 0xFFFFu)));
            sBs[stage][row][i & 1] = d * static_cast<float>(static_cast<int>(ls) - 32);
        }
        __syncthreads();
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
            uint32_t b0[kNF], b1[kNF];
            float wl[kNF], wh[kNF];
#pragma unroll
            for (int nf = 0; nf < kNF; ++nf) {
                const int brow = (warp_n * kTileN) + (nf * 8) + rl;
                const int2 v = iq4nl_lut8(
                    *reinterpret_cast<const uint32_t*>(&sBq[stage][brow][(kb * 16) + (cl * 4)]));
                b0[nf] = static_cast<uint32_t>(v.x);
                b1[nf] = static_cast<uint32_t>(v.y);
                const int ncol = (warp_n * kTileN) + (nf * 8) + (cl * 2);
                wl[nf] = sBs[stage][ncol][kb];
                wh[nf] = sBs[stage][ncol + 1][kb];
            }
#pragma unroll
            for (int mf = 0; mf < kMF; ++mf) {
                const int arow_lo = (warp_m * kTileM) + (mf * 16) + rl;
                const int arow_hi = arow_lo + 8;
                const int acol = (kb * 32) + (cl * 4);
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sA[stage][arow_lo][acol]);
                const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&sA[stage][arow_hi][acol]);
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sA[stage][arow_lo][acol + 16]);
                const uint32_t a3 = *reinterpret_cast<const uint32_t*>(&sA[stage][arow_hi][acol + 16]);
                const float da_lo = __half2float(sAsc[stage][arow_lo][kb]);
                const float da_hi = __half2float(sAsc[stage][arow_hi][kb]);
#pragma unroll
                for (int nf = 0; nf < kNF; ++nf) {
                    int32_t p0 = 0, p1 = 0, p2 = 0, p3 = 0;
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                        : "=r"(p0), "=r"(p1), "=r"(p2), "=r"(p3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0[nf]), "r"(b1[nf]), "r"(0), "r"(0),
                          "r"(0), "r"(0));
                    acc[mf][nf][0] = fmaf(da_lo * wl[nf], static_cast<float>(p0), acc[mf][nf][0]);
                    acc[mf][nf][1] = fmaf(da_lo * wh[nf], static_cast<float>(p1), acc[mf][nf][1]);
                    acc[mf][nf][2] = fmaf(da_hi * wl[nf], static_cast<float>(p2), acc[mf][nf][2]);
                    acc[mf][nf][3] = fmaf(da_hi * wh[nf], static_cast<float>(p3), acc[mf][nf][3]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int mf = 0; mf < kMF; ++mf) {
        const int row_lo = base_m + (warp_m * kTileM) + (mf * 16) + rl;
#pragma unroll
        for (int nf = 0; nf < kNF; ++nf) {
            const int col = base_n + (warp_n * kTileN) + (nf * 8) + (cl * 2);
            if (col + 1 >= N)
                continue;
            auto store = [&](int row, float lo, float hi) {
                if (row >= M)
                    return;
                auto* p = reinterpret_cast<__half2*>(&out[(static_cast<size_t>(row) * N) + col]);
                const __half2 v = __floats2half2_rn(lo, hi);
                *p = BETA1 ? __hadd2(*p, v) : v;
            };
            store(row_lo, acc[mf][nf][0], acc[mf][nf][1]);
            store(row_lo + 8, acc[mf][nf][2], acc[mf][nf][3]);
        }
    }
}

}  // namespace

bool launch_iq4xs_bm128(const uint8_t* w, __half* out_f16, int M, int N, int K, float beta,
                        cudaStream_t stream) {
    const dim3 grid((N + kBN - 1) / kBN, (M + kBM - 1) / kBM, 1);
    if (beta == 1.0f)
        mmq_imma_iq4xs_bm128_kernel<true>
            <<<grid, kThreads, 0, stream>>>(g_imma_act.xs8, g_imma_act.xscale, w, out_f16, M, N, K);
    else
        mmq_imma_iq4xs_bm128_kernel<false>
            <<<grid, kThreads, 0, stream>>>(g_imma_act.xs8, g_imma_act.xscale, w, out_f16, M, N, K);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

}  // namespace imp
