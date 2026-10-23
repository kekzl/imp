// Dense GGUF IMMA kernel for prompt chunks above 32 rows (sm_120a): BM=128 x BN=128 tiles, 64-K steps
// (2 sub-blocks of 32), cp.async staging. Each warp decodes its B fragments once per sub-block and reuses
// them over 2 m-fragments; the 32-row kernel (mmq_q8_imma_gguf_smallm.cu) decodes them per 32 rows.

#include "compute/iq4_lut.cuh"
#include "compute/mmq_q8_imma_internal.cuh"
#include "core/device_info.h"
#include "core/logging.h"

#include <type_traits>

namespace imp {

namespace {

constexpr int kBM = 128;

// Format policy: one 64-K step of a weight row is kPieces pieces of kPiece bytes (piece_off: byte offset in
// the row), staged as a kSegRow-B smem segment (kSegRow / 4 % 32 in {12, 20}: rl rows on distinct banks).
// scale(seg, ks, kb) = (weight scale, min) of sub-block kb of step ks; kMin adds min * activation row sum.

// IQ4_XS: the super-block header (u64) + 32 B of qs (2 sub-blocks) per step.
struct BigIQ4XS {
    static constexpr int kPieces = 5, kPiece = 8, kSegRow = 48;
    static constexpr bool kMin = false;
    static __device__ __forceinline__ int piece_off(int ks, int p) {
        return ((ks >> 2) * 136) + (p == 0 ? 0 : 8 + ((ks & 3) * 32) + ((p - 1) * 8));
    }
    static __host__ __device__ __forceinline__ size_t row_bytes(int K) {
        return static_cast<size_t>(K / 256) * 136;
    }
    static __device__ __forceinline__ uint2 bfrag(const uint8_t* seg, int kb, int cl) {
        const int2 v = iq4nl_lut8(*reinterpret_cast<const uint32_t*>(
            seg + 8 + (static_cast<ptrdiff_t>(kb) * 16) + (static_cast<ptrdiff_t>(cl) * 4)));
        return make_uint2(static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y));
    }
    static __device__ __forceinline__ float2 scale(const uint8_t* seg, int ks, int kb) {
        const uint2 h = *reinterpret_cast<const uint2*>(seg);
        const int j = ((ks & 3) * 2) + kb;
        const uint32_t ls = ((h.y >> (4 * j)) & 0xFu) | (((h.x >> (16 + (2 * j))) & 3u) << 4);
        const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(h.x & 0xFFFFu)));
        return make_float2(d * static_cast<float>(static_cast<int>(ls) - 32), 0.0f);
    }
};

// Q8_0: two 34-B blocks (fp16 d + 32 s8) = 68 B per step, 4-B aligned (K % 256 == 0).
struct BigQ8_0 {
    static constexpr int kPieces = 17, kPiece = 4, kSegRow = 80;
    static constexpr bool kMin = false;
    static __device__ __forceinline__ int piece_off(int ks, int p) { return (ks * 68) + (p * 4); }
    static __host__ __device__ __forceinline__ size_t row_bytes(int K) {
        return static_cast<size_t>(K / 32) * 34;
    }
    static __device__ __forceinline__ uint2 bfrag(const uint8_t* seg, int kb, int cl) {
        const auto* w = reinterpret_cast<const uint32_t*>(seg);
        const int qo = (kb * 34) + 2 + (cl * 4);  // kb 0: 2 mod 4, kb 1: word-aligned
        if (kb != 0)
            return make_uint2(w[qo >> 2], w[(qo + 16) >> 2]);
        return make_uint2(__byte_perm(w[qo >> 2], w[(qo >> 2) + 1], 0x5432),
                          __byte_perm(w[(qo + 16) >> 2], w[((qo + 16) >> 2) + 1], 0x5432));
    }
    static __device__ __forceinline__ float2 scale(const uint8_t* seg, int /*ks*/, int kb) {
        return make_float2(__half2float(
                               reinterpret_cast<const __half*>(seg)[static_cast<ptrdiff_t>(kb) * 17]),
                           0.0f);
    }
};

// Q5_K 6-bit (scale, min) of sub-block j from the 12 scale bytes q (ggml get_scale_min_k4):
// (d * sc, -dmin * m).
__device__ __forceinline__ float2 k_scale_min(const uint8_t* hdr, int j) {
    const auto* h = reinterpret_cast<const __half*>(hdr);
    const uint8_t* q = hdr + 4;
    uint32_t sc, mn;
    if (j < 4) {
        sc = q[j] & 63u;
        mn = q[j + 4] & 63u;
    } else {
        sc = (q[j + 4] & 0xFu) | ((q[j - 4] >> 6) << 4);
        mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
    return make_float2(__half2float(h[0]) * static_cast<float>(sc),
                       -__half2float(h[1]) * static_cast<float>(mn));
}

// Q5_K: 176-B super-block (d, dmin, scales[12], qh[32], qs[128]); step = header + qh + 32 B of qs, element l
// of sub-block j takes bit j of qh[l]. q - 16 enters the MMA (q in 0..31 cancels the min term in fp32).
struct BigQ5K {
    static constexpr int kPieces = 5, kPiece = 16, kSegRow = 80;
    static constexpr bool kMin = true;
    static __device__ __forceinline__ int piece_off(int ks, int p) {
        return ((ks >> 2) * 176) + (p < 3 ? p * 16 : 48 + ((ks & 3) * 32) + ((p - 3) * 16));
    }
    static __host__ __device__ __forceinline__ size_t row_bytes(int K) {
        return static_cast<size_t>(K / 256) * 176;
    }
    static __device__ __forceinline__ uint2 bfrag(const uint8_t* seg, int kb, int cl, int j) {
        const auto* qh = reinterpret_cast<const uint32_t*>(seg + 16);
        const auto* qs = reinterpret_cast<const uint32_t*>(seg + 48);
        const uint32_t sh = static_cast<uint32_t>(kb) * 4;
        return make_uint2(
            __vsub4(((qs[cl] >> sh) & 0x0F0F0F0Fu) | (((qh[cl] >> j) & 0x01010101u) << 4), 0x10101010u),
            __vsub4(((qs[cl + 4] >> sh) & 0x0F0F0F0Fu) | (((qh[cl + 4] >> j) & 0x01010101u) << 4),
                    0x10101010u));
    }
    static __device__ __forceinline__ float2 scale(const uint8_t* seg, int ks, int kb) {
        const float2 s = k_scale_min(seg, ((ks & 3) * 2) + kb);  // q - 16 in the MMA: min += 16 * scale
        return make_float2(s.x, fmaf(16.0f, s.x, s.y));
    }
};

template <class F>
__device__ __forceinline__ uint2 b_of(const uint8_t* seg, int ks, int kb, int cl) {
    if constexpr (std::is_same_v<F, BigQ5K>)
        return F::bfrag(seg, kb, cl, ((ks & 3) * 2) + kb);
    else
        return F::bfrag(seg, kb, cl);
}

// One 64-K step ks: A [128][kRow] s8 + per-32 scales (+ row sums for kMin), B segments [128][F::kSegRow].
template <class F>
__device__ __forceinline__ void load_step(int tid, const int8_t* __restrict__ A,
                                          const __half* __restrict__ Asc, const float* __restrict__ Ars,
                                          const uint8_t* __restrict__ W, int8_t (*sA)[kRow],
                                          __half (*sAsc)[2], float (*sArs)[2], uint8_t (*sB)[F::kSegRow],
                                          int base_m, int M, int K, int base_n, int n_valid, int ks) {
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
        const size_t a = (static_cast<size_t>(valid ? base_m + i : 0) * subs) + (k_base / 32);
        cp_async_ca_4(&sAsc[i][0], Asc + a, valid);
        if constexpr (F::kMin)
            cp_async_ca_8(&sArs[i][0], Ars + a, valid);
    }
    const size_t row_bytes = F::row_bytes(K);
#pragma unroll
    for (int i = tid; i < kBN * F::kPieces; i += kThreads) {
        const int row = i / F::kPieces;
        const int p = i - (row * F::kPieces);
        const bool live = row < n_valid;
        const uint8_t* src = W + ((live ? static_cast<size_t>(base_n + row) : 0) * row_bytes) +
                             F::piece_off(ks, p);
        uint8_t* dst = &sB[row][static_cast<ptrdiff_t>(p) * F::kPiece];
        if constexpr (F::kPiece == 16)
            cp_async_cg_16(dst, src, live);
        else if constexpr (F::kPiece == 8)
            cp_async_ca_8(dst, src, live);
        else
            cp_async_ca_4(dst, src, live);
    }
}

template <class F, bool BETA1, bool SPLITK>
__global__ void __launch_bounds__(kThreads) mmq_imma_gguf_bm128_kernel(
    const int8_t* __restrict__ X_s8, const __half* __restrict__ x_scale, const float* __restrict__ x_rowsum,
    const uint8_t* __restrict__ W, __half* __restrict__ out, int M, int N, int K,
    float* __restrict__ split_out, int ks_per_split) {
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
    __shared__ __align__(16) uint8_t sB[kStages][kBN][F::kSegRow];
    __shared__ __align__(16) __half sAsc[kStages][kBM][2];
    __shared__ __align__(16) float sArs[kStages][F::kMin ? kBM : 1][2];
    __shared__ float2 sBs[kStages][kBN][2];  // (scale, min) per (row, sub-block of the step)

    float acc[kMF][kNF][4];
#pragma unroll
    for (int i = 0; i < kMF; ++i)
#pragma unroll
        for (int j = 0; j < kNF; ++j)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

    // SPLITK: blockIdx.z covers 64-K steps [z * ks_per_split, +ks_per_split), never empty (launch rule).
    const int ks0 = SPLITK ? static_cast<int>(blockIdx.z) * ks_per_split : 0;
    const int ksteps = SPLITK ? min(K / kBK, ks0 + ks_per_split) : K / kBK;
    load_step<F>(tid, X_s8, x_scale, x_rowsum, W, sA[0], sAsc[0], sArs[0], sB[0], base_m, M, K, base_n,
                 n_valid, ks0);
    cp_async_commit();
    for (int ks = ks0; ks < ksteps; ++ks) {
        const int stage = (ks - ks0) & 1;
        if (ks + 1 < ksteps) {
            const int ns = stage ^ 1;
            load_step<F>(tid, X_s8, x_scale, x_rowsum, W, sA[ns], sAsc[ns], sArs[ns], sB[ns], base_m, M, K,
                         base_n, n_valid, ks + 1);
            cp_async_commit();
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();
        for (int i = tid; i < kBN * 2; i += kThreads)
            sBs[stage][i >> 1][i & 1] = F::scale(sB[stage][i >> 1], ks, i & 1);
        __syncthreads();
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
            uint32_t b0[kNF], b1[kNF];
            float2 wl[kNF], wh[kNF];
#pragma unroll
            for (int nf = 0; nf < kNF; ++nf) {
                const uint2 b = b_of<F>(sB[stage][(warp_n * kTileN) + (nf * 8) + rl], ks, kb, cl);
                b0[nf] = b.x;
                b1[nf] = b.y;
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
                [[maybe_unused]] float rs_lo = 0.0f, rs_hi = 0.0f;
                if constexpr (F::kMin) {
                    rs_lo = sArs[stage][arow_lo][kb];
                    rs_hi = sArs[stage][arow_hi][kb];
                }
#pragma unroll
                for (int nf = 0; nf < kNF; ++nf) {
                    int32_t p0 = 0, p1 = 0, p2 = 0, p3 = 0;
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                        : "=r"(p0), "=r"(p1), "=r"(p2), "=r"(p3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0[nf]), "r"(b1[nf]), "r"(0), "r"(0),
                          "r"(0), "r"(0));
                    if constexpr (F::kMin) {
                        acc[mf][nf][0] += da_lo * fmaf(wl[nf].x, static_cast<float>(p0), wl[nf].y * rs_lo);
                        acc[mf][nf][1] += da_lo * fmaf(wh[nf].x, static_cast<float>(p1), wh[nf].y * rs_lo);
                        acc[mf][nf][2] += da_hi * fmaf(wl[nf].x, static_cast<float>(p2), wl[nf].y * rs_hi);
                        acc[mf][nf][3] += da_hi * fmaf(wh[nf].x, static_cast<float>(p3), wh[nf].y * rs_hi);
                    } else {
                        acc[mf][nf][0] = fmaf(da_lo * wl[nf].x, static_cast<float>(p0), acc[mf][nf][0]);
                        acc[mf][nf][1] = fmaf(da_lo * wh[nf].x, static_cast<float>(p1), acc[mf][nf][1]);
                        acc[mf][nf][2] = fmaf(da_hi * wl[nf].x, static_cast<float>(p2), acc[mf][nf][2]);
                        acc[mf][nf][3] = fmaf(da_hi * wh[nf].x, static_cast<float>(p3), acc[mf][nf][3]);
                    }
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
                if constexpr (SPLITK) {  // fp32 partial into this split's [M][N] slice
                    const size_t o = (static_cast<size_t>(blockIdx.z) * M * N) +
                                     (static_cast<size_t>(row) * N) + col;
                    *reinterpret_cast<float2*>(&split_out[o]) = make_float2(lo, hi);
                    return;
                }
                auto* p = reinterpret_cast<__half2*>(&out[(static_cast<size_t>(row) * N) + col]);
                const __half2 v = __floats2half2_rn(lo, hi);
                *p = BETA1 ? __hadd2(*p, v) : v;
            };
            store(row_lo, acc[mf][nf][0], acc[mf][nf][1]);
            store(row_lo + 8, acc[mf][nf][2], acc[mf][nf][3]);
        }
    }
}

template <class F>
void launch_big(const uint8_t* w, __half* out_f16, int M, int N, int K, float beta, bool capturing,
                cudaStream_t stream) {
    const dim3 grid((N + kBN - 1) / kBN, (M + kBM - 1) / kBM, 1);
    const bool beta1 = beta == 1.0f;
    auto* k = beta1 ? mmq_imma_gguf_bm128_kernel<F, true, false>
                    : mmq_imma_gguf_bm128_kernel<F, false, false>;
    // imma_wave_tail: trailing column tiles K-split into one more wave, the leading ones unchanged.
    const ImmaWaveTail t = imma_wave_tail(grid, M, N, K / kBK, device_sm_count_or_0());
    const int n0 = t.nx_big * kBN;
    if (t.used > 0 && imma_ensure_splitk(static_cast<size_t>(M) * (N - n0) * t.used, capturing)) {
        k<<<dim3(t.nx_big, grid.y), kThreads, 0, stream>>>(g_imma_act.xs8, g_imma_act.xscale,
                                                           g_imma_act.xrowsum, w, out_f16, M, N, K, nullptr,
                                                           0);
        IMP_CUDA_CHECK_LAUNCH();
        mmq_imma_gguf_bm128_kernel<F, false, true>
            <<<dim3(grid.x - t.nx_big, grid.y, t.used), kThreads, 0, stream>>>(
                g_imma_act.xs8, g_imma_act.xscale, g_imma_act.xrowsum,
                w + (static_cast<size_t>(n0) * F::row_bytes(K)), out_f16, M, N - n0, K, g_imma_splitk.buf,
                t.per);
        IMP_CUDA_CHECK_LAUNCH();
        imma_splitk_finalize_cols(t.used, M, N - n0, out_f16 + n0, N, beta1, stream);
        return;
    }
    k<<<grid, kThreads, 0, stream>>>(g_imma_act.xs8, g_imma_act.xscale, g_imma_act.xrowsum, w, out_f16, M, N,
                                     K, nullptr, 0);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace

bool launch_gguf_bm128(int qkind, const uint8_t* w, __half* out_f16, int M, int N, int K, float beta,
                       bool capturing, cudaStream_t stream) {
    switch (qkind) {
        case 5:
            launch_big<BigQ8_0>(w, out_f16, M, N, K, beta, capturing, stream);
            return true;
        case 7:
            launch_big<BigIQ4XS>(w, out_f16, M, N, K, beta, capturing, stream);
            return true;
        case 8:
            launch_big<BigQ5K>(w, out_f16, M, N, K, beta, capturing, stream);
            return true;
        default:
            return false;
    }
}

}  // namespace imp
