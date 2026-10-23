// Dense small-M IMMA kernel on GGUF blocks in place (sm_120a): BM=32, one 256-K step per stage, cp.async
// staging, K-split by N and K. Formats with one scale per 32 elements: Q8_0, IQ4_NL, IQ4_XS (symmetric) and
// Q5_K (scale + min, min term via the activation row sums). Launched from gemm_common (qkind 5..8); above
// 32 rows gridDim.x runs 32-row blocks (no K-split).

#include "compute/iq4_lut.cuh"
#include "compute/mmq_q8_imma_internal.cuh"
#include "core/logging.h"

#include <utility>

namespace imp {

namespace {

constexpr int kSubs = 8;         // 32-element sub-blocks per step
constexpr int kARow = 256 + 16;  // A row stride: 68 words, rl lanes on distinct banks
static_assert(kSubs * 32 == 256, "a step is 256 K");

// Block bytes [off, off + 4) of a staged row with off % 4 in {0, 2}: kAligned picks the plain word read.
template <bool kAligned>
__device__ __forceinline__ uint32_t rd_word(const uint32_t* row, int off) {
    if constexpr (kAligned)
        return row[off >> 2];
    else
        return __byte_perm(row[off >> 2], row[(off >> 2) + 1], 0x5432);
}

// Format policy: kStep bytes per 256-K step of a weight row, staged in kChunk-B cp.async pieces at kRow-B
// smem stride (kRow / 4 % 32 in {4, 12}: the 8 rl rows hit distinct banks). bfrag<J>: the m16n8k32 B
// registers (k = 4 cl .. 4 cl + 3 and + 16) of sub-block J; wscale<J>: (scale, min) of sub-block J.

// Q8_0: 8 x 34-B blocks (fp16 d + 32 s8).
struct FmtQ8_0 {
    static constexpr int kStep = 272, kChunk = 16, kRow = 272;
    static constexpr bool kMin = false;
    template <int J>
    static __device__ __forceinline__ uint2 bfrag(const uint8_t* row, int cl) {
        const auto* w = reinterpret_cast<const uint32_t*>(row);
        const int qo = (J * 34) + 2 + (cl * 4);
        return make_uint2(rd_word<(J & 1) != 0>(w, qo), rd_word<(J & 1) != 0>(w, qo + 16));
    }
    template <int J>
    static __device__ __forceinline__ float2 wscale(const uint8_t* row) {
        return make_float2(__half2float(reinterpret_cast<const __half*>(row)[static_cast<ptrdiff_t>(J) * 17]),
                           0.0f);
    }
};

// IQ4_NL: 8 x 18-B blocks (fp16 d + 16 B of codebook nibbles, element e low / e + 16 high nibble of qs[e]).
struct FmtIQ4NL {
    static constexpr int kStep = 144, kChunk = 16, kRow = 144;
    static constexpr bool kMin = false;
    template <int J>
    static __device__ __forceinline__ uint2 bfrag(const uint8_t* row, int cl) {
        const auto* w = reinterpret_cast<const uint32_t*>(row);
        const int2 v = iq4nl_lut8(rd_word<(J & 1) != 0>(w, (J * 18) + 2 + (cl * 4)));
        return make_uint2(static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y));
    }
    template <int J>
    static __device__ __forceinline__ float2 wscale(const uint8_t* row) {
        return make_float2(__half2float(reinterpret_cast<const __half*>(row)[static_cast<ptrdiff_t>(J) * 9]),
                           0.0f);
    }
};

// IQ4_XS: one 136-B super-block (fp16 d, scales_h u16, scales_l u32, qs[128]); scale d * (ls - 32).
// 136 B is 8-B aligned only: 8-B chunks, row padded to 144 B.
struct FmtIQ4XS {
    static constexpr int kStep = 136, kChunk = 8, kRow = 144;
    static constexpr bool kMin = false;
    template <int J>
    static __device__ __forceinline__ uint2 bfrag(const uint8_t* row, int cl) {
        const int2 v = iq4nl_lut8(reinterpret_cast<const uint32_t*>(row)[2 + (4 * J) + cl]);
        return make_uint2(static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y));
    }
    template <int J>
    static __device__ __forceinline__ float2 wscale(const uint8_t* row) {
        const auto* w = reinterpret_cast<const uint32_t*>(row);
        const uint32_t ls = ((w[1] >> (4 * J)) & 0xFu) | (((w[0] >> (16 + (2 * J))) & 3u) << 4);
        const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(w[0] & 0xFFFFu)));
        return make_float2(d * static_cast<float>(static_cast<int>(ls) - 32), 0.0f);
    }
};

// Q5_K: one 176-B super-block (d, dmin, scales[12], qh[32], qs[128]); q - 16 enters the s8 MMA (q in 0..31
// cancels the min term in fp32), x = d sc (q - 16) + 16 d sc - dmin m: (scale, min) = (d sc, 16 d sc - dmin
// m). Element l of sub-block J takes bit J of qh[l].
struct FmtQ5K {
    static constexpr int kStep = 176, kChunk = 16, kRow = 176;
    static constexpr bool kMin = true;
    template <int J>
    static __device__ __forceinline__ uint2 bfrag(const uint8_t* row, int cl) {
        const auto* w = reinterpret_cast<const uint32_t*>(row);
        constexpr int kQs = 12 + (8 * (J >> 1));
        constexpr uint32_t kSh = (J & 1) * 4;
        const uint32_t b0 = ((w[kQs + cl] >> kSh) & 0x0F0F0F0Fu) | (((w[4 + cl] >> J) & 0x01010101u) << 4);
        const uint32_t b1 = ((w[kQs + 4 + cl] >> kSh) & 0x0F0F0F0Fu) |
                            (((w[8 + cl] >> J) & 0x01010101u) << 4);
        return make_uint2(__vsub4(b0, 0x10101010u), __vsub4(b1, 0x10101010u));
    }
    template <int J>
    static __device__ __forceinline__ float2 wscale(const uint8_t* row) {
        const auto* h = reinterpret_cast<const __half*>(row);
        const uint8_t* q = row + 4;
        uint32_t sc, mn;
        if constexpr (J < 4) {
            sc = q[J] & 63u;
            mn = q[J + 4] & 63u;
        } else {
            sc = (q[J + 4] & 0xFu) | ((q[J - 4] >> 6) << 4);
            mn = (q[J + 4] >> 4) | ((q[J] >> 6) << 4);
        }
        const float a = __half2float(h[0]) * static_cast<float>(sc);
        return make_float2(a, fmaf(16.0f, a, -__half2float(h[1]) * static_cast<float>(mn)));
    }
};

// Stage: A [32][kARow] s8, A scales [32][8] fp16, A row sums [32][8] fp32 (kMin only), B [128][F::kRow].
template <class F>
constexpr size_t stage_bytes() {
    return (size_t{32} * kARow) + (size_t{32} * kSubs * sizeof(__half)) +
           (F::kMin ? size_t{32} * kSubs * sizeof(float) : 0) + (static_cast<size_t>(kBN) * F::kRow);
}
template <class F>
constexpr size_t smem_bytes() {
    return kStages * stage_bytes<F>();
}

// One step st: rows >= M and >= n_valid are zero-filled.
template <class F>
__device__ __forceinline__ void load_step(int tid, const int8_t* __restrict__ A,
                                          const __half* __restrict__ Asc, const float* __restrict__ Ars,
                                          const uint8_t* __restrict__ W, uint8_t* sA, __half* sAsc,
                                          float* sArs, uint8_t* sB, int M, int K, int base_n, int n_valid,
                                          int st) {
    const int subs = K / 32;
#pragma unroll
    for (int i = tid; i < 32 * 16; i += kThreads) {
        const int row = i >> 4;
        const int col = (i & 15) * 16;
        cp_async_cg_16(sA + static_cast<ptrdiff_t>((row * kARow) + col),
                       A + (static_cast<size_t>(row) * K) + (static_cast<size_t>(st) * 256) + col, row < M);
    }
    if (tid < 32 * 4) {
        const int row = tid >> 2;
        const int c = (tid & 3) * 2;
        cp_async_ca_4(sAsc + static_cast<ptrdiff_t>((row * kSubs) + c),
                      Asc + (static_cast<size_t>(row) * subs) + (static_cast<size_t>(st) * kSubs) + c,
                      row < M);
    }
    if constexpr (F::kMin) {
        if (tid < 32 * 2) {
            const int row = tid >> 1;
            const int c = (tid & 1) * 4;
            cp_async_cg_16(sArs + static_cast<ptrdiff_t>((row * kSubs) + c),
                           Ars + (static_cast<size_t>(row) * subs) + (static_cast<size_t>(st) * kSubs) + c,
                           row < M);
        }
    }
    constexpr int kChunks = F::kStep / F::kChunk;
    static_assert(F::kStep % F::kChunk == 0, "a step must be whole chunks");
    const size_t row_bytes = static_cast<size_t>(K / 256) * F::kStep;
#pragma unroll
    for (int i = tid; i < kBN * kChunks; i += kThreads) {
        const int row = i / kChunks;
        const int c = i - (row * kChunks);
        const bool live = row < n_valid;
        const uint8_t* src = W + ((live ? static_cast<size_t>(base_n + row) : 0) * row_bytes) +
                             (static_cast<size_t>(st) * F::kStep) + (static_cast<size_t>(c) * F::kChunk);
        uint8_t* dst = sB + static_cast<ptrdiff_t>((row * F::kRow) + (c * F::kChunk));
        if constexpr (F::kChunk == 16)
            cp_async_cg_16(dst, src, live);
        else
            cp_async_ca_8(dst, src, live);
    }
}

template <class F, int J, int kNF>
__device__ __forceinline__ void sub_block(float (&acc)[kNF][4], const uint8_t* sA, const __half* sAsc,
                                          const float* sArs, const uint8_t* sB, int arow_lo, int bcol0,
                                          int ncol0, int rl, int cl) {
    const int arow_hi = arow_lo + 8;
    const int acol = (J * 32) + (cl * 4);
    const uint8_t* arl = sA + static_cast<ptrdiff_t>(arow_lo * kARow);
    const uint8_t* arh = sA + static_cast<ptrdiff_t>(arow_hi * kARow);
    const uint32_t a0 = *reinterpret_cast<const uint32_t*>(arl + acol);
    const uint32_t a1 = *reinterpret_cast<const uint32_t*>(arh + acol);
    const uint32_t a2 = *reinterpret_cast<const uint32_t*>(arl + acol + 16);
    const uint32_t a3 = *reinterpret_cast<const uint32_t*>(arh + acol + 16);
    const float da_lo = __half2float(sAsc[(arow_lo * kSubs) + J]);
    const float da_hi = __half2float(sAsc[(arow_hi * kSubs) + J]);
    [[maybe_unused]] float rs_lo = 0.0f, rs_hi = 0.0f;
    if constexpr (F::kMin) {
        rs_lo = sArs[(arow_lo * kSubs) + J];
        rs_hi = sArs[(arow_hi * kSubs) + J];
    }
#pragma unroll
    for (int nf = 0; nf < kNF; ++nf) {
        const int brow = bcol0 + (nf * 8) + rl;
        const uint2 b = F::template bfrag<J>(sB + (static_cast<ptrdiff_t>(brow) * F::kRow), cl);
        const int ncol = ncol0 + (nf * 8);
        const float2 wl = F::template wscale<J>(sB + (static_cast<ptrdiff_t>(ncol) * F::kRow));
        const float2 wh = F::template wscale<J>(sB + ((static_cast<ptrdiff_t>(ncol) + 1) * F::kRow));
        int32_t p0 = 0, p1 = 0, p2 = 0, p3 = 0;
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=r"(p0), "=r"(p1), "=r"(p2), "=r"(p3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b.x), "r"(b.y), "r"(0), "r"(0), "r"(0), "r"(0));
        if constexpr (F::kMin) {
            acc[nf][0] += da_lo * fmaf(wl.x, static_cast<float>(p0), wl.y * rs_lo);
            acc[nf][1] += da_lo * fmaf(wh.x, static_cast<float>(p1), wh.y * rs_lo);
            acc[nf][2] += da_hi * fmaf(wl.x, static_cast<float>(p2), wl.y * rs_hi);
            acc[nf][3] += da_hi * fmaf(wh.x, static_cast<float>(p3), wh.y * rs_hi);
        } else {
            acc[nf][0] = fmaf(da_lo * wl.x, static_cast<float>(p0), acc[nf][0]);
            acc[nf][1] = fmaf(da_lo * wh.x, static_cast<float>(p1), acc[nf][1]);
            acc[nf][2] = fmaf(da_hi * wl.x, static_cast<float>(p2), acc[nf][2]);
            acc[nf][3] = fmaf(da_hi * wh.x, static_cast<float>(p3), acc[nf][3]);
        }
    }
}

template <class F, int kNF, int... J>
__device__ __forceinline__ void step(float (&acc)[kNF][4], const uint8_t* sA, const __half* sAsc,
                                     const float* sArs, const uint8_t* sB, int arow_lo, int bcol0, int ncol0,
                                     int rl, int cl, std::integer_sequence<int, J...> /*subs*/) {
    (sub_block<F, J, kNF>(acc, sA, sAsc, sArs, sB, arow_lo, bcol0, ncol0, rl, cl), ...);
}

// SPLITK: gridDim.z = K-split, fp32 partials to split_out[z][M][N], st_per_split 256-K steps each.
template <class F, bool BETA1, bool SPLITK>
__global__ void __launch_bounds__(kThreads) mmq_imma_gguf_smallm_kernel(
    const int8_t* __restrict__ X_s8, const __half* __restrict__ x_scale, const float* __restrict__ x_rowsum,
    const uint8_t* __restrict__ W, __half* __restrict__ out, int M, int N, int K,
    float* __restrict__ split_out, int st_per_split) {
    constexpr int kWN = 4;
    constexpr int kTileN = kBN / kWN;
    constexpr int kNF = kTileN / 8;

    // 32-row block blockIdx.x, weight tile blockIdx.y: CTAs of one tile run adjacent and share it in L2.
    const int base_m = static_cast<int>(blockIdx.x) * 32;
    const size_t a_off = static_cast<size_t>(base_m) * (K / 32);
    X_s8 += static_cast<size_t>(base_m) * K;
    x_scale += a_off;
    x_rowsum += a_off;
    out += static_cast<size_t>(base_m) * N;
    M = min(32, M - base_m);
    const int base_n = static_cast<int>(blockIdx.y) * kBN;
    const int n_valid = min(kBN, N - base_n);
    const int steps = K / 256;
    const int st_begin = SPLITK ? static_cast<int>(blockIdx.z) * st_per_split : 0;
    const int st_end = SPLITK ? min(steps, st_begin + st_per_split) : steps;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = warp_id / kWN;
    const int warp_n = warp_id % kWN;
    const int rl = lane >> 2;
    const int cl = lane & 3;
    const int arow_lo = (warp_m * 16) + rl;
    const int bcol0 = warp_n * kTileN;
    const int ncol0 = bcol0 + (cl * 2);

    extern __shared__ __align__(16) uint8_t smem_gguf_sm[];
    constexpr size_t kAsc = size_t{32} * kARow;
    constexpr size_t kArs = kAsc + (size_t{32} * kSubs * sizeof(__half));
    constexpr size_t kB = kArs + (F::kMin ? size_t{32} * kSubs * sizeof(float) : 0);
    auto sA = [&](int s) { return smem_gguf_sm + (static_cast<size_t>(s) * stage_bytes<F>()); };
    auto sAsc = [&](int s) { return reinterpret_cast<__half*>(sA(s) + kAsc); };
    auto sArs = [&](int s) { return reinterpret_cast<float*>(sA(s) + kArs); };
    auto sB = [&](int s) { return sA(s) + kB; };

    float acc[kNF][4];
#pragma unroll
    for (int nf = 0; nf < kNF; ++nf)
        acc[nf][0] = acc[nf][1] = acc[nf][2] = acc[nf][3] = 0.0f;

    const bool rows_live = warp_m * 16 < M;  // M <= 16: the warp_m = 1 half only loads and syncs
    // Launch guarantees st_begin < st_end.
    load_step<F>(tid, X_s8, x_scale, x_rowsum, W, sA(0), sAsc(0), sArs(0), sB(0), M, K, base_n, n_valid,
                 st_begin);
    cp_async_commit();
    for (int st = st_begin; st < st_end; ++st) {
        const int stage = (st - st_begin) & 1;
        if (st + 1 < st_end) {
            const int ns = stage ^ 1;
            load_step<F>(tid, X_s8, x_scale, x_rowsum, W, sA(ns), sAsc(ns), sArs(ns), sB(ns), M, K, base_n,
                         n_valid, st + 1);
            cp_async_commit();
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();
        if (rows_live)
            step<F, kNF>(acc, sA(stage), sAsc(stage), sArs(stage), sB(stage), arow_lo, bcol0, ncol0, rl, cl,
                         std::make_integer_sequence<int, kSubs>{});
        __syncthreads();
    }

    imma_small_m_store<BETA1, SPLITK, kNF>(acc, out,
                                           split_out + (SPLITK ? static_cast<size_t>(blockIdx.z) * M * N : 0),
                                           arow_lo, base_n + ncol0, M, N);
}

template <class F>
bool launch_fmt(const uint8_t* w, __half* out_f16, int M, int N, int K, float beta, bool capturing,
                cudaStream_t stream) {
    using Fn = void (*)(const int8_t*, const __half*, const float*, const uint8_t*, __half*, int, int, int,
                        float*, int);
    const Fn k0 = mmq_imma_gguf_smallm_kernel<F, false, false>;
    const Fn k1 = mmq_imma_gguf_smallm_kernel<F, true, false>;
    const Fn ks = mmq_imma_gguf_smallm_kernel<F, false, true>;
    static bool smem_ok = false;  // per format
    if (!smem_ok)
        smem_ok = imma_smem_opt_in(reinterpret_cast<const void*>(k0), smem_bytes<F>()) &&
                  imma_smem_opt_in(reinterpret_cast<const void*>(k1), smem_bytes<F>()) &&
                  imma_smem_opt_in(reinterpret_cast<const void*>(ks), smem_bytes<F>());
    if (!smem_ok)
        return false;
    const bool beta1 = beta == 1.0f;
    const int n_tiles = (N + kBN - 1) / kBN;
    const int m_blocks = (M + 31) / 32;
    const int steps = K / 256;
    const int s = M <= 32 ? imma_dense_splits(N, steps) : 1;  // K-split for one row block only
    const int per = (steps + s - 1) / s;
    const int used = (steps + per - 1) / per;
    const size_t slice = static_cast<size_t>(M) * N;
    if (used > 1 && imma_ensure_splitk(slice * used, capturing)) {
        ks<<<dim3(1, n_tiles, used), kThreads, smem_bytes<F>(), stream>>>(g_imma_act.xs8, g_imma_act.xscale,
                                                                          g_imma_act.xrowsum, w, out_f16, M,
                                                                          N, K, g_imma_splitk.buf, per);
        IMP_CUDA_CHECK_LAUNCH();
        imma_splitk_finalize(used, static_cast<int>(slice), out_f16, beta1, stream);
        return true;
    }
    (beta1 ? k1 : k0)<<<dim3(m_blocks, n_tiles, 1), kThreads, smem_bytes<F>(), stream>>>(
        g_imma_act.xs8, g_imma_act.xscale, g_imma_act.xrowsum, w, out_f16, M, N, K, nullptr, 0);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

}  // namespace

bool launch_gguf_small_m(int qkind, const uint8_t* w, __half* out_f16, int M, int N, int K, float beta,
                         bool capturing, cudaStream_t stream) {
    switch (qkind) {
        case 5:
            return M > 32 ? launch_gguf_bm128(5, w, out_f16, M, N, K, beta, capturing, stream)
                          : launch_fmt<FmtQ8_0>(w, out_f16, M, N, K, beta, capturing, stream);
        case 6:
            return launch_fmt<FmtIQ4NL>(w, out_f16, M, N, K, beta, capturing, stream);
        case 7:
            return M > 32 ? launch_gguf_bm128(7, w, out_f16, M, N, K, beta, capturing, stream)
                          : launch_fmt<FmtIQ4XS>(w, out_f16, M, N, K, beta, capturing, stream);
        case 8:
            return M > 32 ? launch_gguf_bm128(8, w, out_f16, M, N, K, beta, capturing, stream)
                          : launch_fmt<FmtQ5K>(w, out_f16, M, N, K, beta, capturing, stream);
        default:
            return false;
    }
}

}  // namespace imp
