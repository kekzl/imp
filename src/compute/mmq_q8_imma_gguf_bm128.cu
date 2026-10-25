// Dense GGUF IMMA kernel for prompt chunks above 32 rows (sm_120a): BM=128 x BN=128 tiles, K-steps of
// S 32-K sub-blocks per barrier pair, A via ldmatrix.x4. B: Q8_0 decoded per warp, IQ3_S / IQ4 / Q5_K
// once per CTA into a predecode tile (IQ4_XS gate 298.3 -> 270.6 us). S: Q5_K 4, others 2; S-free math.

#include "compute/iq3s_grid.cuh"
#include "compute/iq4_lut.cuh"
#include "compute/mmq_q8_imma_internal.cuh"
#include "core/device_info.h"
#include "core/logging.h"

#include <type_traits>

namespace imp {

namespace {

constexpr int kBM = 128;

// K-step of S sub-blocks of 32 (S = 2 or 4); kARow 80 / 144 B puts the 8 rl rows on distinct banks.
template <int S>
struct StepGeom {
    static constexpr int kSub = S;
    static constexpr int kStepK = S * 32;
    static constexpr int kARow = kStepK + 16;
    static constexpr int kSpb = 8 / S;  // K-steps per 256-K super-block
};

// Format policy: one K-step of a weight row is kPieces pieces of kPiece bytes (piece_off: byte offset in
// the row), staged as a kSegRow-B smem segment (kSegRow / 4 % 32 in {4, 12, 20, 28}: rl rows on distinct
// banks). scale(seg, ks, kb) = (weight scale, min) of sub-block kb of step ks; kMin adds min * activation
// row sum.

// IQ4_XS: the super-block header (u64) + 16 B of qs per sub-block.
template <int S>
struct BigIQ4XS : StepGeom<S> {
    using G = StepGeom<S>;
    static constexpr int kPieces = 1 + (2 * S), kPiece = 8, kSegRow = S == 4 ? 80 : 48;
    static constexpr bool kMin = false;
    static __device__ __forceinline__ int piece_off(int ks, int p) {
        return ((ks / G::kSpb) * 136) + (p == 0 ? 0 : 8 + ((ks % G::kSpb) * (S * 16)) + ((p - 1) * 8));
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
        const int j = ((ks % G::kSpb) * S) + kb;
        const uint32_t ls = ((h.y >> (4 * j)) & 0xFu) | (((h.x >> (16 + (2 * j))) & 3u) << 4);
        const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(h.x & 0xFFFFu)));
        return make_float2(d * static_cast<float>(static_cast<int>(ls) - 32), 0.0f);
    }
};

// Q8_0: S 34-B blocks (fp16 d + 32 s8) per step, 4-B aligned (K % 256 == 0).
template <int S>
struct BigQ8_0 : StepGeom<S> {
    static constexpr int kPieces = (S * 34) / 4, kPiece = 4, kSegRow = S == 4 ? 144 : 80;
    static constexpr bool kMin = false;
    static __device__ __forceinline__ int piece_off(int ks, int p) { return (ks * S * 34) + (p * 4); }
    static __host__ __device__ __forceinline__ size_t row_bytes(int K) {
        return static_cast<size_t>(K / 32) * 34;
    }
    static __device__ __forceinline__ uint2 bfrag(const uint8_t* seg, int kb, int cl) {
        const auto* w = reinterpret_cast<const uint32_t*>(seg);
        const int qo = (kb * 34) + 2 + (cl * 4);  // even kb: 2 mod 4, odd kb: word-aligned
        if ((kb & 1) != 0)
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

// IQ4_NL: S 18-B blocks (fp16 d + 16 B nibbles, IQ4 LUT) per step, 4-B aligned (K % 256 == 0).
template <int S>
struct BigIQ4NL : StepGeom<S> {
    static constexpr int kPieces = (S * 18) / 4, kPiece = 4, kSegRow = S == 4 ? 80 : 48;
    static constexpr bool kMin = false;
    static __device__ __forceinline__ int piece_off(int ks, int p) { return (ks * S * 18) + (p * 4); }
    static __host__ __device__ __forceinline__ size_t row_bytes(int K) {
        return static_cast<size_t>(K / 32) * 18;
    }
    static __device__ __forceinline__ uint2 bfrag(const uint8_t* seg, int kb, int cl) {
        const auto* w = reinterpret_cast<const uint32_t*>(seg);
        const int qo = (kb * 18) + 2 + (cl * 4);  // even kb: 2 mod 4, odd kb: word-aligned
        const uint32_t q = (kb & 1) != 0 ? w[qo >> 2] : __byte_perm(w[qo >> 2], w[(qo >> 2) + 1], 0x5432);
        const int2 v = iq4nl_lut8(q);
        return make_uint2(static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y));
    }
    static __device__ __forceinline__ float2 scale(const uint8_t* seg, int /*ks*/, int kb) {
        return make_float2(__half2float(reinterpret_cast<const __half*>(seg)[static_cast<ptrdiff_t>(kb) * 9]),
                           0.0f);
    }
};

// IQ3_S (S = 2): 110-B super-block (d, qs[64], qh[8], signs[32], scales[4]), 2-aligned; K % 512 == 0 keeps
// rows 4-aligned. A step stages each field from the word below it: d 1, qs 5, qh 1, signs 3, scale 1 piece
// (all inside the row); a field sits at its row offset mod 4 within its pieces (o = 2 for odd blocks).
template <int S>
struct BigIQ3S : StepGeom<S> {
    static_assert(S == 2, "one scale byte per step");
    using G = StepGeom<S>;
    static constexpr int kPieces = 12, kPiece = 4, kSegRow = 48;  // piece 11 = piece 10: 6 uniform rounds
    static constexpr bool kMin = false;
    static __device__ __forceinline__ int piece_off(int ks, int p) {
        const int base = (ks / G::kSpb) * 110;
        const int j0 = (ks % G::kSpb) * S;
        const int f = p == 0   ? base
                      : p < 6  ? base + 2 + (8 * j0) + (4 * (p - 1))
                      : p == 6 ? base + 66 + j0
                      : p < 10 ? base + 74 + (4 * j0) + (4 * (p - 7))
                               : base + 106 + (j0 / 2);
        return f & ~3;
    }
    static __host__ __device__ __forceinline__ size_t row_bytes(int K) {
        return static_cast<size_t>(K / 256) * 110;
    }
    static __device__ __forceinline__ int odd2(int ks) { return 2 * ((ks / G::kSpb) & 1); }
    // Sub-block kb as 32 s8 (word g = elements 4g..4g+3) into dst (16-B aligned); decoded once per CTA into
    // the predecode tile, one thread per (row, kb).
    static __device__ __forceinline__ void decode_sub(const uint8_t* seg, int ks, int kb, uint32_t* dst,
                                                      const uint32_t* grid) {
        const int o = odd2(ks);
        const int j0 = (ks % G::kSpb) * S;
        const int qo = 4 + ((o + 2) & 3) + (8 * kb);
        const auto* qp = reinterpret_cast<const uint16_t*>(seg + qo);
        const uint32_t q0 = static_cast<uint32_t>(qp[0]) | (static_cast<uint32_t>(qp[1]) << 16);
        const uint32_t q1 = static_cast<uint32_t>(qp[2]) | (static_cast<uint32_t>(qp[3]) << 16);
        const uint32_t qh = seg[24 + ((o + 2 + j0) & 3) + kb];
        const int so = 28 + ((o + 2) & 3) + (4 * kb);
        const auto* sh = reinterpret_cast<const uint16_t*>(seg + so);
        const uint32_t sg = static_cast<uint32_t>(sh[0]) | (static_cast<uint32_t>(sh[1]) << 16);
        uint32_t v[8];
#pragma unroll
        for (int g = 0; g < 8; g++) {
            const uint32_t q = ((g < 4 ? q0 : q1) >> (8 * (g & 3))) & 0xFFu;
            v[g] = static_cast<uint32_t>(iq3s_apply_signs(grid[q | (((qh >> g) & 1u) << 8)], sg >> (4 * g)));
        }
        reinterpret_cast<uint4*>(dst)[0] = make_uint4(v[0], v[4], v[1], v[5]);  // (b0, b1) of cl adjacent
        reinterpret_cast<uint4*>(dst)[1] = make_uint4(v[2], v[6], v[3], v[7]);
    }
    static __device__ __forceinline__ float2 scale(const uint8_t* seg, int ks, int kb) {
        const int o = odd2(ks);
        const int j0 = (ks % G::kSpb) * S;
        const uint32_t sc = (seg[40 + ((o + 2 + (j0 / 2)) & 3)] >> (4 * kb)) & 0xFu;
        return make_float2(__half2float(*reinterpret_cast<const __half*>(seg + o)) *
                               static_cast<float>(1 + (2 * sc)),
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

// Q5_K: 176-B super-block (d, dmin, scales[12], qh[32], qs[128]); step = header + qh + S * 16 B of qs (32-B
// chunks, low nibble = even sub-block), element l of sub-block j takes bit j of qh[l]. q - 16 enters the MMA
// (q in 0..31 cancels the min term in fp32).
template <int S>
struct BigQ5K : StepGeom<S> {
    using G = StepGeom<S>;
    static constexpr int kPieces = 3 + S, kPiece = 16, kSegRow = S == 4 ? 112 : 80;
    static constexpr bool kMin = true;
    static __device__ __forceinline__ int piece_off(int ks, int p) {
        return ((ks / G::kSpb) * 176) + (p < 3 ? p * 16 : 48 + ((ks % G::kSpb) * (S * 16)) + ((p - 3) * 16));
    }
    static __host__ __device__ __forceinline__ size_t row_bytes(int K) {
        return static_cast<size_t>(K / 256) * 176;
    }
    static __device__ __forceinline__ uint2 bfrag(const uint8_t* seg, int kb, int cl, int j) {
        const auto* qh = reinterpret_cast<const uint32_t*>(seg + 16);
        const auto* qs = reinterpret_cast<const uint32_t*>(seg + 48 + (static_cast<ptrdiff_t>(kb >> 1) * 32));
        const uint32_t sh = static_cast<uint32_t>(kb & 1) * 4;
        return make_uint2(
            __vsub4(((qs[cl] >> sh) & 0x0F0F0F0Fu) | (((qh[cl] >> j) & 0x01010101u) << 4), 0x10101010u),
            __vsub4(((qs[cl + 4] >> sh) & 0x0F0F0F0Fu) | (((qh[cl + 4] >> j) & 0x01010101u) << 4),
                    0x10101010u));
    }
    static __device__ __forceinline__ float2 scale(const uint8_t* seg, int ks, int kb) {
        // q - 16 in the MMA: min += 16 * scale
        const float2 s = k_scale_min(seg, ((ks % G::kSpb) * S) + kb);
        return make_float2(s.x, fmaf(16.0f, s.x, s.y));
    }
};

template <class F>
inline constexpr bool kIsQ5K = false;
template <int S>
inline constexpr bool kIsQ5K<BigQ5K<S>> = true;
template <class F>
inline constexpr bool kIsIQ3S = false;
template <int S>
inline constexpr bool kIsIQ3S<BigIQ3S<S>> = true;

template <class F>
__device__ __forceinline__ uint2 b_of(const uint8_t* seg, int ks, int kb, int cl) {
    if constexpr (kIsQ5K<F>)
        return F::bfrag(seg, kb, cl, ((ks % F::kSpb) * F::kSub) + kb);
    else
        return F::bfrag(seg, kb, cl);
}

// Sub-block weight scale as staged in smem: (scale, min) for kMin formats, the scale alone otherwise.
template <class F>
__device__ __forceinline__ std::conditional_t<F::kMin, float2, float> scale_of(const uint8_t* seg, int ks,
                                                                               int kb) {
    if constexpr (F::kMin)
        return F::scale(seg, ks, kb);
    else
        return F::scale(seg, ks, kb).x;
}

template <class F>
inline constexpr bool kIsIQ4 = false;
template <int S>
inline constexpr bool kIsIQ4<BigIQ4XS<S>> = true;
template <int S>
inline constexpr bool kIsIQ4<BigIQ4NL<S>> = true;

// Predecode (IQ3_S, IQ4_XS, IQ4_NL): the stage's B tile decoded once per CTA to s8 [kBN][kDecRow] after the
// scale pass, instead of once per warp_m (4x codebook lookups: IQ3_S 563 / 630 us vs 406 / 457 us).
template <class F>
inline constexpr bool kPredecode = kIsIQ3S<F> || kIsIQ4<F> || kIsQ5K<F>;
template <class F>
inline constexpr int kDecRow = F::kSub == 2 ? 96 : 160;  // row mod 128 B in {32, 96}: LDS.64 conflict-free
template <class F>
constexpr size_t dec_bytes() {
    if constexpr (kIsIQ3S<F>)
        return (static_cast<size_t>(kBN) * kDecRow<F>)+sizeof(kIq3sGridLdg);
    else if constexpr (kPredecode<F>)
        return static_cast<size_t>(kBN) * kDecRow<F>;
    else
        return 0;
}

// Sub-block kb of one row as the 4 lanes' (b0, b1) pairs, cl-major: dst[2 cl], dst[2 cl + 1] (16-B aligned).
template <class F>
__device__ __forceinline__ void predecode_sub(const uint8_t* seg, int ks, int kb, uint32_t* dst,
                                              const uint32_t* grid) {
    if constexpr (kIsIQ3S<F>) {
        F::decode_sub(seg, ks, kb, dst, grid);
    } else {
        uint2 b[4];
#pragma unroll
        for (int cl = 0; cl < 4; ++cl)
            b[cl] = b_of<F>(seg, ks, kb, cl);
        reinterpret_cast<uint4*>(dst)[0] = make_uint4(b[0].x, b[0].y, b[1].x, b[1].y);
        reinterpret_cast<uint4*>(dst)[1] = make_uint4(b[2].x, b[2].y, b[3].x, b[3].y);
    }
}

// Lane (rl, cl) B fragment of sub-block kb: from the predecoded tile, else decoded from the staged segment.
template <class F>
__device__ __forceinline__ uint2 bfrag_at(const uint8_t* seg, const uint32_t* sDec, int brow, int ks, int kb,
                                          int cl) {
    if constexpr (kPredecode<F>)
        return *reinterpret_cast<const uint2*>(sDec + (brow * (kDecRow<F> / 4)) + (kb * 8) + (2 * cl));
    else
        return b_of<F>(seg, ks, kb, cl);
}

// Weight scales of columns ncol, ncol + 1 (one row of the [sub-block][128] scale tile): one 8-B / 16-B load.
__device__ __forceinline__ void bscale_at(const float* row, int ncol, float2& wl, float2& wh) {
    const float2 s = *reinterpret_cast<const float2*>(row + ncol);
    wl = make_float2(s.x, 0.0f);
    wh = make_float2(s.y, 0.0f);
}
__device__ __forceinline__ void bscale_at(const float2* row, int ncol, float2& wl, float2& wh) {
    const float4 s = *reinterpret_cast<const float4*>(row + ncol);
    wl = make_float2(s.x, s.y);
    wh = make_float2(s.z, s.w);
}

// One stage of dynamic smem: A [128][kARow] s8, B segments [128][kSegRow], A per-32 scales, A row sums
// (kMin), B [sub-block][128] scale (kMin: (scale, min)). Every part is a multiple of 16 B.
template <class F>
struct BigSmem {
    static constexpr size_t kA = static_cast<size_t>(kBM) * F::kARow;
    static constexpr size_t kB = static_cast<size_t>(kBN) * F::kSegRow;
    static constexpr size_t kAsc = static_cast<size_t>(kBM) * F::kSub * sizeof(__half);
    static constexpr size_t kArs = F::kMin ? static_cast<size_t>(kBM) * F::kSub * sizeof(float) : 0;
    using BsT = std::conditional_t<F::kMin, float2, float>;
    static constexpr size_t kBs = static_cast<size_t>(kBN) * F::kSub * sizeof(BsT);
    static constexpr size_t kStage = kA + kB + kAsc + kArs + kBs;
    static constexpr size_t kBytes = (kStages * kStage) + dec_bytes<F>();
};

// One K-step ks: A [128][kARow] s8 + per-32 scales (+ row sums for kMin), B segments [128][F::kSegRow].
template <class F>
__device__ __forceinline__ void load_step(int tid, const int8_t* __restrict__ A,
                                          const __half* __restrict__ Asc, const float* __restrict__ Ars,
                                          const uint8_t* __restrict__ W, int8_t (*sA)[F::kARow],
                                          __half (*sAsc)[F::kSub], float (*sArs)[F::kSub],
                                          uint8_t (*sB)[F::kSegRow], int base_m, int M, int K, int base_n,
                                          int n_valid, int ks) {
    constexpr int kChunks = F::kStepK / 16;  // 16-B chunks per A row
    const int subs = K / 32;
    const int k_base = ks * F::kStepK;
#pragma unroll
    for (int i = tid; i < kBM * kChunks; i += kThreads) {
        const int row = i / kChunks;
        const int col = (i % kChunks) * 16;
        const bool valid = (base_m + row) < M;
        cp_async_cg_16(&sA[row][col], A + (static_cast<size_t>(valid ? base_m + row : 0) * K) + k_base + col,
                       valid);
    }
    for (int i = tid; i < kBM; i += kThreads) {
        const bool valid = (base_m + i) < M;
        const size_t a = (static_cast<size_t>(valid ? base_m + i : 0) * subs) + (k_base / 32);
        if constexpr (F::kSub == 4) {
            cp_async_ca_8(&sAsc[i][0], Asc + a, valid);
            if constexpr (F::kMin)
                cp_async_cg_16(&sArs[i][0], Ars + a, valid);
        } else {
            cp_async_ca_4(&sAsc[i][0], Asc + a, valid);
            if constexpr (F::kMin)
                cp_async_ca_8(&sArs[i][0], Ars + a, valid);
        }
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
    using L = BigSmem<F>;
    constexpr int kSub = F::kSub;
    constexpr int kARow = F::kARow;

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

    extern __shared__ __align__(16) uint8_t smem[];
    auto sA = [&](int s) { return reinterpret_cast<int8_t (*)[kARow]>(smem + (s * L::kStage)); };
    auto sB = [&](int s) {
        return reinterpret_cast<uint8_t (*)[F::kSegRow]>(smem + (s * L::kStage) + L::kA);
    };
    auto sAsc = [&](int s) {
        return reinterpret_cast<__half(*)[kSub]>(smem + (s * L::kStage) + L::kA + L::kB);
    };
    auto sArs = [&](int s) {
        return reinterpret_cast<float (*)[kSub]>(smem + (s * L::kStage) + L::kA + L::kB + L::kAsc);
    };
    auto sBs = [&](int s) {
        return reinterpret_cast<typename L::BsT(*)[kBN]>(smem + (s * L::kStage) + L::kA + L::kB + L::kAsc +
                                                         L::kArs);
    };
    // IQ3_S: the codebook in smem after the decoded tile (cp.async.ca pieces share L1 with __ldg).
    [[maybe_unused]] uint32_t* sGrid = nullptr;
    if constexpr (kIsIQ3S<F>) {
        sGrid = reinterpret_cast<uint32_t*>(smem + (kStages * L::kStage) +
                                            (static_cast<size_t>(kBN) * kDecRow<F>));
        for (int i = tid; i < 512; i += kThreads)
            sGrid[i] = __ldg(&kIq3sGridLdg[i]);
    }

    float acc[kMF][kNF][4];
#pragma unroll
    for (int i = 0; i < kMF; ++i)
#pragma unroll
        for (int j = 0; j < kNF; ++j)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

    // SPLITK: blockIdx.z covers K-steps [z * ks_per_split, +ks_per_split), never empty (launch rule).
    const int ks0 = SPLITK ? static_cast<int>(blockIdx.z) * ks_per_split : 0;
    const int ksteps = SPLITK ? min(K / F::kStepK, ks0 + ks_per_split) : K / F::kStepK;
    load_step<F>(tid, X_s8, x_scale, x_rowsum, W, sA(0), sAsc(0), sArs(0), sB(0), base_m, M, K, base_n,
                 n_valid, ks0);
    cp_async_commit();
    for (int ks = ks0; ks < ksteps; ++ks) {
        const int stage = (ks - ks0) & 1;
        cp_async_wait_group<0>();
        __syncthreads();  // step ks landed; every warp left step ks - 1, so stage ^ 1 and sDec are free
        if (ks + 1 < ksteps) {
            const int ns = stage ^ 1;
            load_step<F>(tid, X_s8, x_scale, x_rowsum, W, sA(ns), sAsc(ns), sArs(ns), sB(ns), base_m, M, K,
                         base_n, n_valid, ks + 1);
            cp_async_commit();
        }
        const auto stA = sA(stage);
        const auto stB = sB(stage);
        const auto stAsc = sAsc(stage);
        [[maybe_unused]] const auto stArs = sArs(stage);
        const auto stBs = sBs(stage);
        for (int i = tid; i < kBN * kSub; i += kThreads)
            stBs[i / kBN][i % kBN] = scale_of<F>(stB[i % kBN], ks, i / kBN);
        [[maybe_unused]] auto* sDec = reinterpret_cast<uint32_t*>(smem + (kStages * L::kStage));
        if constexpr (kPredecode<F>) {
            constexpr int kDecW = kDecRow<F> / 4;
            for (int i = tid; i < kBN * kSub; i += kThreads) {  // (row, kb)
                const int dw = ((i / kSub) * kDecW) + ((i % kSub) * 8);
                predecode_sub<F>(stB[i / kSub], ks, i % kSub, sDec + dw, sGrid);
            }
        }
        __syncthreads();
#pragma unroll
        for (int kb = 0; kb < kSub; ++kb) {
            uint32_t b0[kNF], b1[kNF];
            float2 wl[kNF], wh[kNF];  // .y: min term (kMin only)
#pragma unroll
            for (int nf = 0; nf < kNF; ++nf) {
                const int brow = (warp_n * kTileN) + (nf * 8) + rl;
                const uint2 b = bfrag_at<F>(stB[brow], sDec, brow, ks, kb, cl);
                b0[nf] = b.x;
                b1[nf] = b.y;
                bscale_at(stBs[kb], (warp_n * kTileN) + (nf * 8) + (cl * 2), wl[nf], wh[nf]);
            }
#pragma unroll
            for (int mf = 0; mf < kMF; ++mf) {
                const int arow_lo = (warp_m * kTileM) + (mf * 16) + rl;
                const int arow_hi = arow_lo + 8;
                // A: one ldmatrix.x4 = (rows 0-7, 8-15) x (k 0-15, 16-31) of the 16 x 32 fragment
                uint32_t a0, a1, a2, a3;
                {
                    const int m = lane >> 3;
                    const int row = (warp_m * kTileM) + (mf * 16) + (lane & 7) + ((m & 1) * 8);
                    ldmatrix_x4(a0, a1, a2, a3, &stA[row][(kb * 32) + ((m >> 1) * 16)]);
                }
                const float da_lo = __half2float(stAsc[arow_lo][kb]);
                const float da_hi = __half2float(stAsc[arow_hi][kb]);
                [[maybe_unused]] float rs_lo = 0.0f, rs_hi = 0.0f;
                if constexpr (F::kMin) {
                    rs_lo = stArs[arow_lo][kb];
                    rs_hi = stArs[arow_hi][kb];
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

template <typename Kernel>
bool big_smem_opt_in(Kernel k, size_t bytes) {
    const cudaError_t err = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 static_cast<int>(bytes));
    if (err != cudaSuccess)
        IMP_LOG_ERROR("mmq imma gguf bm128: smem opt-in %zu B failed: %s", bytes, cudaGetErrorString(err));
    return err == cudaSuccess;
}

template <class F>
bool launch_big(const uint8_t* w, __half* out_f16, int M, int N, int K, float beta, bool capturing,
                cudaStream_t stream) {
    constexpr size_t smem = BigSmem<F>::kBytes;
    static const bool smem_ok = big_smem_opt_in(mmq_imma_gguf_bm128_kernel<F, false, false>, smem) &&
                                big_smem_opt_in(mmq_imma_gguf_bm128_kernel<F, true, false>, smem) &&
                                big_smem_opt_in(mmq_imma_gguf_bm128_kernel<F, false, true>, smem);
    if (!smem_ok)
        return false;
    const dim3 grid((N + kBN - 1) / kBN, (M + kBM - 1) / kBM, 1);
    const bool beta1 = beta == 1.0f;
    auto* k = beta1 ? mmq_imma_gguf_bm128_kernel<F, true, false>
                    : mmq_imma_gguf_bm128_kernel<F, false, false>;
    // imma_wave_tail: trailing column tiles K-split into one more wave, the leading ones unchanged.
    const ImmaWaveTail t = imma_wave_tail(grid, M, N, K / F::kStepK, device_sm_count_or_0());
    const int n0 = t.nx_big * kBN;
    if (t.used > 0 && imma_ensure_splitk(static_cast<size_t>(M) * (N - n0) * t.used, capturing)) {
        k<<<dim3(t.nx_big, grid.y), kThreads, smem, stream>>>(g_imma_act.xs8, g_imma_act.xscale,
                                                              g_imma_act.xrowsum, w, out_f16, M, N, K,
                                                              nullptr, 0);
        IMP_CUDA_CHECK_LAUNCH();
        mmq_imma_gguf_bm128_kernel<F, false, true>
            <<<dim3(grid.x - t.nx_big, grid.y, t.used), kThreads, smem, stream>>>(
                g_imma_act.xs8, g_imma_act.xscale, g_imma_act.xrowsum,
                w + (static_cast<size_t>(n0) * F::row_bytes(K)), out_f16, M, N - n0, K, g_imma_splitk.buf,
                t.per);
        IMP_CUDA_CHECK_LAUNCH();
        imma_splitk_finalize_cols(t.used, M, N - n0, out_f16 + n0, N, beta1, stream);
        return true;
    }
    k<<<grid, kThreads, smem, stream>>>(g_imma_act.xs8, g_imma_act.xscale, g_imma_act.xrowsum, w, out_f16, M,
                                        N, K, nullptr, 0);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

}  // namespace

bool launch_gguf_bm128(int qkind, const uint8_t* w, __half* out_f16, int M, int N, int K, float beta,
                       bool capturing, cudaStream_t stream) {
    switch (qkind) {
        case 5:
            return launch_big<BigQ8_0<2>>(w, out_f16, M, N, K, beta, capturing, stream);
        case 7:
            return launch_big<BigIQ4XS<2>>(w, out_f16, M, N, K, beta, capturing, stream);
        case 6:
            return launch_big<BigIQ4NL<2>>(w, out_f16, M, N, K, beta, capturing, stream);
        case 8:
            return launch_big<BigQ5K<4>>(w, out_f16, M, N, K, beta, capturing, stream);
        case 9:
            return launch_big<BigIQ3S<2>>(w, out_f16, M, N, K, beta, capturing, stream);
        default:
            return false;
    }
}

}  // namespace imp
