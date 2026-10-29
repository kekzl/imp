// Fused Q4_K / Q5_K prompt GEMM (ROADMAP row 123): the weight-decode tile (gemm_wdq_f16_tile.cuh) with W
// read as raw GGUF Q4_K (144 B) / Q5_K (176 B) blocks. The decoder warps write the swizzled fp16 W slot
// (dl * q - ml in fp32, the dequant_q45k_v2 bits). Replaces the raw IMMA tiles at M > 32 (A stays fp16).

#include "compute/gemm_q45k_f16_tile.h"

#include "compute/gemm_wdq_f16_tile.cuh"

#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>

namespace imp {
namespace {

using namespace wdq_tile;  // NOLINT(google-build-using-namespace): the shared decode-tile skeleton

template <class C>
constexpr int kRowStep = kDecThreads<C> / 4;  // rows per decoder pass: 4 threads (u) per row
template <class C>
constexpr int kUnits = C::kBN / kRowStep<C>;

// One (row, u) unit of half-block g (64-k chunks 2g, 2g + 1 = two stages): d/dmin + the 12 scale bytes,
// qs bytes 8u.. of both chunks (low nibbles = 8-k chunk u, high = u + 4), qh bytes 8u.. (Q5_K only).
struct KqUnit {
    uint4 hd;
    uint2 qs[2];
    uint2 qh;
};

struct KqArgs {
    F16Args o;
    const uint8_t* w;
    size_t row_bytes;
};

template <bool kQ5>
__device__ __forceinline__ void ld_unit(KqUnit& t, const uint8_t* blk, int g, int u) {
    constexpr int kQs = kQ5 ? 48 : 16;
    t.hd = __ldg(reinterpret_cast<const uint4*>(blk));
#pragma unroll
    for (int j = 0; j < 2; ++j)
        t.qs[j] = __ldg(reinterpret_cast<const uint2*>(
            blk + static_cast<ptrdiff_t>(kQs + (((2 * g) + j) * 32) + (u * 8))));
    if constexpr (kQ5)
        t.qh = __ldg(reinterpret_cast<const uint2*>(blk + static_cast<ptrdiff_t>(16 + (u * 8))));
}

// Scale byte b (0..11) of the 12-byte get_scale_min_k4 array (header words y, z, w).
__device__ __forceinline__ int sc_byte(const uint4& hd, int b) {
    const uint32_t w = b < 4 ? hd.y : (b < 8 ? hd.z : hd.w);
    return static_cast<int>((w >> ((b & 3) * 8)) & 0xFFu);
}

// 8-k chunk of sub-block sub (0..7, compile-time after unrolling): nibble half sub & 1 of q, qh bit sub.
template <bool kQ5>
__device__ __forceinline__ uint4 kq_chunk(const KqUnit& t, uint2 q, int sub) {
    int s = 0;
    int m = 0;
    if (sub < 4) {
        s = sc_byte(t.hd, sub) & 63;
        m = sc_byte(t.hd, sub + 4) & 63;
    } else {
        s = (sc_byte(t.hd, sub + 4) & 0xF) | ((sc_byte(t.hd, sub - 4) >> 6) << 4);
        m = (sc_byte(t.hd, sub + 4) >> 4) | ((sc_byte(t.hd, sub) >> 6) << 4);
    }
    const __half2 dm = *reinterpret_cast<const __half2*>(&t.hd.x);
    const float dl = __low2float(dm) * static_cast<float>(s);
    const float ml = __high2float(dm) * static_cast<float>(m);
    const int nshift = (sub & 1) * 4;
    uint32_t o[4];
#pragma unroll
    for (int w = 0; w < 2; ++w) {
        uint32_t v = ((w == 0 ? q.x : q.y) >> nshift) & 0x0F0F0F0Fu;
        if constexpr (kQ5)
            v |= (((w == 0 ? t.qh.x : t.qh.y) >> sub) & 0x01010101u) << 4;
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const float f0 = __uint_as_float(__byte_perm(v, 0x4B000000u, 0x7440 + (2 * p))) - 8388608.0f;
            const float f1 = __uint_as_float(__byte_perm(v, 0x4B000000u, 0x7440 + (2 * p) + 1)) - 8388608.0f;
            const __half2 h = __floats2half2_rn((dl * f0) - ml, (dl * f1) - ml);
            o[(2 * w) + p] = *reinterpret_cast<const uint32_t*>(&h);
        }
    }
    return make_uint4(o[0], o[1], o[2], o[3]);
}

template <class C, bool kQ5>
__device__ __forceinline__ void ld_units(KqUnit (&t)[kUnits<C>], const KqArgs& p, int n0, int r0, int u,
                                         int kb, int g) {
    constexpr int kBytes = kQ5 ? 176 : 144;
#pragma unroll
    for (int i = 0; i < kUnits<C>; ++i) {
        const int n = min(n0 + r0 + (kRowStep<C> * i), p.o.N - 1);
        ld_unit<kQ5>(t[i], p.w + (static_cast<size_t>(n) * p.row_bytes) + (static_cast<size_t>(kb) * kBytes),
                     g, u);
    }
}

// Decoder warps: thread (r0, u) owns units (r0 + kRowStep i, u); one load burst per half-block feeds its
// two BK-64 stages, the next burst issued before this one decodes (as Q6_K). Rows past N decode row N - 1.
template <bool kQ5>
struct KqDec {
    static constexpr int kWideCost = 17;  // wide_wins: 1024x6144x5120 Q5_K 128x128 278 vs 320 us
    using Args = KqArgs;
    template <class C>
    static __device__ __forceinline__ void decode(const KqArgs& p, uint32_t bars, uint8_t* smem, int n0,
                                                  int KT) {
        static_assert(C::kBK == 64 && kUnits<C> <= 4,
                      "four stages per block; the prefetch holds 2 x 32 regs");
        constexpr int kU = kUnits<C>;
        const int dt = decoder_tid();
        const int u = dt & 3;
        const int r0 = dt >> 2;
        const int KB = KT / 4;
        KqUnit t[kU];
        KqUnit nx[kU];
        ld_units<C, kQ5>(t, p, n0, r0, u, 0, 0);
        for (int kb = 0; kb < KB; ++kb) {
#pragma unroll
            for (int g = 0; g < 2; ++g) {
                if (g == 0)
                    ld_units<C, kQ5>(nx, p, n0, r0, u, kb, 1);
                else if (kb + 1 < KB)
                    ld_units<C, kQ5>(nx, p, n0, r0, u, kb + 1, 0);
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int kt = (kb * 4) + (g * 2) + j;
                    const int sub = 2 * ((2 * g) + j);
                    uint8_t* ws = stage_slot<C>(bars, smem, kt);
#pragma unroll
                    for (int i = 0; i < kU; ++i) {
                        const int r = r0 + (kRowStep<C> * i);
                        *reinterpret_cast<uint4*>(
                            ws + (r * C::kRS) + (C::chunk(r, u) * 16)) = kq_chunk<kQ5>(t[i], t[i].qs[j], sub);
                        *reinterpret_cast<uint4*>(ws + (r * C::kRS) +
                                                  (C::chunk(r, u + 4) * 16)) = kq_chunk<kQ5>(t[i], t[i].qs[j],
                                                                                             sub + 1);
                    }
                    stage_done<C>(bars, kt);
                }
#pragma unroll
                for (int i = 0; i < kU; ++i)
                    t[i] = nx[i];
            }
        }
    }
};

}  // namespace

bool gemm_q45k_f16_tile(bool q5k, const void* a, const void* w, void* d, int M, int N, int K, float alpha,
                        float beta, bool acc_fp16, cudaStream_t stream, int variant) {
    if (M <= 0 || N <= 0 || K <= 0 || K % 256 != 0 || N % 8 != 0 || a == nullptr || w == nullptr ||
        d == nullptr || reinterpret_cast<uintptr_t>(a) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(w) % 16 != 0 || reinterpret_cast<uintptr_t>(d) % 16 != 0)
        return false;
    const KqArgs p{{static_cast<half*>(d), alpha, beta, M, N, K},
                   static_cast<const uint8_t*>(w),
                   static_cast<size_t>(K / 256) * (q5k ? 176 : 144)};
    return q5k ? launch_auto<KqDec<true>>(p, a, acc_fp16, variant, stream)
               : launch_auto<KqDec<false>>(p, a, acc_fp16, variant, stream);
}

}  // namespace imp
