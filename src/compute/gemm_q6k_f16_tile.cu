// Fused Q6_K prompt GEMM (ROADMAP row 123): the weight-decode tile (gemm_wdq_f16_tile.cuh) with W read as
// raw GGUF Q6_K blocks. The decoder warps load the stage's Q6_K bytes from L2 and write the swizzled fp16
// W slot (fp32 d * sc * (q - 32), the dequant_q6k_v2 bits). Replaces the whole-weight dequant + cuBLASLt
// pass (Qwen3-14B-Q6_K pp512: dequant 35 % of kernels).

#include "compute/gemm_q6k_f16_tile.h"

#include "compute/gemm_wdq_f16_tile.cuh"

#include <cuda_fp16.h>
#include <cstdint>

namespace imp {
namespace {

using namespace wdq_tile;  // NOLINT(google-build-using-namespace): the shared decode-tile skeleton

template <class C>
constexpr int kRowStep = kDecThreads<C> / 4;  // rows per decoder pass: 4 threads (u) per row
template <class C>
constexpr int kUnits = C::kBN / kRowStep<C>;
constexpr int kQ6Bytes = 210;  // ql[128] qh[64] scales[16] d (fp16) per 256 weights
// Raw bytes of one (row, u) unit of half-block g: ql 8u.. and 32 + 8u.. (low nibbles = stage 2g, high =
// stage 2g + 1 of the block), qh 8u.. (all four bit pairs), the half-block's 8 scales, d.
struct Q6Unit {
    uint32_t qa[2], qb[2], qh[2], sc[2];
    float d;
};

// 8 bytes at blk + off (off % 8 == 0, blk even): three aligned words funnel-shifted by sh; the third
// word only when sh != 0 (the last run ends at the block's last scale byte).
__device__ __forceinline__ void ld_run(uint32_t (&o)[2], const uint32_t* base, int off, uint32_t sh) {
    const uint32_t* p = base + (off >> 2);
    const uint32_t w0 = __ldg(p);
    const uint32_t w1 = __ldg(p + 1);
    const uint32_t w2 = __ldg(sh != 0 ? p + 2 : p + 1);
    o[0] = __funnelshift_r(w0, w1, sh);
    o[1] = __funnelshift_r(w1, w2, sh);
}

__device__ __forceinline__ void ld_unit(Q6Unit& t, const uint8_t* blk, int g, int u) {
    const auto a = reinterpret_cast<uintptr_t>(blk);
    const auto* base = reinterpret_cast<const uint32_t*>(a & ~uintptr_t(3));
    const uint32_t sh = static_cast<uint32_t>(a & 3) * 8;
    ld_run(t.qa, base, (g * 64) + (u * 8), sh);
    ld_run(t.qb, base, (g * 64) + 32 + (u * 8), sh);
    ld_run(t.qh, base, 128 + (g * 32) + (u * 8), sh);
    ld_run(t.sc, base, 192 + (g * 8), sh);
    t.d = __half2float(__ushort_as_half(*reinterpret_cast<const unsigned short*>(blk + 208)));
}

// fp16 pair of bytes k, k + 1 of q (6-bit codes): d*sc * (q - 32) in fp32 (exact: <= 23 bits), one rounding.
__device__ __forceinline__ uint32_t q6_pair(uint32_t q, int k, float ds) {
    const float f0 = __uint_as_float(__byte_perm(q, 0x4B000000u, 0x7440 + k)) - 8388640.0f;
    const float f1 = __uint_as_float(__byte_perm(q, 0x4B000000u, 0x7440 + k + 1)) - 8388640.0f;
    const __half2 h = __floats2half2_rn(ds * f0, ds * f1);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// One 8-k chunk: ql nibble h, qh bit pair `quad`, scale byte jb of scale word h (h is a compile-time
// unroll index, so t.sc stays in registers).
__device__ __forceinline__ uint4 q6_chunk(const uint32_t (&ql)[2], const Q6Unit& t, int h, int quad, int jb) {
    const auto sc = static_cast<int8_t>((t.sc[h] >> (jb * 8)) & 0xFFu);
    const float ds = t.d * static_cast<float>(sc);
    const uint32_t q0 = ((ql[0] >> (h * 4)) & 0x0F0F0F0Fu) | (((t.qh[0] >> (quad * 2)) & 0x03030303u) << 4);
    const uint32_t q1 = ((ql[1] >> (h * 4)) & 0x0F0F0F0Fu) | (((t.qh[1] >> (quad * 2)) & 0x03030303u) << 4);
    return make_uint4(q6_pair(q0, 0, ds), q6_pair(q0, 2, ds), q6_pair(q1, 0, ds), q6_pair(q1, 2, ds));
}

struct Q6Args {
    F16Args o;
    const uint8_t* w;
    size_t row_bytes;
};

// Decoder warps: thread (r0, u) owns units (r0 + kRowStep i, u); one load burst per half-block (128 k)
// feeds its two BK-64 stages: stage j = nibble j, chunks u (quad 2j, ql 8u) and u + 4 (quad 2j + 1,
// ql 32 + 8u). The next half-block's burst is issued before this one decodes (without it the MMA warps
// waited once per burst, 2048x17408x5120 +30 %). Rows past N decode row N - 1 (never stored).
template <class C>
__device__ __forceinline__ void ld_units(Q6Unit (&t)[kUnits<C>], const Q6Args& p, int n0, int r0, int u,
                                         int kp) {
#pragma unroll
    for (int i = 0; i < kUnits<C>; ++i) {
        const int n = min(n0 + r0 + (kRowStep<C> * i), p.o.N - 1);
        ld_unit(t[i],
                p.w + (static_cast<size_t>(n) * p.row_bytes) + (static_cast<size_t>(kp >> 1) * kQ6Bytes),
                kp & 1, u);
    }
}

struct Q6Dec {
    static constexpr bool kWide = true;
    using Args = Q6Args;
    template <class C>
    static __device__ __forceinline__ void decode(const Q6Args& p, uint32_t bars, uint8_t* smem, int n0,
                                                  int KT) {
        static_assert(C::kBK == 64 && kUnits<C> <= 4,
                      "two stages per half-block; the prefetch holds 2 x 36 regs");
        constexpr int kU = kUnits<C>;
        const int dt = decoder_tid();
        const int u = dt & 3;
        const int r0 = dt >> 2;
        Q6Unit t[kU];
        Q6Unit nx[kU];
        ld_units<C>(t, p, n0, r0, u, 0);
        for (int kp = 0; kp < KT / 2; ++kp) {
            if (kp + 1 < KT / 2)
                ld_units<C>(nx, p, n0, r0, u, kp + 1);
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int kt = (kp * 2) + j;
                uint8_t* ws = stage_slot<C>(bars, smem, kt);
#pragma unroll
                for (int i = 0; i < kU; ++i) {
                    const int r = r0 + (kRowStep<C> * i);
                    *reinterpret_cast<uint4*>(ws + (r * C::kRS) +
                                              (C::chunk(r, u) * 16)) = q6_chunk(t[i].qa, t[i], j, 2 * j,
                                                                                u >> 1);
                    *reinterpret_cast<uint4*>(
                        ws + (r * C::kRS) + (C::chunk(r, u + 4) * 16)) = q6_chunk(t[i].qb, t[i], j,
                                                                                  (2 * j) + 1, 2 + (u >> 1));
                }
                stage_done<C>(bars, kt);
            }
#pragma unroll
            for (int i = 0; i < kU; ++i)
                t[i] = nx[i];
        }
    }
};

}  // namespace

bool gemm_q6k_f16_tile(const void* a, const void* w_q6k, void* d, int M, int N, int K, float alpha,
                       float beta, bool acc_fp16, cudaStream_t stream, int variant) {
    if (M <= 0 || N <= 0 || K <= 0 || K % 256 != 0 || N % 8 != 0 || a == nullptr || w_q6k == nullptr ||
        d == nullptr || reinterpret_cast<uintptr_t>(a) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(w_q6k) % 2 != 0 || reinterpret_cast<uintptr_t>(d) % 16 != 0)
        return false;
    const Q6Args p{{static_cast<half*>(d), alpha, beta, M, N, K},
                   static_cast<const uint8_t*>(w_q6k),
                   static_cast<size_t>(K / 256) * kQ6Bytes};
    return launch_auto<Q6Dec>(p, a, acc_fp16, variant, stream);
}

// In situ Qwen3-14B-Q6_K vs dequant + cuBLASLt: pp512 +27 %, pp1536 +5 %, pp2048 -1.7 % (sustained
// 2048x17408x5120: 1319 vs 1277 us, 571 vs 548 W). Grids under 128 tiles lose (512x1024x5120 78 vs 43 us).
// runtime.deterministic: every prompt row (tile-invariant bits); PPL ppl_4k 11.4542 vs 11.4543 off.
bool gemm_q6k_f16_tile_wins(int M, int N, bool deterministic) {
    return deterministic || (M <= 1536 && ((M + 127) / 128) * ((N + 127) / 128) >= 128);
}

}  // namespace imp
