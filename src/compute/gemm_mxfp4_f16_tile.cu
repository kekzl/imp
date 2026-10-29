// Fused MXFP4 W4A16 prompt GEMM (ROADMAP row 123): the weight-decode tile (gemm_wdq_f16_tile.cuh) with W
// read as unpacked E2M1 + UE8M0 bytes instead of a dequant into a scratch weight + gemm() per call.
// E2M1 -> fp16 bits by shifts (magnitude bits at 9..11 = value * 2^-14, sign at 15), then one HMUL2 by
// half2(2^14 * scale): exact for scales 2^-38 .. 2^1, so the dequant_mxfp4_linear_kernel bits.

#include "compute/gemm_mxfp4_f16_tile.h"

#include "compute/gemm_wdq_f16_tile.cuh"

#include <cuda_fp16.h>
#include <cstdint>

namespace imp {
namespace {

using namespace wdq_tile;  // NOLINT(google-build-using-namespace): the shared decode-tile skeleton

struct Mx4Args {
    F16Args o;
    const uint8_t* w;   // E2M1 [N, K/2]
    const uint8_t* sc;  // UE8M0 [N, K/32]
};

// 8 E2M1 codes (one word, even element in the low nibble) -> 4 half2 (one 16 B chunk), times s2.
__device__ __forceinline__ uint4 mx4_chunk(uint32_t w, __half2 s2) {
    uint32_t o[4];
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const uint32_t v = __byte_perm(w, 0u, 0x4440 + b) * 4097u;  // byte | byte << 12: hi nibble at 16
        const uint32_t h = ((v << 9) & 0x0E000E00u) | ((v << 12) & 0x80008000u);
        const __half2 x = __hmul2(*reinterpret_cast<const __half2*>(&h), s2);
        o[b] = *reinterpret_cast<const uint32_t*>(&x);
    }
    return make_uint4(o[0], o[1], o[2], o[3]);
}

// Decoder thread (r0, hb) owns (row r0 + kStep i, 32-k half hb) per BK-64 stage: one 16 B E2M1 load and
// one scale byte -> chunks 4 hb .. 4 hb + 3. The next stage's loads are issued before this one decodes.
// Rows past N decode row N - 1 (never stored).
struct Mx4Dec {
    static constexpr bool kWide = false;
    using Args = Mx4Args;
    template <class C>
    static __device__ __forceinline__ void decode(const Mx4Args& p, uint32_t bars, uint8_t* smem, int n0,
                                                  int KT) {
        static_assert(C::kBK == 64, "a stage is 32 E2M1 bytes + 2 scale bytes per row");
        constexpr int kStep = kDecThreads<C> / 2;
        constexpr int kT = C::kBN / kStep;
        const int dt = decoder_tid();
        const int hb = dt & 1;
        const int r0 = dt >> 1;
        const uint8_t* wr[kT];
        const uint8_t* sr[kT];
#pragma unroll
        for (int i = 0; i < kT; ++i) {
            const int n = min(n0 + r0 + (kStep * i), p.o.N - 1);
            wr[i] = p.w + (static_cast<size_t>(n) * (p.o.K / 2)) + (static_cast<size_t>(hb) * 16);
            sr[i] = p.sc + (static_cast<size_t>(n) * (p.o.K / 32)) + hb;
        }
        uint4 q[kT];
        uint4 qn[kT];
        uint32_t e[kT];
        uint32_t en[kT];
#pragma unroll
        for (int i = 0; i < kT; ++i) {
            q[i] = __ldg(reinterpret_cast<const uint4*>(wr[i]));
            e[i] = __ldg(sr[i]);
        }
        for (int kt = 0; kt < KT; ++kt) {
            if (kt + 1 < KT) {
#pragma unroll
                for (int i = 0; i < kT; ++i) {
                    qn[i] = __ldg(reinterpret_cast<const uint4*>(wr[i] + (static_cast<size_t>(kt + 1) * 32)));
                    en[i] = __ldg(sr[i] + (static_cast<size_t>(kt + 1) * 2));
                }
            }
            uint8_t* ws = stage_slot<C>(bars, smem, kt);
#pragma unroll
            for (int i = 0; i < kT; ++i) {
                const int r = r0 + (kStep * i);
                const __half2 s2 = __float2half2_rn(__uint_as_float(e[i] << 23) * 16384.0f);
                const uint32_t w4[4] = {q[i].x, q[i].y, q[i].z, q[i].w};
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    *reinterpret_cast<uint4*>(ws + (r * C::kRS) +
                                              (C::chunk(r, (4 * hb) + j) * 16)) = mx4_chunk(w4[j], s2);
            }
            stage_done<C>(bars, kt);
#pragma unroll
            for (int i = 0; i < kT; ++i) {
                q[i] = qn[i];
                e[i] = en[i];
            }
        }
    }
};

}  // namespace

bool gemm_mxfp4_f16_tile(const void* a, const void* w_e2m1, const void* scales_ue8m0, void* d, int M, int N,
                         int K, float alpha, float beta, bool acc_fp16, cudaStream_t stream, int variant) {
    if (M <= 0 || N <= 0 || K <= 0 || K % 64 != 0 || N % 8 != 0 || a == nullptr || w_e2m1 == nullptr ||
        scales_ue8m0 == nullptr || d == nullptr || reinterpret_cast<uintptr_t>(a) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(w_e2m1) % 16 != 0 || reinterpret_cast<uintptr_t>(d) % 16 != 0)
        return false;
    const Mx4Args p{{static_cast<half*>(d), alpha, beta, M, N, K},
                    static_cast<const uint8_t*>(w_e2m1),
                    static_cast<const uint8_t*>(scales_ue8m0)};
    return launch_auto<Mx4Dec>(p, a, acc_fp16, variant, stream);
}

}  // namespace imp
