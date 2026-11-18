#pragma once

// NVFP4 decoder policy for the weight-decode tile (gemm_wdq_f16_tile.cuh), plain layout: E2M1 [N, K/2] (even
// element low nibble) + E4M3 micro-scales [N, K/16]. E2M1 -> fp16 bits by shifts (value * 2^-14), one HMUL2
// by half2(e4m3 * 2^6): e4m3 <= 448 keeps it below 28672, value * e4m3 * 2^-8 (<= 6 significant bits,
// >= 2^-18) is exact in fp16. The tensor scale stays in the fp32 epilogue. Users: gemm_nvfp4_f16_tile.cu,
// gemm_grouped_nvfp4_f16_tile.cu.

#include "compute/gemm_wdq_f16_tile.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace imp::wdq_tile {

struct Nv4Args {
    F16Args o;
    const uint8_t* w;   // E2M1 [N, K/2]
    const uint8_t* sc;  // E4M3 [N, K/16]
};

// 8 E2M1 codes (one word, even element in the low nibble) -> 4 half2 (one 16 B chunk), times s2.
__device__ __forceinline__ uint4 nv4_chunk(uint32_t w, __half2 s2) {
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

// E4M3 byte -> half2(e4m3 * 64), exact.
__device__ __forceinline__ __half2 nv4_scale(uint32_t e) {
    const __half h = __half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(e), __NV_E4M3));
    return __half2half2(__hmul(h, __float2half(64.0f)));
}

// Decoder thread (r0, hb) owns (row r0 + kStep i, 32-k half hb) per BK-64 stage: one 16 B E2M1 load and the
// two E4M3 bytes of its two 16-k blocks -> chunks 4 hb .. 4 hb + 3 (chunks 0-1 block 0, 2-3 block 1). The
// next stage's loads are issued before this one decodes. Rows past N decode row N - 1 (never stored).
struct Nv4Dec {
    static constexpr int kWideCost = 0;  // 128x128 only, as MXFP4
    using Args = Nv4Args;
    template <class C>
    static __device__ __forceinline__ void decode(const Nv4Args& p, uint32_t bars, uint8_t* smem, int n0,
                                                  int KT) {
        static_assert(C::kBK == 64, "a stage is 32 E2M1 bytes + 4 scale bytes per row");
        constexpr int kStep = kDecThreads<C> / 2;
        constexpr int kT = C::kBN / kStep;
        const int dt = decoder_tid<C>();
        const int hb = dt & 1;
        const int r0 = dt >> 1;
        const uint8_t* wr[kT];
        const uint16_t* sr[kT];
#pragma unroll
        for (int i = 0; i < kT; ++i) {
            const int n = min(n0 + r0 + (kStep * i), p.o.N - 1);
            wr[i] = p.w + (static_cast<size_t>(n) * (p.o.K / 2)) + (static_cast<size_t>(hb) * 16);
            sr[i] = reinterpret_cast<const uint16_t*>(p.sc + (static_cast<size_t>(n) * (p.o.K / 16))) + hb;
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
                const __half2 s_lo = nv4_scale(e[i] & 0xFFu);
                const __half2 s_hi = nv4_scale(e[i] >> 8);
                const uint32_t w4[4] = {q[i].x, q[i].y, q[i].z, q[i].w};
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    *reinterpret_cast<uint4*>(ws + (r * C::kRS) + (C::chunk(r, (4 * hb) + j) * 16)) =
                        nv4_chunk(w4[j], j < 2 ? s_lo : s_hi);
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

}  // namespace imp::wdq_tile
