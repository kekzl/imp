#pragma once

// MXFP8 decoder policy for the weight-decode tile (gemm_wdq_f16_tile.cuh): E4M3 [N, K] + one exponent
// byte per 32 values [N, K/32], value = e4m3 * 2^(exp - 127). The tile holds e4m3 * 2^(exp - bias) as
// fp16: with bias = (largest exp of the matrix) - 6 the largest entry is 448 * 64 = 28672, and a product
// below 2^-14 rounds into the fp16 subnormals (blocks 2^11 under the matrix's largest). 2^(bias - 127)
// goes into the fp32 epilogue as alpha. User: gemm_grouped_nvfp4_f16_tile.cu (Inkling2-Loop experts).

#include "compute/gemm_wdq_f16_tile.cuh"

#include <cuda_fp16.h>
#include <cstdint>

namespace imp::wdq_tile {

struct Mx8Args {
    F16Args o;
    const uint8_t* w;   // E4M3 [N, K]
    const uint8_t* sc;  // exponent bytes [N, K/32]
    int bias;
};

// E4M3 byte -> fp16 bits, exact: normal = (e + 8) << 10 | m << 7, subnormal m * 2^-9 from a table.
__device__ __forceinline__ uint32_t mx8_half(uint32_t b) {
    const uint32_t mag = b & 0x7Fu;
    const uint32_t m = mag & 7u;
    const uint64_t tab = m < 4u ? 0x1E001C0018000000ull : 0x2300220021002000ull;
    const uint32_t sub = static_cast<uint32_t>(tab >> (16u * (m & 3u))) & 0xFFFFu;
    return (mag < 8u ? sub : (mag << 7) + 0x2000u) | ((b & 0x80u) << 8);
}

// 4 E4M3 bytes -> 2 half2 words, times s2.
__device__ __forceinline__ void mx8_word(uint32_t w, __half2 s2, uint32_t& o0, uint32_t& o1) {
    const uint32_t h0 = mx8_half(w & 0xFFu) | (mx8_half((w >> 8) & 0xFFu) << 16);
    const uint32_t h1 = mx8_half((w >> 16) & 0xFFu) | (mx8_half(w >> 24) << 16);
    const __half2 x0 = __hmul2(*reinterpret_cast<const __half2*>(&h0), s2);
    const __half2 x1 = __hmul2(*reinterpret_cast<const __half2*>(&h1), s2);
    o0 = *reinterpret_cast<const uint32_t*>(&x0);
    o1 = *reinterpret_cast<const uint32_t*>(&x1);
}

// half2(2^ex), ex <= 15; below 2^-24 it is 0.
__device__ __forceinline__ __half2 mx8_scale(int ex) {
    const uint32_t b = ex >= -14 ? static_cast<uint32_t>(ex + 15) << 10 : (ex >= -24 ? 1u << (ex + 24) : 0u);
    const uint32_t both = b | (b << 16);
    return *reinterpret_cast<const __half2*>(&both);
}

// Decoder thread (r0, hb) owns (row r0 + kStep i, 32-k half hb) per BK-64 stage: 32 E4M3 bytes and their
// exponent byte -> chunks 4 hb .. 4 hb + 3. The next stage's loads are issued before this one decodes.
// Rows past N decode row N - 1 (never stored).
struct Mx8Dec {
    static constexpr int kWideCost = 0;
    using Args = Mx8Args;
    static constexpr float kAlpha = 1.0f;
    static __device__ __forceinline__ Args make(const F16Args& o, const uint8_t* w, const uint8_t* sc,
                                                int bias) {
        return {o, w, sc, bias};
    }
    template <class C>
    static __device__ __forceinline__ void decode(const Mx8Args& p, uint32_t bars, uint8_t* smem, int n0,
                                                  int KT) {
        static_assert(C::kBK == 64, "a stage is 64 E4M3 bytes + 2 exponent bytes per row");
        constexpr int kStep = kDecThreads<C> / 2;
        constexpr int kT = C::kBN / kStep;
        const int dt = decoder_tid<C>();
        const int hb = dt & 1;
        const int r0 = dt >> 1;
        const uint8_t* wr[kT];
        const uint8_t* sr[kT];
#pragma unroll
        for (int i = 0; i < kT; ++i) {
            const int n = min(n0 + r0 + (kStep * i), p.o.N - 1);
            wr[i] = p.w + (static_cast<size_t>(n) * p.o.K) + (static_cast<size_t>(hb) * 32);
            sr[i] = p.sc + (static_cast<size_t>(n) * (p.o.K / 32)) + hb;
        }
        uint4 q[kT][2];
        uint4 qn[kT][2];
        uint32_t e[kT];
        uint32_t en[kT];
#pragma unroll
        for (int i = 0; i < kT; ++i) {
            q[i][0] = __ldg(reinterpret_cast<const uint4*>(wr[i]));
            q[i][1] = __ldg(reinterpret_cast<const uint4*>(wr[i] + 16));
            e[i] = __ldg(sr[i]);
        }
        for (int kt = 0; kt < KT; ++kt) {
            if (kt + 1 < KT) {
#pragma unroll
                for (int i = 0; i < kT; ++i) {
                    const uint8_t* nx = wr[i] + (static_cast<size_t>(kt + 1) * 64);
                    qn[i][0] = __ldg(reinterpret_cast<const uint4*>(nx));
                    qn[i][1] = __ldg(reinterpret_cast<const uint4*>(nx + 16));
                    en[i] = __ldg(sr[i] + (static_cast<size_t>(kt + 1) * 2));
                }
            }
            uint8_t* ws = stage_slot<C>(bars, smem, kt);
#pragma unroll
            for (int i = 0; i < kT; ++i) {
                const int r = r0 + (kStep * i);
                const __half2 s2 = mx8_scale(static_cast<int>(e[i]) - p.bias);
                const uint32_t w8[8] = {q[i][0].x, q[i][0].y, q[i][0].z, q[i][0].w,
                                        q[i][1].x, q[i][1].y, q[i][1].z, q[i][1].w};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    uint4 o;
                    mx8_word(w8[2 * j], s2, o.x, o.y);
                    mx8_word(w8[(2 * j) + 1], s2, o.z, o.w);
                    *reinterpret_cast<uint4*>(ws + (r * C::kRS) + (C::chunk(r, (4 * hb) + j) * 16)) = o;
                }
            }
            stage_done<C>(bars, kt);
#pragma unroll
            for (int i = 0; i < kT; ++i) {
                q[i][0] = qn[i][0];
                q[i][1] = qn[i][1];
                e[i] = en[i];
            }
        }
    }
};

}  // namespace imp::wdq_tile
