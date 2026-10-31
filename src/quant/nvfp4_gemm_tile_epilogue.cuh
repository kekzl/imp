#pragma once

// Operands, ring geometry and epilogues of the own dense tile GEMM (nvfp4_gemm_tile.cu, its only includer):
// staged 16 B stores, direct fragment stores, stmatrix + TMA store. sm_120a.

#include "quant/nvfp4_gemm_tile_core.cuh"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace imp::nvfp4_tile {

struct TileArgs {
    const uint8_t* a;
    const uint8_t* a_sf;
    const uint8_t* w;
    const uint8_t* w_sf;
    half* d;
    float alpha;
    int M, N, K;
    // Stream-K (Cfg::kSk): one fp32 partial tile per CTA (workspace); each CTA owns units
    // [b * sk_per, (b + 1) * sk_per) of tiles * KT. Arrival flags: g_sk_flags.
    float* sk_parts;
    int sk_per;
    const float* col_scale;  // MXFP8 only: D column n *= col_scale[n] (FP8 row-scale view), may be null
    // FP8 only: alpha *= *a_scale * *w_scale (device scalars, graph-capturable); D = alpha AW^T + beta D.
    const float* a_scale;
    const float* w_scale;
    float beta;
    int ldd;  // D row stride in elements (0: N); a column block of a wider D
    // Stream-K hybrid plan (sk_dp > 0): CTA b runs whole tiles b, b + grid, ... below sk_dp, then piece
    // b % sk_split of tail tile sk_dp + b / sk_split (piece 0 finishes it). sk_dp 0: contiguous sk_per units.
    int sk_dp;
    int sk_split;
};

// alpha, times col_scale[col], col + 1 for the MXFP8 row-scale view, times the FP8 tensor scales.
template <class C>
__device__ __forceinline__ float2 out_scale(const TileArgs& p, int col) {
    if constexpr (C::kF8) {
        const float s = p.alpha * __ldg(p.a_scale) * __ldg(p.w_scale);
        return make_float2(s, s);
    }
    if constexpr (C::kMx8)
        if (p.col_scale != nullptr && col < p.N)  // N % 8 == 0: col + 1 < N too
            return make_float2(p.alpha * p.col_scale[col], p.alpha * p.col_scale[col + 1]);
    return make_float2(p.alpha, p.alpha);
}

// Per-CTA ring geometry shared by the mainloops and the epilogue.
struct Ring {
    uint8_t* smem;  // 1024-aligned generic base
    uint32_t s0;    // same, shared-window address
    uint32_t bars;  // full[s] at s*8, empty[s] after
    int m0, n0, KT, warp, lane, wm, wn;
};

template <class C>
__device__ __forceinline__ void store_direct(const float (&acc)[4][C::kNF][4], const TileArgs& p, int m0,
                                             int n0, int wm, int wn, int lane) {
    const int T0 = lane & 3;
    const int T1 = lane >> 2;
#pragma unroll
    for (int mf = 0; mf < 4; ++mf)
#pragma unroll
        for (int nf = 0; nf < C::kNF; ++nf) {
            const int col = n0 + C::ncol(wn, nf) + T0 * 2;
            const float2 sc = out_scale<C>(p, col);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm + mf * 16 + T1 + h * 8;
                if (row >= p.M || col >= p.N)
                    continue;
                __half2* dst = reinterpret_cast<__half2*>(p.d + (static_cast<size_t>(row) * p.ldd) + col);
                float2 v = make_float2(acc[mf][nf][h * 2] * sc.x, acc[mf][nf][(h * 2) + 1] * sc.y);
                if constexpr (C::kF8)
                    if (p.beta != 0.0f) {
                        const float2 old = __half22float2(*dst);
                        v = make_float2(fmaf(p.beta, old.x, v.x), fmaf(p.beta, old.y, v.y));
                    }
                *dst = __floats2half2_rn(v.x, v.y);
            }
        }
}

// alpha, fp16, staged in so (stride kOS) for 16 B row-contiguous stores (256 MMA threads). FP8 beta != 0
// stores per fragment: alpha AW^T + beta D in fp32, one rounding.
template <class C>
__device__ __forceinline__ void store_tile(const float (&acc)[4][C::kNF][4], half* so, int m0, int n0, int wm,
                                           int wn, int lane, const TileArgs& p) {
    if constexpr (C::kF8)
        if (p.beta != 0.0f) {
            store_direct<C>(acc, p, m0, n0, wm, wn, lane);
            return;
        }
    asm volatile("bar.sync 1, %0;" ::"n"(kMmaThreads));  // every warp left the ring / the last store loop
    const int T0 = lane & 3;
    const int T1 = lane >> 2;
#pragma unroll
    for (int mf = 0; mf < 4; ++mf)
#pragma unroll
        for (int nf = 0; nf < C::kNF; ++nf) {
            const float2 sc = out_scale<C>(p, n0 + C::ncol(wn, nf) + T0 * 2);
#pragma unroll
            for (int h = 0; h < 2; ++h)
                *reinterpret_cast<__half2*>(so + ((wm + mf * 16 + T1 + h * 8) * C::kOS) + C::ncol(wn, nf) +
                                            T0 * 2) = __floats2half2_rn(acc[mf][nf][h * 2] * sc.x,
                                                                        acc[mf][nf][h * 2 + 1] * sc.y);
        }
    asm volatile("bar.sync 1, %0;" ::"n"(kMmaThreads));
    for (int i = threadIdx.x; i < kBM * (C::kBN / 8); i += kMmaThreads) {
        const int r = i / (C::kBN / 8);
        const int c = (i % (C::kBN / 8)) * 8;
        const int row = m0 + r;
        const int col = n0 + c;
        if (row < p.M && col < p.N)
            *reinterpret_cast<uint4*>(p.d + (static_cast<size_t>(row) * p.ldd) +
                                      col) = *reinterpret_cast<const uint4*>(so + (r * C::kOS) + c);
    }
}

__device__ __forceinline__ uint32_t pack_half2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// TMA epilogue: alpha, fp16, stmatrix into kBN / 64 boxes of 128 rows x 128 B (128B swizzle) in the ring,
// one TMA store per box; rows / columns past M / N clip in the TMA unit. FP8 beta != 0 stores directly.
template <class C>
__device__ __forceinline__ void store_tma(const float (&acc)[4][C::kNF][4], const Ring& g, const TileArgs& p,
                                          const CUtensorMap* md) {
    static_assert(C::kStages * C::kStageBytes >= (C::kBN / 64) * kBM * 128, "boxes must fit the ring");
    if constexpr (C::kF8)
        if (p.beta != 0.0f) {
            store_direct<C>(acc, p, g.m0, g.n0, g.wm, g.wn, g.lane);
            return;
        }
    asm volatile("bar.sync 1, %0;" ::"n"(kMmaThreads));  // every warp left the ring
    const int j = g.lane >> 3;                           // matrix this lane addresses: (h, nf + 1)
    const int T0 = g.lane & 3;
#pragma unroll
    for (int mf = 0; mf < 4; ++mf)
#pragma unroll
        for (int nf = 0; nf < C::kNF; nf += 2) {
            const float2 s0 = out_scale<C>(p, g.n0 + C::ncol(g.wn, nf) + T0 * 2);
            const float2 s1 = out_scale<C>(p, g.n0 + C::ncol(g.wn, nf + 1) + T0 * 2);
            const int row = g.wm + mf * 16 + ((j & 1) * 8) + (g.lane & 7);
            const int col = C::ncol(g.wn, nf + (j >> 1));
            stsm_x4(g.s0 + ((col >> 6) * (kBM * 128)) + (row * 128) + ((((col & 63) >> 3) ^ (row & 7)) * 16),
                    pack_half2(acc[mf][nf][0] * s0.x, acc[mf][nf][1] * s0.y),
                    pack_half2(acc[mf][nf][2] * s0.x, acc[mf][nf][3] * s0.y),
                    pack_half2(acc[mf][nf + 1][0] * s1.x, acc[mf][nf + 1][1] * s1.y),
                    pack_half2(acc[mf][nf + 1][2] * s1.x, acc[mf][nf + 1][3] * s1.y));
        }
    fence_async_smem();
    asm volatile("bar.sync 1, %0;" ::"n"(kMmaThreads));
    if (threadIdx.x == 0) {
#pragma unroll
        for (int b = 0; b < C::kBN / 64; ++b)
            if (g.n0 + (b * 64) < p.N)
                tma_store_2d(md, g.s0 + (b * kBM * 128), g.n0 + (b * 64), g.m0);
        bulk_commit();
        bulk_wait_read();  // the boxes live in shared memory until the TMA unit has read them
    }
}

}  // namespace imp::nvfp4_tile
