#pragma once

// TMA producer stage and stream-K mainloop of the own dense tile GEMM (nvfp4_gemm_tile.cu, its only
// includer). Moved verbatim from nvfp4_gemm_tile.cu (file-size gate). sm_120a.

#include "quant/nvfp4_gemm_tile_core.cuh"
#include "quant/nvfp4_gemm_tile_epilogue.cuh"

#include <cuda.h>
#include <cstdint>

namespace imp::nvfp4_tile {

// Stream-K arrival flags, one per CTA: 0 at load, set by a contributor, reset by the tile's finisher in the
// same launch (no per-launch memset). One stream-K GEMM at a time, as for the shared partial workspace.
constexpr int kMaxSkCtas = 1024;
__device__ int g_sk_flags[kMaxSkCtas];

// TMA producer lane: arm the stage's tx count, then the A/W tiles (OOB rows land as 0) and the
// SF slabs (one per 128 rows; a slab past N re-reads the last one: finite scales on zero rows).
template <class C>
__device__ __forceinline__ void tma_stage(uint32_t s_base, uint32_t full, const CUtensorMap* ma,
                                          const CUtensorMap* mw, const TileArgs& p, int m0, int n0, int kt) {
    const size_t sf_slab = C::sf_slab(p.K);
    const size_t sf_k = C::sf_k(kt);
    // Partial last stage (K % BK = r * 64): TMA zero-fills the data past K, the SfAtoms past r are zeroed
    // here (a stale slot could hold a NaN scale). The empty wait before this call freed the slot.
    uint32_t sf_bytes = C::kSfBytes;
    if constexpr (C::kPartK)
        if ((kt + 1) * C::kBK > p.K) {
            sf_bytes = static_cast<uint32_t>((p.K - (kt * C::kBK)) / 64) * 512;
#pragma unroll
            for (int j = 0; j <= C::kWAtoms; ++j)
                for (uint32_t i = sf_bytes; i < C::kSfBytes; i += 16)
                    asm volatile("st.shared.v4.u32 [%0], {%1, %1, %1, %1};" ::"r"(s_base + C::kSfA +
                                                                                  (j * C::kSfBytes) + i),
                                 "r"(0)
                                 : "memory");
        }
    mbar_expect_tx(full, C::kTxBytes - ((1 + C::kWAtoms) * (C::kSfBytes - sf_bytes)));
    tma_2d(s_base, ma, kt * C::kRB, m0, full);
    tma_2d(s_base + C::kATile, mw, kt * C::kRB, n0, full);
    if constexpr (C::kSfBytes == 0)  // FP8: no SF slabs
        return;
    bulk_1d(s_base + C::kSfA, p.a_sf + (static_cast<size_t>(m0 / kBM) * sf_slab) + sf_k, sf_bytes, full);
#pragma unroll
    for (int j = 0; j < C::kWAtoms; ++j)
        bulk_1d(s_base + C::kSfW + j * C::kSfBytes,
                p.w_sf + (static_cast<size_t>(min(n0 / 128 + j, (p.N - 1) / 128)) * sf_slab) + sf_k, sf_bytes,
                full);
}

// Stream-K. CTA b walks its unit range [begin, end) forward in segments (tile t, k-steps
// [kb, ke)): its first segment can start mid tile (kb > 0): a partial, written right at the start;
// its last can end mid tile (kb == 0, ke < KT): it holds k-step 0, finishes the tile and reads the
// partials of CTAs b+1.. at the end of its range. Whole tiles store directly.
// Partials sum in a fixed order (deterministic, not bit-equal to the data-parallel kernel).
struct SkSeg {
    int t, kb, ke;
};

__device__ __forceinline__ SkSeg sk_seg(int it, int end, int KT) {
    const int t = it / KT;
    const int kb = it - (t * KT);
    return {t, kb, min(KT, kb + (end - it))};
}

__device__ __forceinline__ int ld_acquire(const int* p) {
    int v;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_release(int* p, int v) {
    asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// Partial tile of CTA b in fragment order: float4 j of thread tid at (j * 256 + tid), coalesced.
template <class C>
__device__ __forceinline__ float4* sk_slot(const TileArgs& p, int b) {
    return reinterpret_cast<float4*>(p.sk_parts) + (static_cast<size_t>(b) * 4 * C::kNF * C::kMmaThr);
}

// Calls f(t, kb, ke, last) for every segment (tile t, k-steps [kb, ke)) of CTA b in order; last: the highest
// CTA holding a later piece of t (read by the finisher, the segment with kb == 0 and ke < KT).
template <class F>
__device__ __forceinline__ void sk_walk(const TileArgs& p, int KT, int units, int b, F&& f) {
    if (p.sk_split > 0) {  // hybrid: whole tiles b, b + grid, ..., then one piece of a tail tile
        for (int t = b; t < p.sk_dp; t += static_cast<int>(gridDim.x))
            f(t, 0, KT, b);
        const int pieces = p.sk_split;
        if (b < ((units / KT) - p.sk_dp) * pieces) {
            const int j = b % pieces;
            f(p.sk_dp + (b / pieces), (j * KT) / pieces, ((j + 1) * KT) / pieces, b - j + pieces - 1);
        }
        return;
    }
    const int end = min((b + 1) * p.sk_per, units);
    for (int it = b * p.sk_per; it < end;) {
        const SkSeg sg = sk_seg(it, end, KT);
        f(sg.t, sg.kb, sg.ke, ((sg.t * KT) + KT - 1) / p.sk_per);
        it += sg.ke - sg.kb;
    }
}

template <class C>
__device__ __forceinline__ void sk_mainloop(const Ring& g, const TileArgs& p, const CUtensorMap* ma,
                                            const CUtensorMap* mw, const CUtensorMap* md) {
    const int KT = (p.K + C::kBK - 1) / C::kBK;  // kPartK: a partial last k-step
    const int n_tiles = (p.N + C::kBN - 1) / C::kBN;
    const int units = n_tiles * ((p.M + kBM - 1) / kBM) * KT;
    const int b = blockIdx.x;
    if (threadIdx.x == 0) {
        for (int s = 0; s < C::kStages; ++s) {
            mbar_init(g.bars + s * 8, 1);
            mbar_init(g.bars + (C::kStages + s) * 8, C::kMmaThr / 32);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    int ring = 0;
    if (g.warp == C::kMmaThr / 32) {
        if (g.lane == 0)
            sk_walk(p, KT, units, b, [&](int t, int kb, int ke, int) {
                const int m0 = (t / n_tiles) * kBM;
                const int n0 = (t % n_tiles) * C::kBN;
                for (int kt = kb; kt < ke; ++kt, ++ring) {
                    const int s = ring % C::kStages;
                    if (ring >= C::kStages)
                        mbar_wait_sleep(g.bars + (C::kStages + s) * 8, ((ring / C::kStages) - 1) & 1);
                    tma_stage<C>(g.s0 + s * C::kStageBytes, g.bars + s * 8, ma, mw, p, m0, n0, kt);
                }
            });
        return;
    }
    float acc[4][C::kNF][4];
    sk_walk(p, KT, units, b, [&](int t, int kb, int ke, int last) {
        const int m0 = (t / n_tiles) * kBM;
        const int n0 = (t % n_tiles) * C::kBN;
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < C::kNF; ++j)
#pragma unroll
                for (int r = 0; r < 4; ++r)
                    acc[i][j][r] = 0.0f;
        for (int kt = kb; kt < ke; ++kt, ++ring) {
            const int s = ring % C::kStages;
            mbar_wait_sleep(g.bars + s * 8, (ring / C::kStages) & 1);
            mma_stage<C>(acc, g.s0 + s * C::kStageBytes, g.smem + s * C::kStageBytes, g.wm, g.wn, g.lane,
                         C::sf_g0(kt), n0 & 127);
            __syncwarp();
            if (g.lane == 0)
                mbar_arrive(g.bars + (C::kStages + s) * 8);
        }
        float4* own = sk_slot<C>(p, b);
        if (kb > 0) {  // contributor: this tile's k-step 0 belongs to an earlier CTA
#pragma unroll
            for (int mf = 0; mf < 4; ++mf)
#pragma unroll
                for (int nf = 0; nf < C::kNF; ++nf)
                    own[((mf * C::kNF + nf) * C::kMmaThr) + threadIdx.x] =
                        make_float4(acc[mf][nf][0], acc[mf][nf][1], acc[mf][nf][2], acc[mf][nf][3]);
            __threadfence();
            asm volatile("bar.sync 1, %0;" ::"n"(C::kMmaThr));
            if (threadIdx.x == 0)
                st_release(g_sk_flags + b, 1);
            return;
        }
        if (ke < KT) {  // finisher: CTAs b+1 .. last hold k-steps [ke, KT)
            for (int q = b + 1; q <= last; ++q) {
                if (threadIdx.x == 0) {
                    while (ld_acquire(g_sk_flags + q) == 0) {}
                    g_sk_flags[q] = 0;  // the only reader: every set flag is 0 again when the launch ends
                }
                asm volatile("bar.sync 1, %0;" ::"n"(C::kMmaThr));
                const float4* src = sk_slot<C>(p, q);
#pragma unroll
                for (int mf = 0; mf < 4; ++mf)
#pragma unroll
                    for (int nf = 0; nf < C::kNF; ++nf) {
                        const float4 v = __ldcg(src + ((mf * C::kNF + nf) * C::kMmaThr) + threadIdx.x);
                        acc[mf][nf][0] += v.x;
                        acc[mf][nf][1] += v.y;
                        acc[mf][nf][2] += v.z;
                        acc[mf][nf][3] += v.w;
                    }
            }
        }
        if constexpr (C::kSkStaged) {  // TMA boxes past the barriers: the producer still fills the ring
            static_assert(1024 + ((C::kBN / 64) * kBM * 128) <= 128 + (kBM * C::kOS * 2), "boxes must fit");
            Ring o = g;
            o.s0 = g.bars + 1024;
            o.m0 = m0;
            o.n0 = n0;
            store_tma<C>(acc, o, p, md);
        } else
            store_direct<C>(acc, p, m0, n0, g.wm, g.wn, g.lane);
    });
}

}  // namespace imp::nvfp4_tile
