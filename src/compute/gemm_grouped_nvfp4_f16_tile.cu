// Grouped NVFP4 W4A16 prompt GEMM for MoE experts (ROADMAP B-85): gemm_wdq_f16_tile_kernel's pipeline with
// the plain NVFP4 decoder, one 64-row tile of one expert per CTA. grid.y = ceil(rows / 64) + ne bounds the
// tile count (each expert adds at most one partial tile); warp 0 maps blockIdx.y to (expert, row tile) by a
// scan over the offsets, surplus CTAs exit before the barrier setup. Rows past the expert's end are
// computed from the next expert's A rows and never stored (store_out clips at M = off[e + 1]).

#include "compute/gemm_grouped_nvfp4_f16_tile.h"

#include "compute/gemm_mxfp8_wdq_dec.cuh"
#include "compute/gemm_nvfp4_wdq_dec.cuh"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace imp {
namespace wdq_tile {
namespace {

using GCfg = Cfg<128, 64, 3, false, 0, 64>;  // 64-row tiles: Flint pp512 averages 12 rows per expert

struct GroupedNv4 {
    half* d;
    int N, K, ne;
    const int32_t* off;
    const uint8_t* w;
    size_t w_stride;
    const uint8_t* sc;
    size_t sc_stride;
    const float* ts;
    const int32_t* aux;  // per-expert decoder argument (MXFP8: exponent bias), null = 0
};

template <class C, class D>
__global__ void __launch_bounds__(kThreads<C>, 1) gemm_grouped_nv4_kernel(
    GroupedNv4 g, const __grid_constant__ CUtensorMap map_a) {
    extern __shared__ __align__(128) uint8_t smem_raw[];
    __shared__ int s_e;
    __shared__ int s_mt;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (warp == 0) {
        // Tile t of the launch -> (expert, row tile): running sum of ceil(count / BM) over 32 experts a step.
        const int t = static_cast<int>(blockIdx.y);
        int base = 0;
        int found_e = -1;
        int found_mt = 0;
        for (int e0 = 0; e0 < g.ne && found_e < 0; e0 += 32) {
            const int e = e0 + lane;
            const int cnt = e < g.ne ? (__ldg(g.off + e + 1) - __ldg(g.off + e)) : 0;
            const int tiles = (cnt + C::kBM - 1) / C::kBM;
            int incl = tiles;
#pragma unroll
            for (int s = 1; s < 32; s <<= 1) {
                const int v = __shfl_up_sync(0xffffffffu, incl, s);
                if (lane >= s)
                    incl += v;
            }
            const int start = base + incl - tiles;
            const unsigned hit = __ballot_sync(0xffffffffu, tiles > 0 && t >= start && t < start + tiles);
            if (hit != 0u) {
                const int src = __ffs(static_cast<int>(hit)) - 1;
                found_e = e0 + src;
                found_mt = t - __shfl_sync(0xffffffffu, start, src);
            }
            base += __shfl_sync(0xffffffffu, incl, 31);
        }
        if (lane == 0) {
            s_e = found_e;
            s_mt = found_mt;
        }
    }
    __syncthreads();
    const int e = s_e;
    if (e < 0)
        return;  // surplus CTA: the whole block leaves before any barrier exists
    const int row0 = __ldg(g.off + e);
    const int m0 = row0 + (s_mt * C::kBM);
    const typename D::Args p = D::make(F16Args{g.d, __ldg(g.ts + e) * D::kAlpha, 0.0f, __ldg(g.off + e + 1),
                                               g.N, g.K},
                                       g.w + (static_cast<size_t>(e) * g.w_stride),
                                       g.sc + (static_cast<size_t>(e) * g.sc_stride),
                                       g.aux != nullptr ? __ldg(g.aux + e) : 0);

    const uint32_t raw = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
    const uint32_t pad = ((raw + kAlign - 1) & ~uint32_t(kAlign - 1)) - raw;
    const uint32_t s0 = raw + pad;
    const uint32_t bars = s0 + C::kStages * C::kStageBytes;
    const int n0 = static_cast<int>(blockIdx.x) * C::kBN;
    const int KT = g.K / C::kBK;
    if (threadIdx.x == 0) {
        for (int s = 0; s < C::kStages; ++s) {
            mbar_init(bars + s * 8, 1 + kDecWarps<C>);  // TMA expect_tx + one arrive per decoder warp
            mbar_init(bars + (C::kStages + s) * 8, kMmaThreads / 32);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (warp >= kDecBase<C> / 32) {
        D::template decode<C>(p, bars, smem_raw + pad, n0, KT);
        return;
    }
    if (warp >= kMmaThreads / 32) {  // producer: one lane issues the A tile per stage
        if (warp == kMmaThreads / 32 && lane == 0)
            for (int kt = 0; kt < KT; ++kt) {
                const int s = kt % C::kStages;
                const uint32_t full = bars + s * 8;
                if (kt >= C::kStages)
                    mbar_wait(bars + (C::kStages + s) * 8, ((kt / C::kStages) - 1) & 1);
                mbar_expect_tx(full, C::kATile);
                tma_2d(s0 + s * C::kStageBytes, &map_a, kt * C::kRS, m0, full);
            }
        return;
    }
    const int wm = (warp % C::kWarpsM) * 64;
    const int wn = (warp / C::kWarpsM) * C::kWN;
    Acc<C> acc{};
    for (int kt = 0; kt < KT; ++kt) {
        const int s = kt % C::kStages;
        mbar_wait(bars + s * 8, (kt / C::kStages) & 1);
        mma_stage<C>(acc, s0 + s * C::kStageBytes, wm, wn, lane);
        __threadfence_block();  // as gemm_wdq_f16_tile_kernel: the release must follow the stage's HMMAs
        __syncwarp();
        if (lane == 0)
            mbar_arrive(bars + (C::kStages + s) * 8);
    }
    store_out<C>(acc, smem_raw + pad, p.o, m0, n0, wm, wn, lane);
}

}  // namespace
}  // namespace wdq_tile

namespace {

template <class D>
bool grouped_launch(const GroupedNvfp4F16Args& g, const int32_t* aux, cudaStream_t stream) {
    using namespace wdq_tile;  // NOLINT(google-build-using-namespace): the shared decode-tile skeleton
    using C = GCfg;
    if (g.rows <= 0 || g.ne <= 0 || g.N <= 0 || g.K <= 0 || g.K % 64 != 0 || g.N % 8 != 0 || g.a == nullptr ||
        g.d_offsets == nullptr || g.w == nullptr || g.sc == nullptr || g.d_tensor_scales == nullptr ||
        g.d == nullptr || reinterpret_cast<uintptr_t>(g.a) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(g.w) % 16 != 0 || g.w_stride % 16 != 0 ||
        reinterpret_cast<uintptr_t>(g.sc) % 4 != 0 || g.sc_stride % 4 != 0 ||
        reinterpret_cast<uintptr_t>(g.d) % 16 != 0)
        return false;
    CUtensorMap map_a{};
    if (!encode_2d(&map_a, g.a, g.rows, g.K * 2, C::kRS, C::kBM))
        return false;
    static const bool smem_ok = cudaFuncSetAttribute(gemm_grouped_nv4_kernel<C, D>,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     C::kSmemBytes) == cudaSuccess;
    if (!smem_ok)
        return false;
    const GroupedNv4 k{static_cast<half*>(g.d),
                       g.N,
                       g.K,
                       g.ne,
                       g.d_offsets,
                       static_cast<const uint8_t*>(g.w),
                       g.w_stride,
                       static_cast<const uint8_t*>(g.sc),
                       g.sc_stride,
                       g.d_tensor_scales,
                       aux};
    const dim3 grid((g.N + C::kBN - 1) / C::kBN, ((g.rows + C::kBM - 1) / C::kBM) + g.ne);
    gemm_grouped_nv4_kernel<C, D><<<grid, kThreads<C>, C::kSmemBytes, stream>>>(k, map_a);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

}  // namespace

bool gemm_grouped_nvfp4_f16_tile(const GroupedNvfp4F16Args& g, cudaStream_t stream) {
    return grouped_launch<wdq_tile::Nv4Dec>(g, nullptr, stream);
}

bool gemm_grouped_mxfp8_f16_tile(const GroupedNvfp4F16Args& g, const int32_t* d_exp_bias,
                                 cudaStream_t stream) {
    return d_exp_bias != nullptr && grouped_launch<wdq_tile::Mx8Dec>(g, d_exp_bias, stream);
}

}  // namespace imp
