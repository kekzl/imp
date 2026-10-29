#pragma once

// Weight-decode FP16 tile GEMM skeleton (ROADMAP row 123): the f16 tile core with W decoded per stage by
// decoder warps instead of a TMA'd fp16 W. A producer warp TMAs A; the decoder policy D fills the swizzled
// fp16 W slot of each stage and arrives once per decoder warp on its full barrier. Users: Q6_K
// (gemm_q6k_f16_tile.cu), MXFP4 (gemm_mxfp4_f16_tile.cu).

#include "compute/gemm_f16_tile_core.cuh"
#include "core/logging.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace imp::wdq_tile {

using namespace f16_tile;  // NOLINT(google-build-using-namespace): the shared tile core

// Decoder warps per config: 8 where the MMA warps leave the registers (128x128 fp16 acc: 96 regs x 544),
// else 4 (13 warps cap ptxas at 128 regs: 48 B stack on 256x128; the TMA folded into decoder thread 0
// removed it but cost +10 % at 2048x17408x5120). 128x128 without decode: 992 us there (cuBLASLt 1011).
template <class C>
constexpr int kDecWarps = (C::kBM == 128 && C::kBN == 128 && C::kAcc16) ? 8 : 4;
template <class C>
constexpr int kDecThreads = kDecWarps<C> * 32;
template <class C>
constexpr int kThreads = kMmaThreads + 32 + kDecThreads<C>;

// Decoder thread index (0 .. kDecThreads - 1).
__device__ __forceinline__ int decoder_tid() { return static_cast<int>(threadIdx.x) - kMmaThreads - 32; }

// Decoder side of stage kt: wait for its slot (free after the MMA warps consumed stage kt - kStages),
// return the W slot to fill. stage_done() releases it to the MMA warps.
template <class C>
__device__ __forceinline__ uint8_t* stage_slot(uint32_t bars, uint8_t* smem, int kt) {
    const int s = kt % C::kStages;
    if (kt >= C::kStages)
        mbar_wait(bars + (C::kStages + s) * 8, ((kt / C::kStages) - 1) & 1);
    return smem + (s * C::kStageBytes) + C::kATile;
}
template <class C>
__device__ __forceinline__ void stage_done(uint32_t bars, int kt) {
    __syncwarp();
    if ((threadIdx.x & 31) == 0)
        mbar_arrive(bars + (kt % C::kStages) * 8);
}

// D: `Args` (with `F16Args o`), `kWide` (256x128 allowed) and `template <class C> static void decode(const
// Args&, bars, smem, n0, KT)`.
template <class C, class D>
__global__ void __launch_bounds__(kThreads<C>, 1) gemm_wdq_f16_tile_kernel(
    typename D::Args p, const __grid_constant__ CUtensorMap map_a) {
    extern __shared__ __align__(128) uint8_t smem_raw[];
    const uint32_t raw = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
    const uint32_t pad = ((raw + kAlign - 1) & ~uint32_t(kAlign - 1)) - raw;
    const uint32_t s0 = raw + pad;
    const uint32_t bars = s0 + C::kStages * C::kStageBytes;
    int mt = 0;
    int nt = 0;
    tile_coords<C>(mt, nt);
    const int n0 = nt * C::kBN;
    const int m0 = mt * C::kBM;
    const int KT = p.o.K / C::kBK;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (threadIdx.x == 0) {
        for (int s = 0; s < C::kStages; ++s) {
            mbar_init(bars + s * 8, 1 + kDecWarps<C>);  // TMA expect_tx + one arrive per decoder warp
            mbar_init(bars + (C::kStages + s) * 8, kMmaThreads / 32);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (warp > kMmaThreads / 32) {
        D::template decode<C>(p, bars, smem_raw + pad, n0, KT);
        return;
    }
    if (warp == kMmaThreads / 32) {  // producer: one lane issues the A tile per stage
        if (lane == 0)
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
        // ptxas hoists the release above the stage's last HMMAs, right behind its last LDSM; the decoders
        // store into the slot at once (no TMA latency), so without the fence 19/20 runs differed.
        __threadfence_block();
        __syncwarp();
        if (lane == 0)
            mbar_arrive(bars + (C::kStages + s) * 8);
    }
    store_out<C>(acc, smem_raw + pad, p.o, m0, n0, wm, wn, lane);
}

template <class C, class D>
[[nodiscard]] bool launch(const typename D::Args& p, const void* a, cudaStream_t stream) {
    CUtensorMap map_a{};
    if (!encode_2d(&map_a, a, p.o.M, p.o.K * 2, C::kRS, C::kBM))
        return false;
    static const bool smem_ok = cudaFuncSetAttribute(gemm_wdq_f16_tile_kernel<C, D>,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     C::kSmemBytes) == cudaSuccess;
    if (!smem_ok)
        return false;
    const dim3 grid((p.o.N + C::kBN - 1) / C::kBN, (p.o.M + C::kBM - 1) / C::kBM);
    gemm_wdq_f16_tile_kernel<C, D><<<grid, kThreads<C>, C::kSmemBytes, stream>>>(p, map_a);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

// The shipped configs: 256x128 (each decoded W row serves 256 A rows) once its grid fills a wave, else
// 128x128 with 8 decoder warps; fp32 accumulate 128x128. 512x17408x5120 Q6_K: 324 vs 128x256 341 us;
// 512x5120x17408: 128x128 331 vs 256x128 443. D::kWide = false keeps 128x128 (MXFP4: 2048x9216x2560 271
// vs 393 us). variant: -1 auto, 1 128x128, 2 256x128.
template <class D>
[[nodiscard]] bool launch_auto(const typename D::Args& p, const void* a, bool acc_fp16, int variant,
                               cudaStream_t stream) {
    if (!acc_fp16)
        return launch<Cfg<128, 64, 3, false>, D>(p, a, stream);
    if (variant < 0 && !D::kWide)
        variant = 1;
    if (variant < 0)
        variant = (((p.o.M + 255) / 256) * ((p.o.N + 127) / 128) >= 170) ? 2 : 1;
    return variant == 2 ? launch<Cfg<128, 64, 2, true, 16, 256>, D>(p, a, stream)
                        : launch<Cfg<128, 64, 3, true, 16>, D>(p, a, stream);
}

}  // namespace imp::wdq_tile
