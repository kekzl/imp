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
// else 4 (256x128 under kRegSplit; fp32 accumulate 128x128). 128x128 without decode: 992 us at
// 2048x17408x5120 (cuBLASLt 1011).
template <class C>
constexpr int kDecWarps = (C::kBM == 128 && C::kBN == 128 && C::kAcc16) ? 8 : 4;
template <class C>
constexpr int kDecThreads = kDecWarps<C> * 32;
// 256x128: 16 warps as 4 warpgroups (MMA 0-1, producer 2 with 3 parked warps, decoders 3) so setmaxnreg moves
// registers to the MMA warps; with 13 warps all sat at 128 (5 LDL per stage per MMA warp, 151 MB local
// loads per 512x17408x5120 Q6_K launch).
template <class C>
constexpr bool kRegSplit = C::kBM == 256;
template <class C>
constexpr int kThreads = kRegSplit<C> ? 512 : kMmaThreads + 32 + kDecThreads<C>;
template <class C>
constexpr int kDecBase = kRegSplit<C> ? 384 : kMmaThreads + 32;

// Decoder thread index (0 .. kDecThreads - 1).
template <class C>
__device__ __forceinline__ int decoder_tid() {
    return static_cast<int>(threadIdx.x) - kDecBase<C>;
}

// First W row of decoder thread dt with 4 threads (dt & 3) per row: a quarter warp takes rows r and r + 4,
// whose swizzled chunks c ^ (r & 7) are disjoint (rows r, r + 1 shared 4 chunks: 2-way STS.128 conflict,
// ncu 4.2 excess wavefronts per store). Bijective on 0 .. rows - 1 for rows % 8 == 0.
__device__ __forceinline__ int decoder_row4(int dt) {
    const int q = dt >> 3;
    return (q & 3) + ((q >> 2) * 8) + (((dt >> 2) & 1) * 4);
}

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

// D: `Args` (with `F16Args o`), `kWideCost` (wide_wins; 0 = 128x128 only) and `template <class C> static
// void decode(const Args&, bars, smem, n0, KT)`.
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
    // kRegSplit budget: 256 x 176 (MMA) + 128 x 40 (producer group) + 128 x 120 (decoders) = 65536.
    if (warp >= kDecBase<C> / 32) {
        if constexpr (kRegSplit<C>)
            asm volatile("setmaxnreg.dec.sync.aligned.u32 120;\n" ::: "memory");
        D::template decode<C>(p, bars, smem_raw + pad, n0, KT);
        return;
    }
    if (warp >= kMmaThreads / 32) {  // producer: one lane issues the A tile per stage
        if constexpr (kRegSplit<C>)
            asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
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
    if constexpr (kRegSplit<C>)
        asm volatile("setmaxnreg.inc.sync.aligned.u32 176;\n" ::: "memory");
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

// 256x128 (1 CTA/SM) when its waves x D::kWideCost / 10 (one 256x128 wave in 128x128 waves) undercut the
// 128x128 waves. Q5_K flushed: 512x6144x5120 150 vs 187 us (1 vs 2 waves), 1024x6144x5120 320 vs 278 (2 vs
// 3); Q6_K 1536x5120x5120 296 vs 313 (2 vs 3).
[[nodiscard]] inline bool wide_wins(int M, int N, int cost10) {
    static const int sms = [] {
        int dev = 0;
        int n = 0;
        return (cudaGetDevice(&dev) == cudaSuccess &&
                cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess && n > 0)
                   ? n
                   : 170;
    }();
    const int tn = (N + 127) / 128;
    const int w128 = ((((M + 127) / 128) * tn) + sms - 1) / sms;
    const int w256 = ((((M + 255) / 256) * tn) + sms - 1) / sms;
    return w256 * cost10 < w128 * 10;
}

// The shipped configs: 256x128 (each decoded W row serves 256 A rows) where wide_wins, else 128x128 with
// 8 decoder warps; fp32 accumulate 128x128. D::kWideCost = 0 keeps 128x128 (MXFP4: 2048x9216x2560 271 vs
// 393 us). variant: -1 auto, 1 128x128, 2 256x128.
template <class D>
[[nodiscard]] bool launch_auto(const typename D::Args& p, const void* a, bool acc_fp16, int variant,
                               cudaStream_t stream) {
    if (!acc_fp16)
        return launch<Cfg<128, 64, 3, false>, D>(p, a, stream);
    if (variant < 0 && D::kWideCost == 0)
        variant = 1;
    if (variant < 0)
        variant = wide_wins(p.o.M, p.o.N, D::kWideCost) ? 2 : 1;
    return variant == 2 ? launch<Cfg<128, 64, 2, true, 16, 256>, D>(p, a, stream)
                        : launch<Cfg<128, 64, 3, true, 16>, D>(p, a, stream);
}

}  // namespace imp::wdq_tile
