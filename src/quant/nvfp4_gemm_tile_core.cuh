#pragma once

// Shared device side of the own NVFP4 tile GEMMs (nvfp4_gemm_tile.cu dense, the grouped MoE
// kernel): tile config, SfAtom offsets, the per-stage MMA loop. sm_120a.

#include "quant/nvfp4_gemm_tile_ptx.cuh"

#include <cuda_fp16.h>
#include <cstdint>

namespace imp::nvfp4_tile {

constexpr int kBM = 128;
constexpr int kMmaThreads = 256;

template <int BN, int BK, int STAGES, bool TMA, bool PW = false, bool SK = false>
struct Cfg {
    static constexpr int kBN = BN;
    static constexpr int kBK = BK;
    static constexpr int kStages = STAGES;
    static constexpr bool kTma = TMA;
    // PW: a 9th warp issues the TMA loads. 9 warps put 3 on one SM sub-partition, which caps
    // ptxas at 168 registers (128x256 spilled 140 B); without it thread 0 refills each slot.
    static constexpr bool kPw = TMA && PW;
    static constexpr bool kSk = SK;  // stream-K: grid = SMs over (tile, k-step) units, direct epilogue
    static_assert(!SK || (TMA && PW), "stream-K runs on the producer-warp mainloop");
    static constexpr int kThreads = kMmaThreads + (kPw ? 32 : 0);
    static constexpr int kWN = BN / 4;  // warp tile 64 x kWN
    static constexpr int kNF = kWN / 8;
    static constexpr int kChunks = BK / 32;  // 16 B chunks per row per stage
    // TMA: dense rows, swizzled chunks (64B / 128B mode); sync: rows padded by 16 B. Both put
    // the 8 rows of an ldmatrix on 32 distinct banks.
    static constexpr int kRS = TMA ? BK / 2 : BK / 2 + 16;
    static constexpr int kSfBytes = (BK / 64) * 512;  // SfAtoms per 128-row slab per stage
    static constexpr int kATile = kBM * kRS;
    static constexpr int kWTile = BN * kRS;
    static constexpr int kSfA = kATile + kWTile;  // stage offsets of the SF slabs
    static constexpr int kSfW = kSfA + kSfBytes;
    static constexpr int kStageBytes = kSfW + (BN / 128) * kSfBytes;
    static constexpr uint32_t kTxBytes = (kBM + BN) * (BK / 2) + (1 + BN / 128) * kSfBytes;
    static constexpr int kOS = BN + 8;   // epilogue staging stride in halves: half2 writes conflict-free
    static constexpr int kAlign = 1024;  // 128B-swizzle atom; stage and tile offsets keep it
    // Stream-K stages the epilogue in its own buffer behind the mbarriers when it fits 99 KiB: the
    // producer refills the ring meanwhile.
    static constexpr bool kSkStaged = SK && (STAGES * kStageBytes) + 128 + (kBM * kOS * 2) + kAlign <= 101376;
    static constexpr int kSmemBytes = STAGES * kStageBytes + 128 + (kSkStaged ? kBM * kOS * 2 : 0) + kAlign;
    static_assert(STAGES * kStageBytes >= kBM * kOS * 2, "epilogue staging must fit the ring");
    static_assert(2 * STAGES * 8 <= 128, "mbarrier area");
    static_assert(TMA || BN == 128, "sync mainloop loads one 128-row W slab");
    static_assert(!TMA || (kATile % kAlign == 0 && kStageBytes % kAlign == 0), "TMA tiles 1024-aligned");

    // Physical 16 B chunk of logical chunk c in row r (CU_TENSOR_MAP_SWIZZLE_64B / _128B).
    __device__ static __forceinline__ int chunk(int r, int c) {
        if constexpr (!TMA)
            return c;
        else if constexpr (BK == 128)
            return c ^ ((r >> 1) & 3);
        else
            return c ^ (r & 7);
    }
};

// Byte of row r's 4 K-group scales inside a stage's SF slabs (r < 128 per slab).
template <class C>
__device__ __forceinline__ int sf_off(int r, int c) {
    return ((r >> 7) * C::kSfBytes) + (c * 512) + ((r & 31) * 16) + (((r & 127) >> 5) * 4);
}

// One stage of MMAs for a 64 x kWN warp tile: BK/64 k64 steps of 4 x kNF m16n8k64.
template <class C>
__device__ __forceinline__ void mma_stage(float (&acc)[4][C::kNF][4], uint32_t sa_s, const uint8_t* sst,
                                          int wm, int wn, int lane) {
    const int T1 = lane >> 2;
    const int lr = lane & 7;
    const int lm = lane >> 3;
    const int sfa_lane = ((lane & 1) * 8) + (lane >> 2);
#pragma unroll
    for (int c = 0; c < C::kBK / 64; ++c) {
        uint32_t af[4][4];
        uint32_t sfa[4];
#pragma unroll
        for (int mf = 0; mf < 4; ++mf) {
            const int r = wm + mf * 16 + lr + (lm & 1) * 8;
            ldsm_x4(af[mf], sa_s + (r * C::kRS) + (C::chunk(r, (c * 2) + (lm >> 1)) * 16));
            sfa[mf] = *reinterpret_cast<const uint32_t*>(sst + C::kSfA +
                                                         sf_off<C>(wm + mf * 16 + sfa_lane, c));
        }
#pragma unroll
        for (int np = 0; np < C::kNF / 2; ++np) {
            uint32_t bf[4];
            const int r = wn + (np * 2 + (lm >> 1)) * 8 + lr;
            ldsm_x4(bf, sa_s + C::kATile + (r * C::kRS) + (C::chunk(r, (c * 2) + (lm & 1)) * 16));
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int nf = np * 2 + h;
                const uint32_t sfb = *reinterpret_cast<const uint32_t*>(sst + C::kSfW +
                                                                        sf_off<C>(wn + nf * 8 + T1, c));
#pragma unroll
                for (int mf = 0; mf < 4; ++mf)
                    mma_mxf4nvf4(acc[mf][nf], af[mf], h == 0 ? bf[0] : bf[2], h == 0 ? bf[1] : bf[3], sfa[mf],
                                 sfb);
            }
        }
    }
}

}  // namespace imp::nvfp4_tile
