#pragma once

// Shared device side of the own NVFP4 tile GEMMs (nvfp4_gemm_tile.cu dense, the grouped MoE
// kernel): tile config, SfAtom offsets, the per-stage MMA loop. sm_120a.

#include "quant/nvfp4_gemm_tile_ptx.cuh"

#include <cuda_fp16.h>
#include <cstdint>

namespace imp::nvfp4_tile {

constexpr int kBM = 128;
constexpr int kMmaThreads = 256;
// UE8M0 1.0: per-tensor FP8 runs the block-scaled MMA; plain kind::f8f6f4 f32-acc reached 424 vs 647 TFLOPS
// (2048x12288x4096, flushed bench).
constexpr uint32_t kUe8m0One = 0x7F7F7F7Fu;

// Operand format: NVFP4 (E2M1, UE4M3 per 16 K, atom = 64 K), MXFP4 (E2M1, UE8M0 per 32 K, atom = 128 K),
// MXFP8 (E4M3 bytes, UE8M0 per 32 K, atom = 128 K), FP8 (E4M3 bytes, per-tensor scales in the epilogue, no
// SF).
enum class Fmt { kNv4, kMx4, kMx8, kF8 };

template <int BN, int BK, int STAGES, bool TMA, bool PW = false, bool SK = false, Fmt F = Fmt::kNv4>
struct Cfg {
    static constexpr int kBN = BN;
    static constexpr int kBK = BK;
    // A stage moves kRB bytes per row in every format, so ldmatrix fragments, swizzle and TMA boxes are
    // shared.
    static constexpr bool kMx8 = F == Fmt::kMx8;
    static constexpr bool kMx4 = F == Fmt::kMx4;
    static constexpr bool kF8 = F == Fmt::kF8;  // byte geometry of MXFP8, no SF slabs
    static constexpr bool kB8 = kMx8 || kF8;    // one byte per element
    static constexpr int kRB = kB8 ? BK : BK / 2;
    static_assert(!kB8 || BK == 64 || BK % 128 == 0, "8-bit stage = half or whole SfAtoms");
    static_assert(!kMx4 || BK % 128 == 0, "MXFP4 stage = whole SfAtoms");
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
    static constexpr int kChunks = kRB / 16;  // 16 B chunks per row per stage
    // TMA: dense rows, swizzled chunks (64B / 128B mode); sync: rows padded by 16 B. Both put
    // the 8 rows of an ldmatrix on 32 distinct banks.
    static constexpr int kRS = TMA ? kRB : kRB + 16;
    // SfAtoms per 128-row slab per stage; an MXFP8 BK=64 stage loads its whole atom (two stages share it).
    static constexpr int kSfBytes = kF8    ? 0
                                    : kMx8 ? ((BK + 127) / 128) * 512
                                           : (BK / (kMx4 ? 128 : 64)) * 512;
    // Operand geometry per K: row bytes, SF bytes per 128-row slab, SF offset and first K group of stage kt.
    static constexpr size_t row_bytes(int K) {
        return kB8 ? static_cast<size_t>(K) : static_cast<size_t>(K) / 2;
    }
    static constexpr size_t sf_slab(int K) {
        return kF8            ? 0
               : kMx8 || kMx4 ? static_cast<size_t>((K + 127) / 128) * 512
                              : static_cast<size_t>(K / 64) * 512;
    }
    __host__ __device__ static constexpr size_t sf_k(int kt) {
        return kMx8 ? static_cast<size_t>((kt * BK) / 128) * 512 : static_cast<size_t>(kt) * kSfBytes;
    }
    __host__ __device__ static constexpr int sf_g0(int kt) { return kMx8 ? ((kt * BK) % 128) / 32 : 0; }
    static constexpr int kATile = kBM * kRS;
    static constexpr int kWTile = BN * kRS;
    static constexpr int kSfA = kATile + kWTile;  // stage offsets of the SF slabs
    static constexpr int kSfW = kSfA + kSfBytes;
    // Rounded to the TMA alignment (MXFP8 128x256: 25.5 KiB of tiles + SF).
    static constexpr int kStageBytes = TMA ? ((kSfW + (BN / 128) * kSfBytes + 1023) / 1024) * 1024
                                           : kSfW + (BN / 128) * kSfBytes;
    static constexpr uint32_t kTxBytes = (kBM + BN) * kRB + (1 + BN / 128) * kSfBytes;
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
        else if constexpr (kRB == 64)
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

// Row r's scale for 32-byte K step c: NVFP4 the 4 UE4M3 of atom c, MXFP4 the 2 UE8M0 of groups 2c, 2c + 1
// in the low half, MXFP8 the UE8M0 of group g0 + c.
template <class C>
__device__ __forceinline__ uint32_t sf_word(const uint8_t* slab, int r, int c, int g0) {
    if constexpr (C::kMx8) {
        const int g = g0 + c;
        return slab[sf_off<C>(r, g >> 2) + (g & 3)];
    } else if constexpr (C::kMx4) {
        return *reinterpret_cast<const uint32_t*>(slab + sf_off<C>(r, c >> 1)) >> ((c & 1) * 16);
    } else {
        return *reinterpret_cast<const uint32_t*>(slab + sf_off<C>(r, c));
    }
}

// One stage of MMAs for a 64 x kWN warp tile: kRB/32 steps of 4 x kNF m16n8k64 (NVFP4) or m16n8k32
// (MXFP8); both read 32 bytes of K per row. g0: the stage's first MXFP8 K group in its atom.
template <class C>
__device__ __forceinline__ void mma_stage(float (&acc)[4][C::kNF][4], uint32_t sa_s, const uint8_t* sst,
                                          int wm, int wn, int lane, int g0 = 0) {
    const int T1 = lane >> 2;
    const int lr = lane & 7;
    const int lm = lane >> 3;
    const int sfa_lane = ((lane & 1) * 8) + (lane >> 2);
#pragma unroll
    for (int c = 0; c < C::kRB / 32; ++c) {
        uint32_t af[4][4];
        uint32_t sfa[4] = {};
#pragma unroll
        for (int mf = 0; mf < 4; ++mf) {
            const int r = wm + mf * 16 + lr + (lm & 1) * 8;
            ldsm_x4(af[mf], sa_s + (r * C::kRS) + (C::chunk(r, (c * 2) + (lm >> 1)) * 16));
            if constexpr (!C::kF8)  // FP8: no SF slabs in the stage
                sfa[mf] = sf_word<C>(sst + C::kSfA, wm + mf * 16 + sfa_lane, c, g0);
        }
#pragma unroll
        for (int np = 0; np < C::kNF / 2; ++np) {
            uint32_t bf[4];
            const int r = wn + (np * 2 + (lm >> 1)) * 8 + lr;
            ldsm_x4(bf, sa_s + C::kATile + (r * C::kRS) + (C::chunk(r, (c * 2) + (lm & 1)) * 16));
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int nf = np * 2 + h;
                const uint32_t sfb = C::kF8 ? 0u : sf_word<C>(sst + C::kSfW, wn + nf * 8 + T1, c, g0);
#pragma unroll
                for (int mf = 0; mf < 4; ++mf) {
                    if constexpr (C::kF8)
                        mma_mxf8(acc[mf][nf], af[mf], h == 0 ? bf[0] : bf[2], h == 0 ? bf[1] : bf[3],
                                 kUe8m0One, kUe8m0One);
                    else if constexpr (C::kMx8)
                        mma_mxf8(acc[mf][nf], af[mf], h == 0 ? bf[0] : bf[2], h == 0 ? bf[1] : bf[3], sfa[mf],
                                 sfb);
                    else if constexpr (C::kMx4)
                        mma_mxf4(acc[mf][nf], af[mf], h == 0 ? bf[0] : bf[2], h == 0 ? bf[1] : bf[3], sfa[mf],
                                 sfb);
                    else
                        mma_mxf4nvf4(acc[mf][nf], af[mf], h == 0 ? bf[0] : bf[2], h == 0 ? bf[1] : bf[3],
                                     sfa[mf], sfb);
                }
            }
        }
    }
}

}  // namespace imp::nvfp4_tile
