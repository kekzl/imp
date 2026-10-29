#pragma once

// Device core of the own FP16 tile GEMM (gemm_f16_tile.cu), shared with the fused GGUF-dequant
// GEMM (gemm_q6k_f16_tile.cu): tile config, accumulator, the per-stage MMA loop and the epilogue.
// Moved verbatim from gemm_f16_tile.cu.

#include "quant/nvfp4_gemm_tile_ptx.cuh"

#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>

namespace imp::f16_tile {

using namespace nvfp4_tile;  // NOLINT(google-build-using-namespace): the shared PTX helpers

constexpr int kBM = 128;
constexpr int kMmaThreads = 256;
constexpr int kAlign = 1024;

template <int BN, int BK, int STAGES, bool ACC16, int GROUP_M = 0, int BM = kBM>
struct Cfg {
    static constexpr int kBN = BN;
    static constexpr int kBK = BK;
    static constexpr int kStages = STAGES;
    static constexpr bool kAcc16 = ACC16;
    // > 0: a wave walks N inside bands of GROUP_M row tiles, so the W slab it streams stays in L2
    // (FP16 W of 17408x5120 is 178 MB > 96 MB L2; N-fastest re-read it from DRAM per 2.5 rows).
    static constexpr int kGroupM = GROUP_M;
    static constexpr int kBM = BM;
    static constexpr int kWarpsM = BM / 64;  // 8 MMA warps of 64 x kWN
    static constexpr int kWN = BN / (8 / kWarpsM);
    static constexpr int kNF = kWN / 8;
    static constexpr int kRS = BK * 2;  // row bytes per stage
    static constexpr int kATile = BM * kRS;
    static constexpr int kStageBytes = kATile + (BN * kRS);
    static constexpr uint32_t kTxBytes = kStageBytes;
    static constexpr int kOS = BN + 8;  // epilogue staging stride in halves
    static constexpr int kSmemBytes = STAGES * kStageBytes + 128 + kAlign;
    static_assert(STAGES * kStageBytes >= BM * kOS * 2, "epilogue staging must fit the ring");
    static_assert(kATile % kAlign == 0 && kStageBytes % kAlign == 0, "TMA tiles 1024-aligned");
    static_assert(ACC16 || BN == 128, "an fp32 128x256 accumulator exceeds the 9-warp register cap");
    // Physical 16 B chunk of chunk c in row r (SWIZZLE_64B for 64 B rows, SWIZZLE_128B for 128 B).
    __device__ static __forceinline__ int chunk(int r, int c) {
        if constexpr (BK == 32)
            return c ^ ((r >> 1) & 3);
        else
            return c ^ (r & 7);
    }
};

template <class C>
struct Acc {
    using T = std::conditional_t<C::kAcc16, uint32_t, float>;
    T v[4][C::kNF][C::kAcc16 ? 2 : 4];  // fp16: packed half2 [row T1, row T1 + 8]
};

__device__ __forceinline__ void mma16(uint32_t (&d)[2], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};\n"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma32(float (&d)[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <class C>
__device__ __forceinline__ void mma_stage(Acc<C>& acc, uint32_t sa, int wm, int wn, int lane) {
    const int lr = lane & 7;
    const int lm = lane >> 3;
#pragma unroll
    for (int c = 0; c < C::kBK / 16; ++c) {
        uint32_t af[4][4];
#pragma unroll
        for (int mf = 0; mf < 4; ++mf) {
            const int r = wm + mf * 16 + lr + (lm & 1) * 8;
            ldsm_x4(af[mf], sa + (r * C::kRS) + (C::chunk(r, (c * 2) + (lm >> 1)) * 16));
        }
#pragma unroll
        for (int np = 0; np < C::kNF / 2; ++np) {
            uint32_t bf[4];
            const int r = wn + (np * 2 + (lm >> 1)) * 8 + lr;
            ldsm_x4(bf, sa + C::kATile + (r * C::kRS) + (C::chunk(r, (c * 2) + (lm & 1)) * 16));
#pragma unroll
            for (int mf = 0; mf < 4; ++mf) {
                if constexpr (C::kAcc16) {
                    mma16(acc.v[mf][np * 2], af[mf], bf[0], bf[1]);
                    mma16(acc.v[mf][(np * 2) + 1], af[mf], bf[2], bf[3]);
                } else {
                    mma32(acc.v[mf][np * 2], af[mf], bf[0], bf[1]);
                    mma32(acc.v[mf][(np * 2) + 1], af[mf], bf[2], bf[3]);
                }
            }
        }
    }
}

template <class C>
__device__ __forceinline__ float2 acc_pair(const Acc<C>& acc, int mf, int nf, int h) {
    if constexpr (C::kAcc16)
        return __half22float2(*reinterpret_cast<const __half2*>(&acc.v[mf][nf][h]));
    else
        return make_float2(acc.v[mf][nf][h * 2], acc.v[mf][nf][(h * 2) + 1]);
}

struct F16Args {
    half* d;
    float alpha, beta;  // beta != 0: D is read before the store, the add in fp32
    int M, N, K;
};

template <class C>
__device__ __forceinline__ void store_out(const Acc<C>& acc, uint8_t* smem, const F16Args& p, int m0, int n0,
                                          int wm, int wn, int lane) {
    asm volatile("bar.sync 1, %0;" ::"n"(kMmaThreads));  // ring dead: stage the output
    const int T0 = lane & 3;
    const int T1 = lane >> 2;
    half* so = reinterpret_cast<half*>(smem);
#pragma unroll
    for (int mf = 0; mf < 4; ++mf)
#pragma unroll
        for (int nf = 0; nf < C::kNF; ++nf)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const float2 v = acc_pair<C>(acc, mf, nf, h);
                *reinterpret_cast<__half2*>(so + ((wm + mf * 16 + T1 + h * 8) * C::kOS) + wn + nf * 8 +
                                            T0 * 2) = __floats2half2_rn(v.x * p.alpha, v.y * p.alpha);
            }
    asm volatile("bar.sync 1, %0;" ::"n"(kMmaThreads));
    for (int i = static_cast<int>(threadIdx.x); i < C::kBM * (C::kBN / 8); i += kMmaThreads) {
        const int r = i / (C::kBN / 8);
        const int c = (i % (C::kBN / 8)) * 8;
        const int row = m0 + r;
        const int col = n0 + c;
        if (row >= p.M || col >= p.N)
            continue;
        uint4 out = *reinterpret_cast<const uint4*>(so + (r * C::kOS) + c);
        const size_t off = (static_cast<size_t>(row) * p.N) + col;
        if (p.beta != 0.0f) {
            const uint4 old = *reinterpret_cast<const uint4*>(p.d + off);
            const __half2* o2 = reinterpret_cast<const __half2*>(&old);
            __half2* n2 = reinterpret_cast<__half2*>(&out);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 a = __half22float2(n2[j]);
                const float2 b = __half22float2(o2[j]);
                n2[j] = __floats2half2_rn(a.x + (p.beta * b.x), a.y + (p.beta * b.y));
            }
        }
        *reinterpret_cast<uint4*>(p.d + off) = out;
    }
}

// Tile coordinates of this CTA: grouped-M raster when C::kGroupM > 0.
template <class C>
__device__ __forceinline__ void tile_coords(int& mt, int& nt) {
    mt = static_cast<int>(blockIdx.y);
    nt = static_cast<int>(blockIdx.x);
    if constexpr (C::kGroupM > 0) {
        const int nts = static_cast<int>(gridDim.x);
        const int bid = (mt * nts) + nt;
        const int first = (bid / (C::kGroupM * nts)) * C::kGroupM;
        const int gsz = min(static_cast<int>(gridDim.y) - first, C::kGroupM);
        mt = first + ((bid % (C::kGroupM * nts)) % gsz);
        nt = (bid % (C::kGroupM * nts)) / gsz;
    }
}

}  // namespace imp::f16_tile
