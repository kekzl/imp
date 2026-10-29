// Own FP16 dense GEMM (ROADMAP row 123): D[M,N] fp16 = alpha * A W^T + beta * D, A [M,K] / W [N,K]
// row-major fp16. The nvfp4_gemm_tile.cu skeleton: CTA tile 128 x BN, BK-half stages (TMA
// swizzle by row bytes), mbarrier ring fed by a producer warp, 8 MMA warps of 64 x BN/4,
// ldmatrix x4 fragments (one 16 B chunk = 8 halves), mma.sync m16n8k16 with an fp16 (cuBLASLt
// COMPUTE_16F, full rate) or fp32 accumulator. FP16 moves 4x the bytes per flop of NVFP4:
// 128x128 BK64 is L2-bound (N=17408 238 vs cuBLASLt 370 TFLOPS), hence the 128x256 tile.

#include "compute/gemm_f16_tile.h"

#include "core/logging.h"
#include "core/process_diag.h"
#include "core/tensor.h"
#include "quant/nvfp4_gemm_tile_ptx.cuh"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace imp {
namespace {

using namespace nvfp4_tile;  // NOLINT(google-build-using-namespace): the shared PTX helpers

constexpr int kBM = 128;
constexpr int kMmaThreads = 256;
constexpr int kThreads = kMmaThreads + 32;
constexpr int kAlign = 1024;

template <int BN, int BK, int STAGES, bool ACC16, int GROUP_M = 0>
struct Cfg {
    static constexpr int kBN = BN;
    static constexpr int kBK = BK;
    static constexpr int kStages = STAGES;
    static constexpr bool kAcc16 = ACC16;
    // > 0: a wave walks N inside bands of GROUP_M row tiles, so the W slab it streams stays in L2
    // (FP16 W of 17408x5120 is 178 MB > 96 MB L2; N-fastest re-read it from DRAM per 2.5 rows).
    static constexpr int kGroupM = GROUP_M;
    static constexpr int kWN = BN / 4;
    static constexpr int kNF = kWN / 8;
    static constexpr int kRS = BK * 2;  // row bytes per stage
    static constexpr int kATile = kBM * kRS;
    static constexpr int kStageBytes = kATile + (BN * kRS);
    static constexpr uint32_t kTxBytes = kStageBytes;
    static constexpr int kOS = BN + 8;  // epilogue staging stride in halves
    static constexpr int kSmemBytes = STAGES * kStageBytes + 128 + kAlign;
    static_assert(STAGES * kStageBytes >= kBM * kOS * 2, "epilogue staging must fit the ring");
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
    for (int i = static_cast<int>(threadIdx.x); i < kBM * (C::kBN / 8); i += kMmaThreads) {
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

template <class C>
__global__ void __launch_bounds__(kThreads, 1) gemm_f16_tile_kernel(
    F16Args p, const __grid_constant__ CUtensorMap map_a, const __grid_constant__ CUtensorMap map_w) {
    extern __shared__ __align__(128) uint8_t smem_raw[];
    const uint32_t raw = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
    const uint32_t pad = ((raw + kAlign - 1) & ~uint32_t(kAlign - 1)) - raw;
    const uint32_t s0 = raw + pad;
    const uint32_t bars = s0 + C::kStages * C::kStageBytes;
    int mt = static_cast<int>(blockIdx.y);
    int nt = static_cast<int>(blockIdx.x);
    if constexpr (C::kGroupM > 0) {
        const int nts = static_cast<int>(gridDim.x);
        const int bid = (mt * nts) + nt;
        const int first = (bid / (C::kGroupM * nts)) * C::kGroupM;
        const int gsz = min(static_cast<int>(gridDim.y) - first, C::kGroupM);
        mt = first + ((bid % (C::kGroupM * nts)) % gsz);
        nt = (bid % (C::kGroupM * nts)) / gsz;
    }
    const int n0 = nt * C::kBN;
    const int m0 = mt * kBM;
    const int KT = p.K / C::kBK;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (threadIdx.x == 0) {
        for (int s = 0; s < C::kStages; ++s) {
            mbar_init(bars + s * 8, 1);
            mbar_init(bars + (C::kStages + s) * 8, kMmaThreads / 32);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (warp == kMmaThreads / 32) {  // producer: one lane issues 2 TMA tiles per stage
        if (lane == 0)
            for (int kt = 0; kt < KT; ++kt) {
                const int s = kt % C::kStages;
                const uint32_t full = bars + s * 8;
                if (kt >= C::kStages)
                    mbar_wait(bars + (C::kStages + s) * 8, ((kt / C::kStages) - 1) & 1);
                mbar_expect_tx(full, C::kTxBytes);
                tma_2d(s0 + s * C::kStageBytes, &map_a, kt * C::kRS, m0, full);
                tma_2d(s0 + s * C::kStageBytes + C::kATile, &map_w, kt * C::kRS, n0, full);
            }
        return;
    }
    const int wm = (warp & 1) * 64;
    const int wn = (warp >> 1) * C::kWN;
    Acc<C> acc{};
    for (int kt = 0; kt < KT; ++kt) {
        const int s = kt % C::kStages;
        mbar_wait(bars + s * 8, (kt / C::kStages) & 1);
        mma_stage<C>(acc, s0 + s * C::kStageBytes, wm, wn, lane);
        __syncwarp();
        if (lane == 0)
            mbar_arrive(bars + (C::kStages + s) * 8);
    }
    store_out<C>(acc, smem_raw + pad, p, m0, n0, wm, wn, lane);
}

template <class C>
bool launch(const F16Args& p, const void* a, const void* w, cudaStream_t stream) {
    if (p.K % C::kBK != 0)
        return false;
    CUtensorMap map_a{};
    CUtensorMap map_w{};
    if (!encode_2d(&map_a, a, p.M, p.K * 2, C::kRS, kBM) ||
        !encode_2d(&map_w, w, p.N, p.K * 2, C::kRS, C::kBN))
        return false;
    static const bool smem_ok = cudaFuncSetAttribute(gemm_f16_tile_kernel<C>,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     C::kSmemBytes) == cudaSuccess;
    if (!smem_ok)
        return false;
    const dim3 grid((p.N + C::kBN - 1) / C::kBN, (p.M + kBM - 1) / kBM);
    gemm_f16_tile_kernel<C><<<grid, kThreads, C::kSmemBytes, stream>>>(p, map_a, map_w);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

}  // namespace

bool gemm_f16_tile(const void* a, const void* w, void* d, int M, int N, int K, float alpha, float beta,
                   bool acc_fp16, cudaStream_t stream, int variant) {
    if (M <= 0 || N <= 0 || K <= 0 || K % 32 != 0 || N % 8 != 0 || a == nullptr || w == nullptr ||
        d == nullptr || reinterpret_cast<uintptr_t>(a) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(w) % 16 != 0 || reinterpret_cast<uintptr_t>(d) % 16 != 0)
        return false;
    const F16Args p{static_cast<half*>(d), alpha, beta, M, N, K};
    if (!acc_fp16)
        return launch<Cfg<128, 64, 3, false>>(p, a, w, stream);
    // 128x256 BK64 x2, grouped-M 16: 4096x17408x5120 2102.6 us (no group 2382.1, group 8 2183.8,
    // BK32 x4 2249.4, 128x128 3034.7; cuBLASLt 1990.9). variant 1: 128x128 (bench).
    return variant == 1 ? launch<Cfg<128, 64, 3, true, 16>>(p, a, w, stream)
                        : launch<Cfg<256, 64, 2, true, 16>>(p, a, w, stream);
}

// gemm.f16_tile: own fp16-accumulate GEMM (bit-equal to COMPUTE_16F, capture-safe) for deep K
// (K >= N) once its 128x256 grid fills a wave. In situ Qwen3-14B-Q6_K pp2048: the down proj
// (2048x5120x17408) wins, the N=17408 gate/up lose 1 %; small grids lose (512x1024 120 vs 30.7 us).
bool gemm_f16_tile_route(const Tensor& A, const Tensor& B, Tensor& C, float alpha, float beta,
                         cudaStream_t stream) {
    const int64_t M = A.shape[0];
    const int64_t K = A.shape[1];
    const int64_t N = B.shape[0];
    if (!process_diag_f16_tile() || K < N || ((N + 255) / 256) * ((M + 127) / 128) < 170 ||
        !gemm_f16_tile(A.data, B.data, C.data, static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                       alpha, beta, /*acc_fp16=*/true, stream))
        return false;
    static bool logged = false;  // activity proof, once
    if (!logged) {
        logged = true;
        IMP_LOG_INFO("gemm.f16_tile: own FP16 GEMM ACTIVE (M=%ld N=%ld K=%ld)", (long)M, (long)N, (long)K);
    }
    return true;
}

}  // namespace imp
