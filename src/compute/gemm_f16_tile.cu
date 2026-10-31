// Own FP16 dense GEMM (ROADMAP row 123): D[M,N] fp16 = alpha * A W^T + beta * D, A [M,K] / W [N,K]
// row-major fp16. The nvfp4_gemm_tile.cu skeleton: CTA tile 128 x BN, BK-half stages (TMA
// swizzle by row bytes), mbarrier ring fed by a producer warp, 8 MMA warps of 64 x BN/4,
// ldmatrix x4 fragments (one 16 B chunk = 8 halves), mma.sync m16n8k16 with an fp16 (cuBLASLt
// COMPUTE_16F, full rate) or fp32 accumulator. FP16 moves 4x the bytes per flop of NVFP4:
// 128x128 BK64 is L2-bound (N=17408 238 vs cuBLASLt 370 TFLOPS), hence the 128x256 tile.

#include "compute/gemm_f16_tile.h"

#include "compute/gemm_f16_tile_core.cuh"
#include "core/device_info.h"
#include "core/logging.h"
#include "core/process_diag.h"
#include "core/tensor.h"

#include <cuda.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>

namespace imp {
namespace {

using namespace f16_tile;  // NOLINT(google-build-using-namespace): the shared tile core

constexpr int kThreads = kMmaThreads + 32;

template <class C>
__global__ void __launch_bounds__(kThreads, 1) gemm_f16_tile_kernel(
    F16Args p, const __grid_constant__ CUtensorMap map_a, const __grid_constant__ CUtensorMap map_w) {
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
    const int wm = (warp % C::kWarpsM) * 64;
    const int wn = (warp / C::kWarpsM) * C::kWN;
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
    if (!encode_2d(&map_a, a, p.M, p.K * 2, C::kRS, C::kBM) ||
        !encode_2d(&map_w, w, p.N, p.K * 2, C::kRS, C::kBN))
        return false;
    static const bool smem_ok = cudaFuncSetAttribute(gemm_f16_tile_kernel<C>,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     C::kSmemBytes) == cudaSuccess;
    if (!smem_ok)
        return false;
    const dim3 grid((p.N + C::kBN - 1) / C::kBN, (p.M + C::kBM - 1) / C::kBM);
    gemm_f16_tile_kernel<C><<<grid, kThreads, C::kSmemBytes, stream>>>(p, map_a, map_w);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

constexpr int kMaxSplit = 8;  // portable cluster size

// Cluster split-K (blockIdx.z = rank): the S CTAs of a cluster run k-step ranges of one 128x128
// fp32-accumulate tile and park their partials fragment-native in ws (L2); after the cluster barrier CTA r
// sums its share of the fragments over the S partials in rank order: deterministic, one launch. Remote DSMEM
// reads instead: 26.8k vs 1.1k cycles for the 2048x128x2816 reduction (sm_120).
template <class C>
__global__ void __launch_bounds__(kThreads, 1) gemm_f16_splitk_kernel(
    F16Args p, float4* ws, const __grid_constant__ CUtensorMap map_a,
    const __grid_constant__ CUtensorMap map_w) {
    static_assert(!C::kAcc16 && C::kBN == 128, "fp32 128x128 partials");
    constexpr int kFrags = 4 * C::kNF;  // float4 fragments per MMA thread
    extern __shared__ __align__(128) uint8_t smem_raw[];
    namespace cg = cooperative_groups;
    const cg::cluster_group cluster = cg::this_cluster();
    const int S = static_cast<int>(cluster.num_blocks());
    const int rank = static_cast<int>(cluster.block_rank());
    const uint32_t raw = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
    const uint32_t pad = ((raw + kAlign - 1) & ~uint32_t(kAlign - 1)) - raw;
    const uint32_t s0 = raw + pad;
    const uint32_t bars = s0 + C::kStages * C::kStageBytes;
    const int n0 = static_cast<int>(blockIdx.x) * C::kBN;
    const int m0 = static_cast<int>(blockIdx.y) * kBM;
    const int KT = p.K / C::kBK;
    const int kb = (rank * KT) / S;
    const int steps = (((rank + 1) * KT) / S) - kb;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    float4* tile_ws = ws +
                      (static_cast<size_t>((blockIdx.y * gridDim.x) + blockIdx.x) * S * kFrags * kMmaThreads);
    if (threadIdx.x == 0) {
        for (int s = 0; s < C::kStages; ++s) {
            mbar_init(bars + s * 8, 1);
            mbar_init(bars + (C::kStages + s) * 8, kMmaThreads / 32);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (warp == kMmaThreads / 32) {  // producer; stays for the cluster barrier
        if (lane == 0) {
            asm volatile("prefetch.tensormap [%0];" ::"l"(&map_a) : "memory");
            asm volatile("prefetch.tensormap [%0];" ::"l"(&map_w) : "memory");
        }
        if (lane == 0)
            for (int i = 0; i < steps; ++i) {
                const int s = i % C::kStages;
                const uint32_t full = bars + s * 8;
                if (i >= C::kStages)
                    mbar_wait(bars + (C::kStages + s) * 8, ((i / C::kStages) - 1) & 1);
                mbar_expect_tx(full, C::kTxBytes);
                tma_2d(s0 + s * C::kStageBytes, &map_a, (kb + i) * C::kRS, m0, full);
                tma_2d(s0 + s * C::kStageBytes + C::kATile, &map_w, (kb + i) * C::kRS, n0, full);
            }
    } else {
        const int wm = (warp % C::kWarpsM) * 64;
        const int wn = (warp / C::kWarpsM) * C::kWN;
        Acc<C> acc{};
        for (int i = 0; i < steps; ++i) {
            const int s = i % C::kStages;
            mbar_wait(bars + s * 8, (i / C::kStages) & 1);
            mma_stage<C>(acc, s0 + s * C::kStageBytes, wm, wn, lane);
            __syncwarp();
            if (lane == 0)
                mbar_arrive(bars + (C::kStages + s) * 8);
        }
        float4* own = tile_ws + (static_cast<size_t>(rank) * kFrags * kMmaThreads);
#pragma unroll
        for (int mf = 0; mf < 4; ++mf)
#pragma unroll
            for (int nf = 0; nf < C::kNF; ++nf)
                __stcg(own + (((mf * C::kNF) + nf) * kMmaThreads) + threadIdx.x,
                       make_float4(acc.v[mf][nf][0], acc.v[mf][nf][1], acc.v[mf][nf][2], acc.v[mf][nf][3]));
    }
    cluster.sync();  // barrier.cluster arrive.release / wait.acquire: the peers' partials are visible
    const int i0 = (rank * kFrags * kMmaThreads) / S;
    const int i1 = ((rank + 1) * kFrags * kMmaThreads) / S;
    for (int i = i0 + static_cast<int>(threadIdx.x); i < i1; i += kThreads) {
        float4 v[kMaxSplit];  // every peer's load in flight before the first add
#pragma unroll
        for (int q = 0; q < kMaxSplit; ++q)
            if (q < S)
                v[q] = __ldcg(tile_ws + (static_cast<size_t>(q) * kFrags * kMmaThreads) + i);
        float4 sum = v[0];
#pragma unroll
        for (int q = 1; q < kMaxSplit; ++q)
            if (q < S) {
                sum.x += v[q].x;
                sum.y += v[q].y;
                sum.z += v[q].z;
                sum.w += v[q].w;
            }
        const int f = i / kMmaThreads;  // fragment (mf, nf) of MMA thread t
        const int t = i % kMmaThreads;
        const int tw = t >> 5;
        const int row = m0 + ((tw % C::kWarpsM) * 64) + ((f / C::kNF) * 16) + ((t & 31) >> 2);
        const int col = n0 + ((tw / C::kWarpsM) * C::kWN) + ((f % C::kNF) * 8) + ((t & 3) * 2);
        if (col < p.N) {
            if (row < p.M)
                *reinterpret_cast<__half2*>(p.d + (static_cast<size_t>(row) * p.N) +
                                            col) = __floats2half2_rn(sum.x * p.alpha, sum.y * p.alpha);
            if (row + 8 < p.M)
                *reinterpret_cast<__half2*>(p.d + (static_cast<size_t>(row + 8) * p.N) +
                                            col) = __floats2half2_rn(sum.z * p.alpha, sum.w * p.alpha);
        }
    }
}

}  // namespace

size_t gemm_f16_splitk_workspace_bytes(int M, int N) {
    return static_cast<size_t>((M + kBM - 1) / kBM) * ((N + 127) / 128) * kMaxSplit * kBM * 128 *
           sizeof(float);
}

bool gemm_f16_splitk(const void* a, const void* w, void* d, int M, int N, int K, float alpha, void* ws,
                     size_t ws_bytes, cudaStream_t stream) {
    using C = Cfg<128, 64, 3, false>;
    if (M <= 0 || N <= 0 || K < C::kBK || K % C::kBK != 0 || N % 8 != 0 || a == nullptr || w == nullptr ||
        d == nullptr || ws == nullptr || reinterpret_cast<uintptr_t>(a) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(w) % 16 != 0 || reinterpret_cast<uintptr_t>(d) % 4 != 0 ||
        reinterpret_cast<uintptr_t>(ws) % 16 != 0)
        return false;
    const int tiles = ((N + C::kBN - 1) / C::kBN) * ((M + kBM - 1) / kBM);
    int split = std::clamp(device_sm_count_or_0() / tiles, 1, std::min(kMaxSplit, K / C::kBK));
    while (split > 1 && ws_bytes < static_cast<size_t>(tiles) * split * kBM * C::kBN * sizeof(float))
        --split;
    if (ws_bytes < static_cast<size_t>(tiles) * split * kBM * C::kBN * sizeof(float))
        return false;
    CUtensorMap map_a{};
    CUtensorMap map_w{};
    if (!encode_2d(&map_a, a, M, K * 2, C::kRS, kBM) || !encode_2d(&map_w, w, N, K * 2, C::kRS, C::kBN))
        return false;
    static const bool smem_ok = cudaFuncSetAttribute(gemm_f16_splitk_kernel<C>,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     C::kSmemBytes) == cudaSuccess;
    if (!smem_ok)
        return false;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3((N + C::kBN - 1) / C::kBN, (M + kBM - 1) / kBM, split);
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = C::kSmemBytes;
    cfg.stream = stream;
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = 1;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = static_cast<unsigned>(split);
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    const F16Args p{static_cast<half*>(d), alpha, 0.0f, M, N, K};
    if (cudaLaunchKernelEx(&cfg, gemm_f16_splitk_kernel<C>, p, static_cast<float4*>(ws), map_a, map_w) !=
        cudaSuccess) {
        (void)cudaGetLastError();  // refused (cluster does not fit): the caller keeps its path
        return false;
    }
    return true;
}

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
// (K >= N) once its 128x256 grid fills >= 80 % of its last wave. Wins: 2048x5120x17408 (320 tiles,
// 94 %); loses: 2048x2688x4096 Nemotron out_proj (176 tiles, 52 %) 220 vs cuBLASLt 122 us in situ.
bool gemm_f16_tile_route(const Tensor& A, const Tensor& B, Tensor& C, float alpha, float beta,
                         cudaStream_t stream) {
    const int64_t M = A.shape[0];
    const int64_t K = A.shape[1];
    const int64_t N = B.shape[0];
    constexpr int64_t kSms = 170;
    const int64_t tiles = ((N + 255) / 256) * ((M + 127) / 128);
    const int64_t wave_slots = ((tiles + kSms - 1) / kSms) * kSms;
    if (!process_diag_f16_tile() || K < N || tiles < kSms || tiles * 5 < wave_slots * 4 ||
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
