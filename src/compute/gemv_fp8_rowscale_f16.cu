// FP16-output per-row-scale FP8 GEMM for 2 <= M <= 32 rows: GDN / attention projections with a released F16
// source (executor_gemm_released.cpp) and the DSpark drafter (dspark_drafter.cu). The streamed kernel reads
// the weight in full 128 B lines (cp.async ring) instead of the split-K kernel's 8 rows x 64 B fragment
// loads; same chunk MMAs and quarter fold, so every output bit equals the split-K twin's.

#include "compute/gemm.h"
#include "compute/gemv_fp8_rowscale_mma.cuh"
#include "core/logging.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

namespace imp {

namespace {

using namespace fp8_rowscale;  // NOLINT(google-build-using-namespace): the TU's own helpers

// Split-K like fp8_head_mma_kernel (kNT * 8 rows per block, warp w = K quarter w); each warp streams its
// quarter through a kS-stage cp.async ring, 4 rows x 128 B per instruction, and loads the activation
// fragments one stage ahead. Same chunk MMAs, same w0 + w1 + w2 + w3 fold as the split-K kernel.
template <int MT, int kNT, int kS>
__global__ void __launch_bounds__(kThreads) fp8_proj_cpasync_kernel(const uint8_t* __restrict__ W,
                                                                    const float* __restrict__ row_scales,
                                                                    const half* __restrict__ x,
                                                                    half* __restrict__ y, int M, int K,
                                                                    int n_rows) {
    constexpr int kC = kStageBytes / 64;  // MMA chunks per stage
    constexpr int kR = kNT * 8;
    constexpr int kLoads = kR / 4;
    constexpr int kSlot = kR * kRowPitch;
    __shared__ float s_red[kWarps - 1][MT * kNT * 4][32];
    extern __shared__ __align__(16) uint8_t ring_smem[];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int n_base = blockIdx.x * kR;
    const int k_begin = warp * (K / kWarps);
    const int n_stages = K / (kWarps * kStageBytes);
    const uint32_t ring = static_cast<uint32_t>(__cvta_generic_to_shared(ring_smem)) + (warp * kS * kSlot);

    const uint8_t* src[kLoads];
    uint32_t dst[kLoads];
#pragma unroll
    for (int i = 0; i < kLoads; ++i) {
        const int row = (i * 4) + (lane >> 3);
        const int r = min(n_base + row, M - 1);  // clamped rows are never stored
        src[i] = W + (static_cast<int64_t>(r) * K) + k_begin + static_cast<int64_t>((lane & 7) * 16);
        dst[i] = (row * kRowPitch) + ((lane & 7) * 16);
    }
    auto issue = [&](int st) {
        if (st < n_stages) {
            const uint32_t slot = ring + ((st % kS) * kSlot);
#pragma unroll
            for (int i = 0; i < kLoads; ++i)
                cp_async16(slot + dst[i], src[i] + (static_cast<int64_t>(st) * kStageBytes));
        }
        cp_async_commit();
    };
    const uint4* xp[MT][2];
    bool x_ok[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = (mt * 16) + (h * 8) + g;
            x_ok[mt][h] = r < n_rows;
            xp[mt][h] = reinterpret_cast<const uint4*>(x + (static_cast<int64_t>(x_ok[mt][h] ? r : 0) * K));
        }
#pragma unroll
    for (int s = 0; s < kS - 1; ++s)
        issue(s);

    float acc[MT][kNT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int j = 0; j < kNT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e)
                acc[mt][j][e] = 0.0f;
    uint4 xn[kC][MT][2][2];
    auto load_x_stage = [&](int st) {
#pragma unroll
        for (int cc = 0; cc < kC; ++cc)
            load_x_chunk<MT>(xn[cc], xp, x_ok, k_begin + (st * kStageBytes) + (cc * 64), t);
    };
    load_x_stage(0);
    for (int st = 0; st < n_stages; ++st) {
        uint4 xc[kC][MT][2][2];
#pragma unroll
        for (int cc = 0; cc < kC; ++cc)
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    xc[cc][mt][h][0] = xn[cc][mt][h][0];
                    xc[cc][mt][h][1] = xn[cc][mt][h][1];
                }
        if (st + 1 < n_stages)
            load_x_stage(st + 1);
        cp_async_wait<kS - 2>();
        __syncwarp();
        issue(st + kS - 1);  // refills the slot consumed by stage st - 1
        // NOLINTNEXTLINE(bugprone-implicit-widening-of-multiplication-result): smem offsets < 25 KB
        const uint8_t* slot = ring_smem + (warp * kS * kSlot) + ((st % kS) * kSlot);
#pragma unroll
        for (int cc = 0; cc < kC; ++cc) {
            uint4 wr[kNT];
#pragma unroll
            for (int j = 0; j < kNT; ++j)
                // NOLINTNEXTLINE(bugprone-implicit-widening-of-multiplication-result): smem offsets < 25 KB
                wr[j] = *reinterpret_cast<const uint4*>(slot + (((8 * j) + g) * kRowPitch) + (cc * 64) +
                                                        (16 * t));
            mma_chunk_x<MT, kNT>(acc, wr, xc[cc]);
        }
        __syncwarp();
    }
    cp_async_wait<0>();

    if (warp > 0)
        stash_partials<MT, kNT>(acc, s_red[warp - 1], lane);
    __syncthreads();
    if (warp != 0)
        return;
    add_partials<MT, kNT>(acc, s_red, lane);
    store_rows<MT, kNT>(acc, row_scales, y, M, n_rows, n_base, g, t);
}

// Rows per block by pass size: <= 8 rows 8 (Flash-Next GDN out 2560 rows = 320 blocks), 9..16 rows 16,
// 17..32 rows 32: more rows per block cut the activation re-reads per weight byte. 2 stages each, chosen by
// FP8ProjStreamTest.Sweep over 8 shapes x 6 row counts (docs/audit/fp8_proj_stream_2026_11_04/).
constexpr int kStages = 2;

template <int MT, int kNT>
bool set_smem() {
    return cudaFuncSetAttribute(fp8_proj_cpasync_kernel<MT, kNT, kStages>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                cpasync_smem_bytes<kNT, kStages>()) == cudaSuccess;
}

bool proj_stream_ready() {
    static const bool ok = set_smem<1, 1>() && set_smem<1, 2>() && set_smem<2, 4>();
    return ok;
}

template <int MT, int kNT>
void launch_pass(const uint8_t* W, const float* d_row_scales, const half* x, half* y, int M, int K, int nb,
                 cudaStream_t stream) {
    fp8_proj_cpasync_kernel<MT, kNT, kStages>
        <<<(M + (kNT * 8) - 1) / (kNT * 8), kThreads, cpasync_smem_bytes<kNT, kStages>(), stream>>>(
            W, d_row_scales, x, y, M, K, nb);
    IMP_CUDA_CHECK_LAUNCH();
}

// Whole 128 B stages per K quarter: K % 512 == 0, else the caller takes the split-K twin.
bool launch_proj_stream(const void* W_fp8, const float* d_row_scales, const half* x, half* y, int M, int K,
                        int n_rows, cudaStream_t stream) {
    if (!W_fp8 || !d_row_scales || !x || !y || M <= 0 || K <= 0 || (K % (kStageBytes * kWarps)) != 0 ||
        n_rows <= 0 || !proj_stream_ready())
        return false;
    const auto* W = static_cast<const uint8_t*>(W_fp8);
    for (int r0 = 0; r0 < n_rows; r0 += kRowsPerPass) {
        const int nb = std::min(kRowsPerPass, n_rows - r0);
        const half* xr = x + (static_cast<int64_t>(r0) * K);
        half* yr = y + (static_cast<int64_t>(r0) * M);
        if (nb <= 8)
            launch_pass<1, 1>(W, d_row_scales, xr, yr, M, K, nb, stream);
        else if (nb <= 16)
            launch_pass<1, 2>(W, d_row_scales, xr, yr, M, K, nb, stream);
        else
            launch_pass<2, 4>(W, d_row_scales, xr, yr, M, K, nb, stream);
    }
    return true;
}

}  // namespace

bool gemv_fp8_rowscale_f16(const void* W_fp8, const float* d_row_scales, const half* x, half* y, int M, int K,
                           int n_rows, cudaStream_t stream) {
    return launch_proj_stream(W_fp8, d_row_scales, x, y, M, K, n_rows, stream) ||
           fp8_rowscale::split_k_f16(W_fp8, d_row_scales, x, y, M, K, n_rows, stream);
}

bool gemv_fp8_rowscale_f16_tuned(const void* W_fp8, const float* d_row_scales, const half* x, half* y, int M,
                                 int K, int n_rows, int variant, cudaStream_t stream) {
    switch (variant) {
        case 0:
            return gemv_fp8_rowscale_f16(W_fp8, d_row_scales, x, y, M, K, n_rows, stream);
        case 1:
            return fp8_rowscale::split_k_f16(W_fp8, d_row_scales, x, y, M, K, n_rows, stream);
        case 2:
            return launch_proj_stream(W_fp8, d_row_scales, x, y, M, K, n_rows, stream);
        default:
            return false;
    }
}

}  // namespace imp
