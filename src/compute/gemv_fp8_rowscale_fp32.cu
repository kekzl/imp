// Per-row-scale FP8 E4M3 GEMV with FP32 output for the FP8 LM head (gemm.nvfp4_lm_head=fp8,
// #2156): y[r][row] = scale[row] * dot(W[row], x[r]). One warp per weight row, up to
// kMaxRows activation rows per weight pass.

#include "compute/gemm.h"
#include "core/logging.h"
#include "quant/fp8_row_quant.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

namespace imp {

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kMaxRows = 8;

// Row r's FMA chain depends only on (lane, K), never on NB or r: per-row bits are identical
// for every n_rows (#2152 rule), so decode n=1, batched decode and --perplexity agree.
template <int NB>
__global__ void __launch_bounds__(kThreads) gemv_fp8_rowscale_fp32_kernel(
    const uint8_t* __restrict__ W, const float* __restrict__ row_scales, const half* __restrict__ x,
    float* __restrict__ y, int M, int K) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5);
    if (row >= M)
        return;

    const uint4* w = reinterpret_cast<const uint4*>(W + static_cast<int64_t>(row) * K);
    const int k_vec = K / 16;
    float acc[NB];
#pragma unroll
    for (int b = 0; b < NB; ++b)
        acc[b] = 0.0f;

    for (int i = lane; i < k_vec; i += 32) {
        const uint4 wr = w[i];
        const uint8_t* wb = reinterpret_cast<const uint8_t*>(&wr);
        float wf[16];
#pragma unroll
        for (int j = 0; j < 16; ++j)
            wf[j] = fp8_rows::decode(wb[j]);
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            const uint4* xb = reinterpret_cast<const uint4*>(x + static_cast<int64_t>(b) * K) + 2 * i;
            const uint4 x0 = xb[0];
            const uint4 x1 = xb[1];
            const half* h0 = reinterpret_cast<const half*>(&x0);
            const half* h1 = reinterpret_cast<const half*>(&x1);
#pragma unroll
            for (int j = 0; j < 8; ++j)
                acc[b] = fmaf(wf[j], __half2float(h0[j]), acc[b]);
#pragma unroll
            for (int j = 0; j < 8; ++j)
                acc[b] = fmaf(wf[8 + j], __half2float(h1[j]), acc[b]);
        }
    }

    const float s = row_scales[row];
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        float v = acc[b];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0)
            y[static_cast<int64_t>(b) * M + row] = v * s;
    }
}

template <int NB>
void launch_rows(const uint8_t* W, const float* s, const half* x, float* y, int M, int K,
                 cudaStream_t stream) {
    const int blocks = (M + kWarps - 1) / kWarps;
    gemv_fp8_rowscale_fp32_kernel<NB><<<blocks, kThreads, 0, stream>>>(W, s, x, y, M, K);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace

bool gemv_fp8_rowscale_fp32(const void* W_fp8, const float* d_row_scales, const half* x, float* y, int M,
                            int K, int n_rows, cudaStream_t stream) {
    if (!W_fp8 || !d_row_scales || !x || !y || M <= 0 || K <= 0 || (K % 16) != 0 || n_rows <= 0)
        return false;
    const auto* W = static_cast<const uint8_t*>(W_fp8);
    for (int r0 = 0; r0 < n_rows; r0 += kMaxRows) {
        const int nb = std::min(kMaxRows, n_rows - r0);
        const half* xr = x + static_cast<int64_t>(r0) * K;
        float* yr = y + static_cast<int64_t>(r0) * M;
        switch (nb) {
            case 1:
                launch_rows<1>(W, d_row_scales, xr, yr, M, K, stream);
                break;
            case 2:
                launch_rows<2>(W, d_row_scales, xr, yr, M, K, stream);
                break;
            case 3:
                launch_rows<3>(W, d_row_scales, xr, yr, M, K, stream);
                break;
            case 4:
                launch_rows<4>(W, d_row_scales, xr, yr, M, K, stream);
                break;
            case 5:
                launch_rows<5>(W, d_row_scales, xr, yr, M, K, stream);
                break;
            case 6:
                launch_rows<6>(W, d_row_scales, xr, yr, M, K, stream);
                break;
            case 7:
                launch_rows<7>(W, d_row_scales, xr, yr, M, K, stream);
                break;
            default:
                launch_rows<8>(W, d_row_scales, xr, yr, M, K, stream);
                break;
        }
    }
    return true;
}

}  // namespace imp
