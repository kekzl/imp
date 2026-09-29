// FP8 E4M3 block-scale MoE decode GEMV. Math and layout: gemv_fp8_block_moe.h.
#include "compute/gemv_fp8_block_moe.h"

#include "compute/gemm_internal.cuh"
#include "core/logging.h"

#include <cuda_fp8.h>

namespace imp {

namespace {

// 16 E4M3 weights (one uint4) dotted with 16 FP16 inputs, FP32 accumulation.
__device__ __forceinline__ float fp8x16_dot(const uint4 w, const uint4 xa, const uint4 xb) {
    const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
    const uint32_t xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(wv[i] & 0xFFFFu),
                                                          __NV_E4M3);
        const __half2_raw hi =
            __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(wv[i] >> 16), __NV_E4M3);
        const float2 wl = __half22float2(*reinterpret_cast<const __half2*>(&lo));
        const float2 wh = __half22float2(*reinterpret_cast<const __half2*>(&hi));
        const float2 x0 = __half22float2(
            *reinterpret_cast<const __half2*>(&xv[static_cast<ptrdiff_t>(2 * i)]));
        const float2 x1 = __half22float2(*reinterpret_cast<const __half2*>(&xv[2 * i + 1]));
        acc += wl.x * x0.x + wl.y * x0.y + wh.x * x1.x + wh.y * x1.y;
    }
    return acc;
}

__global__ void gemv_fp8_block_moe_kernel(Fp8BlockMoeArgs a, int blocks_per_proj) {
    const int warps = blockDim.x / 32;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int sp = blockIdx.x / blocks_per_proj;  // slot * n_proj + proj
    const int slot = sp / a.n_proj;
    const int proj = sp % a.n_proj;
    const int row = (blockIdx.x % blocks_per_proj) * warps + warp;
    if (row >= a.rows)
        return;
    const int e = a.expert_ids[slot];
    const int tab = e * a.n_proj + proj;
    const uint8_t* w_row = a.w[tab] + static_cast<size_t>(row) * a.K;
    const int kb = a.K / kFp8BlockSize;
    const float* s_row = a.scales + (static_cast<size_t>(tab) * (a.rows / kFp8BlockSize) + row / kFp8BlockSize) * kb;
    const half* x = a.x + static_cast<size_t>(slot) * a.x_stride;

    float sum = 0.0f;
    const int chunks = a.K / 16;
    for (int c = lane; c < chunks; c += 32) {
        const uint4 w = *reinterpret_cast<const uint4*>(w_row + static_cast<ptrdiff_t>(c * 16));
        const uint4 xa = *reinterpret_cast<const uint4*>(x + static_cast<ptrdiff_t>(c * 16));
        const uint4 xb = *reinterpret_cast<const uint4*>(x + static_cast<ptrdiff_t>(c * 16) + 8);
        sum += fp8x16_dot(w, xa, xb) * s_row[(c * 16) / kFp8BlockSize];
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0)
        a.y[(static_cast<size_t>(slot) * a.n_proj + proj) * a.rows + row] = __float2half(sum);
}

__global__ void swiglu_packed_rows_kernel(const half* __restrict__ gu, half* __restrict__ act, int eff, int n) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n)
        return;
    const int s = t / eff;
    const int j = t % eff;
    const float g = __half2float(gu[static_cast<size_t>(s) * 2 * eff + j]);
    const float u = __half2float(gu[static_cast<size_t>(s) * 2 * eff + eff + j]);
    act[t] = __float2half(g / (1.0f + __expf(-g)) * u);
}

}  // namespace

bool gemv_fp8_block_moe(const Fp8BlockMoeArgs& a, cudaStream_t stream) {
    if (a.w == nullptr || a.scales == nullptr || a.expert_ids == nullptr || a.x == nullptr || a.y == nullptr ||
        a.rows <= 0 || a.K <= 0 || a.top_k <= 0 || a.n_proj <= 0 || a.rows % kFp8BlockSize != 0 ||
        a.K % kFp8BlockSize != 0)
        return false;
    const int blocks_per_proj = gemv_blocks(a.rows);
    gemv_fp8_block_moe_kernel<<<a.top_k * a.n_proj * blocks_per_proj, kGemvThreads, 0, stream>>>(a,
                                                                                                blocks_per_proj);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

void swiglu_packed_rows(const half* gu, half* act, int eff, int top_k, cudaStream_t stream) {
    const int n = eff * top_k;
    swiglu_packed_rows_kernel<<<(n + 255) / 256, 256, 0, stream>>>(gu, act, eff, n);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
