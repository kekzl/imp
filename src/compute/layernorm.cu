#include "compute/layernorm.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

namespace imp {

// --------------------------------------------------------------------------
// Warp-level reduction: sum across a warp using shuffle intrinsics
// --------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// --------------------------------------------------------------------------
// Block-level reduction using shared memory (up to 8 warps = 256 threads)
// --------------------------------------------------------------------------
__device__ float block_reduce_sum(float val) {
    __shared__ float shared[8];  // one slot per warp, max 256 threads = 8 warps
    const int lane   = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces the warp-level results
    const int num_warps = (blockDim.x + 31) / 32;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

// --------------------------------------------------------------------------
// RMSNorm kernel for FP32
// One block per row. Block size 256.
// --------------------------------------------------------------------------
__global__ void rmsnorm_fp32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int d_model,
    float eps)
{
    const int row = blockIdx.x;
    const float* x_row = x + static_cast<int64_t>(row) * d_model;
    float*     out_row = out + static_cast<int64_t>(row) * d_model;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        float v = x_row[i];
        sum_sq += v * v;
    }
    sum_sq = block_reduce_sum(sum_sq);

    // Broadcast the inverse RMS
    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(sum_sq / static_cast<float>(d_model) + eps);
    }
    __syncthreads();
    const float inv_rms = s_inv_rms;

    // Normalize and scale
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        out_row[i] = x_row[i] * inv_rms * weight[i];
    }
}

// --------------------------------------------------------------------------
// RMSNorm kernel for FP16 (read half, compute in float, write half)
// --------------------------------------------------------------------------
__global__ void rmsnorm_fp16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ weight,
    __half* __restrict__ out,
    int d_model,
    float eps)
{
    const int row = blockIdx.x;
    const __half* x_row = x + static_cast<int64_t>(row) * d_model;
    __half*     out_row = out + static_cast<int64_t>(row) * d_model;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        float v = __half2float(x_row[i]);
        sum_sq += v * v;
    }
    sum_sq = block_reduce_sum(sum_sq);

    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(sum_sq / static_cast<float>(d_model) + eps);
    }
    __syncthreads();
    const float inv_rms = s_inv_rms;

    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        float v = __half2float(x_row[i]);
        float w = __half2float(weight[i]);
        out_row[i] = __float2half(v * inv_rms * w);
    }
}

// --------------------------------------------------------------------------
// Fused RMSNorm + residual for FP32
// out = rmsnorm(x + residual) * weight
// x is updated in-place to (x + residual)
// --------------------------------------------------------------------------
__global__ void rmsnorm_residual_fp32_kernel(
    float* __restrict__ x,
    const float* __restrict__ residual,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int d_model,
    float eps)
{
    const int row = blockIdx.x;
    float*       x_row = x + static_cast<int64_t>(row) * d_model;
    const float* r_row = residual + static_cast<int64_t>(row) * d_model;
    float*     out_row = out + static_cast<int64_t>(row) * d_model;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        float v = x_row[i] + r_row[i];
        x_row[i] = v;  // store x + residual back
        sum_sq += v * v;
    }
    sum_sq = block_reduce_sum(sum_sq);

    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(sum_sq / static_cast<float>(d_model) + eps);
    }
    __syncthreads();
    const float inv_rms = s_inv_rms;

    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        out_row[i] = x_row[i] * inv_rms * weight[i];
    }
}

// --------------------------------------------------------------------------
// Fused RMSNorm + residual for FP16
// --------------------------------------------------------------------------
__global__ void rmsnorm_residual_fp16_kernel(
    __half* __restrict__ x,
    const __half* __restrict__ residual,
    const __half* __restrict__ weight,
    __half* __restrict__ out,
    int d_model,
    float eps)
{
    const int row = blockIdx.x;
    __half*       x_row = x + static_cast<int64_t>(row) * d_model;
    const __half* r_row = residual + static_cast<int64_t>(row) * d_model;
    __half*     out_row = out + static_cast<int64_t>(row) * d_model;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        float v = __half2float(x_row[i]) + __half2float(r_row[i]);
        x_row[i] = __float2half(v);
        sum_sq += v * v;
    }
    sum_sq = block_reduce_sum(sum_sq);

    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(sum_sq / static_cast<float>(d_model) + eps);
    }
    __syncthreads();
    const float inv_rms = s_inv_rms;

    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        float v = __half2float(x_row[i]);
        float w = __half2float(weight[i]);
        out_row[i] = __float2half(v * inv_rms * w);
    }
}

// --------------------------------------------------------------------------
// Host dispatch: rmsnorm
// --------------------------------------------------------------------------
void rmsnorm(const Tensor& x, const Tensor& weight, Tensor& out,
             float eps, cudaStream_t stream)
{
    const int rows    = static_cast<int>(x.shape[0]);
    const int d_model = static_cast<int>(x.shape[1]);
    const int block   = 256;

    if (rows == 0 || d_model == 0) return;

    switch (x.dtype) {
        case DType::FP32:
            rmsnorm_fp32_kernel<<<rows, block, 0, stream>>>(
                static_cast<const float*>(x.data),
                static_cast<const float*>(weight.data),
                static_cast<float*>(out.data),
                d_model, eps);
            break;
        case DType::FP16:
            rmsnorm_fp16_kernel<<<rows, block, 0, stream>>>(
                static_cast<const __half*>(x.data),
                static_cast<const __half*>(weight.data),
                static_cast<__half*>(out.data),
                d_model, eps);
            break;
        default:
            break;
    }
}

// --------------------------------------------------------------------------
// Host dispatch: rmsnorm_residual
// --------------------------------------------------------------------------
void rmsnorm_residual(const Tensor& x, const Tensor& residual,
                      const Tensor& weight, Tensor& out,
                      float eps, cudaStream_t stream)
{
    const int rows    = static_cast<int>(x.shape[0]);
    const int d_model = static_cast<int>(x.shape[1]);
    const int block   = 256;

    if (rows == 0 || d_model == 0) return;

    switch (x.dtype) {
        case DType::FP32:
            rmsnorm_residual_fp32_kernel<<<rows, block, 0, stream>>>(
                static_cast<float*>(x.data),
                static_cast<const float*>(residual.data),
                static_cast<const float*>(weight.data),
                static_cast<float*>(out.data),
                d_model, eps);
            break;
        case DType::FP16:
            rmsnorm_residual_fp16_kernel<<<rows, block, 0, stream>>>(
                static_cast<__half*>(x.data),
                static_cast<const __half*>(residual.data),
                static_cast<const __half*>(weight.data),
                static_cast<__half*>(out.data),
                d_model, eps);
            break;
        default:
            break;
    }
}

} // namespace imp
