#include "compute/layernorm.h"
#include "compute/warp_reduce.cuh"
#include "runtime/pdl.h"
#include "core/tensor.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include "compute/pdl_device.cuh"

namespace imp {

// --------------------------------------------------------------------------
// Block-level reduction using shared memory (up to 32 warps = 1024 threads)
// --------------------------------------------------------------------------
__device__ float block_reduce_sum(float val) {
    __shared__ float shared[32];  // one slot per warp, up to 1024 threads = 32 warps
    const int lane = threadIdx.x & 31;
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
__global__ void rmsnorm_fp32_kernel(const float* __restrict__ x, const float* __restrict__ weight,
                                    float* __restrict__ out, int d_model, float eps, float weight_offset) {
    const int row = blockIdx.x;
    const float* x_row = x + static_cast<int64_t>(row) * d_model;
    float* out_row = out + static_cast<int64_t>(row) * d_model;

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
        out_row[i] = x_row[i] * inv_rms * (weight[i] + weight_offset);
    }
}

// --------------------------------------------------------------------------
// RMSNorm kernel for FP16 — vectorized float4 loads (8 halfs per load)
// Requires d_model % 8 == 0 (true for all supported models).
// --------------------------------------------------------------------------
__global__ void rmsnorm_fp16_kernel(const __half* __restrict__ x, const __half* __restrict__ weight,
                                    __half* __restrict__ out, int d_model, float eps, float weight_offset) {
    const int row = blockIdx.x;
    const int d_vec = d_model >> 3;  // d_model / 8
    const float4* x_vec = reinterpret_cast<const float4*>(x + static_cast<int64_t>(row) * d_model);
    const float4* w_vec = reinterpret_cast<const float4*>(weight);
    float4* out_vec = reinterpret_cast<float4*>(out + static_cast<int64_t>(row) * d_model);

    // Pass 1: vectorized sum-of-squares
    float sum_sq = 0.0f;
    pdl_wait();
    for (int i = threadIdx.x; i < d_vec; i += blockDim.x) {
        float4 v = x_vec[i];
        half2 h0 = *reinterpret_cast<half2*>(&v.x);
        half2 h1 = *reinterpret_cast<half2*>(&v.y);
        half2 h2 = *reinterpret_cast<half2*>(&v.z);
        half2 h3 = *reinterpret_cast<half2*>(&v.w);
        float2 f0 = __half22float2(h0), f1 = __half22float2(h1);
        float2 f2 = __half22float2(h2), f3 = __half22float2(h3);
        sum_sq += f0.x * f0.x + f0.y * f0.y + f1.x * f1.x + f1.y * f1.y + f2.x * f2.x + f2.y * f2.y +
                  f3.x * f3.x + f3.y * f3.y;
    }
    sum_sq = block_reduce_sum(sum_sq);

    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(sum_sq / static_cast<float>(d_model) + eps);
    }
    __syncthreads();
    pdl_trigger();
    const float inv_rms = s_inv_rms;

    // Pass 2: vectorized normalize + scale (x re-read hits L1 cache)
    for (int i = threadIdx.x; i < d_vec; i += blockDim.x) {
        float4 xv = x_vec[i];
        float4 wv = w_vec[i];

        half2 xh0 = *reinterpret_cast<half2*>(&xv.x);
        half2 xh1 = *reinterpret_cast<half2*>(&xv.y);
        half2 xh2 = *reinterpret_cast<half2*>(&xv.z);
        half2 xh3 = *reinterpret_cast<half2*>(&xv.w);

        half2 wh0 = *reinterpret_cast<half2*>(&wv.x);
        half2 wh1 = *reinterpret_cast<half2*>(&wv.y);
        half2 wh2 = *reinterpret_cast<half2*>(&wv.z);
        half2 wh3 = *reinterpret_cast<half2*>(&wv.w);

        float2 xf0 = __half22float2(xh0), wf0 = __half22float2(wh0);
        float2 xf1 = __half22float2(xh1), wf1 = __half22float2(wh1);
        float2 xf2 = __half22float2(xh2), wf2 = __half22float2(wh2);
        float2 xf3 = __half22float2(xh3), wf3 = __half22float2(wh3);

        float4 result;
        *reinterpret_cast<half2*>(&result.x) = __float22half2_rn(
            make_float2(xf0.x * inv_rms * (wf0.x + weight_offset),
                        xf0.y * inv_rms * (wf0.y + weight_offset)));
        *reinterpret_cast<half2*>(&result.y) = __float22half2_rn(
            make_float2(xf1.x * inv_rms * (wf1.x + weight_offset),
                        xf1.y * inv_rms * (wf1.y + weight_offset)));
        *reinterpret_cast<half2*>(&result.z) = __float22half2_rn(
            make_float2(xf2.x * inv_rms * (wf2.x + weight_offset),
                        xf2.y * inv_rms * (wf2.y + weight_offset)));
        *reinterpret_cast<half2*>(&result.w) = __float22half2_rn(
            make_float2(xf3.x * inv_rms * (wf3.x + weight_offset),
                        xf3.y * inv_rms * (wf3.y + weight_offset)));

        out_vec[i] = result;
    }
}

// --------------------------------------------------------------------------
// Warp-per-row FP16 RMSNorm — the batch-prefill variant (#602).
//
// The block-per-row kernel above runs 512 threads on rows of d_vec =
// d_model/8 float4s (256 for d=2048): half the threads idle and every row
// pays two __syncthreads + a smem round-trip — measured 8-10% of DRAM BW
// at 7-8% of the NVFP4 prefill window (latency-bound, not bandwidth).
// One WARP per row needs only shuffle reductions: no barrier, no smem.
// Eight rows per 256-thread block.
// --------------------------------------------------------------------------
__global__ void rmsnorm_fp16_warp_kernel(const __half* __restrict__ x,
                                         const __half* __restrict__ weight,
                                         __half* __restrict__ out, int rows, int d_model,
                                         float eps, float weight_offset) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;
    const int d_vec = d_model >> 3;
    const float4* x_vec = reinterpret_cast<const float4*>(x + static_cast<int64_t>(row) * d_model);
    const float4* w_vec = reinterpret_cast<const float4*>(weight);
    float4* out_vec = reinterpret_cast<float4*>(out + static_cast<int64_t>(row) * d_model);

    float sum_sq = 0.0f;
    pdl_wait();
    for (int i = lane; i < d_vec; i += 32) {
        float4 v = x_vec[i];
        half2 h0 = *reinterpret_cast<half2*>(&v.x);
        half2 h1 = *reinterpret_cast<half2*>(&v.y);
        half2 h2 = *reinterpret_cast<half2*>(&v.z);
        half2 h3 = *reinterpret_cast<half2*>(&v.w);
        float2 f0 = __half22float2(h0), f1 = __half22float2(h1);
        float2 f2 = __half22float2(h2), f3 = __half22float2(h3);
        sum_sq += f0.x * f0.x + f0.y * f0.y + f1.x * f1.x + f1.y * f1.y + f2.x * f2.x +
                  f2.y * f2.y + f3.x * f3.x + f3.y * f3.y;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFFu, sum_sq, off);
    const float inv_rms = rsqrtf(sum_sq / static_cast<float>(d_model) + eps);
    pdl_trigger();

    for (int i = lane; i < d_vec; i += 32) {
        float4 xv = x_vec[i];
        float4 wv = w_vec[i];
        half2 xh0 = *reinterpret_cast<half2*>(&xv.x);
        half2 xh1 = *reinterpret_cast<half2*>(&xv.y);
        half2 xh2 = *reinterpret_cast<half2*>(&xv.z);
        half2 xh3 = *reinterpret_cast<half2*>(&xv.w);
        half2 wh0 = *reinterpret_cast<half2*>(&wv.x);
        half2 wh1 = *reinterpret_cast<half2*>(&wv.y);
        half2 wh2 = *reinterpret_cast<half2*>(&wv.z);
        half2 wh3 = *reinterpret_cast<half2*>(&wv.w);
        float2 xf0 = __half22float2(xh0), wf0 = __half22float2(wh0);
        float2 xf1 = __half22float2(xh1), wf1 = __half22float2(wh1);
        float2 xf2 = __half22float2(xh2), wf2 = __half22float2(wh2);
        float2 xf3 = __half22float2(xh3), wf3 = __half22float2(wh3);
        float4 result;
        *reinterpret_cast<half2*>(&result.x) = __float22half2_rn(
            make_float2(xf0.x * inv_rms * (wf0.x + weight_offset),
                        xf0.y * inv_rms * (wf0.y + weight_offset)));
        *reinterpret_cast<half2*>(&result.y) = __float22half2_rn(
            make_float2(xf1.x * inv_rms * (wf1.x + weight_offset),
                        xf1.y * inv_rms * (wf1.y + weight_offset)));
        *reinterpret_cast<half2*>(&result.z) = __float22half2_rn(
            make_float2(xf2.x * inv_rms * (wf2.x + weight_offset),
                        xf2.y * inv_rms * (wf2.y + weight_offset)));
        *reinterpret_cast<half2*>(&result.w) = __float22half2_rn(
            make_float2(xf3.x * inv_rms * (wf3.x + weight_offset),
                        xf3.y * inv_rms * (wf3.y + weight_offset)));
        out_vec[i] = result;
    }
}

// Row-block batched-decode kernels (plain + NVFP4 producer fusion) live in
// layernorm_rowblock.cu — see rmsnorm_fp16_rowblock() / rmsnorm_nvfp4().

// --------------------------------------------------------------------------
// Fused RMSNorm + residual for FP32
// out = rmsnorm(x + residual) * weight
// x is updated in-place to (x + residual)
// --------------------------------------------------------------------------
__global__ void rmsnorm_residual_fp32_kernel(float* __restrict__ x, const float* __restrict__ residual,
                                             const float* __restrict__ weight, float* __restrict__ out,
                                             int d_model, float eps, float weight_offset) {
    const int row = blockIdx.x;
    float* x_row = x + static_cast<int64_t>(row) * d_model;
    const float* r_row = residual + static_cast<int64_t>(row) * d_model;
    float* out_row = out + static_cast<int64_t>(row) * d_model;

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
        out_row[i] = x_row[i] * inv_rms * (weight[i] + weight_offset);
    }
}

// --------------------------------------------------------------------------
// Fused RMSNorm + residual for FP16 — vectorized float4 loads
// --------------------------------------------------------------------------
__global__ void rmsnorm_residual_fp16_kernel(__half* __restrict__ x, const __half* __restrict__ residual,
                                             const __half* __restrict__ weight, __half* __restrict__ out,
                                             int d_model, float eps, float weight_offset) {
    const int row = blockIdx.x;
    const int d_vec = d_model >> 3;
    float4* x_vec = reinterpret_cast<float4*>(x + static_cast<int64_t>(row) * d_model);
    const float4* r_vec = reinterpret_cast<const float4*>(residual + static_cast<int64_t>(row) * d_model);
    const float4* w_vec = reinterpret_cast<const float4*>(weight);
    float4* out_vec = reinterpret_cast<float4*>(out + static_cast<int64_t>(row) * d_model);

    float sum_sq = 0.0f;
    pdl_wait();
    for (int i = threadIdx.x; i < d_vec; i += blockDim.x) {
        float4 xv = x_vec[i];
        float4 rv = r_vec[i];

        half2 xh0 = *reinterpret_cast<half2*>(&xv.x);
        half2 xh1 = *reinterpret_cast<half2*>(&xv.y);
        half2 xh2 = *reinterpret_cast<half2*>(&xv.z);
        half2 xh3 = *reinterpret_cast<half2*>(&xv.w);

        half2 rh0 = *reinterpret_cast<half2*>(&rv.x);
        half2 rh1 = *reinterpret_cast<half2*>(&rv.y);
        half2 rh2 = *reinterpret_cast<half2*>(&rv.z);
        half2 rh3 = *reinterpret_cast<half2*>(&rv.w);

        float2 xf0 = __half22float2(xh0), rf0 = __half22float2(rh0);
        float2 xf1 = __half22float2(xh1), rf1 = __half22float2(rh1);
        float2 xf2 = __half22float2(xh2), rf2 = __half22float2(rh2);
        float2 xf3 = __half22float2(xh3), rf3 = __half22float2(rh3);

        float2 s0 = make_float2(xf0.x + rf0.x, xf0.y + rf0.y);
        float2 s1 = make_float2(xf1.x + rf1.x, xf1.y + rf1.y);
        float2 s2 = make_float2(xf2.x + rf2.x, xf2.y + rf2.y);
        float2 s3 = make_float2(xf3.x + rf3.x, xf3.y + rf3.y);

        // Write x + residual back
        float4 sv;
        *reinterpret_cast<half2*>(&sv.x) = __float22half2_rn(s0);
        *reinterpret_cast<half2*>(&sv.y) = __float22half2_rn(s1);
        *reinterpret_cast<half2*>(&sv.z) = __float22half2_rn(s2);
        *reinterpret_cast<half2*>(&sv.w) = __float22half2_rn(s3);
        x_vec[i] = sv;

        sum_sq += s0.x * s0.x + s0.y * s0.y + s1.x * s1.x + s1.y * s1.y + s2.x * s2.x + s2.y * s2.y +
                  s3.x * s3.x + s3.y * s3.y;
    }
    sum_sq = block_reduce_sum(sum_sq);

    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(sum_sq / static_cast<float>(d_model) + eps);
    }
    __syncthreads();
    pdl_trigger();
    const float inv_rms = s_inv_rms;

    for (int i = threadIdx.x; i < d_vec; i += blockDim.x) {
        float4 xv = x_vec[i];  // re-read x+residual from L1
        float4 wv = w_vec[i];

        half2 xh0 = *reinterpret_cast<half2*>(&xv.x);
        half2 xh1 = *reinterpret_cast<half2*>(&xv.y);
        half2 xh2 = *reinterpret_cast<half2*>(&xv.z);
        half2 xh3 = *reinterpret_cast<half2*>(&xv.w);

        half2 wh0 = *reinterpret_cast<half2*>(&wv.x);
        half2 wh1 = *reinterpret_cast<half2*>(&wv.y);
        half2 wh2 = *reinterpret_cast<half2*>(&wv.z);
        half2 wh3 = *reinterpret_cast<half2*>(&wv.w);

        float2 xf0 = __half22float2(xh0), wf0 = __half22float2(wh0);
        float2 xf1 = __half22float2(xh1), wf1 = __half22float2(wh1);
        float2 xf2 = __half22float2(xh2), wf2 = __half22float2(wh2);
        float2 xf3 = __half22float2(xh3), wf3 = __half22float2(wh3);

        float4 result;
        *reinterpret_cast<half2*>(&result.x) = __float22half2_rn(
            make_float2(xf0.x * inv_rms * (wf0.x + weight_offset),
                        xf0.y * inv_rms * (wf0.y + weight_offset)));
        *reinterpret_cast<half2*>(&result.y) = __float22half2_rn(
            make_float2(xf1.x * inv_rms * (wf1.x + weight_offset),
                        xf1.y * inv_rms * (wf1.y + weight_offset)));
        *reinterpret_cast<half2*>(&result.z) = __float22half2_rn(
            make_float2(xf2.x * inv_rms * (wf2.x + weight_offset),
                        xf2.y * inv_rms * (wf2.y + weight_offset)));
        *reinterpret_cast<half2*>(&result.w) = __float22half2_rn(
            make_float2(xf3.x * inv_rms * (wf3.x + weight_offset),
                        xf3.y * inv_rms * (wf3.y + weight_offset)));

        out_vec[i] = result;
    }
}

// --------------------------------------------------------------------------
// Host dispatch: rmsnorm
// --------------------------------------------------------------------------
void rmsnorm(const Tensor& x, const Tensor& weight, Tensor& out, float eps, cudaStream_t stream,
             float weight_offset) {
    const int rows = static_cast<int>(x.shape[0]);
    const int d_model = static_cast<int>(x.shape[1]);
    const int block = 512;

    if (rows == 0 || d_model == 0)
        return;

    switch (x.qtype) {
        case QType::F32:
            pdl::launch(rmsnorm_fp32_kernel, dim3(rows), dim3(block), 0, stream,
                        static_cast<const float*>(x.data), static_cast<const float*>(weight.data),
                        static_cast<float*>(out.data), d_model, eps, weight_offset);
            break;
        case QType::F16:
            // Batched decode (2..64 rows): row-block kernel, register-
            // resident single DRAM pass — the warp kernel reads every row
            // twice and fields 4 CTAs at rows=32 (see its comment).
            if (rows >= 2 && rows <= 64 && (d_model & 7) == 0 && (d_model >> 3) <= 1024) {
                rmsnorm_fp16_rowblock(x, weight, out, rows, d_model, eps, stream, weight_offset);
                break;
            }
            // Batch prefill (#602): warp-per-row — no barriers/smem; the
            // block-per-row kernel was latency-bound at 8-10% BW.
            if (rows >= 16 && (d_model & 7) == 0) {
                const int warps_per_block = 8;
                const int grid = (rows + warps_per_block - 1) / warps_per_block;
                pdl::launch(rmsnorm_fp16_warp_kernel, dim3(grid), dim3(warps_per_block * 32), 0,
                            stream, static_cast<const __half*>(x.data),
                            static_cast<const __half*>(weight.data),
                            static_cast<__half*>(out.data), rows, d_model, eps, weight_offset);
                break;
            }
            pdl::launch(rmsnorm_fp16_kernel, dim3(rows), dim3(block), 0, stream,
                        static_cast<const __half*>(x.data), static_cast<const __half*>(weight.data),
                        static_cast<__half*>(out.data), d_model, eps, weight_offset);
            break;
        default:
            break;
    }
}

// --------------------------------------------------------------------------
// RMSNorm with FP32 input and FP16 output — used for Gemma-4 to avoid
// losing precision on the attention/FFN entry when the residual stream is
// kept in FP32 (`fp32_hidden_`) but the downstream GEMM expects FP16 input.
// --------------------------------------------------------------------------
__global__ void rmsnorm_fp32_in_fp16_out_kernel(const float* __restrict__ x,
                                                const __half* __restrict__ weight, __half* __restrict__ out,
                                                int d_model, float eps, float weight_offset) {
    const int row = blockIdx.x;
    const float* x_row = x + static_cast<int64_t>(row) * d_model;
    __half* out_row = out + static_cast<int64_t>(row) * d_model;

    // Pass 1: sum of squares in FP32.
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        float v = x_row[i];
        sum_sq += v * v;
    }
    sum_sq = block_reduce_sum(sum_sq);

    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(sum_sq / static_cast<float>(d_model) + eps);
    }
    __syncthreads();
    const float inv_rms = s_inv_rms;

    // Pass 2: normalize + weight, cast to FP16 at store.
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        float v = x_row[i] * inv_rms * (__half2float(weight[i]) + weight_offset);
        out_row[i] = __float2half(v);
    }
}

void rmsnorm_fp32_to_fp16(const Tensor& x_fp32, const Tensor& weight, Tensor& out_fp16, float eps,
                          cudaStream_t stream, float weight_offset) {
    const int rows = static_cast<int>(x_fp32.shape[0]);
    const int d_model = static_cast<int>(x_fp32.shape[1]);
    if (rows == 0 || d_model == 0)
        return;
    const int block = 512;
    pdl::launch(rmsnorm_fp32_in_fp16_out_kernel, dim3(rows), dim3(block), 0, stream,
                static_cast<const float*>(x_fp32.data), static_cast<const __half*>(weight.data),
                static_cast<__half*>(out_fp16.data), d_model, eps, weight_offset);
}

// FP32 input, FP32 output, FP16 weight. Used for Gemma-4 ggml MMVQ prefill:
// keeps full FP32 precision through norm → Q8_1 quantization, matching llama's
// FP32→Q8_1 path and avoiding the ~0.03% per-element FP16 truncation.
__global__ void rmsnorm_fp32_in_fp32_out_kernel(const float* __restrict__ x,
                                                const __half* __restrict__ weight, float* __restrict__ out,
                                                int d_model, float eps, float weight_offset) {
    const int row = blockIdx.x;
    const float* x_row = x + static_cast<int64_t>(row) * d_model;
    float* out_row = out + static_cast<int64_t>(row) * d_model;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        float v = x_row[i];
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
        out_row[i] = x_row[i] * inv_rms * (__half2float(weight[i]) + weight_offset);
    }
}

void rmsnorm_fp32_to_fp32(const Tensor& x_fp32, const Tensor& weight, float* out_fp32, int rows, int d_model,
                          float eps, cudaStream_t stream, float weight_offset) {
    if (rows == 0 || d_model == 0)
        return;
    const int block = 512;
    pdl::launch(rmsnorm_fp32_in_fp32_out_kernel, dim3(rows), dim3(block), 0, stream,
                static_cast<const float*>(x_fp32.data), static_cast<const __half*>(weight.data), out_fp32,
                d_model, eps, weight_offset);
}

// --------------------------------------------------------------------------
// Host dispatch: rmsnorm_residual
// --------------------------------------------------------------------------
void rmsnorm_residual(const Tensor& x, const Tensor& residual, const Tensor& weight, Tensor& out, float eps,
                      cudaStream_t stream, float weight_offset) {
    const int rows = static_cast<int>(x.shape[0]);
    const int d_model = static_cast<int>(x.shape[1]);
    const int block = 512;

    if (rows == 0 || d_model == 0)
        return;

    switch (x.qtype) {
        case QType::F32:
            pdl::launch(rmsnorm_residual_fp32_kernel, dim3(rows), dim3(block), 0, stream,
                        static_cast<float*>(x.data), static_cast<const float*>(residual.data),
                        static_cast<const float*>(weight.data), static_cast<float*>(out.data), d_model, eps,
                        weight_offset);
            break;
        case QType::F16:
            pdl::launch(rmsnorm_residual_fp16_kernel, dim3(rows), dim3(block), 0, stream,
                        static_cast<__half*>(x.data), static_cast<const __half*>(residual.data),
                        static_cast<const __half*>(weight.data), static_cast<__half*>(out.data), d_model, eps,
                        weight_offset);
            break;
        default:
            break;
    }
}

// --------------------------------------------------------------------------
// PDL registration
// --------------------------------------------------------------------------
void layernorm_pdl_register() {
    // Only kernels that call pdl_wait() may be registered (compute/pdl_device.cuh).
    pdl::enable(reinterpret_cast<const void*>(&rmsnorm_fp16_kernel));
    pdl::enable(reinterpret_cast<const void*>(&rmsnorm_fp16_warp_kernel));
    pdl::enable(reinterpret_cast<const void*>(&rmsnorm_residual_fp16_kernel));
}

// --------------------------------------------------------------------------
// True LayerNorm with bias + optional residual (#836, encoder post-LN).
// One CTA per row; the row is staged in dynamic shared memory (d_model * 4 B
// = 3 KiB at nomic's 768), mean/var via two block reductions.
// --------------------------------------------------------------------------
namespace {
__device__ __forceinline__ float ln_param(const void* p, bool f32, int i) {
    return f32 ? static_cast<const float*>(p)[i]
               : __half2float(static_cast<const __half*>(p)[i]);
}

__device__ float ln_block_sum(float v) {
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffff, v, m);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    const int n_warps = (blockDim.x + 31) / 32;
    float total = (threadIdx.x < n_warps) ? warp_sums[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32)
        for (int m = 16; m > 0; m >>= 1) total += __shfl_xor_sync(0xffffffff, total, m);
    if (threadIdx.x == 0) warp_sums[0] = total;
    __syncthreads();
    return warp_sums[0];
}

__global__ void layernorm_residual_kernel(const __half* __restrict__ x,
                                          const __half* __restrict__ residual,
                                          const void* __restrict__ w, bool w_f32,
                                          const void* __restrict__ b, bool b_f32,
                                          __half* __restrict__ out, int d, float eps) {
    extern __shared__ float row[];
    const size_t base = static_cast<size_t>(blockIdx.x) * d;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float v = __half2float(x[base + i]);
        if (residual) v += __half2float(residual[base + i]);
        row[i] = v;
        sum += v;
    }
    __syncthreads();
    const float mean = ln_block_sum(sum) / d;
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const float c = row[i] - mean;
        var_sum += c * c;
    }
    __syncthreads();
    const float rstd = rsqrtf(ln_block_sum(var_sum) / d + eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const float bias = b ? ln_param(b, b_f32, i) : 0.0f;
        out[base + i] = __float2half((row[i] - mean) * rstd * ln_param(w, w_f32, i) + bias);
    }
}
}  // namespace

void layernorm_residual(const Tensor& x, const Tensor& residual, const Tensor& weight,
                        const Tensor& bias, Tensor& out, float eps, cudaStream_t stream) {
    const int rows = static_cast<int>(x.shape[0]);
    const int d = static_cast<int>(x.shape[1]);
    const bool w_f32 = weight.qtype == QType::F32;
    const bool b_f32 = bias.qtype == QType::F32;
    const size_t shmem = static_cast<size_t>(d) * sizeof(float);
    layernorm_residual_kernel<<<rows, 256, shmem, stream>>>(
        static_cast<const __half*>(x.data),
        residual.data ? static_cast<const __half*>(residual.data) : nullptr, weight.data, w_f32,
        bias.data, b_f32, static_cast<__half*>(out.data), d, eps);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
