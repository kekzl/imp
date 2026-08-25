// gdn_gated_norm.cu — the RMSNormGated + SiLU family (FP16, the NVFP4
// producer-fusion variant, FP32-in and FP32-in/out). Split out of gdn.cu
// (one logical unit per TU; the hard-review size gate was the trigger).
// Kernels moved VERBATIM from gdn.cu.
#include "compute/gdn.h"
#include "quant/nvfp4_pack.cuh"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace imp {

// ---------------------------------------------------------------------------
// Fused RMSNormGated + SiLU kernel.
// Computes: y[t,h,:] = rmsnorm(y[t,h,:], weight) * silu(gate[t,h,:])
//
// Grid:  (n_tokens, n_heads)
// Block: (head_dim)
// ---------------------------------------------------------------------------
__global__ void gdn_rmsnorm_gated_silu_kernel(
    half* __restrict__ y,             // [n_tokens, n_heads * head_dim] in/out
    const half* __restrict__ gate,    // [n_tokens, n_heads * head_dim]
    const half* __restrict__ weight,  // [head_dim] shared norm weight
    float eps, int n_heads, int head_dim) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    if (d >= head_dim)
        return;

    const int inner = n_heads * head_dim;
    const int base = t * inner + h * head_dim;

    // Load y value
    float val = __half2float(y[base + d]);

    // Parallel sum-of-squares for RMSNorm
    extern __shared__ float s_buf[];
    s_buf[d] = val * val;
    __syncthreads();
    for (int stride = head_dim / 2; stride > 0; stride >>= 1) {
        if (d < stride)
            s_buf[d] += s_buf[d + stride];
        __syncthreads();
    }
    float inv_rms = rsqrtf(s_buf[0] / static_cast<float>(head_dim) + eps);

    // RMSNorm: normalize and scale by weight
    float normed = val * inv_rms * __half2float(weight[d]);

    // SiLU on gate and multiply
    float g = __half2float(gate[base + d]);
    float silu_g = g / (1.0f + expf(-g));

    y[base + d] = __float2half(normed * silu_g);
}

// ---------------------------------------------------------------------------
// Fused RMSNormGated + SiLU + NVFP4 activation quantize — the batched-decode
// producer fusion for the ssm_out projection input (same mechanism as
// rmsnorm_nvfp4 / swiglu_quantize_nvfp4, see quant/nvfp4_pack.cuh). Norm and
// gate arithmetic identical to the kernel above (y bytes unchanged); each
// 16-lane segment additionally emits its micro-block's packed nibbles + FP8
// micro-scale from the ROUNDED fp16 values — bit-identical to
// quantize_fp16_to_nvfp4_into on the stored y. head_dim % 32 == 0 keeps the
// segments warp-aligned; the caller guarantees it.
// ---------------------------------------------------------------------------
__global__ void gdn_rmsnorm_gated_silu_nvfp4_kernel(
    half* __restrict__ y,             // [n_tokens, n_heads * head_dim] in/out
    const half* __restrict__ gate,    // [n_tokens, n_heads * head_dim]
    const half* __restrict__ weight,  // [head_dim] shared norm weight
    uint8_t* __restrict__ xq_packed,  // [n_tokens, inner/2]
    uint8_t* __restrict__ xq_scales,  // [n_tokens, inner/16]
    float eps, int n_heads, int head_dim) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    if (d >= head_dim)
        return;

    const int inner = n_heads * head_dim;
    const int base = t * inner + h * head_dim;

    // Load y value
    float val = __half2float(y[base + d]);

    // Parallel sum-of-squares for RMSNorm
    extern __shared__ float s_buf[];
    s_buf[d] = val * val;
    __syncthreads();
    for (int stride = head_dim / 2; stride > 0; stride >>= 1) {
        if (d < stride)
            s_buf[d] += s_buf[d + stride];
        __syncthreads();
    }
    float inv_rms = rsqrtf(s_buf[0] / static_cast<float>(head_dim) + eps);

    // RMSNorm: normalize and scale by weight
    float normed = val * inv_rms * __half2float(weight[d]);

    // SiLU on gate and multiply
    float g = __half2float(gate[base + d]);
    float silu_g = g / (1.0f + expf(-g));

    const half r = __float2half(normed * silu_g);
    y[base + d] = r;

    // Quantize epilogue: 16 consecutive lanes = one micro-block.
    const float rv = __half2float(r);
    float amax = fabsf(rv);
#pragma unroll
    for (int off = 8; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
    uint8_t s8;
    const float actual = nvfp4_encode_micro_scale(amax, /*tensor_scale=*/1.0f, &s8);
    const int64_t elem = static_cast<int64_t>(t) * inner + h * head_dim + d;
    if ((d & 15) == 0)
        xq_scales[elem >> 4] = s8;
    const float inv = 1.0f / actual;
    const float partner = __shfl_xor_sync(0xFFFFFFFFu, rv, 1);
    if ((d & 1) == 0)
        xq_packed[elem >> 1] = nvfp4_pack_pair_hw(rv * inv, partner * inv);
}

// FP32-input variant: reads y as FP32, writes FP16. Used together with
// `gdn_scan_fused_fp32out` so the RMS reduction sees full-precision scan output
// (without FP16 subnormal truncation at ~6e-5).
__global__ void gdn_rmsnorm_gated_silu_fp32in_kernel(
    half* __restrict__ y_fp16_out,        // [n_tokens, n_heads * head_dim]
    const float* __restrict__ y_fp32_in,  // [n_tokens, n_heads * head_dim]
    const half* __restrict__ gate, const half* __restrict__ weight, float eps, int n_heads, int head_dim) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    if (d >= head_dim)
        return;

    const int inner = n_heads * head_dim;
    const int base = t * inner + h * head_dim;

    float val = y_fp32_in[base + d];

    extern __shared__ float s_buf[];
    s_buf[d] = val * val;
    __syncthreads();
    for (int stride = head_dim / 2; stride > 0; stride >>= 1) {
        if (d < stride)
            s_buf[d] += s_buf[d + stride];
        __syncthreads();
    }
    float inv_rms = rsqrtf(s_buf[0] / static_cast<float>(head_dim) + eps);

    float normed = val * inv_rms * __half2float(weight[d]);

    float g = __half2float(gate[base + d]);
    float silu_g = g / (1.0f + expf(-g));

    y_fp16_out[base + d] = __float2half(normed * silu_g);
}

void gdn_rmsnorm_gated_silu_fp32in(half* y_fp16_out, const float* y_fp32_in, const half* gate,
                                   const half* weight, float eps, int n_tokens, int n_heads, int head_dim,
                                   cudaStream_t stream) {
    size_t smem = head_dim * sizeof(float);
    dim3 grid(n_tokens, n_heads);
    gdn_rmsnorm_gated_silu_fp32in_kernel<<<grid, head_dim, smem, stream>>>(y_fp16_out, y_fp32_in, gate,
                                                                           weight, eps, n_heads, head_dim);
    IMP_CUDA_CHECK_LAUNCH();
}

// FP32-in, FP32-out: keeps full precision through gated norm so ssm_out GEMM
// sees FP32 input (fixes 6% accumulation drift in FP16-input matmul).
__global__ void gdn_rmsnorm_gated_silu_fp32inout_kernel(float* __restrict__ y_fp32_out,
                                                        const float* __restrict__ y_fp32_in,
                                                        const half* __restrict__ gate,
                                                        const half* __restrict__ weight, float eps,
                                                        int n_heads, int head_dim) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    if (d >= head_dim)
        return;

    const int inner = n_heads * head_dim;
    const int base = t * inner + h * head_dim;

    float val = y_fp32_in[base + d];

    extern __shared__ float s_buf[];
    s_buf[d] = val * val;
    __syncthreads();
    for (int stride = head_dim / 2; stride > 0; stride >>= 1) {
        if (d < stride)
            s_buf[d] += s_buf[d + stride];
        __syncthreads();
    }
    float inv_rms = rsqrtf(s_buf[0] / static_cast<float>(head_dim) + eps);

    float normed = val * inv_rms * __half2float(weight[d]);

    float g = __half2float(gate[base + d]);
    float silu_g = g / (1.0f + expf(-g));

    y_fp32_out[base + d] = normed * silu_g;
}

void gdn_rmsnorm_gated_silu_fp32inout(float* y_fp32_out, const float* y_fp32_in, const half* gate,
                                      const half* weight, float eps, int n_tokens, int n_heads, int head_dim,
                                      cudaStream_t stream) {
    size_t smem = head_dim * sizeof(float);
    dim3 grid(n_tokens, n_heads);
    gdn_rmsnorm_gated_silu_fp32inout_kernel<<<grid, head_dim, smem, stream>>>(y_fp32_out, y_fp32_in, gate,
                                                                              weight, eps, n_heads, head_dim);
    IMP_CUDA_CHECK_LAUNCH();
}

// Fused RMSNormGated + SiLU
void gdn_rmsnorm_gated_silu(half* y, const half* gate, const half* weight, float eps, int n_tokens,
                            int n_heads, int head_dim, cudaStream_t stream) {
    size_t smem = head_dim * sizeof(float);
    dim3 grid(n_tokens, n_heads);
    gdn_rmsnorm_gated_silu_kernel<<<grid, head_dim, smem, stream>>>(y, gate, weight, eps, n_heads, head_dim);
    IMP_CUDA_CHECK_LAUNCH();
}

bool gdn_rmsnorm_gated_silu_nvfp4(half* y, const half* gate, const half* weight, uint8_t* xq_packed,
                                  uint8_t* xq_scales, float eps, int n_tokens, int n_heads,
                                  int head_dim, cudaStream_t stream) {
    // Warp-aligned 16-lane segments need head_dim % 32 == 0; the smem tree
    // (verbatim from the unfused kernel) needs a power of two.
    if (head_dim < 32 || head_dim > 1024 || (head_dim & 31) != 0 ||
        (head_dim & (head_dim - 1)) != 0)
        return false;
    size_t smem = head_dim * sizeof(float);
    dim3 grid(n_tokens, n_heads);
    gdn_rmsnorm_gated_silu_nvfp4_kernel<<<grid, head_dim, smem, stream>>>(
        y, gate, weight, xq_packed, xq_scales, eps, n_heads, head_dim);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

}  // namespace imp
