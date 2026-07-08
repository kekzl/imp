#include "exec/executor_kernels.h"
#include "runtime/pdl.h"
#include "compute/warp_reduce.cuh"  // kWarpSize

#include <cuda_bf16.h>
#include <algorithm>

namespace imp {

// ---------------------------------------------------------------------------
// Small CUDA kernels used by the executor
// ---------------------------------------------------------------------------

// Broadcast bias addition: out[row, col] += bias[col] for rows x cols elements
__global__ __launch_bounds__(256) void broadcast_add_bias_fp16_kernel(half* __restrict__ out,
                                                                      const half* __restrict__ bias, int rows,
                                                                      int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int col = i % cols;
        out[i] = __hadd(out[i], bias[col]);
    }
}

// Element-wise scale: out[i] *= scale, for FP16 data (Gemma embedding scaling)
__global__ __launch_bounds__(256) void scale_fp16_kernel(half* __restrict__ data, half scale, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t n2 = n / 2;
    half2 s2 = __half2half2(scale);
    if (idx < n2) {
        half2* d2 = reinterpret_cast<half2*>(data);
        d2[idx] = __hmul2(d2[idx], s2);
    }
    // Handle odd element
    if (idx == n2 && (n & 1)) {
        data[n - 1] = __hmul(data[n - 1], scale);
    }
}

// Element-wise addition: a[i] += b[i], for FP16 data
__global__ __launch_bounds__(256) void elementwise_add_fp16_kernel(half* __restrict__ a,
                                                                   const half* __restrict__ b, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t n2 = n / 2;
    if (idx < n2) {
        half2* a2 = reinterpret_cast<half2*>(a);
        const half2* b2 = reinterpret_cast<const half2*>(b);
        a2[idx] = __hadd2(a2[idx], b2[idx]);
    }
    if (idx == 0 && (n & 1)) {
        a[n - 1] = __hadd(a[n - 1], b[n - 1]);
    }
}

// Element-wise add-store: out[i] = a[i] + b[i], for FP16 data
__global__ __launch_bounds__(256) void elementwise_add_store_fp16_kernel(const half* __restrict__ a,
                                                                         const half* __restrict__ b,
                                                                         half* __restrict__ out, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t n2 = n / 2;
    if (idx < n2) {
        const half2* a2 = reinterpret_cast<const half2*>(a);
        const half2* b2 = reinterpret_cast<const half2*>(b);
        half2* o2 = reinterpret_cast<half2*>(out);
        o2[idx] = __hadd2(a2[idx], b2[idx]);
    }
    if (idx == 0 && (n & 1)) {
        out[n - 1] = __hadd(a[n - 1], b[n - 1]);
    }
}

// FP32 accumulator += FP16 branch: accum[i] += __half2float(branch[i])
__global__ __launch_bounds__(256) void fp32_accum_add_fp16_kernel(float* __restrict__ accum,
                                                                  const half* __restrict__ branch,
                                                                  int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        accum[idx] += __half2float(branch[idx]);
    }
}

// Convert FP32 → FP16 with per-row dynamic scaling.
// Each row is independently scaled so max_abs maps to ≤65000, preserving
// the ratio between elements.  Since subsequent operations (RMSNorm) are
// scale-invariant per row, this produces correct normalized output even
// when the FP32 residual stream far exceeds FP16 range.
// Launch: <<<n_rows, 256, 256 * sizeof(float)>>>
__global__ __launch_bounds__(256) void fp32_to_fp16_rowscale_kernel(const float* __restrict__ in,
                                                                    half* __restrict__ out, int rows,
                                                                    int cols) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows)
        return;

    const float* row_in = in + static_cast<int64_t>(row) * cols;
    half* row_out = out + static_cast<int64_t>(row) * cols;

    // Phase 1: parallel reduction to find max |value| in this row
    float local_max = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        local_max = fmaxf(local_max, fabsf(row_in[c]));

    smem[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = smem[0];

    // Only scale if values actually exceed safe FP16 range
    float inv_scale = (row_max > 65000.0f) ? (65000.0f / row_max) : 1.0f;

    // Phase 2: scale and convert to FP16
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        row_out[c] = __float2half(row_in[c] * inv_scale);
}

// Fused RMSNorm + FP32 accumulator add + FP32→FP16 row-scale conversion.
// Replaces 3 separate kernels in the post-norm FP32 accumulator path:
//   rmsnorm(input, weight, tmp) → fp32_accum_add(accum, tmp) → fp32_to_fp16_rowscale(accum, out)
// Saves 2 kernel launches + 2 DRAM round-trips per invocation.
// Uses same register-cached, warp-level reduction pattern as rmsnorm_quantize_q8_1.
// Launch: <<<n_rows, 256>>>
__global__ __launch_bounds__(512) void rmsnorm_fp32_accum_to_fp16_kernel(
    const half* __restrict__ input,   // [n, d_model] pre-norm data (e.g. GEMV output)
    const half* __restrict__ norm_w,  // [d_model] RMSNorm weights
    float* __restrict__ fp32_accum,   // [n, d_model] FP32 accumulator (read-modify-write)
    half* __restrict__ output,        // [n, d_model] FP16 output for next layer
    int d_model, float eps, float weight_offset) {
    __shared__ float warp_reduce[kWarpSize];  // support up to 1024 threads (32 warps)
    __shared__ float s_inv_rms;
    __shared__ float s_row_max;

    const int tid = threadIdx.x;
    const int lane = tid % kWarpSize;
    const int warp_id = tid / kWarpSize;
    const int n_warps = blockDim.x / kWarpSize;
    const int row = blockIdx.x;

    // Vectorized: process 8 halfs (1 float4 = 2 half2) per iteration.
    const int d_model_v = d_model / 8;  // number of float4-sized chunks

    const float4* x_row4 = reinterpret_cast<const float4*>(input + static_cast<int64_t>(row) * d_model);
    const float4* nw_row4 = reinterpret_cast<const float4*>(norm_w);
    float4* accum_row4 = reinterpret_cast<float4*>(fp32_accum + static_cast<int64_t>(row) * d_model);
    float4* out_row4 = reinterpret_cast<float4*>(output + static_cast<int64_t>(row) * d_model);

    // Phase 1: Load input (half→float via float4 loads), compute sum of squares.
    // Each thread handles d_model_v / blockDim.x chunks, each chunk = 8 halfs.
    float sum_sq = 0.0f;
    for (int i = tid; i < d_model_v; i += blockDim.x) {
        float4 h4 = x_row4[i];  // 8 halfs packed as float4
        const half2* h2 = reinterpret_cast<const half2*>(&h4);
        float2 f0 = __half22float2(h2[0]);
        float2 f1 = __half22float2(h2[1]);
        float2 f2 = __half22float2(h2[2]);
        float2 f3 = __half22float2(h2[3]);
        sum_sq += f0.x * f0.x + f0.y * f0.y + f1.x * f1.x + f1.y * f1.y + f2.x * f2.x + f2.y * f2.y +
                  f3.x * f3.x + f3.y * f3.y;
    }

// Block reduce sum_sq
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, off);
    if (lane == 0)
        warp_reduce[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float total = (lane < n_warps) ? warp_reduce[lane] : 0.0f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            total += __shfl_xor_sync(0xFFFFFFFF, total, off);
        if (lane == 0)
            s_inv_rms = rsqrtf(total / static_cast<float>(d_model) + eps);
    }
    __syncthreads();
    float inv_rms = s_inv_rms;

    // Phase 2: Normalize, add to FP32 accumulator, find max_abs.
    // Vectorized: read float4 from accum (4 floats), half2×4 from input/norm_w.
    float local_max = 0.0f;
    for (int i = tid; i < d_model_v; i += blockDim.x) {
        // Re-read input (small enough to stay in L1/L2)
        float4 h4 = x_row4[i];
        const half2* h2 = reinterpret_cast<const half2*>(&h4);
        float4 nw4 = nw_row4[i];
        const half2* nw2 = reinterpret_cast<const half2*>(&nw4);

        // Read FP32 accumulator (2 float4s = 8 floats)
        float4 acc_lo = accum_row4[i * 2];
        float4 acc_hi = accum_row4[i * 2 + 1];
        float* acc_f = reinterpret_cast<float*>(&acc_lo);
        float* acc_f_hi = reinterpret_cast<float*>(&acc_hi);

        float2 f0 = __half22float2(h2[0]);
        float2 f1 = __half22float2(h2[1]);
        float2 f2 = __half22float2(h2[2]);
        float2 f3 = __half22float2(h2[3]);
        float2 w0 = __half22float2(nw2[0]);
        float2 w1 = __half22float2(nw2[1]);
        float2 w2 = __half22float2(nw2[2]);
        float2 w3 = __half22float2(nw2[3]);

        acc_f[0] += f0.x * inv_rms * (w0.x + weight_offset);
        acc_f[1] += f0.y * inv_rms * (w0.y + weight_offset);
        acc_f[2] += f1.x * inv_rms * (w1.x + weight_offset);
        acc_f[3] += f1.y * inv_rms * (w1.y + weight_offset);
        acc_f_hi[0] += f2.x * inv_rms * (w2.x + weight_offset);
        acc_f_hi[1] += f2.y * inv_rms * (w2.y + weight_offset);
        acc_f_hi[2] += f3.x * inv_rms * (w3.x + weight_offset);
        acc_f_hi[3] += f3.y * inv_rms * (w3.y + weight_offset);

        accum_row4[i * 2] = acc_lo;
        accum_row4[i * 2 + 1] = acc_hi;

        local_max = fmaxf(local_max, fmaxf(fmaxf(fabsf(acc_f[0]), fabsf(acc_f[1])),
                                           fmaxf(fabsf(acc_f[2]), fabsf(acc_f[3]))));
        local_max = fmaxf(local_max, fmaxf(fmaxf(fabsf(acc_f_hi[0]), fabsf(acc_f_hi[1])),
                                           fmaxf(fabsf(acc_f_hi[2]), fabsf(acc_f_hi[3]))));
    }

// Block reduce max_abs
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, off));
    if (lane == 0)
        warp_reduce[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        float m = (lane < n_warps) ? warp_reduce[lane] : 0.0f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, off));
        if (lane == 0)
            s_row_max = m;
    }
    __syncthreads();
    float inv_scale = (s_row_max > 65000.0f) ? (65000.0f / s_row_max) : 1.0f;

    // Phase 3: Scale FP32 accum → FP16 output (vectorized float4 reads, half2×4 writes).
    for (int i = tid; i < d_model_v; i += blockDim.x) {
        float4 acc_lo = accum_row4[i * 2];
        float4 acc_hi = accum_row4[i * 2 + 1];
        float* af = reinterpret_cast<float*>(&acc_lo);
        float* af_hi = reinterpret_cast<float*>(&acc_hi);

        float4 out4;
        half2* oh2 = reinterpret_cast<half2*>(&out4);
        oh2[0] = __floats2half2_rn(af[0] * inv_scale, af[1] * inv_scale);
        oh2[1] = __floats2half2_rn(af[2] * inv_scale, af[3] * inv_scale);
        oh2[2] = __floats2half2_rn(af_hi[0] * inv_scale, af_hi[1] * inv_scale);
        oh2[3] = __floats2half2_rn(af_hi[2] * inv_scale, af_hi[3] * inv_scale);
        out_row4[i] = out4;
    }
}

// FP32-input variant of rmsnorm_fp32_accum_to_fp16_kernel.
// Input is FP32 (e.g. attention output projection kept in FP32 to preserve
// cuBLAS internal accumulator precision). Same accum + overflow protection as
// the FP16-input variant. Used by the overrides.gemma4.fp32_gemm_out config flag
// (was IMP_GEMMA4_FP32_GEMM_OUT env) for attention.
__global__ __launch_bounds__(512) void rmsnorm_fp32in_fp32_accum_to_fp16_kernel(
    const float* __restrict__ input,  // [n, d_model] FP32 pre-norm data
    const half* __restrict__ norm_w,  // [d_model] RMSNorm weights
    float* __restrict__ fp32_accum,   // [n, d_model] FP32 accumulator (RMW)
    half* __restrict__ output,        // [n, d_model] FP16 output for next layer
    int d_model, float eps, float weight_offset) {
    __shared__ float warp_reduce[32];
    __shared__ float s_inv_rms;
    __shared__ float s_row_max;

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp_id = tid / 32;
    const int n_warps = blockDim.x / 32;
    const int row = blockIdx.x;

    const float* x_row = input + static_cast<int64_t>(row) * d_model;
    const half* nw = norm_w;
    float* accum_row = fp32_accum + static_cast<int64_t>(row) * d_model;
    half* out_row = output + static_cast<int64_t>(row) * d_model;

    // Phase 1: sum of squares (input already FP32)
    float sum_sq = 0.0f;
    for (int i = tid; i < d_model; i += blockDim.x) {
        float v = x_row[i];
        sum_sq += v * v;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, off);
    if (lane == 0)
        warp_reduce[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        float total = (lane < n_warps) ? warp_reduce[lane] : 0.0f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            total += __shfl_xor_sync(0xFFFFFFFF, total, off);
        if (lane == 0)
            s_inv_rms = rsqrtf(total / static_cast<float>(d_model) + eps);
    }
    __syncthreads();
    float inv_rms = s_inv_rms;

    // Phase 2: accum += norm(x) * weight; track max
    float local_max = 0.0f;
    for (int i = tid; i < d_model; i += blockDim.x) {
        float v = x_row[i];
        float w = __half2float(nw[i]) + weight_offset;
        float val = v * inv_rms * w;
        float new_acc = accum_row[i] + val;
        accum_row[i] = new_acc;
        local_max = fmaxf(local_max, fabsf(new_acc));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, off));
    if (lane == 0)
        warp_reduce[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        float m = (lane < n_warps) ? warp_reduce[lane] : 0.0f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, off));
        if (lane == 0)
            s_row_max = m;
    }
    __syncthreads();
    float inv_scale = (s_row_max > 65000.0f) ? (65000.0f / s_row_max) : 1.0f;

    // Phase 3: write FP16 output (with overflow scaling)
    for (int i = tid; i < d_model; i += blockDim.x) {
        out_row[i] = __float2half(accum_row[i] * inv_scale);
    }
}

// Convert FP16 → FP32: out[i] = __half2float(in[i])
__global__ __launch_bounds__(256) void fp16_to_fp32_kernel(const half* __restrict__ in,
                                                           float* __restrict__ out, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __half2float(in[idx]);
    }
}

// Element-wise addition: a[i] += b[i], for FP32 data
__global__ __launch_bounds__(256) void elementwise_add_fp32_kernel(float* __restrict__ a,
                                                                   const float* __restrict__ b, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (int64_t i = idx; i < n; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        a[i] += b[i];
    }
}

// Add FP16 bias to each row of FP32 matrix: out[i,j] += bias[j]
// Grid: n_tokens, Block: 256, each thread handles multiple expert indices.
__global__ __launch_bounds__(256) void add_fp16_bias_to_fp32_kernel(float* __restrict__ data,
                                                                    const half* __restrict__ bias,
                                                                    int n_tokens, int n_cols) {
    int token = blockIdx.x;
    if (token >= n_tokens)
        return;
    float* row = data + static_cast<int64_t>(token) * n_cols;
    for (int j = threadIdx.x; j < n_cols; j += blockDim.x) {
        row[j] += __half2float(bias[j]);
    }
}

// Scale FP32 expert weights in-place: weights[i] *= scale
__global__ __launch_bounds__(256) void scale_fp32_kernel(float* __restrict__ data, float scale, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

// Logit soft-capping: logit = softcap * tanh(logit / softcap)  (Gemma-2/3)
__global__ __launch_bounds__(256) void logit_softcap_fp32_kernel(float* __restrict__ data, float softcap,
                                                                 float inv_softcap, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = softcap * tanhf(data[idx] * inv_softcap);
    }
}

// FP32 -> FP16 conversion kernel (for scatter output back to compute_dtype)
__global__ __launch_bounds__(256) void fp32_to_fp16_kernel(const float* __restrict__ in,
                                                           half* __restrict__ out, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);
    }
}

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------

void elementwise_add(Tensor& a, const Tensor& b, cudaStream_t stream) {
    int64_t n = a.numel();
    if (a.qtype == QType::F16) {
        int64_t n2 = (n + 1) / 2;
        int threads = 256;
        int blocks = static_cast<int>((n2 + threads - 1) / threads);
        pdl::launch(elementwise_add_fp16_kernel, dim3(blocks), dim3(threads), 0, stream,
                    static_cast<half*>(a.data), static_cast<const half*>(b.data), n);
    } else {
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        pdl::launch(elementwise_add_fp32_kernel, dim3(blocks), dim3(threads), 0, stream,
                    static_cast<float*>(a.data), static_cast<const float*>(b.data), n);
    }
}

// Element-wise add-store: out[i] = a[i] + b[i] — avoids in-place + copy pattern
void elementwise_add_store(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t stream) {
    int64_t n = a.numel();
    int64_t n2 = (n + 1) / 2;
    int threads = 256;
    int blocks = static_cast<int>((n2 + threads - 1) / threads);
    pdl::launch(elementwise_add_store_fp16_kernel, dim3(blocks), dim3(threads), 0, stream,
                static_cast<const half*>(a.data), static_cast<const half*>(b.data),
                static_cast<half*>(out.data), n);
}

// Add 1D bias to each row of a 2D output: out[row, col] += bias[col]
void add_bias(Tensor& out, const Tensor& bias, cudaStream_t stream) {
    if (bias.data == nullptr)
        return;
    int rows = static_cast<int>(out.shape[0]);
    int cols = static_cast<int>(bias.shape[0]);
    if (rows == 0 || cols == 0)
        return;
    int total = rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    broadcast_add_bias_fp16_kernel<<<blocks, threads, 0, stream>>>(static_cast<half*>(out.data),
                                                                   static_cast<const half*>(bias.data), rows,
                                                                   cols);
}

// Fused 3-way bias add: applies up to 3 biases in a single kernel launch.
// blockIdx.y selects which output/bias pair (0, 1, or 2).
__global__ __launch_bounds__(256) void add_bias_3way_kernel(
    half* __restrict__ out0, const half* __restrict__ bias0, int cols0, half* __restrict__ out1,
    const half* __restrict__ bias1, int cols1, half* __restrict__ out2, const half* __restrict__ bias2,
    int cols2, int rows) {
    int which = blockIdx.y;
    half* out;
    const half* bias;
    int cols;
    if (which == 0) {
        out = out0;
        bias = bias0;
        cols = cols0;
    } else if (which == 1) {
        out = out1;
        bias = bias1;
        cols = cols1;
    } else {
        out = out2;
        bias = bias2;
        cols = cols2;
    }
    if (!out || !bias)
        return;

    int total = rows * cols;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int col = i % cols;
        out[i] = __hadd(out[i], bias[col]);
    }
}

void add_bias_3way(Tensor& out_a, const Tensor& bias_a, Tensor& out_b, const Tensor& bias_b, Tensor& out_c,
                   const Tensor& bias_c, cudaStream_t stream) {
    // Count how many actually have biases
    int n_active = (bias_a.data ? 1 : 0) + (bias_b.data ? 1 : 0) + (bias_c.data ? 1 : 0);
    if (n_active == 0)
        return;

    // Fall back to individual calls if only 1-2 biases
    if (n_active <= 2) {
        add_bias(out_a, bias_a, stream);
        add_bias(out_b, bias_b, stream);
        add_bias(out_c, bias_c, stream);
        return;
    }

    int rows = static_cast<int>(out_a.shape[0]);
    int cols_a = bias_a.data ? static_cast<int>(bias_a.shape[0]) : 0;
    int cols_b = bias_b.data ? static_cast<int>(bias_b.shape[0]) : 0;
    int cols_c = bias_c.data ? static_cast<int>(bias_c.shape[0]) : 0;

    int max_cols = std::max({cols_a, cols_b, cols_c});
    int total = rows * max_cols;
    int threads = 256;
    int blocks_x = (total + threads - 1) / threads;
    dim3 grid(blocks_x, 3);

    add_bias_3way_kernel<<<grid, threads, 0, stream>>>(
        bias_a.data ? static_cast<half*>(out_a.data) : nullptr,
        bias_a.data ? static_cast<const half*>(bias_a.data) : nullptr, cols_a,
        bias_b.data ? static_cast<half*>(out_b.data) : nullptr,
        bias_b.data ? static_cast<const half*>(bias_b.data) : nullptr, cols_b,
        bias_c.data ? static_cast<half*>(out_c.data) : nullptr,
        bias_c.data ? static_cast<const half*>(bias_c.data) : nullptr, cols_c, rows);
}

// Fused residual add + RMSNorm: hidden += residual; output = rmsnorm(hidden, weight).
// Saves 1 DRAM round-trip: reads hidden+residual+weight, writes hidden+output.
// Launch: <<<n_rows, 256>>>
__global__ __launch_bounds__(256) void residual_add_rmsnorm_kernel(
    half* __restrict__ hidden,          // [n, d_model] — modified in-place (residual added)
    const half* __restrict__ residual,  // [n, d_model]
    const half* __restrict__ weight,    // [d_model] RMSNorm weight
    half* __restrict__ output,          // [n, d_model] normalized output
    int d_model, float eps, float weight_offset) {
    __shared__ float warp_reduce[kWarpSize];
    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    half* h_row = hidden + static_cast<int64_t>(row) * d_model;
    const half* r_row = residual + static_cast<int64_t>(row) * d_model;
    half* o_row = output + static_cast<int64_t>(row) * d_model;

    // Phase 1: Add residual to hidden + compute sum of squares
    float sum_sq = 0.0f;
    for (int d = tid; d < d_model; d += blockDim.x) {
        float h = __half2float(h_row[d]) + __half2float(r_row[d]);
        h_row[d] = __float2half(h);
        sum_sq += h * h;
    }

// Warp-level reduction
#pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, off);

    int warp_id = tid / kWarpSize;
    int lane = tid % kWarpSize;
    if (lane == 0)
        warp_reduce[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        int n_warps = blockDim.x / kWarpSize;
        float total = (lane < n_warps) ? warp_reduce[lane] : 0.0f;
#pragma unroll
        for (int off = kWarpSize / 2; off > 0; off >>= 1)
            total += __shfl_xor_sync(0xFFFFFFFF, total, off);
        if (lane == 0)
            warp_reduce[0] = rsqrtf(total / static_cast<float>(d_model) + eps);
    }
    __syncthreads();
    float inv_rms = warp_reduce[0];

    // Phase 2: Apply normalization
    for (int d = tid; d < d_model; d += blockDim.x) {
        float h = __half2float(h_row[d]);
        float w = __half2float(weight[d]) + weight_offset;
        o_row[d] = __float2half(h * inv_rms * w);
    }
}

void residual_add_rmsnorm(Tensor& hidden, const Tensor& residual, const Tensor& weight, Tensor& output,
                          float eps, cudaStream_t stream, float weight_offset) {
    int n = static_cast<int>(hidden.shape[0]);
    int d_model = static_cast<int>(hidden.shape[hidden.ndim - 1]);
    residual_add_rmsnorm_kernel<<<n, 256, 0, stream>>>(static_cast<half*>(hidden.data),
                                                       static_cast<const half*>(residual.data),
                                                       static_cast<const half*>(weight.data),
                                                       static_cast<half*>(output.data), d_model, eps,
                                                       weight_offset);
}

// Fused add-store + RMSNorm in-place: hidden = rmsnorm(a + b, weight).
// Saves 2 kernel launches + 1 memcpy vs separate add_store + rmsnorm + copy.
// Launch: <<<n_rows, 256>>>
__global__ __launch_bounds__(256) void add_rmsnorm_inplace_kernel(
    const half* __restrict__ a,       // [n, d_model]
    const half* __restrict__ b,       // [n, d_model]
    half* __restrict__ hidden,        // [n, d_model] — output (a + b, then normalized)
    const half* __restrict__ weight,  // [d_model] RMSNorm weight
    int d_model, float eps, float weight_offset) {
    __shared__ float warp_reduce[kWarpSize];
    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    const half* a_row = a + static_cast<int64_t>(row) * d_model;
    const half* b_row = b + static_cast<int64_t>(row) * d_model;
    half* h_row = hidden + static_cast<int64_t>(row) * d_model;

    // Phase 1: Compute a+b and sum of squares in one pass
    float sum_sq = 0.0f;
    for (int d = tid; d < d_model; d += blockDim.x) {
        float h = __half2float(a_row[d]) + __half2float(b_row[d]);
        h_row[d] = __float2half(h);  // store sum for phase 2
        sum_sq += h * h;
    }

// Warp-level reduction
#pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, off);

    int warp_id = tid / kWarpSize;
    int lane = tid % kWarpSize;
    if (lane == 0)
        warp_reduce[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        int n_warps = blockDim.x / kWarpSize;
        float total = (lane < n_warps) ? warp_reduce[lane] : 0.0f;
#pragma unroll
        for (int off = kWarpSize / 2; off > 0; off >>= 1)
            total += __shfl_xor_sync(0xFFFFFFFF, total, off);
        if (lane == 0)
            warp_reduce[0] = rsqrtf(total / static_cast<float>(d_model) + eps);
    }
    __syncthreads();
    float inv_rms = warp_reduce[0];

    // Phase 2: Normalize in-place
    for (int d = tid; d < d_model; d += blockDim.x) {
        float h = __half2float(h_row[d]);
        float w = __half2float(weight[d]) + weight_offset;
        h_row[d] = __float2half(h * inv_rms * w);
    }
}

void add_rmsnorm_inplace(const Tensor& a, const Tensor& b, Tensor& hidden, const Tensor& weight, float eps,
                         cudaStream_t stream, float weight_offset) {
    int n = static_cast<int>(a.shape[0]);
    int d_model = static_cast<int>(a.shape[a.ndim - 1]);
    add_rmsnorm_inplace_kernel<<<n, 256, 0, stream>>>(static_cast<const half*>(a.data),
                                                      static_cast<const half*>(b.data),
                                                      static_cast<half*>(hidden.data),
                                                      static_cast<const half*>(weight.data), d_model, eps,
                                                      weight_offset);
}

// Fused RMSNorm + residual add: output = rmsnorm(input, weight) + residual.
// Launch: <<<n_rows, 256>>>
__global__ __launch_bounds__(256) void rmsnorm_add_residual_kernel(
    const half* __restrict__ input,     // [n, d_model]
    const half* __restrict__ weight,    // [d_model]
    const half* __restrict__ residual,  // [n, d_model]
    half* __restrict__ output,          // [n, d_model]
    int d_model, float eps, float weight_offset) {
    __shared__ float warp_reduce[kWarpSize];
    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    const half* in_row = input + static_cast<int64_t>(row) * d_model;
    const half* r_row = residual + static_cast<int64_t>(row) * d_model;
    half* o_row = output + static_cast<int64_t>(row) * d_model;

    // Phase 1: Compute sum of squares of input
    float sum_sq = 0.0f;
    for (int d = tid; d < d_model; d += blockDim.x) {
        float v = __half2float(in_row[d]);
        sum_sq += v * v;
    }

#pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, off);

    int warp_id = tid / kWarpSize;
    int lane = tid % kWarpSize;
    if (lane == 0)
        warp_reduce[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        int n_warps = blockDim.x / kWarpSize;
        float total = (lane < n_warps) ? warp_reduce[lane] : 0.0f;
#pragma unroll
        for (int off = kWarpSize / 2; off > 0; off >>= 1)
            total += __shfl_xor_sync(0xFFFFFFFF, total, off);
        if (lane == 0)
            warp_reduce[0] = rsqrtf(total / static_cast<float>(d_model) + eps);
    }
    __syncthreads();
    float inv_rms = warp_reduce[0];

    // Phase 2: Normalize + add residual
    for (int d = tid; d < d_model; d += blockDim.x) {
        float v = __half2float(in_row[d]);
        float w = __half2float(weight[d]) + weight_offset;
        float r = __half2float(r_row[d]);
        o_row[d] = __float2half(v * inv_rms * w + r);
    }
}

void rmsnorm_add_residual(const Tensor& input, const Tensor& weight, const Tensor& residual, Tensor& output,
                          float eps, cudaStream_t stream, float weight_offset) {
    int n = static_cast<int>(input.shape[0]);
    int d_model = static_cast<int>(input.shape[input.ndim - 1]);
    rmsnorm_add_residual_kernel<<<n, 256, 0, stream>>>(static_cast<const half*>(input.data),
                                                       static_cast<const half*>(weight.data),
                                                       static_cast<const half*>(residual.data),
                                                       static_cast<half*>(output.data), d_model, eps,
                                                       weight_offset);
}

// Create a view of the first n_tokens rows from a [max_tokens, cols] buffer.
// Never modifies the source tensor.
Tensor slice_rows(const Tensor& buf, int n_tokens) {
    if (n_tokens == static_cast<int>(buf.shape[0]))
        return buf;
    // buf.slice(0, n) returns a view with shape[0] = n, same data pointer.
    return buf.slice(0, n_tokens);
}
}  // namespace imp
