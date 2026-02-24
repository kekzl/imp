#include "compute/ssm.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

namespace imp {

// ---------------------------------------------------------------------------
// SiLU in-place kernel
// ---------------------------------------------------------------------------
__global__ void silu_inplace_fp16_kernel(half* __restrict__ x, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __half2float(x[idx]);
        val = val / (1.0f + expf(-val));
        x[idx] = __float2half(val);
    }
}

void silu_inplace(Tensor& x, cudaStream_t stream) {
    int64_t n = x.numel();
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    if (x.dtype == DType::FP16) {
        silu_inplace_fp16_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<half*>(x.data), n);
    }
}

// ---------------------------------------------------------------------------
// Squared ReLU: out[i] = max(0, x[i])^2
// Used by Nemotron-H expert FFN (non-gated) instead of SiLU.
// ---------------------------------------------------------------------------
__global__ void relu_sqr_inplace_fp16_kernel(half* __restrict__ x, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __half2float(x[idx]);
        val = fmaxf(val, 0.0f);
        val = val * val;
        x[idx] = __float2half(val);
    }
}

void relu_sqr_inplace(Tensor& x, cudaStream_t stream) {
    int64_t n = x.numel();
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    if (x.dtype == DType::FP16) {
        relu_sqr_inplace_fp16_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<half*>(x.data), n);
    }
}

// ---------------------------------------------------------------------------
// Element-wise multiply kernel
// ---------------------------------------------------------------------------
__global__ void elementwise_mul_fp16_kernel(const half* __restrict__ a,
                                             const half* __restrict__ b,
                                             half* __restrict__ out,
                                             int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(__half2float(a[idx]) * __half2float(b[idx]));
    }
}

void elementwise_mul(const Tensor& a, const Tensor& b, Tensor& out,
                     cudaStream_t stream) {
    int64_t n = a.numel();
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    if (a.dtype == DType::FP16) {
        elementwise_mul_fp16_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const half*>(a.data),
            static_cast<const half*>(b.data),
            static_cast<half*>(out.data), n);
    }
}

// ---------------------------------------------------------------------------
// Conv1d decode: single token
// ---------------------------------------------------------------------------

// Each thread handles one channel.
// Shift the conv_state window left by 1, insert new value, compute dot product.
__global__ void ssm_conv1d_decode_kernel(
    float* __restrict__ conv_state,       // [channels, kernel_size]
    const half* __restrict__ x_in,        // [channels]
    const half* __restrict__ weight,      // [channels, kernel_size] from GGUF (ne[0]=K, ne[1]=C)
    const half* __restrict__ bias,        // [channels] or nullptr
    half* __restrict__ x_out,             // [channels]
    int channels, int kernel_size)
{
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    float* state = conv_state + ch * kernel_size;

    // Shift state left by 1
    for (int k = 0; k < kernel_size - 1; k++) {
        state[k] = state[k + 1];
    }
    // Insert new value
    state[kernel_size - 1] = __half2float(x_in[ch]);

    // Compute conv: sum(state[k] * weight[ch, k]) + bias
    // Weight layout: [channels, kernel_size] — kernel_size is contiguous per channel
    float sum = 0.0f;
    for (int k = 0; k < kernel_size; k++) {
        sum += state[k] * __half2float(weight[ch * kernel_size + k]);
    }
    if (bias) {
        sum += __half2float(bias[ch]);
    }

    x_out[ch] = __float2half(sum);
}

void ssm_conv1d_decode(void* conv_state, const Tensor& x_in,
                       const Tensor& weight, const Tensor& bias,
                       Tensor& x_out, int conv_kernel,
                       cudaStream_t stream) {
    int channels = static_cast<int>(x_in.shape[x_in.ndim - 1]);
    int threads = 256;
    int blocks = (channels + threads - 1) / threads;

    ssm_conv1d_decode_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<float*>(conv_state),
        static_cast<const half*>(x_in.data),
        static_cast<const half*>(weight.data),
        bias.data ? static_cast<const half*>(bias.data) : nullptr,
        static_cast<half*>(x_out.data),
        channels, conv_kernel);
}

// ---------------------------------------------------------------------------
// Conv1d prefill: full sequence causal convolution
// ---------------------------------------------------------------------------

// Grid: (n_tokens), Block: 256
// Each block handles all channels for one token using a loop.
__global__ void ssm_conv1d_prefill_kernel(
    float* __restrict__ conv_state,       // [channels, kernel_size] — updated with last K values
    const half* __restrict__ x_in,        // [n_tokens, channels]
    const half* __restrict__ weight,      // [channels, kernel_size] from GGUF (ne[0]=K, ne[1]=C)
    const half* __restrict__ bias,        // [channels] or nullptr
    half* __restrict__ x_out,             // [n_tokens, channels]
    int n_tokens, int channels, int kernel_size)
{
    int token = blockIdx.x;
    if (token >= n_tokens) return;

    for (int ch = threadIdx.x; ch < channels; ch += blockDim.x) {
        float sum = 0.0f;

        for (int k = 0; k < kernel_size; k++) {
            int src_t = token - (kernel_size - 1) + k;
            float val = 0.0f;
            if (src_t >= 0) {
                val = __half2float(x_in[src_t * channels + ch]);
            }
            // Weight layout: [channels, kernel_size] — kernel_size is contiguous per channel
            sum += val * __half2float(weight[ch * kernel_size + k]);
        }
        if (bias) {
            sum += __half2float(bias[ch]);
        }
        x_out[token * channels + ch] = __float2half(sum);

        // For the last token, write conv_state
        if (token == n_tokens - 1) {
            float* state = conv_state + ch * kernel_size;
            for (int k = 0; k < kernel_size; k++) {
                int src_t = n_tokens - kernel_size + k;
                state[k] = (src_t >= 0) ? __half2float(x_in[src_t * channels + ch]) : 0.0f;
            }
        }
    }
}

void ssm_conv1d_prefill(void* conv_state, const Tensor& x_in,
                        const Tensor& weight, const Tensor& bias,
                        Tensor& x_out, int conv_kernel,
                        cudaStream_t stream) {
    int n_tokens = static_cast<int>(x_in.shape[0]);
    int channels = static_cast<int>(x_in.shape[1]);

    ssm_conv1d_prefill_kernel<<<n_tokens, 256, 0, stream>>>(
        static_cast<float*>(conv_state),
        static_cast<const half*>(x_in.data),
        static_cast<const half*>(weight.data),
        bias.data ? static_cast<const half*>(bias.data) : nullptr,
        static_cast<half*>(x_out.data),
        n_tokens, channels, conv_kernel);
}

// ---------------------------------------------------------------------------
// Mamba2 SSM scan decode (single step)
// ---------------------------------------------------------------------------

// One block per head. Threads iterate over head_dim_ssm.
// For each (head, dim): reduce over state_size using shared memory.
//
// Mamba2 scan equations (discrete-time):
//   dt_h = softplus(dt_raw[h] + dt_bias[h])
//   a_bar = exp(dt_h * A_log[h])     // A_log is log(A), so exp(dt*A_log) = A^dt
//   For each d in [0, head_dim_ssm):
//     For each s in [0, state_size):
//       h_state[h,d,s] = a_bar * h_state[h,d,s] + dt_h * x[h*hd+d] * B[g*S+s]
//     y[h*hd+d] = sum_s(h_state[h,d,s] * C[g*S+s]) + D[h] * x[h*hd+d]
//
// Template parameter H_FP16: when true, h_state is stored as FP16 (loaded/stored
// via __half2float/__float2half) but all computation remains in FP32.
template <bool H_FP16>
__global__ void ssm_scan_decode_kernel(
    const half* __restrict__ x,          // [inner_size]
    const half* __restrict__ B_in,       // [n_groups * state_size]
    const half* __restrict__ C_in,       // [n_groups * state_size]
    const half* __restrict__ dt_raw,     // [n_heads]
    const float* __restrict__ A_log,     // [n_heads]
    const float* __restrict__ D_skip,    // [n_heads]
    const float* __restrict__ dt_bias,   // [n_heads]
    void* __restrict__ h_state,          // [n_heads, head_dim_ssm, state_size]
    half* __restrict__ y,                // [inner_size]
    int n_heads, int head_dim_ssm, int state_size, int n_groups)
{
    int h = blockIdx.x;
    if (h >= n_heads) return;

    int heads_per_group = n_heads / n_groups;
    int g = h / heads_per_group;

    // Compute dt
    float dt_val = __half2float(dt_raw[h]) + dt_bias[h];
    // softplus: log(1 + exp(x))
    dt_val = (dt_val > 20.0f) ? dt_val : logf(1.0f + expf(dt_val));

    float a_bar = expf(dt_val * A_log[h]);
    float d_val = D_skip[h];

    int64_t h_offset = static_cast<int64_t>(h) * head_dim_ssm * state_size;
    const half* B_g = B_in + g * state_size;
    const half* C_g = C_in + g * state_size;

    for (int d = threadIdx.x; d < head_dim_ssm; d += blockDim.x) {
        float x_val = __half2float(x[h * head_dim_ssm + d]);
        int64_t state_off = h_offset + d * state_size;

        float y_acc = 0.0f;
        for (int s = 0; s < state_size; s++) {
            float b_s = __half2float(B_g[s]);
            float c_s = __half2float(C_g[s]);
            float h_old;
            if constexpr (H_FP16) {
                h_old = __half2float(static_cast<half*>(h_state)[state_off + s]);
            } else {
                h_old = static_cast<float*>(h_state)[state_off + s];
            }
            float h_new = a_bar * h_old + dt_val * x_val * b_s;
            if constexpr (H_FP16) {
                static_cast<half*>(h_state)[state_off + s] = __float2half(h_new);
            } else {
                static_cast<float*>(h_state)[state_off + s] = h_new;
            }
            y_acc += h_new * c_s;
        }

        y[h * head_dim_ssm + d] = __float2half(y_acc + d_val * x_val);
    }
}

void ssm_scan_decode(const Tensor& x, const Tensor& B, const Tensor& C,
                     const Tensor& dt, const Tensor& A_log, const Tensor& D,
                     const Tensor& dt_bias, void* h_state,
                     Tensor& y, int n_heads, int head_dim_ssm,
                     int state_size, int n_groups,
                     DType h_dtype,
                     cudaStream_t stream) {
    int threads = std::min(head_dim_ssm, 256);
    if (h_dtype == DType::FP16) {
        ssm_scan_decode_kernel<true><<<n_heads, threads, 0, stream>>>(
            static_cast<const half*>(x.data),
            static_cast<const half*>(B.data),
            static_cast<const half*>(C.data),
            static_cast<const half*>(dt.data),
            static_cast<const float*>(A_log.data),
            static_cast<const float*>(D.data),
            static_cast<const float*>(dt_bias.data),
            h_state,
            static_cast<half*>(y.data),
            n_heads, head_dim_ssm, state_size, n_groups);
    } else {
        ssm_scan_decode_kernel<false><<<n_heads, threads, 0, stream>>>(
            static_cast<const half*>(x.data),
            static_cast<const half*>(B.data),
            static_cast<const half*>(C.data),
            static_cast<const half*>(dt.data),
            static_cast<const float*>(A_log.data),
            static_cast<const float*>(D.data),
            static_cast<const float*>(dt_bias.data),
            h_state,
            static_cast<half*>(y.data),
            n_heads, head_dim_ssm, state_size, n_groups);
    }
}

void ssm_scan_prefill(const Tensor& x, const Tensor& B, const Tensor& C,
                      const Tensor& dt, const Tensor& A_log, const Tensor& D,
                      const Tensor& dt_bias, void* h_state,
                      Tensor& y, int n_tokens, int n_heads, int head_dim_ssm,
                      int state_size, int n_groups,
                      DType h_dtype,
                      cudaStream_t stream) {
    int inner_size = n_heads * head_dim_ssm;
    int BC_size = n_groups * state_size;

    for (int t = 0; t < n_tokens; t++) {
        // Create per-token views
        const char* x_base = static_cast<const char*>(x.data) + t * inner_size * sizeof(half);
        const char* B_base = static_cast<const char*>(B.data) + t * BC_size * sizeof(half);
        const char* C_base = static_cast<const char*>(C.data) + t * BC_size * sizeof(half);
        const char* dt_base = static_cast<const char*>(dt.data) + t * n_heads * sizeof(half);
        char* y_base = static_cast<char*>(y.data) + t * inner_size * sizeof(half);

        int64_t x_shape[1] = {inner_size};
        int64_t BC_shape[1] = {BC_size};
        int64_t dt_shape[1] = {n_heads};

        Tensor x_t(const_cast<void*>(static_cast<const void*>(x_base)), DType::FP16, 1, x_shape, true);
        Tensor B_t(const_cast<void*>(static_cast<const void*>(B_base)), DType::FP16, 1, BC_shape, true);
        Tensor C_t(const_cast<void*>(static_cast<const void*>(C_base)), DType::FP16, 1, BC_shape, true);
        Tensor dt_t(const_cast<void*>(static_cast<const void*>(dt_base)), DType::FP16, 1, dt_shape, true);
        Tensor y_t(y_base, DType::FP16, 1, x_shape, true);

        ssm_scan_decode(x_t, B_t, C_t, dt_t, A_log, D, dt_bias, h_state,
                        y_t, n_heads, head_dim_ssm, state_size, n_groups, h_dtype, stream);
    }
}

// ---------------------------------------------------------------------------
// Group RMSNorm
// ---------------------------------------------------------------------------

// One block per (token, group). Threads reduce over group_size.
__global__ void group_rmsnorm_fp16_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ out,
    int total_dim, int n_groups, float eps)
{
    int token = blockIdx.x;
    int group = blockIdx.y;
    int group_size = total_dim / n_groups;
    int offset = token * total_dim + group * group_size;

    // Compute sum of squares
    extern __shared__ float sdata[];
    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float val = __half2float(x[offset + i]);
        local_ss += val * val;
    }

    sdata[threadIdx.x] = local_ss;
    __syncthreads();

    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    float rms = rsqrtf(sdata[0] / static_cast<float>(group_size) + eps);

    // Apply norm + weight
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float val = __half2float(x[offset + i]);
        float w = __half2float(weight[group * group_size + i]);
        out[offset + i] = __float2half(val * rms * w);
    }
}

void group_rmsnorm(const Tensor& x, const Tensor& weight, Tensor& out,
                   int n_groups, float eps, cudaStream_t stream) {
    int n_tokens = static_cast<int>(x.shape[0]);
    int total_dim = static_cast<int>(x.shape[1]);
    int group_size = total_dim / n_groups;

    int threads = std::min(group_size, 256);
    // Power of 2 for reduction
    threads = 1;
    while (threads * 2 <= std::min(group_size, 256)) threads *= 2;

    dim3 grid(n_tokens, n_groups);
    size_t smem = threads * sizeof(float);

    group_rmsnorm_fp16_kernel<<<grid, threads, smem, stream>>>(
        static_cast<const half*>(x.data),
        static_cast<const half*>(weight.data),
        static_cast<half*>(out.data),
        total_dim, n_groups, eps);
}

} // namespace imp
