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
    if (x.qtype == QType::F16) {
        silu_inplace_fp16_kernel<<<blocks, threads, 0, stream>>>(static_cast<half*>(x.data), n);
        IMP_CUDA_CHECK_LAUNCH();
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
    if (x.qtype == QType::F16) {
        relu_sqr_inplace_fp16_kernel<<<blocks, threads, 0, stream>>>(static_cast<half*>(x.data), n);
        IMP_CUDA_CHECK_LAUNCH();
    }
}

// ---------------------------------------------------------------------------
// Element-wise multiply kernel
// ---------------------------------------------------------------------------
// Sigmoid multiply: out[i] = a[i] * sigmoid(b[i])
// Used by Qwen3.5 attention output gate.
__global__ void sigmoid_mul_fp16_kernel(const half* __restrict__ a, const half* __restrict__ b,
                                        half* __restrict__ out, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val_a = __half2float(a[idx]);
        float val_b = __half2float(b[idx]);
        float sig = 1.0f / (1.0f + expf(-val_b));
        out[idx] = __float2half(val_a * sig);
    }
}

void sigmoid_mul(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t stream) {
    int64_t n = a.numel();
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    if (a.qtype == QType::F16) {
        sigmoid_mul_fp16_kernel<<<blocks, threads, 0, stream>>>(static_cast<const half*>(a.data),
                                                                static_cast<const half*>(b.data),
                                                                static_cast<half*>(out.data), n);
        IMP_CUDA_CHECK_LAUNCH();
    }
}

// ---------------------------------------------------------------------------
__global__ void elementwise_mul_fp16_kernel(const half* __restrict__ a, const half* __restrict__ b,
                                            half* __restrict__ out, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(__half2float(a[idx]) * __half2float(b[idx]));
    }
}

void elementwise_mul(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t stream) {
    int64_t n = a.numel();
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    if (a.qtype == QType::F16) {
        elementwise_mul_fp16_kernel<<<blocks, threads, 0, stream>>>(static_cast<const half*>(a.data),
                                                                    static_cast<const half*>(b.data),
                                                                    static_cast<half*>(out.data), n);
        IMP_CUDA_CHECK_LAUNCH();
    }
}

// ---------------------------------------------------------------------------
// Conv1d decode: single token
// ---------------------------------------------------------------------------

// Each thread handles one channel.
// Shift the conv_state window left by 1, insert new value, compute dot product.
__global__ void ssm_conv1d_decode_kernel(
    float* __restrict__ conv_state,   // [channels, kernel_size]
    const half* __restrict__ x_in,    // [channels]
    const half* __restrict__ weight,  // [channels, kernel_size] from GGUF (ne[0]=K, ne[1]=C)
    const half* __restrict__ bias,    // [channels] or nullptr
    half* __restrict__ x_out,         // [channels]
    int channels, int kernel_size) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels)
        return;

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

// FP32-output conv1d decode: reads FP16 input, produces FP32 output with fused SiLU.
// Used by GDN layers for full FP32 pipeline (matching llama.cpp precision).
__global__ void ssm_conv1d_decode_f32_silu_kernel(
    float* __restrict__ conv_state,   // [channels, kernel_size]
    const half* __restrict__ x_in,    // [channels] FP16
    const half* __restrict__ weight,  // [channels, kernel_size] FP16
    const half* __restrict__ bias,    // [channels] or nullptr
    float* __restrict__ x_out,        // [channels] FP32
    int channels, int kernel_size,
    // --- batched decode over independent sequences ---
    // seq_slots: per-blockIdx.y conv-state slot, or nullptr for the
    // single-sequence call (gridDim.y == 1, every offset below is zero).
    const int* __restrict__ seq_slots = nullptr, int64_t conv_state_seq_stride = 0) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels)
        return;
    const int seq = blockIdx.y;
    if (seq_slots)
        conv_state += static_cast<size_t>(seq_slots[seq]) * static_cast<size_t>(conv_state_seq_stride);
    x_in += static_cast<size_t>(seq) * channels;
    x_out += static_cast<size_t>(seq) * channels;

    float* state = conv_state + ch * kernel_size;
    for (int k = 0; k < kernel_size - 1; k++)
        state[k] = state[k + 1];
    state[kernel_size - 1] = __half2float(x_in[ch]);

    float sum = 0.0f;
    for (int k = 0; k < kernel_size; k++)
        sum += state[k] * __half2float(weight[ch * kernel_size + k]);
    if (bias)
        sum += __half2float(bias[ch]);

    // Fused SiLU: x / (1 + exp(-x))
    x_out[ch] = sum / (1.0f + expf(-sum));
}

// Batched conv1d decode over N independent sequences. Same contract as the GDN
// scan's batched form: sequences on blockIdx.y, slot table selects each one's
// conv state, x_in/x_out are [n_seq, channels] sequence-major.
void ssm_conv1d_decode_f32_silu_batched(void* conv_state_pool, const int* seq_slots,
                                        int64_t conv_state_seq_stride, const half* x_in,
                                        const Tensor& weight, const Tensor& bias, float* x_out_f32,
                                        int n_seq, int channels, int conv_kernel, cudaStream_t stream) {
    if (n_seq <= 0)
        return;
    int threads = 256;
    dim3 blocks((channels + threads - 1) / threads, n_seq);
    ssm_conv1d_decode_f32_silu_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<float*>(conv_state_pool), x_in, static_cast<const half*>(weight.data),
        bias.data ? static_cast<const half*>(bias.data) : nullptr, x_out_f32, channels, conv_kernel,
        seq_slots, conv_state_seq_stride);
    IMP_CUDA_CHECK_LAUNCH();
}

void ssm_conv1d_decode_f32_silu(void* conv_state, const Tensor& x_in, const Tensor& weight,
                                const Tensor& bias, float* x_out_f32, int conv_kernel, cudaStream_t stream) {
    int channels = static_cast<int>(x_in.shape[x_in.ndim - 1]);
    int threads = 256;
    int blocks = (channels + threads - 1) / threads;
    ssm_conv1d_decode_f32_silu_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<float*>(conv_state), static_cast<const half*>(x_in.data),
        static_cast<const half*>(weight.data), bias.data ? static_cast<const half*>(bias.data) : nullptr,
        x_out_f32, channels, conv_kernel);
    IMP_CUDA_CHECK_LAUNCH();
}

void ssm_conv1d_decode(void* conv_state, const Tensor& x_in, const Tensor& weight, const Tensor& bias,
                       Tensor& x_out, int conv_kernel, cudaStream_t stream) {
    int channels = static_cast<int>(x_in.shape[x_in.ndim - 1]);
    int threads = 256;
    int blocks = (channels + threads - 1) / threads;

    ssm_conv1d_decode_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<float*>(conv_state), static_cast<const half*>(x_in.data),
        static_cast<const half*>(weight.data), bias.data ? static_cast<const half*>(bias.data) : nullptr,
        static_cast<half*>(x_out.data), channels, conv_kernel);
    IMP_CUDA_CHECK_LAUNCH();
}

// ---------------------------------------------------------------------------
// Conv1d prefill: full sequence causal convolution
// ---------------------------------------------------------------------------

// Grid: (n_tokens), Block: 256
// Each block handles all channels for one token using a loop.
__global__ void ssm_conv1d_prefill_kernel(
    float* __restrict__ conv_state,   // [channels, kernel_size] — updated with last K values
    const half* __restrict__ x_in,    // [n_tokens, channels]
    const half* __restrict__ weight,  // [channels, kernel_size] from GGUF (ne[0]=K, ne[1]=C)
    const half* __restrict__ bias,    // [channels] or nullptr
    half* __restrict__ x_out,         // [n_tokens, channels]
    int n_tokens, int channels, int kernel_size,
    const int* __restrict__ d_real_n,       // device chunk length (padded verify chunk) or nullptr
    float* __restrict__ conv_snap,          // second conv-state commit, or nullptr
    const int* __restrict__ d_snap_n,       // row count conv_snap is taken at
    const float* __restrict__ conv_prev) {  // pre-chunk conv_state copy
    int token = blockIdx.x;
    if (token >= n_tokens)
        return;
    const int real_n = d_real_n ? min(n_tokens, __ldg(d_real_n)) : n_tokens;
    // Second commit row for the speculative verify (see the h_state snapshot in
    // gdn.cu): the conv window as of snap_n rows. 0 disables it.
    const int snap_n = (conv_snap && d_snap_n) ? min(n_tokens, __ldg(d_snap_n)) : 0;

    for (int ch = threadIdx.x; ch < channels; ch += blockDim.x) {
        float sum = 0.0f;

        for (int k = 0; k < kernel_size; k++) {
            int src_t = token - (kernel_size - 1) + k;
            float val;
            if (src_t >= 0) {
                val = __half2float(x_in[src_t * channels + ch]);
            } else if (conv_state) {
                // Chunked prefill: read trailing context from previous chunk's
                // conv_state instead of zero-padding. conv_state[ch*K + s] holds
                // the input at global position (chunk_offset - K + s).
                int state_idx = src_t + kernel_size;  // maps to [1..K-1]
                val = (state_idx >= 0 && state_idx < kernel_size)
                          ? conv_state[ch * kernel_size + state_idx]
                          : 0.0f;
            } else {
                val = 0.0f;
            }
            // Weight layout: [channels, kernel_size] — kernel_size is contiguous per channel
            sum += val * __half2float(weight[ch * kernel_size + k]);
        }
        if (bias) {
            sum += __half2float(bias[ch]);
        }
        x_out[token * channels + ch] = __float2half(sum);

        // For the last (real) token, write conv_state. Chunks shorter than
        // kernel_size shift the missing leading values in from the previous
        // chunk's conv_state (per-thread read index real_n+k stays ahead of
        // every write index k, so the in-place shift is ordered correctly).
        if (token == real_n - 1 && conv_state) {
            float* state = conv_state + ch * kernel_size;
            for (int k = 0; k < kernel_size; k++) {
                int src_t = real_n - kernel_size + k;
                state[k] = (src_t >= 0) ? __half2float(x_in[src_t * channels + ch])
                                        : state[src_t + kernel_size];
            }
        }
        // Same window, taken at snap_n rows. Its leading values come from the
        // PRE-chunk state (conv_prev), not the live one: the real-row commit
        // above may already have run on another block. snap_n == real_n needs
        // no second copy — the two commits coincide.
        if (token == snap_n - 1 && conv_snap && conv_prev && snap_n != real_n) {
            float* snap = conv_snap + ch * kernel_size;
            const float* prev = conv_prev + ch * kernel_size;
            for (int k = 0; k < kernel_size; k++) {
                int src_t = snap_n - kernel_size + k;
                snap[k] = (src_t >= 0) ? __half2float(x_in[src_t * channels + ch])
                                       : prev[src_t + kernel_size];
            }
        }
    }
}

void ssm_conv1d_prefill(void* conv_state, const Tensor& x_in, const Tensor& weight, const Tensor& bias,
                        Tensor& x_out, int conv_kernel, cudaStream_t stream, const int* d_real_n,
                        void* conv_snap, const int* d_snap_n, const void* conv_prev) {
    int n_tokens = static_cast<int>(x_in.shape[0]);
    int channels = static_cast<int>(x_in.shape[1]);

    ssm_conv1d_prefill_kernel<<<n_tokens, 256, 0, stream>>>(
        static_cast<float*>(conv_state), static_cast<const half*>(x_in.data),
        static_cast<const half*>(weight.data), bias.data ? static_cast<const half*>(bias.data) : nullptr,
        static_cast<half*>(x_out.data), n_tokens, channels, conv_kernel, d_real_n,
        static_cast<float*>(conv_snap), d_snap_n, static_cast<const float*>(conv_prev));
    IMP_CUDA_CHECK_LAUNCH();
}

// ---------------------------------------------------------------------------
// Fused conv1d + SiLU + FP32 output for prefill.
// Replaces 3 separate kernels (conv → SiLU → FP16→FP32) with one.
// ---------------------------------------------------------------------------
__global__ void ssm_conv1d_prefill_f32_silu_kernel(
    float* __restrict__ conv_state, const half* __restrict__ x_in, const half* __restrict__ weight,
    const half* __restrict__ bias, float* __restrict__ x_out_f32, int n_tokens, int channels, int kernel_size,
    const int* __restrict__ d_real_n, float* __restrict__ conv_snap, const int* __restrict__ d_snap_n,
    const float* __restrict__ conv_prev) {
    int token = blockIdx.x;
    if (token >= n_tokens)
        return;
    const int real_n = d_real_n ? min(n_tokens, __ldg(d_real_n)) : n_tokens;
    // Second commit row for the speculative verify (see the h_state snapshot in
    // gdn.cu): the conv window as of snap_n rows. 0 disables it.
    const int snap_n = (conv_snap && d_snap_n) ? min(n_tokens, __ldg(d_snap_n)) : 0;

    // Channels are split across blockIdx.y as well as threads. With the grid
    // keyed on tokens alone, a speculative verify chunk launched 2-4 blocks
    // total and left 166 of 170 SMs idle while each block walked every channel:
    // measured 2.10 ms for a 2-row chunk against 0.10 ms for the single-token
    // decode kernel, a 21x ratio on two rows. Tokens are independent here (each
    // reads its own window from x_in, and conv_state only for the leading
    // positions), and the two state commits below are per channel, so widening
    // the grid changes no result. Prefill keeps the same mapping, with more
    // blocks.
    for (int ch = blockIdx.y * blockDim.x + threadIdx.x; ch < channels;
         ch += gridDim.y * blockDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; k++) {
            int src_t = token - (kernel_size - 1) + k;
            float val;
            if (src_t >= 0) {
                val = __half2float(x_in[src_t * channels + ch]);
            } else if (conv_state) {
                // Chunked prefill: read trailing context from previous chunk's
                // conv_state instead of zero-padding.  conv_state[ch*K + s]
                // holds the input at global position (chunk_offset - K + s).
                int state_idx = src_t + kernel_size;  // maps to [1..K-1]
                val = (state_idx >= 0 && state_idx < kernel_size) ? conv_state[ch * kernel_size + state_idx]
                                                                  : 0.0f;
            } else {
                val = 0.0f;
            }
            sum += val * __half2float(weight[ch * kernel_size + k]);
        }
        if (bias)
            sum += __half2float(bias[ch]);

        // Fused SiLU + FP32 output
        x_out_f32[token * channels + ch] = sum / (1.0f + expf(-sum));

        // Update conv_state for the last (real) token; short chunks shift the
        // missing leading values in from the previous chunk's conv_state (see
        // ssm_conv1d_prefill_kernel).
        if (token == real_n - 1 && conv_state) {
            float* state = conv_state + ch * kernel_size;
            for (int k = 0; k < kernel_size; k++) {
                int src_t = real_n - kernel_size + k;
                state[k] = (src_t >= 0) ? __half2float(x_in[src_t * channels + ch])
                                        : state[src_t + kernel_size];
            }
        }
        // Same window, taken at snap_n rows. Its leading values come from the
        // state BEFORE this chunk, and they must be read from the caller's
        // pre-chunk copy rather than from conv_state: the commit above writes
        // conv_state from a different block, and nothing orders the two. Read
        // the live buffer here and the snapshot silently picks up whichever
        // block ran first — measured as a drop in draft acceptance from 2.26 to
        // 2.03 emitted per verify, with no error anywhere.
        if (token == snap_n - 1 && conv_snap && conv_prev && snap_n != real_n) {
            const float* src_state = conv_prev + ch * kernel_size;
            float* snap = conv_snap + ch * kernel_size;
            for (int k = 0; k < kernel_size; k++) {
                int src_t = snap_n - kernel_size + k;
                snap[k] = (src_t >= 0) ? __half2float(x_in[src_t * channels + ch])
                                       : src_state[src_t + kernel_size];
            }
        }
    }
}

void ssm_conv1d_prefill_f32_silu(void* conv_state, const Tensor& x_in, const Tensor& weight,
                                 const Tensor& bias, float* x_out_f32, int conv_kernel, cudaStream_t stream,
                                 const int* d_real_n, void* conv_snap, const int* d_snap_n,
                                 const void* conv_prev) {
    int n_tokens = static_cast<int>(x_in.shape[0]);
    int channels = static_cast<int>(x_in.shape[1]);
    // Grid over (tokens, channel tiles). A 2-row verify chunk used to occupy
    // two SMs; the y extent gives it the whole card, the same way the
    // single-token decode launcher already does.
    constexpr int kThreads = 256;
    dim3 grid(n_tokens, (channels + kThreads - 1) / kThreads);
    ssm_conv1d_prefill_f32_silu_kernel<<<grid, kThreads, 0, stream>>>(
        static_cast<float*>(conv_state), static_cast<const half*>(x_in.data),
        static_cast<const half*>(weight.data), bias.data ? static_cast<const half*>(bias.data) : nullptr,
        x_out_f32, n_tokens, channels, conv_kernel, d_real_n, static_cast<float*>(conv_snap), d_snap_n,
        static_cast<const float*>(conv_prev));
    IMP_CUDA_CHECK_LAUNCH();
}

// ---------------------------------------------------------------------------
// Mamba2 SSM scan — optimized fused multi-token kernel
// ---------------------------------------------------------------------------
//
// One block per head. Threads organized as (d_tid, s_tid) to parallelize
// both head_dim_ssm and state_size dimensions.
//
// Key optimizations over v1:
// 1. Transposed h_state layout: [n_heads, state_size, head_dim_ssm]
//    Adjacent d-threads access adjacent memory → coalesced reads/writes.
// 2. State-dimension parallelism: s_tiles threads per d-value reduce the
//    inner loop from state_size to state_size/s_tiles iterations.
// 3. Optional fused gating: when z is non-null, computes y * SiLU(z)
//    inline, eliminating 2 kernel launches (silu_inplace + elementwise_mul).
//
// Mamba2 scan equations (discrete-time):
//   dt_h = softplus(dt_raw[h] + dt_bias[h])
//   a_bar = exp(dt_h * A_log[h])
//   For each d in [0, head_dim_ssm):
//     For each s in [0, state_size):
//       h_state[h,s,d] = a_bar * h_state[h,s,d] + dt_h * x[h*hd+d] * B[g*S+s]
//     y[h*hd+d] = sum_s(h_state[h,s,d] * C[g*S+s]) + D[h] * x[h*hd+d]
//   If z provided: y[h*hd+d] *= SiLU(z[h*hd+d])
//
// Template parameters:
//   H_FP16: h_state stored as FP16 (all compute in FP32)
//   FUSE_GATE: fuse y * SiLU(z) into output
template <bool H_FP16, bool FUSE_GATE>
__global__ void ssm_scan_kernel(
    const half* __restrict__ x,         // [n_tokens, inner_size]
    const half* __restrict__ B_in,      // [n_tokens, n_groups * state_size]
    const half* __restrict__ C_in,      // [n_tokens, n_groups * state_size]
    const half* __restrict__ dt_raw,    // [n_tokens, n_heads]
    const float* __restrict__ A_log,    // [n_heads]
    const float* __restrict__ D_skip,   // [n_heads]
    const float* __restrict__ dt_bias,  // [n_heads]
    void* __restrict__ h_state,         // [n_heads, state_size, head_dim_ssm] (transposed)
    half* __restrict__ y,               // [n_tokens, inner_size]
    const half* __restrict__ z,         // [n_tokens, inner_size] (gate, only if FUSE_GATE)
    int n_tokens, int n_heads, int head_dim_ssm, int state_size, int n_groups, int s_tiles,
    const int* __restrict__ d_real_n,    // device chunk length (padded verify chunk) or nullptr
    void* __restrict__ h_snap,           // second state output, or nullptr
    const int* __restrict__ d_snap_n) {  // row count h_snap is taken at
    int h = blockIdx.x;
    if (h >= n_heads)
        return;
    // Padded verify chunk (#847): y is produced for every row (padding rows
    // are causally invisible downstream), but h_state stops advancing after
    // the real last row so the committed recurrent state ignores the pads.
    const int real_n = d_real_n ? min(n_tokens, __ldg(d_real_n)) : n_tokens;
    // Second commit row for the speculative verify (see the h_state snapshot in
    // gdn.cu): the state as of snap_n rows. 0 disables it.
    const int snap_n = (h_snap && d_snap_n) ? min(n_tokens, __ldg(d_snap_n)) : 0;

    int heads_per_group = n_heads / n_groups;
    int g = h / heads_per_group;
    float a_log_h = A_log[h];
    float d_val = D_skip[h];
    float dt_b = dt_bias[h];
    int inner_size = n_heads * head_dim_ssm;
    int BC_size = n_groups * state_size;

    // Thread indexing: d_tid selects head dimension, s_tid selects state chunk
    int d_tid = threadIdx.x % head_dim_ssm;
    int s_tid = threadIdx.x / head_dim_ssm;
    int s_chunk = (state_size + s_tiles - 1) / s_tiles;
    int s_start = s_tid * s_chunk;
    int s_end = s_start + s_chunk;
    if (s_end > state_size)
        s_end = state_size;

    // h_state transposed layout: [n_heads, state_size, head_dim_ssm]
    // For coalesced access: h_state[h, s, d] = base[h * S * hd + s * hd + d]
    int64_t h_base = static_cast<int64_t>(h) * state_size * head_dim_ssm;

    // Shared memory for y_acc reduction across s-tiles: [head_dim_ssm * s_tiles]
    extern __shared__ float smem[];

    for (int t = 0; t < n_tokens; t++) {
        const half* B_g = B_in + t * BC_size + g * state_size;
        const half* C_g = C_in + t * BC_size + g * state_size;

        // Compute dt for this token (shared across all threads in this head)
        float dt_val = __half2float(dt_raw[t * n_heads + h]) + dt_b;
        dt_val = (dt_val > 20.0f) ? dt_val : logf(1.0f + expf(dt_val));
        float a_bar = expf(dt_val * a_log_h);

        float x_val = __half2float(x[t * inner_size + h * head_dim_ssm + d_tid]);

        float y_partial = 0.0f;
        for (int s = s_start; s < s_end; s++) {
            float b_s = __half2float(B_g[s]);
            float c_s = __half2float(C_g[s]);
            // Transposed access: adjacent d-threads read adjacent memory (coalesced)
            int64_t idx = h_base + static_cast<int64_t>(s) * head_dim_ssm + d_tid;
            float h_old;
            if constexpr (H_FP16) {
                h_old = __half2float(static_cast<half*>(h_state)[idx]);
            } else {
                h_old = static_cast<float*>(h_state)[idx];
            }
            float h_new = a_bar * h_old + dt_val * x_val * b_s;
            if (t < real_n) {
                if constexpr (H_FP16) {
                    static_cast<half*>(h_state)[idx] = __float2half(h_new);
                } else {
                    static_cast<float*>(h_state)[idx] = h_new;
                }
            }
            if (t == snap_n - 1) {
                if constexpr (H_FP16) {
                    static_cast<half*>(h_snap)[idx] = __float2half(h_new);
                } else {
                    static_cast<float*>(h_snap)[idx] = h_new;
                }
            }
            y_partial += h_new * c_s;
        }

        if (s_tiles > 1) {
            // Reduce y_partial across s-tiles for this d
            smem[d_tid * s_tiles + s_tid] = y_partial;
            __syncthreads();

            for (int stride = s_tiles / 2; stride > 0; stride >>= 1) {
                if (s_tid < stride) {
                    smem[d_tid * s_tiles + s_tid] += smem[d_tid * s_tiles + s_tid + stride];
                }
                __syncthreads();
            }

            if (s_tid == 0) {
                float y_val = smem[d_tid * s_tiles] + d_val * x_val;
                if constexpr (FUSE_GATE) {
                    float z_val = __half2float(z[t * inner_size + h * head_dim_ssm + d_tid]);
                    z_val = z_val / (1.0f + expf(-z_val));
                    y_val *= z_val;
                }
                y[t * inner_size + h * head_dim_ssm + d_tid] = __float2half(y_val);
            }
            // Barrier before next token iteration (all threads must finish writing smem)
            if (n_tokens > 1)
                __syncthreads();
        } else {
            // s_tiles == 1: no reduction needed
            float y_val = y_partial + d_val * x_val;
            if constexpr (FUSE_GATE) {
                float z_val = __half2float(z[t * inner_size + h * head_dim_ssm + d_tid]);
                z_val = z_val / (1.0f + expf(-z_val));
                y_val *= z_val;
            }
            y[t * inner_size + h * head_dim_ssm + d_tid] = __float2half(y_val);
        }
    }
}

static void ssm_scan_launch(const half* x, const half* B, const half* C, const half* dt, const float* A_log,
                            const float* D, const float* dt_bias, void* h_state, half* y, const half* z,
                            int n_tokens, int n_heads, int head_dim_ssm, int state_size, int n_groups,
                            QType h_dtype, cudaStream_t stream, const int* d_real_n = nullptr,
                            void* h_snap = nullptr, const int* d_snap_n = nullptr) {
    // Pick s_tiles to maximize thread count per block (power of 2, up to 1024 threads)
    int hd = std::max(head_dim_ssm, 1);
    int max_s_tiles = std::min(state_size, 1024 / hd);
    int s_tiles = 1;
    while (s_tiles * 2 <= max_s_tiles)
        s_tiles *= 2;

    int threads = hd * s_tiles;
    size_t smem_bytes = (s_tiles > 1) ? static_cast<size_t>(hd) * s_tiles * sizeof(float) : 0;

    bool fp16 = (h_dtype == QType::F16);
    bool fused = (z != nullptr);

#define SSM_SCAN_LAUNCH(H_FP16_V, FUSE_V)                                                              \
    do {                                                                                               \
        ssm_scan_kernel<H_FP16_V, FUSE_V>                                                              \
            <<<n_heads, threads, smem_bytes, stream>>>(x, B, C, dt, A_log, D, dt_bias, h_state, y, z,  \
                                                       n_tokens, n_heads, head_dim_ssm, state_size,    \
                                                       n_groups, s_tiles, d_real_n, h_snap, d_snap_n); \
        IMP_CUDA_CHECK_LAUNCH();                                                                       \
    } while (0)

    if (fp16) {
        if (fused)
            SSM_SCAN_LAUNCH(true, true);
        else
            SSM_SCAN_LAUNCH(true, false);
    } else {
        if (fused)
            SSM_SCAN_LAUNCH(false, true);
        else
            SSM_SCAN_LAUNCH(false, false);
    }
#undef SSM_SCAN_LAUNCH
}

void ssm_scan_decode(const Tensor& x, const Tensor& B, const Tensor& C, const Tensor& dt, const Tensor& A_log,
                     const Tensor& D, const Tensor& dt_bias, void* h_state, Tensor& y, const void* z,
                     int n_heads, int head_dim_ssm, int state_size, int n_groups, QType h_dtype,
                     cudaStream_t stream) {
    ssm_scan_launch(static_cast<const half*>(x.data), static_cast<const half*>(B.data),
                    static_cast<const half*>(C.data), static_cast<const half*>(dt.data),
                    static_cast<const float*>(A_log.data), static_cast<const float*>(D.data),
                    static_cast<const float*>(dt_bias.data), h_state, static_cast<half*>(y.data),
                    static_cast<const half*>(z), 1, n_heads, head_dim_ssm, state_size, n_groups, h_dtype,
                    stream);
}

void ssm_scan_prefill(const Tensor& x, const Tensor& B, const Tensor& C, const Tensor& dt,
                      const Tensor& A_log, const Tensor& D, const Tensor& dt_bias, void* h_state, Tensor& y,
                      const void* z, int n_tokens, int n_heads, int head_dim_ssm, int state_size,
                      int n_groups, QType h_dtype, cudaStream_t stream, const int* d_real_n, void* h_snap,
                      const int* d_snap_n) {
    ssm_scan_launch(static_cast<const half*>(x.data), static_cast<const half*>(B.data),
                    static_cast<const half*>(C.data), static_cast<const half*>(dt.data),
                    static_cast<const float*>(A_log.data), static_cast<const float*>(D.data),
                    static_cast<const float*>(dt_bias.data), h_state, static_cast<half*>(y.data),
                    static_cast<const half*>(z), n_tokens, n_heads, head_dim_ssm, state_size, n_groups,
                    h_dtype, stream, d_real_n, h_snap, d_snap_n);
}

// ---------------------------------------------------------------------------
// Group RMSNorm
// ---------------------------------------------------------------------------

// One block per (token, group). Threads reduce over group_size.
__global__ void group_rmsnorm_fp16_kernel(const half* __restrict__ x, const half* __restrict__ weight,
                                          half* __restrict__ out, int total_dim, int n_groups, float eps) {
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
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
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

void group_rmsnorm(const Tensor& x, const Tensor& weight, Tensor& out, int n_groups, float eps,
                   cudaStream_t stream) {
    int n_tokens = static_cast<int>(x.shape[0]);
    int total_dim = static_cast<int>(x.shape[1]);
    int group_size = total_dim / n_groups;

    // Power of 2 for reduction
    int threads = 1;
    while (threads * 2 <= std::min(group_size, 256))
        threads *= 2;

    dim3 grid(n_tokens, n_groups);
    size_t smem = threads * sizeof(float);

    group_rmsnorm_fp16_kernel<<<grid, threads, smem, stream>>>(static_cast<const half*>(x.data),
                                                               static_cast<const half*>(weight.data),
                                                               static_cast<half*>(out.data), total_dim,
                                                               n_groups, eps);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
