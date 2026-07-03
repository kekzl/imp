#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

// Conv1d decode: shift conv_state sliding window, insert new value, compute output.
// conv_state: [conv_channels, conv_kernel] float (updated in-place)
// x_in:       [n_tokens, conv_channels] compute_dtype (input)
// weight:     [conv_channels, conv_kernel] float (or FP16)
// bias:       [conv_channels] float (or FP16), can be nullptr
// x_out:      [n_tokens, conv_channels] compute_dtype (output)
// FP32-output conv1d decode with fused SiLU (for GDN FP32 pipeline)
void ssm_conv1d_decode_f32_silu(void* conv_state, const Tensor& x_in, const Tensor& weight,
                                const Tensor& bias, float* x_out_f32, int conv_kernel, cudaStream_t stream);

// For decode: n_tokens = 1 per sequence
void ssm_conv1d_decode(void* conv_state, const Tensor& x_in, const Tensor& weight, const Tensor& bias,
                       Tensor& x_out, int conv_kernel, cudaStream_t stream);

// Conv1d prefill: causal 1D convolution over full sequence.
// Also updates conv_state with the last conv_kernel values (for chunks
// shorter than conv_kernel the missing leading values shift in from the
// previous chunk's conv_state).
// conv_state: [conv_channels, conv_kernel] float (written with last values)
// x_in:       [n_tokens, conv_channels] compute_dtype
// weight:     [conv_channels, conv_kernel] float
// bias:       [conv_channels] float, can be nullptr
// x_out:      [n_tokens, conv_channels] compute_dtype
// d_real_n:   optional device int — real chunk length when the chunk is
//             padded for a captured verify graph (#847). Rows past it still
//             produce conv outputs (causally invisible padding), but the
//             conv_state tail is written from the real last rows only.
void ssm_conv1d_prefill(void* conv_state, const Tensor& x_in, const Tensor& weight, const Tensor& bias,
                        Tensor& x_out, int conv_kernel, cudaStream_t stream,
                        const int* d_real_n = nullptr);

// Fused conv1d + SiLU + FP32 output for prefill (GDN layers).
// Replaces 3 separate kernels (conv → SiLU → FP16→FP32) with one launch.
// d_real_n: see ssm_conv1d_prefill.
void ssm_conv1d_prefill_f32_silu(void* conv_state, const Tensor& x_in, const Tensor& weight,
                                 const Tensor& bias, float* x_out_f32, int conv_kernel, cudaStream_t stream,
                                 const int* d_real_n = nullptr);

// Mamba2 SSM scan decode (single step per sequence).
// x:        [inner_size] compute_dtype — input after conv + SiLU
// B:        [n_groups * state_size] compute_dtype
// C:        [n_groups * state_size] compute_dtype
// dt:       [n_heads] compute_dtype (raw dt before softplus)
// A_log:    [n_heads] float
// D:        [n_heads] float
// dt_bias:  [n_heads] float
// h_state:  [n_heads, state_size, head_dim_ssm] float/FP16 (transposed for coalescing)
// y:        [inner_size] compute_dtype (output)
// z:        [inner_size] compute_dtype (gate input, nullptr = no fusion)
//           When non-null, output is y * SiLU(z) (fused gating, saves 2 kernels).
// h_dtype: QType of h_state storage (FP32 default, FP16 for VRAM savings).
// Computation always in FP32; FP16 only affects load/store.
void ssm_scan_decode(const Tensor& x, const Tensor& B, const Tensor& C, const Tensor& dt, const Tensor& A_log,
                     const Tensor& D, const Tensor& dt_bias, void* h_state, Tensor& y, const void* z,
                     int n_heads, int head_dim_ssm, int state_size, int n_groups, QType h_dtype = QType::F32,
                     cudaStream_t stream = nullptr);

// SSM scan prefill: iterate scan_decode over all tokens sequentially.
// z: [n_tokens, inner_size] gate input (nullptr = no fusion).
// d_real_n: optional device int — real chunk length when the chunk is padded
//           for a captured verify graph (#847). y is written for all rows,
//           but h_state stops advancing after the real last row.
void ssm_scan_prefill(const Tensor& x, const Tensor& B, const Tensor& C, const Tensor& dt,
                      const Tensor& A_log, const Tensor& D, const Tensor& dt_bias, void* h_state, Tensor& y,
                      const void* z, int n_tokens, int n_heads, int head_dim_ssm, int state_size,
                      int n_groups, QType h_dtype = QType::F32, cudaStream_t stream = nullptr,
                      const int* d_real_n = nullptr);

// Group RMSNorm: normalize each of n_groups groups independently.
// x:      [n_tokens, dim] compute_dtype (dim = n_groups * group_size)
// weight: [dim] compute_dtype
// out:    [n_tokens, dim] compute_dtype
void group_rmsnorm(const Tensor& x, const Tensor& weight, Tensor& out, int n_groups, float eps,
                   cudaStream_t stream);

// Standalone SiLU: out[i] = x[i] * sigmoid(x[i])
void silu_inplace(Tensor& x, cudaStream_t stream);

// Squared ReLU: out[i] = max(0, x[i])^2  (Nemotron-H expert activation)
void relu_sqr_inplace(Tensor& x, cudaStream_t stream);

// Element-wise multiply: out[i] = a[i] * b[i]
void elementwise_mul(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t stream);

// Sigmoid multiply: out[i] = a[i] * sigmoid(b[i])
void sigmoid_mul(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t stream);

}  // namespace imp
