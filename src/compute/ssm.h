#pragma once

#include <cuda_fp16.h>

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

// Conv1d decode: shift conv_state sliding window, insert new value, compute output.
// conv_state:[conv_channels,conv_kernel] float; x_in:[n_tokens,conv_channels]; weight:
// [conv_channels,conv_kernel]; bias:[conv_channels] (nullable); x_out same shape as x_in.
// FP32-output variant with fused SiLU (GDN FP32 pipeline).
void ssm_conv1d_decode_f32_silu(void* conv_state, const Tensor& x_in, const Tensor& weight,
                                const Tensor& bias, float* x_out_f32, int conv_kernel, cudaStream_t stream);

// Batched conv1d decode over N independent sequences (concurrent GDN decode).
// x_in / x_out_f32 are [n_seq, channels] sequence-major; seq_slots selects each
// sequence's conv state out of the pool.
void ssm_conv1d_decode_f32_silu_batched(void* conv_state_pool, const int* seq_slots,
                                        int64_t conv_state_seq_stride, const half* x_in,
                                        const Tensor& weight, const Tensor& bias, float* x_out_f32,
                                        int n_seq, int channels, int conv_kernel, cudaStream_t stream);

// For decode: n_tokens = 1 per sequence
void ssm_conv1d_decode(void* conv_state, const Tensor& x_in, const Tensor& weight, const Tensor& bias,
                       Tensor& x_out, int conv_kernel, cudaStream_t stream);

// Conv1d prefill: causal 1D convolution over the full sequence; also updates conv_state
// with the last conv_kernel values (shorter chunks shift in missing leading values from
// the previous conv_state).
// d_real_n: optional device int, real chunk length when padded for a captured verify graph
// (#847); rows past it still produce outputs, but the conv_state tail comes from real rows only.
// conv_snap/d_snap_n/conv_prev: second commit at d_snap_n rows, the state a fully rejected
// speculative chunk falls back to; conv_prev is needed because a snapshot row shorter than
// kernel_size takes leading values from before the chunk, and the live buffer has moved on.
void ssm_conv1d_prefill(void* conv_state, const Tensor& x_in, const Tensor& weight, const Tensor& bias,
                        Tensor& x_out, int conv_kernel, cudaStream_t stream, const int* d_real_n = nullptr,
                        void* conv_snap = nullptr, const int* d_snap_n = nullptr,
                        const void* conv_prev = nullptr);

// Fused conv1d + SiLU + FP32 output for prefill (GDN layers).
// Replaces 3 separate kernels (conv → SiLU → FP16→FP32) with one launch.
// d_real_n: see ssm_conv1d_prefill.
void ssm_conv1d_prefill_f32_silu(void* conv_state, const Tensor& x_in, const Tensor& weight,
                                 const Tensor& bias, float* x_out_f32, int conv_kernel, cudaStream_t stream,
                                 const int* d_real_n = nullptr, void* conv_snap = nullptr,
                                 const int* d_snap_n = nullptr, const void* conv_prev = nullptr);

// Grouped form (multi-candidate verify chunk on a hybrid): x_in holds n_seq groups of
// n_tokens rows; group z reads/commits the conv window of pool slot seq_slots[z]
// (slot_stride in floats). d_real_n bounds every group's commit row; the snapshot
// (conv_snap/d_snap_n/conv_prev) is written from group 0 only.
// out_slots/snap_slots (batched speculative verify): group z commits at d_real_n rows into
// out_slots[z] instead of in place, and at d_snap_n rows into snap_slots[z]. snap_slots
// requires out_slots; snap_slots[z] may equal seq_slots[z].
void ssm_conv1d_prefill_f32_silu_grouped(void* conv_state_pool, const int* seq_slots, int64_t slot_stride,
                                         int n_seq, const Tensor& x_in, const Tensor& weight,
                                         const Tensor& bias, float* x_out_f32, int conv_kernel,
                                         cudaStream_t stream, const int* d_real_n = nullptr,
                                         void* conv_snap = nullptr, const int* d_snap_n = nullptr,
                                         const void* conv_prev = nullptr, const int* out_slots = nullptr,
                                         const int* snap_slots = nullptr);

// Mamba2 SSM scan decode (single step/sequence). x:[inner_size]; B,C:[n_groups*state_size];
// dt:[n_heads] (raw, pre-softplus); A_log/D/dt_bias:[n_heads] float;
// h_state:[n_heads,state_size,head_dim_ssm] float/FP16 (transposed for coalescing);
// y:[inner_size]; z:[inner_size] gate (nullptr=no fusion, else y*=SiLU(z), saves 2 kernels).
// h_dtype: FP32 default, FP16 for VRAM savings; computation always FP32, dtype affects
// load/store only.
void ssm_scan_decode(const Tensor& x, const Tensor& B, const Tensor& C, const Tensor& dt, const Tensor& A_log,
                     const Tensor& D, const Tensor& dt_bias, void* h_state, Tensor& y, const void* z,
                     int n_heads, int head_dim_ssm, int state_size, int n_groups, QType h_dtype = QType::F32,
                     cudaStream_t stream = nullptr);

// SSM scan prefill: iterates scan_decode over all tokens sequentially. z:[n_tokens,
// inner_size] gate (nullptr=no fusion).
// d_real_n: optional device int, real chunk length for a padded captured verify graph
// (#847); y written for all rows, h_state stops advancing after the real last row.
// h_snap/d_snap_n: second state output at d_snap_n rows alongside the committed one
// (mirrors the GDN scan's snapshot, gdn.cu) - needed when a rejected speculative draft
// falls back to the state after the chunk's first row.
void ssm_scan_prefill(const Tensor& x, const Tensor& B, const Tensor& C, const Tensor& dt,
                      const Tensor& A_log, const Tensor& D, const Tensor& dt_bias, void* h_state, Tensor& y,
                      const void* z, int n_tokens, int n_heads, int head_dim_ssm, int state_size,
                      int n_groups, QType h_dtype = QType::F32, cudaStream_t stream = nullptr,
                      const int* d_real_n = nullptr, void* h_snap = nullptr, const int* d_snap_n = nullptr);

// Group RMSNorm: normalizes each of n_groups groups independently. x:[n_tokens,dim]
// (dim=n_groups*group_size); weight:[dim]; out same shape as x.
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

// Factored conv window for the batched speculative verify
// (compute/ssm_conv_tap.cu, docs/plans/2026-09-12-factored-verify-spare.md).
// The drafted row shifts the window by one, so only the single new tap (channels halfs)
// need survive the step, not the whole window (channels*kernel_size floats). Both pools
// indexed by recurrent SLOT so a request keeps its row while moving within the batch.
// stash: x_in is [n_seq,n_tokens,channels] half, row taken is real_n-1.
// apply: advances slot's window by its stashed tap (single-row ssm_conv1d_commit_kernel).
void ssm_conv_tap_stash(void* tap_pool, const void* x_in, int n_tokens, int channels, const int* d_real_n,
                        const int* slots, int n_seq, cudaStream_t stream);
void ssm_conv_tap_apply(void* conv_pool, int64_t slot_stride_floats, const void* tap_pool, int channels,
                        int kernel_size, const int* slots, int n_seq, cudaStream_t stream);

}  // namespace imp
