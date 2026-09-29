#pragma once
// Qwen4Exp PLE (n-gram per-layer embedding) kernels. FP16 in/out, FP32 math. The projections
// and grouped norms around them reuse gemm() and hc_grouped_rmsnorm(); executor_ple.cu wires it.

#include "core/tensor.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

// In place over q: gv[t, s*d + j] = sigmoid(g) * value[t, j] with
// g = dot(key[t, s, :], q[t, s, :]) / sqrt(d), g = sqrt(max(|g|, 1e-6)) * sign(g).
void ple_gate_value(const Tensor& key, Tensor& q_gv, const Tensor& value, int hc, int d, cudaStream_t stream);

// hidden[t, c] += gv[t, c] + silu(sum_k w[c, k] * x[t - (kernel-1-k)*dilation, c]) with
// x = conv_state (state_len = (kernel-1)*dilation rows, the sequence's past) ++ gvn; then
// conv_state <- the last state_len rows of x. w is depthwise [channels, kernel].
// n_seq sequences of n/n_seq rows each; sequence s's [state_len, channels] FP16 rows start at
// conv_state + (slots ? slots[s] : 0) * slot_stride halves (slots: DEVICE, graph-stable).
// Single sequence: d_real_n (device) caps the committed rows (pad rows), snap_state gets the state
// after d_snap_n rows (verify row-0 snapshot). All three nullable.
void ple_conv_add(const Tensor& gv, const Tensor& gvn, const Tensor& w, void* conv_state, int64_t slot_stride,
                  const int* slots, int n_seq, Tensor& hidden, int channels, int kernel, int dilation,
                  cudaStream_t stream, void* snap_state = nullptr, const int* d_snap_n = nullptr,
                  const int* d_real_n = nullptr);

}  // namespace imp
