#pragma once

// Qwen4Exp gated residual (hyper-connections). The residual stream is hc x d wide; each block
// reads one d-wide mix of the streams and writes its output back through hc scalar gates.
//
//   normed = grouped_rmsnorm(hidden, w)                       [n, hc*d], (1 + w), one RMS per (row, stream)
//   low    = silu(normed @ down^T / hc)                        [n, lowrank]
//   mixw   = sigmoid(low @ up^T)                               [n, hc*d]
//   mixed  = mean_s(mixw[:, s] * normed[:, s])                 [n, d]         -> block input
//   inj    = 2 * sigmoid(normed @ inject^T / hc)               [n, hc]
//   hidden[:, s] += block_out * inj[:, s]                      for every stream s
//
// The two GEMMs go through gemm(); everything else is here. FP16 in and out, FP32 arithmetic.

#include <cuda_runtime.h>

#include "core/tensor.h"

namespace imp {

// out[t, s*d + j] = x[t, s*d + j] / rms(x[t, s*d : s*d + d]) * (1 + w[s*d + j]).
void hc_grouped_rmsnorm(const Tensor& x, const Tensor& w, Tensor& out, int hc, int d, float eps,
                        cudaStream_t stream);

// x = silu(x / hc), in place. [n, lowrank] FP16.
void hc_silu_div(Tensor& x, int hc, cudaStream_t stream);

// out[t, j] = mean over s of sigmoid(mixw[t, s*d + j]) * normed[t, s*d + j]; out2 (optional) gets a copy.
void hc_mix(const Tensor& mixw, const Tensor& normed, Tensor& out, int hc, int d, cudaStream_t stream,
            Tensor* out2 = nullptr);

// n = 1: grouped_rmsnorm + low = silu(normed @ down^T / hc) + inj = 2*sigmoid(normed @ inject^T / hc) in
// one kernel, bit-identical to the separate kernels. False (nothing launched): n != 1, hc > 8, d % 8,
// hc*d % 16, hc*d > 24576, an x/w/normed pointer not 16-byte aligned or a non-FP16 operand.
[[nodiscard]] bool hc_read_decode(const Tensor& x, const Tensor& w, const Tensor& down, const Tensor* inject,
                                  Tensor& normed, Tensor& low, Tensor& inj, int hc, int d, float eps,
                                  cudaStream_t stream);

// hidden[t, s*d + j] += (h - mixed)[t, j] * inj[t, s]: hc_sub + hc_inject_add in one kernel.
void hc_write(Tensor& hidden, const Tensor& h, const Tensor& mixed, const Tensor& inj, int hc, int d,
              cudaStream_t stream);

// inj = 2 * sigmoid(inj / hc), in place. [n, hc] FP16.
void hc_inject_weights(Tensor& inj, int hc, cudaStream_t stream);

// hidden[t, s*d + j] += out[t, j] * inj[t, s].
void hc_inject_add(Tensor& hidden, const Tensor& out, const Tensor& inj, int hc, int d, cudaStream_t stream);

// out = a - b, elementwise FP16 (block output recovered from the block's h + out convention).
void hc_sub(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t stream);

// dst[t, s*d + j] = src[t, j] for every stream s (the embedding repeated hc times).
void hc_repeat(const Tensor& src, Tensor& dst, int hc, int d, cudaStream_t stream);

}  // namespace imp
