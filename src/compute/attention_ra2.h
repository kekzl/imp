#pragma once

#include <cuda_runtime.h>

#include "core/tensor.h"

namespace imp {

// RA2 prefill (attention.ra2_prefill, opt-in): INT8 QK^T + NVFP4 P x two-term NVFP4 V on mma.sync, Q/K/V
// quantized per call. q [n, nh*hd], k/v [kv_len, nkv*hd] FP16, o [n, nh*hd]; causal, Q row 0 at
// q_offset, kv_len = q_offset + n. hd 64/128/256, nh % nkv == 0. false = declined (scratch or shape).
[[nodiscard]] bool attention_ra2_prefill(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& o, int n,
                                         int kv_len, int nh, int nkv, int hd, float scale, int q_offset,
                                         cudaStream_t stream);

// Pre-cudaDeviceReset hook: drops the arena scratch pointer.
void attention_ra2_reset_static_cuda_state();

}  // namespace imp
