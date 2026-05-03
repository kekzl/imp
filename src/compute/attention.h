#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

// Runtime-dispatched attention prefill (SM120 / Blackwell).
// Dispatch: MXFP4 FMHA -> FP8 FMHA -> FP16 FMHA -> Blackwell WMMA 128x64.
void attention_prefill_dispatch(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                                bool causal = true, int sliding_window = 0, float softcap = 0.0f,
                                cudaStream_t stream = nullptr);

// Query the compute capability of the current device.
int get_device_sm_version();

}  // namespace imp
