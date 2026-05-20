#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

struct RuntimeConfig;  // fwd, defined in runtime/config.h

// Runtime-dispatched attention prefill (SM120 / Blackwell).
// Dispatch: MXFP4 FMHA -> FP8 FMHA -> FP16 FMHA -> Blackwell WMMA 128x64.
// rcfg is read for attention.{fp8_fmha,fmha_sm120} ladder gates.
void attention_prefill_dispatch(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                                bool causal, int sliding_window, float softcap, cudaStream_t stream,
                                const RuntimeConfig& rcfg);

// Query the compute capability of the current device.
int get_device_sm_version();

}  // namespace imp
