#pragma once

#include "core/dispatch_policy.h"

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

struct RuntimeConfig;  // fwd, defined in runtime/config.h

// Runtime-dispatched attention prefill (SM120 / Blackwell).
// Dispatch: MXFP4 FMHA -> FP8 FMHA -> FP16 FMHA -> Blackwell WMMA 128x64.
// rcfg is read for attention.{fp8_fmha,fmha_sm120} ladder gates.
// attn_sinks (optional, [n_heads] FP16 device ptr): learned attention sinks
// (gpt-oss #547). Only the FP16 WMMA FMHA tier understands them (#992) —
// when set, every other tier is skipped, and an FMHA decline throws instead
// of falling through to a sink-blind kernel (silent-wrong-output guard).
void attention_prefill_dispatch(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                                bool causal, int sliding_window, float softcap, cudaStream_t stream,
                                const DispatchPolicy& rcfg, int q_offset = 0,
                                const half* attn_sinks = nullptr);

// Query the compute capability of the current device.
int get_device_sm_version();

}  // namespace imp
