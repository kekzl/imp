#pragma once

#include "core/dispatch_policy.h"

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

struct RuntimeConfig;  // fwd, defined in runtime/config.h

// Runtime-dispatched attention prefill (SM120/Blackwell): FA2 -> FP8 FMHA (opt-in) -> FP16 FMHA ->
// WMMA 128x64. Gated by attention.{fmha_fa2,fp8_fmha,fmha_sm120} in rcfg. attn_sinks (gpt-oss #547): only
// fp16-qk FA2 and the FP16 FMHA fold them (#992); if set, every other tier is skipped and decline throws.
void attention_prefill_dispatch(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                                bool causal, int sliding_window, float softcap, cudaStream_t stream,
                                const DispatchPolicy& rcfg, int q_offset = 0,
                                const half* attn_sinks = nullptr);

}  // namespace imp
