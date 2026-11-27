#pragma once

#include <cuda_runtime.h>

namespace imp {

struct MoeRoutingResult;
struct Tensor;

// GraphExecutor::run_moe_decode_fast / run_moe_decode_nvfp4_host arguments in their old parameter
// order (audit D-11); routing and Tensors by pointer, residual_fused the out flag.
struct MoeDecodeFastArgs {
    int layer = 0;
    cudaStream_t stream = nullptr;
    int n = 0;
    int d = 0;
    int eff = 0;
    int top_k = 0;
    const MoeRoutingResult* routing = nullptr;
    const Tensor* no = nullptr;
    Tensor* h = nullptr;
    const Tensor* r = nullptr;
    bool moe_use_fp32_residual = false;
    bool moe_fused_norm_q8 = false;
    bool will_skip_residual_copy = false;
    bool* residual_fused = nullptr;
};

// n_rows 1 = the single-row decode step (the old default).
struct MoeDecodeNvfp4HostArgs {
    int layer = 0;
    cudaStream_t stream = nullptr;
    int d = 0;
    int eff = 0;
    int top_k = 0;
    const MoeRoutingResult* routing = nullptr;
    const Tensor* no = nullptr;
    Tensor* h = nullptr;
    const Tensor* r = nullptr;
    bool moe_use_fp32_residual = false;
    bool will_skip_residual_copy = false;
    bool* residual_fused = nullptr;
    bool non_gated_experts = false;
    int n_rows = 1;
};

}  // namespace imp
