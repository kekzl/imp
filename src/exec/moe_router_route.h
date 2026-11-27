#pragma once

// Fused-router dispatch from compute_moe_routing: rows n >= 2 through moe_router_fused, one decode row of a
// sigmoid router through moe_router_decode, ids / weights straight into the routing buffers; false = not
// taken (the caller keeps GEMM + top-k).

#include "compute/moe_routing.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace imp {

// What the fused router cannot serve: policy off, group limit, gpt-oss logit bias.
struct RouterFusedGate {
    bool group_limit;
    bool logit_bias_profile;
    const void* logit_bias;
};

// moe_router_decode: fp32 logits row (moe_.gate_logits) and its last-CTA ticket (nullptr = not allocated).
struct RouterDecodeScratch {
    float* logits;
    void* ticket;
};

// Arguments in the old parameter order (audit D-11); Tensors and buffers by pointer, gate and scratch by
// value.
struct RouterFusedRouteArgs {
    const Tensor* in = nullptr;
    const Tensor* gate = nullptr;
    QType compute{};
    int n = 0;
    int ne = 0;
    int top_k = 0;
    bool use_sigmoid = false;
    bool norm_weights = false;
    const void* score_bias = nullptr;
    RouterFusedGate g{};
    MoeRoutingBuffers* buf = nullptr;
    void* ws = nullptr;
    size_t ws_bytes = 0;
    RouterDecodeScratch dec{};
    cudaStream_t stream = nullptr;
};
[[nodiscard]] bool router_fused_route(const RouterFusedRouteArgs& a);

}  // namespace imp
