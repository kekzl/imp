#pragma once

// moe.router_fused dispatch from compute_moe_routing: rows n >= 2 through moe_router_fused, ids / weights
// straight into the routing buffers; false = not taken (the caller keeps GEMM + top-k).

#include "compute/moe_routing.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace imp {

// What the fused router cannot serve: policy off, logit dumps (no logits), group limit, gpt-oss logit bias.
struct RouterFusedGate {
    bool policy_on;
    bool dump_logits;
    bool group_limit;
    bool logit_bias_profile;
    const void* logit_bias;
};

[[nodiscard]] bool router_fused_route(const Tensor& in, const Tensor& gate, QType compute, int n, int ne,
                                      int top_k, bool use_sigmoid, bool norm_weights, const void* score_bias,
                                      const RouterFusedGate& g, MoeRoutingBuffers& buf, void* ws,
                                      size_t ws_bytes, cudaStream_t stream);

}  // namespace imp
