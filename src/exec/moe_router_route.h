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

[[nodiscard]] bool router_fused_route(const Tensor& in, const Tensor& gate, QType compute, int n, int ne,
                                      int top_k, bool use_sigmoid, bool norm_weights, const void* score_bias,
                                      const RouterFusedGate& g, MoeRoutingBuffers& buf, void* ws,
                                      size_t ws_bytes, const RouterDecodeScratch& dec, cudaStream_t stream);

}  // namespace imp
