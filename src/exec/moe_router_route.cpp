#include "exec/moe_router_route.h"

#include "compute/moe_router_fused.h"
#include "core/logging.h"
#include "core/process_diag.h"
#include "exec/executor_debug.h"

#include <cuda_fp16.h>

#include <utility>

namespace imp {

namespace {
constexpr int kRouterFusedFp16AccMaxRows = 1024;
}  // namespace

bool router_fused_route(const Tensor& in, const Tensor& gate, QType compute, int n, int ne, int top_k,
                        bool use_sigmoid, bool norm_weights, const void* score_bias, const RouterFusedGate& g,
                        MoeRoutingBuffers& buf, void* ws, size_t ws_bytes, cudaStream_t stream) {
    const int d = static_cast<int>(in.shape[1]);
    if (!g.policy_on || g.dump_logits || g.group_limit || (g.logit_bias_profile && g.logit_bias != nullptr) ||
        debug_forward_enabled())
        return false;
    // Deterministic mode keeps the row-invariant router (a row must not depend on M, #2152).
    if (n < 2 || compute != QType::F16 || gate.qtype != QType::F16 || buf.pool == nullptr ||
        process_diag_deterministic() || in.stride[0] != d || gate.shape[1] != d || n > buf.max_tokens)
        return false;
    // From 1024 rows the full-rate fp16 split-K of cuBLASLt (gemm.cublas_fp16_acc) keeps the router:
    // n 2048 Nemotron fused 28.56 vs 23.9 us chain, Qwen3-30B 26.65 vs 26.4; Gemma-4 (fp32 accumulate) gains.
    if (n >= kRouterFusedFp16AccMaxRows && process_diag_cublas_fp16_acc())
        return false;
    MoeRouterFusedArgs a;
    a.x = static_cast<const half*>(in.data);
    a.w = static_cast<const half*>(gate.data);
    a.M = n;
    a.K = d;
    a.ne = ne;
    a.top_k = top_k;
    a.use_sigmoid = use_sigmoid;
    a.normalize = norm_weights;
    a.score_bias = static_cast<const half*>(score_bias);
    a.ids = buf.expert_indices;
    a.weights = buf.expert_weights;
    if (!moe_router_fused(a, ws, ws_bytes, stream))
        return false;
    static bool logged = false;  // activity proof, once
    if (!std::exchange(logged, true))
        IMP_LOG_INFO("moe.router_fused: fused router ACTIVE (n=%d ne=%d d=%d top_k=%d)", n, ne, d, top_k);
    return true;
}

}  // namespace imp
