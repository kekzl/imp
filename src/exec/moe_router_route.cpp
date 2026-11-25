#include "exec/moe_router_route.h"

#include "compute/moe_router_decode.h"
#include "compute/moe_router_fused.h"
#include "core/log_once.h"
#include "core/logging.h"
#include "core/process_diag.h"
#include "exec/executor_debug.h"

#include <cuda_fp16.h>

#include <utility>

namespace imp {

namespace {
constexpr int kRouterFusedFp16AccMaxRows = 1024;

// One decode row of a sigmoid router (Flint 256 experts, over moe_router_fused's 128): GEMV + top-k in one
// launch, logits bit-identical to gemv_gate_fp32. Runs under runtime.deterministic too (one row).
bool router_decode_route(const Tensor& in, const Tensor& gate, int ne, int top_k, bool use_sigmoid,
                         bool norm_weights, const void* score_bias, MoeRoutingBuffers& buf,
                         const RouterDecodeScratch& dec, cudaStream_t stream) {
    const int d = static_cast<int>(in.shape[1]);
    if (!use_sigmoid || dec.logits == nullptr || dec.ticket == nullptr || gate.shape[0] != ne)
        return false;
    const MoeRouterDecodeArgs a{static_cast<const half*>(gate.data),
                                static_cast<const half*>(in.data),
                                dec.logits,
                                ne,
                                d,
                                top_k,
                                norm_weights,
                                static_cast<const half*>(score_bias),
                                buf.expert_indices,
                                buf.expert_weights,
                                static_cast<unsigned int*>(dec.ticket)};
    if (!moe_router_decode(a, stream))
        return false;
    static LogOnce logged;  // activity proof, once
    if (logged.first())
        IMP_LOG_INFO("decode MoE router ACTIVE (ne=%d d=%d top_k=%d)", ne, d, top_k);
    return true;
}
}  // namespace

bool router_fused_route(const Tensor& in, const Tensor& gate, QType compute, int n, int ne, int top_k,
                        bool use_sigmoid, bool norm_weights, const void* score_bias, const RouterFusedGate& g,
                        MoeRoutingBuffers& buf, void* ws, size_t ws_bytes, const RouterDecodeScratch& dec,
                        cudaStream_t stream) {
    const int d = static_cast<int>(in.shape[1]);
    if (g.group_limit || (g.logit_bias_profile && g.logit_bias != nullptr) || debug_forward_enabled())
        return false;
    if (n == 1 && compute == QType::F16 && gate.qtype == QType::F16 && buf.pool != nullptr &&
        in.stride[0] == d && gate.shape[1] == d)
        return router_decode_route(in, gate, ne, top_k, use_sigmoid, norm_weights, score_bias, buf, dec,
                                   stream);
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
    static LogOnce logged;  // activity proof, once
    if (logged.first())
        IMP_LOG_INFO("fused MoE router ACTIVE (n=%d ne=%d d=%d top_k=%d)", n, ne, d, top_k);
    return true;
}

}  // namespace imp
