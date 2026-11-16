#pragma once
// Decode (M=1) MoE router for sigmoid routers wider than moe_router_fused takes (ne > 128, Flint 256):
// router GEMV (fp32 logits, bit-identical to gemv_gate_fp32) + sigmoid (+ selection bias) + top-k + renorm
// in one launch; the last CTA to finish its rows selects. Replaces gemv_gate_fp32 + topk_gating_kernel.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

struct MoeRouterDecodeArgs {
    const half* w = nullptr;  // [ne, K] router weight
    const half* x = nullptr;  // [K] router input
    float* logits = nullptr;  // [ne] fp32 logits, written as gemv_gate_fp32 does
    int ne = 0;               // <= 512
    int K = 0;                // 1024, 2048, 3072 or 4096
    int top_k = 0;            // 1..16, <= ne
    bool normalize = true;
    const half* score_bias = nullptr;  // [ne], selection only
    int32_t* ids = nullptr;            // [top_k]
    float* weights = nullptr;          // [top_k]
    unsigned int* ticket = nullptr;    // one counter, zeroed once, self-resetting
};

// False = shape refused; the caller keeps gemv_gate_fp32 + moe_topk_gating. Ties pick the lowest expert.
[[nodiscard]] bool moe_router_decode(const MoeRouterDecodeArgs& args, cudaStream_t stream);

}  // namespace imp
