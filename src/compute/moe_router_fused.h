#pragma once

// Fused MoE router (moe.router_fused): logits GEMM x W^T + softmax / sigmoid (+ selection bias) + top-k
// + renorm in one launch, writes only expert ids and weights (no fp16 logits, no split-K reduce, no
// conversion, no top-k launch). Semantics of topk_gating_kernel; ties pick the lowest expert index.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace imp {

struct MoeRouterFusedArgs {
    const half* x = nullptr;  // [M, K] row-major, 16 B aligned rows
    const half* w = nullptr;  // [ne, K] router weight
    int M = 0;
    int K = 0;
    int ne = 0;  // <= 128, % 8 == 0
    int top_k = 0;
    bool use_sigmoid = false;
    bool normalize = true;
    const half* score_bias = nullptr;  // [ne], selection only (DeepSeek / Nemotron-H)
    int32_t* ids = nullptr;            // [M, top_k]
    float* weights = nullptr;          // [M, top_k]
};

// ws: per-tile split-K tickets (zeroed once at allocation, self-resetting) + fp32 partials.
// False = refused (shape, alignment, workspace): the caller keeps its GEMM + top-k path.
[[nodiscard]] bool moe_router_fused(const MoeRouterFusedArgs& args, void* ws, size_t ws_bytes,
                                    cudaStream_t stream);

// Test / bench hook: force the row tile (16, 32, 64) and the K split (1..16); 0 = heuristic.
void moe_router_fused_tune(int bm, int split);

// Workspace that serves every M up to max_rows at the largest split the heuristic picks.
[[nodiscard]] size_t moe_router_fused_workspace_bytes(int max_rows);

}  // namespace imp
