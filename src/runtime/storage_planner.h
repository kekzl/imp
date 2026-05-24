#pragma once

#include "core/qtype.h"
#include "core/tensor_kind.h"
#include "core/storage_tier.h"

#include <cstdint>
#include <string>
#include <vector>

namespace imp {

class Model;
struct ModelConfig;

struct PlanHints {
    bool prefer_nvfp4_decode = false;
    bool dual_path_attn_fp8_ffn_nvfp4 = false;
    // When true, prefer FP8 over FP16 for weight cache (mirrors WeightCaches::use_fp8).
    bool prefer_fp8 = false;
    size_t vram_budget_bytes = 0;
};

struct StoragePlan {
    struct Entry {
        TensorID id;
        TensorKind kind;
        QType source_qtype;  // source storage; feeds effective_capabilities() in downgrade loop
        StorageTier tier;
        int64_t bytes;
        int64_t rows;
        int64_t cols;
    };
    std::vector<Entry> entries;
    size_t projected_vram_bytes = 0;
    bool failed = false;
    std::string failure_reason;
};

// Pure function — no GPU allocations, no side effects. Determines per-tensor
// storage tier based on capabilities + budget + hints.
//
// SCOPE (Phase 4 Option C, decided 2026-04-24): the plan describes the
// **overlay layer** — tensors whose storage tier is a runtime decision
// (NVFP4 decode cache, FP8 prefill cache, CUTLASS layouts). Native GGUF
// block formats (Q4_K_M, Q5_K_M, Q6_K, Q8_0, MXFP4) stay as mmap'd blocks
// owned by `Model::gpu_allocations_` and are dequantized per-kernel call.
// They bypass the plan/registry entirely.
//
// Today the plan enumerates the **ideal** overlay (every quantize-able
// tensor at its preferred tier). The runtime caching policy in
// `pre_dequant_weights` is VRAM-budget-aware and may decide not to cache
// some entries. The Phase-4 parity diagnostic surfaces the resulting
// gap as informational; it is not an error.
StoragePlan plan_storage(const Model& model, const ModelConfig& cfg, const PlanHints& hints);

}  // namespace imp
