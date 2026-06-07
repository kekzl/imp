#pragma once

#include "core/qtype.h"
#include "core/tensor_kind.h"
#include "core/storage_tier.h"

#include <cstdint>
#include <string>
#include <unordered_map>
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
        const void* source_data = nullptr;  // GGUF/source pointer — the key the
                                            // pre-dequant phases dispatch on.
        // Stage 1: arch rule (gemma-3) requires the NVFP4 decode cache to be
        // built FROM an FP16 companion copy, not from scratch. When true, an
        // NVFP4-tier entry also keeps an FP16 cache entry alive.
        bool fp16_companion = false;
    };
    std::vector<Entry> entries;
    size_t projected_vram_bytes = 0;
    bool failed = false;
    std::string failure_reason;

    // O(1) lookup of the planned tier for a source pointer. Returns
    // StorageTier::Undefined if the pointer is not in the plan (e.g. native
    // GGUF blocks that bypass the overlay layer). Built lazily on first call.
    StorageTier tier_of(const void* src) const;
    const Entry* entry_of(const void* src) const;

  private:
    mutable std::unordered_map<const void*, const Entry*> by_src_;
    mutable bool index_built_ = false;
    void build_index_() const;
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
