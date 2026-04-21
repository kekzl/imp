#pragma once

#include "imp/tensor_kind.h"
#include "imp/storage_tier.h"

#include <cstdint>
#include <string>
#include <vector>

namespace imp {

class Model;
struct ModelConfig;

struct PlanHints {
    bool   prefer_nvfp4_decode = false;
    bool   dual_path_attn_fp8_ffn_nvfp4 = false;
    size_t vram_budget_bytes = 0;
};

struct StoragePlan {
    struct Entry {
        TensorID    id;
        TensorKind  kind;
        StorageTier tier;
        int64_t     bytes;
        int64_t     rows;
        int64_t     cols;
    };
    std::vector<Entry> entries;
    size_t projected_vram_bytes = 0;
    bool   failed = false;
    std::string failure_reason;
};

// Pure function — no GPU allocations, no side effects. Determines per-tensor
// storage tier based on capabilities + budget + hints.
StoragePlan plan_storage(const Model& model, const ModelConfig& cfg,
                         const PlanHints& hints);

} // namespace imp
