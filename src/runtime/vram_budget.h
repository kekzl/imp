#pragma once

#include "model/model.h"
#include "memory/kv_cache.h"
#include <cstddef>

namespace imp {

struct EngineConfig;  // forward declaration

// VRAM budget for weight cache allocation (computed by Engine::plan_vram_budget).
// Replaces ad-hoc "remaining_budget" with per-phase caps computed upfront.
struct VRAMBudget {
    enum Strategy { FP8_PREFILL_NVFP4_DECODE, NVFP4_DECODE_ONLY, FP16_ONLY };
    Strategy strategy = FP16_ONLY;
    size_t kv_cache_bytes = 0;
    size_t fp8_cache_bytes = 0;  // 0 for sub-8-bit models
    size_t nvfp4_cache_bytes = 0;
    size_t reserve_bytes = 1024ULL * 1024 * 1024;  // 1 GiB safety
    int kv_max_blocks = 0;
    bool nvfp4_second_pass = false;  // true → re-run NVFP4 after FP16-Free
};

// Pure computation: plan VRAM allocation split between KV cache, FP8 prefill
// cache, and NVFP4 decode cache based on model characteristics and config.
// No GPU allocation — just arithmetic.
VRAMBudget compute_vram_budget(const Model& model, const EngineConfig& config, int n_kv_layers, int head_dim,
                               size_t free_vram);

}  // namespace imp
