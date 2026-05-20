#pragma once

#include "exec/executor.h"  // VRAMBudget
#include "model/model.h"
#include "memory/kv_cache.h"
#include <cstddef>

namespace imp {

struct EngineConfig;  // forward declaration

// Pure computation: plan VRAM allocation split between KV cache, FP8 prefill
// cache, and NVFP4 decode cache based on model characteristics and config.
// No GPU allocation — just arithmetic.
VRAMBudget compute_vram_budget(const Model& model, const EngineConfig& config, int n_kv_layers, int head_dim,
                               size_t free_vram);

}  // namespace imp
