#pragma once

// Native MoE NVFP4 copy (B-135): an expert block it releases takes its nvfp4 and nvfp4_sfatom
// entries along, so no entry stays keyed by, or views, a freed block.
#include "exec/weight_caches.h"

namespace imp {

// Erases the entries keyed by `block` in both maps; returns how many (0..2). An entry releases
// only what it owns: an nvfp4 view (Phase 0b) nothing, an owned SfAtom buffer once.
inline int erase_released_expert_keys(WeightCaches& wc, const void* block) {
    return static_cast<int>(wc.nvfp4.erase(block) + wc.nvfp4_sfatom.erase(block));
}

}  // namespace imp
