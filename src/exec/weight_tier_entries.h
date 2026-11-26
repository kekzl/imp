#pragma once

// A handle's cache entries (audit D-3): bound once from the maps, checked against its tiers.

#include "core/weight_handle.h"
#include "exec/weight_caches.h"

namespace imp {

// Binds h.entries to the fp16 / fp8 / nvfp4 entries keyed by h.source_data (null where a map has
// none). The map nodes are stable after Phase 4 (they change only in free_buffers).
inline void bind_cache_entries(WeightHandle& h, const WeightCaches& wc) {
    h.entries = {};
    if (h.source_data == nullptr)
        return;
    if (const auto it = wc.fp16.find(h.source_data); it != wc.fp16.end())
        h.entries.fp16 = &it->second;
    if (const auto it = wc.fp8.find(h.source_data); it != wc.fp8.end())
        h.entries.fp8 = &it->second;
    if (const auto it = wc.nvfp4.find(h.source_data); it != wc.nvfp4.end())
        h.entries.nvfp4 = &it->second;
}

// The first tier of `h` whose dispatch reads a cache entry that h.entries lacks, or nullptr.
// Checked: NVFP4 primary / decode (nvfp4), FP8 decode / prefill (fp8). An FP16 prefill tier
// without an fp16 entry is a designed state (the source is the weight), not checked.
[[nodiscard]] inline const char* missing_tier_entry(const WeightHandle& h) {
    const StorageTier decode = h.decode_tier != StorageTier::Undefined ? h.decode_tier : h.primary_tier;
    const StorageTier prefill = h.prefill_tier != StorageTier::Undefined ? h.prefill_tier : h.primary_tier;
    if ((h.primary_tier == StorageTier::NVFP4 || decode == StorageTier::NVFP4) && h.entries.nvfp4 == nullptr)
        return "nvfp4";
    if ((decode == StorageTier::FP8 || prefill == StorageTier::FP8) && h.entries.fp8 == nullptr)
        return "fp8";
    return nullptr;
}

}  // namespace imp
