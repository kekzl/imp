#pragma once

// A handle's cache entries (audit D-3): bound once from the maps, checked against its tiers.

#include "core/weight_handle.h"
#include "exec/weight_caches.h"

#include <type_traits>

namespace imp {

// Binds h.entries to the entries the source-keyed maps hold under h.source_data (null / false
// where a map has none). Map nodes are stable after Phase 4c (they change only in free_buffers);
// Phase 4b / 4c fill released_fp8_view / q8_planes_only after registration and rebind.
inline void bind_cache_entries(WeightHandle& h, const WeightCaches& wc) {
    h.entries = {};
    if (h.source_data == nullptr)
        return;
    const auto entry = [&](const auto& map) -> const typename std::decay_t<decltype(map)>::mapped_type* {
        const auto it = map.find(h.source_data);
        return it != map.end() ? &it->second : nullptr;
    };
    h.entries.fp16 = entry(wc.fp16);
    h.entries.fp8 = entry(wc.fp8);
    h.entries.nvfp4 = entry(wc.nvfp4);
    h.entries.nvfp4_sfatom = entry(wc.nvfp4_sfatom);
    h.entries.nvfp4_prefill = entry(wc.nvfp4_sfatom_prefill);
    h.entries.mxfp8_prefill = entry(wc.mxfp8_sfatom_prefill);
    h.entries.mxfp4_sfatom = entry(wc.mxfp4_sfatom);
    h.entries.released_fp8 = entry(wc.released_fp8_view);
    h.entries.q8_planes_only = wc.q8_planes_only.contains(h.source_data);
}

// The NVFP4 decode-cache entry of a handle that may be absent (no handle = no data = no entry).
[[nodiscard]] inline const NvFP4CacheEntry* nvfp4_entry_of(const WeightHandle* h) {
    return h != nullptr ? h->entries.nvfp4 : nullptr;
}

// The first tier of `h` whose dispatch reads a cache entry that h.entries lacks, or nullptr.
// Checked: NVFP4 primary / decode (nvfp4), FP8 decode / prefill (fp8). An FP16 prefill tier
// without an fp16 entry is a designed state (the source is the weight), not checked.
[[nodiscard]] inline const char* missing_tier_entry(const WeightHandle& h) {
    const StorageTier decode = h.decode_tier != StorageTier::Undefined ? h.decode_tier : h.primary_tier;
    const StorageTier prefill = h.prefill_tier != StorageTier::Undefined ? h.prefill_tier : h.primary_tier;
    if ((h.primary_tier == StorageTier::NVFP4_LINEAR || decode == StorageTier::NVFP4_LINEAR) &&
        h.entries.nvfp4 == nullptr)
        return "nvfp4";
    if ((decode == StorageTier::FP8 || prefill == StorageTier::FP8) && h.entries.fp8 == nullptr)
        return "fp8";
    return nullptr;
}

}  // namespace imp
