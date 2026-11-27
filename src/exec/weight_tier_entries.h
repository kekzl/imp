#pragma once

// A handle's cache entries (audit D-3): bound once from the maps, checked against its tiers.

#include "core/weight_handle.h"
#include "exec/weight_caches.h"

#include <type_traits>

namespace imp {

// Binds the release entries Phase 4b / 4c key by handle id, never by the source pointer they free
// (a later allocation can reuse that address).
inline void bind_release_entries(WeightHandle& h, const WeightCaches& wc) {
    const auto it = wc.released_fp8_view.find(h.id);
    h.entries.released_fp8 = it != wc.released_fp8_view.end() ? &it->second : nullptr;
    h.entries.q8_planes_only = wc.q8_planes_only.contains(h.id);
}

// Binds h.entries to the entries the source-keyed maps hold under h.source_data (null / false
// where a map has none) and the release entries under h.id. Map nodes are stable after Phase 4c
// (they change only in free_buffers).
inline void bind_cache_entries(WeightHandle& h, const WeightCaches& wc) {
    h.entries = {};
    bind_release_entries(h, wc);
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
}

// The tier the bound entries offer: SfAtom NVFP4 > MXFP4 > linear NVFP4 > FP8 > FP16 > none.
[[nodiscard]] inline StorageTier cached_tier(const WeightCacheRefs& e) {
    if (e.nvfp4_sfatom != nullptr)
        return StorageTier::NVFP4_SFATOM;
    if (e.mxfp4_sfatom != nullptr)
        return StorageTier::MXFP4;
    if (e.nvfp4 != nullptr)
        return StorageTier::NVFP4_LINEAR;
    if (e.fp8 != nullptr)
        return StorageTier::FP8;
    if (e.fp16 != nullptr)
        return StorageTier::FP16;
    return StorageTier::Undefined;
}

struct HandleTiers {
    StorageTier primary = StorageTier::Undefined;
    StorageTier prefill = StorageTier::Undefined;  // M > 1
    StorageTier decode = StorageTier::Undefined;   // M = 1
};

// Phase 4's tiers of a handle, from its bound entries (audit D-3). An FP8 entry is a decode sidecar
// (FP16 primary / prefill, FP8 decode) on an F16 source, on a native-FP8 source beside an FP16 copy,
// and with per-row scales on a dequantable GGUF source; else it is primary.
[[nodiscard]] inline HandleTiers plan_handle_tiers(const WeightCacheRefs& e, QType source_qtype,
                                                   bool source_dequantable) {
    const StorageTier tier = cached_tier(e);
    bool fp8_sidecar = false;
    if (tier == StorageTier::FP8) {
        if (e.fp8->native_source)
            fp8_sidecar = e.fp16 != nullptr;
        else if (source_qtype == QType::F16)
            fp8_sidecar = true;
        else if (source_dequantable)
            fp8_sidecar = e.fp8->d_row_scales != nullptr;
    }
    if (fp8_sidecar)
        return {.primary = StorageTier::FP16, .prefill = StorageTier::FP16, .decode = StorageTier::FP8};
    HandleTiers t{.primary = tier, .prefill = tier, .decode = tier};
    if (tier == StorageTier::NVFP4_SFATOM && e.fp16 != nullptr) {
        t.prefill = StorageTier::FP16;  // decode stays on the source NVFP4 GEMV
    } else if (tier == StorageTier::NVFP4_SFATOM && e.fp8 != nullptr) {
        t.prefill = StorageTier::FP8;
        t.decode = StorageTier::FP8;
    } else if (tier == StorageTier::NVFP4_SFATOM && e.nvfp4 != nullptr) {
        t.decode = StorageTier::NVFP4_LINEAR;
    }
    return t;
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
