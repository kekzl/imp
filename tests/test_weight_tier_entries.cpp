// WeightHandle::entries (audit D-3): the cache entries a handle's tiers read are bound once from
// the maps, and Phase 4 refuses a decode / prefill tier whose entry is missing. CPU lane.

#include <gtest/gtest.h>

#include "core/weight_handle.h"
#include "exec/weight_caches.h"
#include "exec/weight_tier_entries.h"

namespace {

using imp::StorageTier;
using imp::WeightCaches;
using imp::WeightHandle;

int g_src_a = 0, g_src_b = 0, g_src_none = 0;

WeightCaches caches() {
    WeightCaches wc;
    wc.fp16.emplace(&g_src_a, imp::Fp16CacheEntry{});
    wc.fp8.emplace(&g_src_a, imp::FP8CacheEntry{});
    wc.nvfp4.emplace(&g_src_a, imp::NvFP4CacheEntry{});
    wc.nvfp4.emplace(&g_src_b, imp::NvFP4CacheEntry{});
    return wc;
}

WeightHandle handle(const void* src, StorageTier primary, StorageTier decode, StorageTier prefill) {
    WeightHandle h;
    h.source_data = src;
    h.primary_tier = primary;
    h.decode_tier = decode;
    h.prefill_tier = prefill;
    return h;
}

TEST(WeightTierEntries, BindTakesTheMapNodesOfTheSource) {
    const WeightCaches wc = caches();
    WeightHandle h = handle(&g_src_a, StorageTier::FP16, StorageTier::FP8, StorageTier::FP16);
    imp::bind_cache_entries(h, wc);
    EXPECT_EQ(h.entries.fp16, &wc.fp16.at(&g_src_a));
    EXPECT_EQ(h.entries.fp8, &wc.fp8.at(&g_src_a));
    EXPECT_EQ(h.entries.nvfp4, &wc.nvfp4.at(&g_src_a));

    // A rebound source (layer offload) drops the entries the new pointer has none of.
    h.source_data = &g_src_b;
    imp::bind_cache_entries(h, wc);
    EXPECT_EQ(h.entries.fp16, nullptr);
    EXPECT_EQ(h.entries.fp8, nullptr);
    EXPECT_EQ(h.entries.nvfp4, &wc.nvfp4.at(&g_src_b));

    h.source_data = nullptr;
    imp::bind_cache_entries(h, wc);
    EXPECT_EQ(h.entries.nvfp4, nullptr);
}

// Slice 2: the CUTLASS / prefill-copy maps and the Phase 4b / 4c release maps bind too; a bind
// taken before 4b / 4c keyed the released source sees none of them until the rebind.
TEST(WeightTierEntries, BindTakesTheCopyAndReleaseMaps) {
    WeightCaches wc;
    wc.cutlass_nvfp4.emplace(&g_src_a, imp::CutlassNvFP4CacheEntry{});
    wc.cutlass_nvfp4_prefill.emplace(&g_src_a, imp::CutlassNvFP4CacheEntry{});
    wc.cutlass_mxfp8_prefill.emplace(&g_src_a, imp::CutlassMxFP8CacheEntry{});
    wc.cutlass_mxfp4.emplace(&g_src_a, imp::CutlassMxFP4CacheEntry{});
    WeightHandle h = handle(&g_src_a, StorageTier::FP16, StorageTier::Undefined, StorageTier::Undefined);
    imp::bind_cache_entries(h, wc);
    EXPECT_EQ(h.entries.cutlass_nvfp4, &wc.cutlass_nvfp4.at(&g_src_a));
    EXPECT_EQ(h.entries.nvfp4_prefill, &wc.cutlass_nvfp4_prefill.at(&g_src_a));
    EXPECT_EQ(h.entries.mxfp8_prefill, &wc.cutlass_mxfp8_prefill.at(&g_src_a));
    EXPECT_EQ(h.entries.cutlass_mxfp4, &wc.cutlass_mxfp4.at(&g_src_a));
    EXPECT_EQ(h.entries.released_fp8, nullptr);
    EXPECT_FALSE(h.entries.q8_planes_only);

    wc.released_fp8_view.emplace(&g_src_a, imp::FP8CacheEntry{});
    wc.q8_planes_only.insert(&g_src_a);
    EXPECT_EQ(h.entries.released_fp8, nullptr);  // stale until the rebind after Phase 4c
    imp::bind_cache_entries(h, wc);
    EXPECT_EQ(h.entries.released_fp8, &wc.released_fp8_view.at(&g_src_a));
    EXPECT_TRUE(h.entries.q8_planes_only);

    // Another source sees none of them.
    WeightHandle other = handle(&g_src_b, StorageTier::FP16, StorageTier::Undefined, StorageTier::Undefined);
    imp::bind_cache_entries(other, wc);
    EXPECT_EQ(other.entries.cutlass_nvfp4, nullptr);
    EXPECT_EQ(other.entries.mxfp8_prefill, nullptr);
    EXPECT_EQ(other.entries.released_fp8, nullptr);
    EXPECT_FALSE(other.entries.q8_planes_only);

    // The attention / FFN probes: an absent handle has no nvfp4 entry.
    const WeightCaches nv = caches();
    WeightHandle hb = handle(&g_src_b, StorageTier::FP16, StorageTier::Undefined, StorageTier::Undefined);
    imp::bind_cache_entries(hb, nv);
    EXPECT_EQ(imp::nvfp4_entry_of(&hb), &nv.nvfp4.at(&g_src_b));
    EXPECT_EQ(imp::nvfp4_entry_of(nullptr), nullptr);
}

TEST(WeightTierEntries, EveryReadTierNeedsItsEntry) {
    const WeightCaches wc = caches();
    auto bound = [&](const void* src, StorageTier p, StorageTier d, StorageTier f) {
        WeightHandle h = handle(src, p, d, f);
        imp::bind_cache_entries(h, wc);
        return h;
    };
    constexpr auto U = StorageTier::Undefined;
    // All entries present.
    EXPECT_EQ(imp::missing_tier_entry(
                  bound(&g_src_a, StorageTier::NVFP4_SFATOM, StorageTier::FP8, StorageTier::FP8)),
              nullptr);
    EXPECT_EQ(imp::missing_tier_entry(bound(&g_src_b, StorageTier::NVFP4_LINEAR, U, U)), nullptr);
    // NVFP4 decode (explicit or through the primary tier) without an nvfp4 entry.
    EXPECT_STREQ(imp::missing_tier_entry(
                     bound(&g_src_none, StorageTier::NVFP4_SFATOM, StorageTier::NVFP4_LINEAR, U)),
                 "nvfp4");
    EXPECT_STREQ(imp::missing_tier_entry(bound(&g_src_none, StorageTier::NVFP4_LINEAR, U, U)), "nvfp4");
    // FP8 decode or FP8 prefill without an fp8 entry.
    EXPECT_STREQ(imp::missing_tier_entry(bound(&g_src_b, StorageTier::FP16, StorageTier::FP8, U)), "fp8");
    EXPECT_STREQ(imp::missing_tier_entry(
                     bound(&g_src_b, StorageTier::NVFP4_SFATOM, StorageTier::NVFP4_LINEAR, StorageTier::FP8)),
                 "fp8");
    // FP16 prefill without an fp16 entry: the source is the weight, a designed state.
    EXPECT_EQ(imp::missing_tier_entry(bound(&g_src_none, StorageTier::FP16, U, U)), nullptr);
    EXPECT_EQ(imp::missing_tier_entry(bound(&g_src_none, StorageTier::Undefined, U, U)), nullptr);
}

}  // namespace
