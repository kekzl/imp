// WeightHandle::entries (audit D-3): the cache entries a handle's tiers read are bound once from
// the maps, and Phase 4 refuses a decode / prefill tier whose entry is missing. CPU lane.

#include <gtest/gtest.h>

#include "core/weight_handle.h"
#include "exec/moe_ffn_context.h"
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

// Slice 2: the SfAtom / prefill-copy maps and the Phase 4b / 4c release maps bind too; a bind
// taken before 4b / 4c keyed the released handle sees none of them until the rebind.
TEST(WeightTierEntries, BindTakesTheCopyAndReleaseMaps) {
    WeightCaches wc;
    wc.nvfp4_sfatom.emplace(&g_src_a, imp::NvFP4SfAtomCacheEntry{});
    wc.nvfp4_sfatom_prefill.emplace(&g_src_a, imp::NvFP4SfAtomCacheEntry{});
    wc.mxfp8_sfatom_prefill.emplace(&g_src_a, imp::MxFP8SfAtomCacheEntry{});
    wc.mxfp4_sfatom.emplace(&g_src_a, imp::MxFP4SfAtomCacheEntry{});
    WeightHandle h = handle(&g_src_a, StorageTier::FP16, StorageTier::Undefined, StorageTier::Undefined);
    imp::bind_cache_entries(h, wc);
    EXPECT_EQ(h.entries.nvfp4_sfatom, &wc.nvfp4_sfatom.at(&g_src_a));
    EXPECT_EQ(h.entries.nvfp4_prefill, &wc.nvfp4_sfatom_prefill.at(&g_src_a));
    EXPECT_EQ(h.entries.mxfp8_prefill, &wc.mxfp8_sfatom_prefill.at(&g_src_a));
    EXPECT_EQ(h.entries.mxfp4_sfatom, &wc.mxfp4_sfatom.at(&g_src_a));
    EXPECT_EQ(h.entries.released_fp8, nullptr);
    EXPECT_FALSE(h.entries.q8_planes_only);

    // Phase 4b / 4c key the release maps by handle id; the rebind after 4c binds only those.
    h.id = 7;
    wc.released_fp8_view.emplace(7, imp::FP8CacheEntry{});
    wc.q8_planes_only.insert(7);
    EXPECT_EQ(h.entries.released_fp8, nullptr);  // stale until the rebind after Phase 4c
    imp::bind_release_entries(h, wc);
    EXPECT_EQ(h.entries.released_fp8, &wc.released_fp8_view.at(7));
    EXPECT_TRUE(h.entries.q8_planes_only);
    EXPECT_EQ(h.entries.nvfp4_sfatom, &wc.nvfp4_sfatom.at(&g_src_a));  // source-keyed binds kept

    // Another handle on the same (freed, reused) source pointer inherits no release (audit D-3).
    WeightHandle reuse = handle(&g_src_a, StorageTier::FP16, StorageTier::Undefined, StorageTier::Undefined);
    reuse.id = 8;
    imp::bind_cache_entries(reuse, wc);
    EXPECT_EQ(reuse.entries.released_fp8, nullptr);
    EXPECT_FALSE(reuse.entries.q8_planes_only);

    // Another source sees none of them.
    WeightHandle other = handle(&g_src_b, StorageTier::FP16, StorageTier::Undefined, StorageTier::Undefined);
    imp::bind_cache_entries(other, wc);
    EXPECT_EQ(other.entries.nvfp4_sfatom, nullptr);
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

// Phase 4 reads a handle's tiers from its bound entries, not from which map holds the source.
TEST(WeightTierEntries, TiersComeFromTheBoundEntries) {
    using imp::QType;
    constexpr auto U = StorageTier::Undefined, F16 = StorageTier::FP16, F8 = StorageTier::FP8;
    constexpr auto SF = StorageTier::NVFP4_SFATOM, LIN = StorageTier::NVFP4_LINEAR;
    const imp::Fp16CacheEntry fp16;
    imp::FP8CacheEntry fp8;
    const imp::NvFP4CacheEntry nvfp4;
    const imp::NvFP4SfAtomCacheEntry sfatom;
    const imp::MxFP4SfAtomCacheEntry mxfp4;
    const float row_scales[1] = {1.0f};
    auto expect = [](const imp::HandleTiers& t, StorageTier p, StorageTier f, StorageTier d) {
        EXPECT_EQ(t.primary, p);
        EXPECT_EQ(t.prefill, f);
        EXPECT_EQ(t.decode, d);
    };
    imp::WeightCacheRefs e;
    expect(imp::plan_handle_tiers(e, QType::Q4_K, true), U, U, U);
    e.fp16 = &fp16;
    expect(imp::plan_handle_tiers(e, QType::Q4_K, true), F16, F16, F16);
    // GGUF source: a per-tensor FP8 entry is primary, a per-row one a decode sidecar.
    e = {};
    e.fp8 = &fp8;
    expect(imp::plan_handle_tiers(e, QType::Q4_K, true), F8, F8, F8);
    fp8.d_row_scales = row_scales;
    expect(imp::plan_handle_tiers(e, QType::Q4_K, true), F16, F16, F8);
    expect(imp::plan_handle_tiers(e, QType::NVFP4, false), F8, F8, F8);
    fp8.d_row_scales = nullptr;
    // F16 source: always the sidecar.
    expect(imp::plan_handle_tiers(e, QType::F16, false), F16, F16, F8);
    // Native FP8 source: primary alone, the sidecar beside an FP16 copy.
    fp8.native_source = true;
    expect(imp::plan_handle_tiers(e, QType::FP8_E4M3, false), F8, F8, F8);
    e.fp16 = &fp16;
    expect(imp::plan_handle_tiers(e, QType::FP8_E4M3, false), F16, F16, F8);
    fp8.native_source = false;
    // SfAtom NVFP4 leads; its FP16 copy takes prefill, else FP8 both, else NVFP4 decode.
    e = {};
    e.nvfp4_sfatom = &sfatom;
    e.mxfp4_sfatom = &mxfp4;
    e.nvfp4 = &nvfp4;
    expect(imp::plan_handle_tiers(e, QType::NVFP4, false), SF, SF, LIN);
    e.fp8 = &fp8;
    expect(imp::plan_handle_tiers(e, QType::NVFP4, false), SF, F8, F8);
    e.fp16 = &fp16;
    expect(imp::plan_handle_tiers(e, QType::NVFP4, false), SF, F16, SF);
    e.nvfp4_sfatom = nullptr;
    expect(imp::plan_handle_tiers(e, QType::MXFP4, false), StorageTier::MXFP4, StorageTier::MXFP4,
           StorageTier::MXFP4);
    e.mxfp4_sfatom = nullptr;
    expect(imp::plan_handle_tiers(e, QType::Q4_K, true), LIN, LIN, LIN);
}

// The legacy MoE fallback reads the packed-stack entry Phase 4 bound per projection.
TEST(WeightTierEntries, MoeEntryIsTheLayersBoundProjection) {
    imp::TransformerLayer ly;
    EXPECT_EQ(imp::nvfp4_moe_entry(ly, imp::ExpertProj::Up), nullptr);
    const imp::NvFP4MoEQuantResult gate, up, down;
    ly.nvfp4_moe_gate_ptr = &gate;
    ly.nvfp4_moe_up_ptr = &up;
    ly.nvfp4_moe_down_ptr = &down;
    EXPECT_EQ(imp::nvfp4_moe_entry(ly, imp::ExpertProj::Gate), &gate);
    EXPECT_EQ(imp::nvfp4_moe_entry(ly, imp::ExpertProj::Up), &up);
    EXPECT_EQ(imp::nvfp4_moe_entry(ly, imp::ExpertProj::Down), &down);
}

}  // namespace
