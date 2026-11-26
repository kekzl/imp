// Inkling2-Loop per-slot recurrent slab (exec/inkling_slab_layout.h, B-111 S3): what one slot holds, where
// each application's conv state and K/V ring sit in it, and which slot a request's rows reach.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <utility>
#include <vector>

#include "exec/inkling_slab_layout.h"
#include "exec/inkling_slot_table.h"

using namespace imp;

namespace {

// Inkling-2-Mini's shape: 24 layers, 6..17 looped three times, full attention on 5, 11, 17 and 23.
InklingConfig mini() {
    InklingConfig c;
    c.global.assign(static_cast<size_t>(c.n_layers), false);
    for (int l : {5, 11, 17, 23})
        c.global[static_cast<size_t>(l)] = true;
    return c;
}

constexpr size_t kNone = InklingSlabLayout::kNone;

// Inkling-2-Mini's mtp_config: 3 layers, 0 and 2 local, 4 taps.
InklingDraftConfig mini_draft() {
    InklingDraftConfig d;
    d.taps = {5, 23, 41, 48};
    d.local = {true, false, true};
    return d;
}
constexpr int kVocab = 201024;

// The bytes the forward allocated per slot before S3: one conv block and a ring of window + segment +
// slack rows per sliding application, K and V. The slab holds exactly these, no padding.
TEST(InklingSlabLayout, SlotHoldsTheConvStateAndFortySlidingRings) {
    const InklingConfig c = mini();
    const InklingSlabLayout l = inkling_slab_layout(c, 8192);
    EXPECT_EQ(l.ring, 1024 + 1024 + 5120);
    EXPECT_EQ(l.kv_row_halves, 1024u);
    EXPECT_EQ(l.conv_app_floats, 3u * (2 * 1024 + 2 * 2304));
    EXPECT_EQ(l.conv_bytes, 48u * 19968 * 4);
    int sliding = 0;
    for (size_t k : l.k_off)
        sliding += k != kNone ? 1 : 0;
    EXPECT_EQ(sliding, 40);
    EXPECT_EQ(l.ring_bytes(), 7168u * 1024 * 2);
    EXPECT_EQ(l.slot_bytes, l.conv_bytes + 80 * l.ring_bytes());
    EXPECT_EQ(l.slot_bytes, 1178238976u);
    EXPECT_EQ(l.total_bytes(2), 2356477952u);
    EXPECT_EQ(l.total_bytes(0), 0u);
    std::printf("inkling slab: %zu B per slot (%.2f MiB), 2 slots %zu B\n", l.slot_bytes,
                l.slot_bytes / 1048576.0, l.total_bytes(2));
    // A call of fewer rows than a segment shrinks the ring by the difference.
    EXPECT_EQ(inkling_slab_layout(c, 512).ring, 1024 + 512 + 5120);
}

// Conv state at offset 0; every ring aligned, inside the slot, disjoint from the conv block and every
// other ring; full-attention applications hold none (their K/V are pool layers).
TEST(InklingSlabLayout, RegionsAreAlignedAndDisjoint) {
    const InklingConfig c = mini();
    const InklingSlabLayout l = inkling_slab_layout(c, 1024);
    ASSERT_EQ(static_cast<int>(l.k_off.size()), c.n_apps());
    ASSERT_EQ(static_cast<int>(l.v_off.size()), c.n_apps());
    EXPECT_EQ(l.slot_bytes % InklingSlabLayout::kAlign, 0u);
    std::vector<std::pair<size_t, size_t>> regions = {{0, l.conv_bytes}};
    for (int a = 0; a < c.n_apps(); ++a) {
        const size_t i = static_cast<size_t>(a);
        const bool global = c.global[static_cast<size_t>(c.app_layer(a))];
        EXPECT_EQ(l.k_off[i] == kNone, global) << "app " << a;
        EXPECT_EQ(l.v_off[i] == kNone, global) << "app " << a;
        for (size_t off : {l.k_off[i], l.v_off[i]}) {
            if (off == kNone)
                continue;
            EXPECT_EQ(off % InklingSlabLayout::kAlign, 0u) << "app " << a;
            EXPECT_LE(off + l.ring_bytes(), l.slot_bytes) << "app " << a;
            regions.emplace_back(off, off + l.ring_bytes());
        }
    }
    std::sort(regions.begin(), regions.end());
    for (size_t i = 1; i < regions.size(); ++i)
        EXPECT_LE(regions[i - 1].second, regions[i].first) << "region " << i << " overlaps";
}

// A request's rows reach the slab region of the slot the table bound it to: slot s at s strides, none
// past the slab; a released slot is the one the next request binds (its region kept for the prefix match).
TEST(InklingSlabLayout, BoundRequestReachesItsSlotRegion) {
    const InklingSlabLayout l = inkling_slab_layout(mini(), 1024);
    constexpr int kSlots = 2;
    InklingSlotTable t(kSlots);
    const auto any = [](size_t) { return 0; };
    const int s7 = t.bind(7, any), s9 = t.bind(9, any);
    ASSERT_EQ(s7, 0);
    ASSERT_EQ(s9, 1);
    EXPECT_EQ(l.slot_offset(t.slot_of(7), kSlots), 0u);
    EXPECT_EQ(l.slot_offset(t.slot_of(9), kSlots), l.slot_bytes);
    EXPECT_EQ(l.slot_offset(kSlots, kSlots), kNone);
    EXPECT_EQ(l.slot_offset(-1, kSlots), kNone);
    EXPECT_EQ(l.slot_offset(t.find(5), kSlots), kNone);  // an unbound id reaches no region
    const int ids[2] = {9, 7};
    std::vector<int> rows;
    t.rows_to_slots(ids, 2, rows);
    EXPECT_EQ(l.slot_offset(rows[0], kSlots), l.slot_bytes);
    EXPECT_EQ(l.slot_offset(rows[1], kSlots), 0u);
    t.release(9);  // the teardown row inkling_slot
    EXPECT_EQ(t.bind(11, any), 1);
    EXPECT_EQ(l.slot_offset(t.slot_of(11), kSlots), l.slot_bytes);
}

// S3b: the bytes the first forward allocated per lookahead slot before (InklingDraft::init's 2 local rings
// of 2 x (window + block + cap + kSeg + kSlack) rows, enable_lookahead's 6 buffers, the runtime's
// [kGraphRows][vocab] logits) sit in one region; 224 B of it pad the 32 B token ids to 256.
TEST(InklingSlabLayout, LookRegionHoldsTheDraftRingsAndLookaheadPlanes) {
    const InklingConfig c = mini();
    const InklingDraftConfig d = mini_draft();
    const InklingSlabLayout l = inkling_slab_layout(c, 8192, &d, kVocab);
    const InklingLookLayout& k = l.look;
    EXPECT_EQ(k.draft_ring, 1024 + 8 + 1024 + 1024 + 5120);
    const size_t ring = 2 * 8200 * 1024 * 2, H = 2304, n = 48 * 19968, R = 8;
    EXPECT_EQ(4 * ring, 134348800u);
    const size_t before = 4 * ring + 4 * 1024 * H * 4 + 2 * n * 4 + n / 3 * R * 4 + H * R * 2 + R * 4 +
                          R * kVocab * 4;
    EXPECT_EQ(k.bytes, before + 224);
    EXPECT_EQ(k.bytes, 196458752u);
    EXPECT_EQ(l.look_slots(2), 2);
    EXPECT_EQ(l.look_slots(4), kInklingLookaheadSlots);
    EXPECT_EQ(l.total_bytes(2), 2356477952u);  // the slots alone: S3a's bytes
    EXPECT_EQ(l.total_bytes(2, 2), 2356477952u + 2 * 196458752u);
    std::printf("inkling slab: lookahead region %zu B (%.2f MiB), 2 slots + 2 regions %zu B\n", k.bytes,
                k.bytes / 1048576.0, l.total_bytes(2, 2));
    // No draft (speculative.mtp_k = 0, no mtp.safetensors): no region, S3a's slab.
    const InklingSlabLayout off = inkling_slab_layout(c, 8192);
    EXPECT_EQ(off.look.bytes, 0u);
    EXPECT_EQ(off.look_slots(2), 0);
    EXPECT_EQ(off.slot_bytes, l.slot_bytes);
    // The draft ring follows the call's rows as InklingDraft's cap = min(max_tokens, kSeg) does.
    EXPECT_EQ(inkling_slab_layout(c, 512, &d, kVocab).look.draft_ring, 1024 + 8 + 512 + 1024 + 5120);
}

// Every plane and ring of a region is aligned, inside it and disjoint from the others; the full-context
// draft layer (the pool's) holds none.
TEST(InklingSlabLayout, LookRegionPlanesAreAlignedAndDisjoint) {
    const InklingDraftConfig d = mini_draft();
    const InklingLookLayout k = inkling_slab_layout(mini(), 1024, &d, kVocab).look;
    ASSERT_EQ(k.dk.size(), 3u);
    EXPECT_EQ(k.dk[1], kNone);
    EXPECT_EQ(k.dv[1], kNone);
    const size_t ring = 2 * static_cast<size_t>(k.draft_ring) * 1024 * 2, H = 2304, n = 48 * 19968;
    std::vector<std::pair<size_t, size_t>> regions = {
        {k.feat, 4 * 1024 * H * 4},        {k.conv_pre, n * 4},   {k.conv_tmp, n * 4},
        {k.conv_log, n / 3 * 32},          {k.look_h, H * 8 * 2}, {k.ids, 8 * 4},
        {k.logits, 8 * size_t{kVocab} * 4}};
    for (size_t i : {0u, 2u})
        regions.insert(regions.end(), {{k.dk[i], ring}, {k.dv[i], ring}});
    for (auto& [off, len] : regions) {
        EXPECT_EQ(off % InklingSlabLayout::kAlign, 0u) << "at " << off;
        EXPECT_LE(off + len, k.bytes) << "at " << off;
        len += off;
    }
    EXPECT_EQ(k.bytes % InklingSlabLayout::kAlign, 0u);
    std::sort(regions.begin(), regions.end());
    for (size_t i = 1; i < regions.size(); ++i)
        EXPECT_LE(regions[i - 1].second, regions[i].first) << "plane " << i << " overlaps";
}

// Slot s < kInklingLookaheadSlots has its region after every slot's stride, the others none; a slab
// allocated without regions (the card took the slots only) has none at all.
TEST(InklingSlabLayout, LookRegionsSitAfterEverySlot) {
    const InklingDraftConfig d = mini_draft();
    const InklingSlabLayout l = inkling_slab_layout(mini(), 1024, &d, kVocab);
    for (int slots = 1; slots <= 4; ++slots) {
        const int looks = l.look_slots(slots);
        EXPECT_EQ(looks, std::min(slots, 2));
        for (int s = 0; s < slots; ++s) {
            const size_t off = l.look_offset(s, slots, looks);
            if (s >= looks) {
                EXPECT_EQ(off, kNone) << slots << " slots, slot " << s;
                continue;
            }
            EXPECT_EQ(off, l.total_bytes(slots) + static_cast<size_t>(s) * l.look.bytes);
            EXPECT_LE(off + l.look.bytes, l.total_bytes(slots, looks));
            EXPECT_EQ(off % InklingSlabLayout::kAlign, 0u);
        }
        EXPECT_EQ(l.look_offset(0, slots, 0), kNone);
        EXPECT_EQ(l.look_offset(-1, slots, looks), kNone);
    }
}

}  // namespace
