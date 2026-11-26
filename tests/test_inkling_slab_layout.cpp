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

}  // namespace
