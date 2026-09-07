// The serving metadata pool (engine_kv_cache_init.cpp) hands every forward
// path a fixed region of one allocation so nothing allocates while serving
// (invariant I2). The engine relies on three things the layout function
// promises: regions are 256-byte aligned, they do not overlap, and they end
// inside `total`. Mutation: drop the rounding in `region()` and Aligned fails;
// shrink any region below its byte count and Disjoint fails.

#include "runtime/serving_metadata_layout.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

namespace {

using imp::ServingMetadataLayout;

struct Region {
    const char* name;
    size_t off;
    size_t bytes;
};

std::vector<Region> regions(const ServingMetadataLayout& l, int rows, int seq_cap, int bt_cap, bool swa,
                            size_t scratch) {
    const size_t r = static_cast<size_t>(rows), s = static_cast<size_t>(seq_cap);
    const size_t bt = static_cast<size_t>(bt_cap) * sizeof(int);
    const size_t swa_bt = swa ? bt : 0;
    return {
        {"pf_tok", l.pf_tok, r * 4},
        {"pf_pos", l.pf_pos, r * 4},
        {"pf_bt", l.pf_bt, bt},
        {"pf_bt_swa", l.pf_bt_swa, swa_bt},
        {"pf_ctx", l.pf_ctx, 4},
        {"rg_tok", l.rg_tok, r * 4},
        {"rg_pos", l.rg_pos, r * 4},
        {"rg_bt", l.rg_bt, s * bt},
        {"rg_ctx", l.rg_ctx, s * 4},
        {"rg_soff", l.rg_soff, (s + 1) * 4},
        {"rg_slots", l.rg_slots, s * 4},
        {"gl_bt", l.gl_bt, bt},
        {"gl_bt_swa", l.gl_bt_swa, swa_bt},
        {"agl_bt", l.agl_bt, bt},
        {"agl_bt_swa", l.agl_bt_swa, swa_bt},
        {"cp_bt", l.cp_bt, bt},
        {"cp_token", l.cp_token, scratch},
        {"cp_pos", l.cp_pos, 4},
        {"cp_ctx", l.cp_ctx, 4},
    };
}

void check(int rows, int seq_cap, int bt_cap, bool swa, size_t scratch) {
    const auto l = ServingMetadataLayout::compute(rows, seq_cap, bt_cap, swa, scratch);
    auto rs = regions(l, rows, seq_cap, bt_cap, swa, scratch);
    for (const auto& x : rs) {
        EXPECT_EQ(x.off % ServingMetadataLayout::kAlign, 0u) << x.name;
        EXPECT_LE(x.off + x.bytes, l.total) << x.name;
    }
    std::sort(rs.begin(), rs.end(), [](const Region& a, const Region& b) { return a.off < b.off; });
    for (size_t i = 1; i < rs.size(); ++i) {
        // Non-empty regions must not overlap the previous non-empty one.
        size_t j = i;
        while (j > 0 && rs[j - 1].bytes == 0)
            --j;
        if (j == 0 || rs[i].bytes == 0)
            continue;
        EXPECT_GE(rs[i].off, rs[j - 1].off + rs[j - 1].bytes) << rs[j - 1].name << " -> " << rs[i].name;
    }
}

TEST(ServingMetadataLayout, Aligned) {
    const auto l = ServingMetadataLayout::compute(131072, 64, 8192, true, 4096);
    for (size_t off : {l.pf_tok,     l.pf_pos, l.pf_bt,    l.pf_bt_swa, l.pf_ctx, l.rg_tok,    l.rg_pos,
                       l.rg_bt,      l.rg_ctx, l.rg_soff,  l.rg_slots,  l.gl_bt,  l.gl_bt_swa, l.agl_bt,
                       l.agl_bt_swa, l.cp_bt,  l.cp_token, l.cp_pos,    l.cp_ctx, l.total})
        EXPECT_EQ(off % 256, 0u);
}

TEST(ServingMetadataLayout, Disjoint) {
    check(131072, 64, 8192, true, 4096);
    check(131072, 64, 8192, false, 4096);
    check(4096, 1, 16, false, 64);
    check(2048, 32, 1000, true, 4096);  // bt_cap not a multiple of 64 ints
}

TEST(ServingMetadataLayout, SizeIsSmall) {
    // The largest default config: 128k rows, 64 members, 8192-block tables.
    // Twice 1 MiB of rows, 2 MiB of ragged tables, the rest in KiB.
    const auto l = ServingMetadataLayout::compute(131072, 64, 8192, true, 4096);
    EXPECT_LT(l.total, 8u << 20);
    EXPECT_GT(l.total, 4u << 20);
}

TEST(ServingMetadataLayout, SwaOffAddsNothing) {
    const auto on = ServingMetadataLayout::compute(4096, 8, 256, true, 4096);
    const auto off = ServingMetadataLayout::compute(4096, 8, 256, false, 4096);
    EXPECT_EQ(on.total - off.total, 3u * 1024);  // three 1 KiB mirrors (256 ints each)
    EXPECT_EQ(off.pf_bt_swa, off.pf_ctx);        // an empty region has no extent
}

}  // namespace
