// Inkling2-Loop global K/V in the paged pool (exec/inkling_kv_layout.h, B-111 S2b): which pool layer each
// layer application writes, how much prefix a request may reuse, how large the pool gets.

#include <gtest/gtest.h>

#include <memory>
#include <set>
#include <vector>

#include "exec/inkling_kv_layout.h"
#include "memory/kv_cache.h"
#include "memory/kv_cache_manager.h"

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

TEST(InklingKvLayout, GlobalAppsOwnOneKvLayerEach) {
    const InklingConfig c = mini();
    const InklingKvLayout l = inkling_kv_apps(c, true);
    ASSERT_EQ(static_cast<int>(l.app_layer.size()), c.n_apps());
    EXPECT_EQ(c.n_apps(), 48);
    // 5 and 23 once, 11 and 17 in each of the three passes.
    EXPECT_EQ(l.n_global, 8);
    EXPECT_EQ(l.draft_layer, 8);
    EXPECT_EQ(l.n_layers(), 9);
    std::set<int> seen;
    int last = -1;
    for (int a = 0; a < c.n_apps(); ++a) {
        const int p = l.app_layer[static_cast<size_t>(a)];
        EXPECT_EQ(p >= 0, static_cast<bool>(c.global[static_cast<size_t>(c.app_layer(a))])) << "app " << a;
        if (p < 0)
            continue;
        EXPECT_GT(p, last) << "app order";
        last = p;
        seen.insert(p);
    }
    EXPECT_EQ(static_cast<int>(seen.size()), l.n_global);
    EXPECT_EQ(*seen.rbegin(), l.n_global - 1);
    EXPECT_EQ(l.app_layer[5], 0);
    EXPECT_EQ(l.app_layer[47], 7);  // layer 17, third pass
    // Without the draft the pool holds the applications only.
    const InklingKvLayout n = inkling_kv_apps(c, false);
    EXPECT_EQ(n.draft_layer, -1);
    EXPECT_EQ(n.n_layers(), 8);
    EXPECT_EQ(n.app_layer, l.app_layer);
    // app_layer inverts app_index over every pass.
    for (int layer = 0; layer < c.n_layers; ++layer)
        for (int pass = 0; pass < (c.is_loop(layer) ? c.passes : 1); ++pass)
            EXPECT_EQ(c.app_layer(c.app_index(layer, pass)), layer);
}

TEST(InklingPrefixCap, NeverPastSlotMatch) {
    constexpr int bs = 16, grid = 256;
    for (int match : {0, 100, 256, 511, 512, 4096})
        for (int cached : {0, 1, 15, 16, 31, 32, 300})
            for (int cap : {0, 16, 40, 1000}) {
                const int b = inkling_prefix_cap_blocks(match, cached, cap, bs, grid);
                EXPECT_GE(b, 0);
                EXPECT_LE(b * bs, match);
                EXPECT_LE(b, cached);
                EXPECT_LE(b, cap);
                EXPECT_EQ(b * bs % grid, 0) << match << " " << cached << " " << cap;
            }
    // The largest grid position all three allow.
    EXPECT_EQ(inkling_prefix_cap_blocks(512, 300, 1000, bs, grid), 32);
    EXPECT_EQ(inkling_prefix_cap_blocks(4096, 31, 1000, bs, grid), 16);  // 496 cached tokens -> 256
    EXPECT_EQ(inkling_prefix_cap_blocks(4096, 300, 40, bs, grid), 32);   // prompt cap 640 -> 512
    EXPECT_EQ(inkling_prefix_cap_blocks(255, 300, 1000, bs, grid), 0);
    // A grid that is not a multiple of the block reuses nothing.
    EXPECT_EQ(inkling_prefix_cap_blocks(512, 300, 1000, 24, grid), 0);
}

// The S2b acceptance start (2 slots x 81920, 32 GB card): 9137 MiB free, the I8B32 copies 655 MiB, the
// pool's ceiling 5761 MiB. Charging the pool dropped the copies: other texts, decode 236 -> 206 tok/s.
TEST(InklingDense8, FitDoesNotChargeTheKvPool) {
    constexpr size_t MiB = size_t{1} << 20;
    EXPECT_TRUE(inkling_dense8_fits(9137 * MiB, 655 * MiB));
    EXPECT_FALSE(inkling_dense8_fits(9137 * MiB - 5761 * MiB, 655 * MiB));
    EXPECT_FALSE(inkling_dense8_fits(655 * MiB + kInklingDense8Spare - 1, 655 * MiB));
    EXPECT_TRUE(inkling_dense8_fits(655 * MiB + kInklingDense8Spare, 655 * MiB));
}

TEST(InklingPrefixCap, PoolHoldsEverySlotAtMaxSeqLenPlusTheLookaheadRows) {
    EXPECT_EQ(inkling_pool_blocks(2, 81920, 16), 2 * 5121);
    EXPECT_EQ(inkling_pool_blocks(2, 32768, 16), 2 * 2049);
    EXPECT_EQ(inkling_pool_blocks(1, 9, 16), 1);
    EXPECT_EQ(inkling_pool_blocks(1, 10, 16), 2);
    EXPECT_EQ(inkling_pool_blocks(3, 100, 0), 0);
}

// A snapshot saved at pos finds the request whose prefix cache probe hashes the same blocks (B-111 S4): its
// key is the chain hash of block pos / 16 - 1 under the request's salt, blind to the tokens after pos.
TEST(InklingKvLayout, SnapshotKeyIsThePrefixCacheHashOfItsLastBlock) {
    auto kvm = std::make_unique<KVCacheManager>(KVCache::for_accounting(9, 8, 128, QType::F16, 64));
    std::vector<int32_t> t(600);
    for (size_t i = 0; i < t.size(); ++i)
        t[i] = static_cast<int32_t>(i * 31 % 997);
    const size_t salt = 0x5eedu;
    const InklingBlockHash h = &KVCacheManager::compute_block_hash;
    for (size_t s : {size_t{0}, salt}) {
        std::vector<size_t> hashes;
        (void)kvm->longest_cached_prefix_blocks(t, hashes, s);
        ASSERT_EQ(hashes.size(), 37u);
        EXPECT_EQ(inkling_snapshot_key(t, 512, 16, s, h), hashes[31]) << "salt " << s;
        EXPECT_EQ(inkling_snapshot_key(t, 256, 16, s, h), hashes[15]) << "salt " << s;
    }
    const size_t k = inkling_snapshot_key(t, 512, 16, salt, h);
    EXPECT_NE(k, inkling_snapshot_key(t, 512, 16, 0, h));
    std::vector<int32_t> u = t;
    u[600 - 1] += 1;  // after pos: the same prefix
    EXPECT_EQ(inkling_snapshot_key(u, 512, 16, salt, h), k);
    u[3] += 1;  // inside the first block
    EXPECT_NE(inkling_snapshot_key(u, 512, 16, salt, h), k);
    EXPECT_EQ(inkling_snapshot_key(t, 500, 16, salt, h), 0u);  // off the block grid
    EXPECT_EQ(inkling_snapshot_key(t, 608, 16, salt, h), 0u);  // past the tokens
    EXPECT_EQ(inkling_snapshot_key(t, 0, 16, salt, h), 0u);
}

}  // namespace
