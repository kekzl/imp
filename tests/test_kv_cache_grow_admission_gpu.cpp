// Growable KV pool on the admission path: a refused growth is not asked again per block
// (cfg::kKvGrowRetryMs), and the grow-ahead step covers the last admission. GPU (VMM) tests,
// test-kv.

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "memory/kv_cache.h"
#include "memory/kv_cache_manager.h"
#include "memory/vram_query.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

#include "test_cuda_skip.h"

using namespace imp;

namespace {

// n full 16-token blocks, hashed, freed: all of them cached and reclaimable.
void make_cached_seq(KVCacheManager* mgr, int seq_id, int n_full_blocks, int token_base) {
    std::vector<int32_t> tokens(static_cast<size_t>(n_full_blocks) * 16);
    std::iota(tokens.begin(), tokens.end(), token_base);
    ASSERT_EQ(mgr->allocate_blocks_with_prefix(seq_id, tokens), 0);
    mgr->register_block_hashes(seq_id, tokens);
    mgr->free_sequence(seq_id);
}

// A pool capped by free VRAM below its ceiling takes a prompt's blocks from the prefix cache. At
// retry 0 every block off the empty free list read free VRAM first (Gemma-4 pp8192: 334-503
// readings, 22.9-36.0 ms queue); with the window one refused reading covers the request.
TEST(KVCacheGrowAdmissionTest, CappedPoolReclaimsWithoutAFreeVramReadPerBlock) {
    SKIP_IF_NO_CUDA();
    for (const int retry_ms : {0, 100}) {
        SCOPED_TRACE(retry_ms);
        auto cache = std::make_unique<KVCache>(/*n_layers=*/2, /*n_kv_heads=*/4, /*head_dim=*/64, QType::F16,
                                               /*max_blocks=*/8, /*block_size=*/16, /*alloc=*/nullptr,
                                               /*ceiling_blocks=*/4096);
        if (!cache->growable())
            GTEST_SKIP() << "no VMM backend on this device";
        KVCache* kvc = cache.get();
        kvc->set_grow_retry_ms(retry_ms);
        KVCacheManager mgr(std::move(cache));
        mgr.set_prefix_caching_enabled(true);
        vram_budget_install(16);  // a 16 MiB card: the pool caps far below its 4096-block ceiling
        const int capped = kvc->try_grow_to_now(4096);
        ASSERT_GT(capped, 64);
        ASSERT_LT(capped, 4096);
        // Every block cached and reclaimable, the free list empty: the warm prefix-cache server.
        make_cached_seq(&mgr, 0, capped, 1000);
        ASSERT_EQ(mgr.num_free_blocks(), 0);
        ASSERT_EQ(mgr.num_reclaimable_cached_blocks(), capped);
        ASSERT_TRUE(mgr.can_allocate(capped / 2)) << "the request fits on reclaimable blocks";
        const uint64_t reads0 = kvc->free_vram_queries();
        const auto t0 = std::chrono::steady_clock::now();
        ASSERT_TRUE(mgr.allocate_blocks(1, capped / 2));
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const uint64_t reads = kvc->free_vram_queries() - reads0;
        vram_budget_install(0);
        EXPECT_EQ(kvc->total_blocks(), capped) << "nothing was affordable: the pool must not grow";
        if (retry_ms == 0) {
            EXPECT_GE(reads, static_cast<uint64_t>(capped / 4)) << "retry 0: one reading per reclaimed block";
        } else {
            EXPECT_LE(reads, 1u) << "a refused growth was asked again for every reclaimed block";
            EXPECT_LT(ms, 5.0) << capped / 2 << " reclaimable blocks took " << ms << " ms to admit";
        }
        mgr.free_sequence(1);
    }
}

// A pool with 100 free blocks after a 150-block admission wakes the worker for the next 150 (a
// second prompt that size), though 100 is above the 64-block step. No admission: no wake, the
// second prompt grew on the admission path (Gemma-4 pp8192 fresh: 21.0-36.3 ms queue).
TEST(KVCacheGrowAdmissionTest, GrowAheadStepCoversTheLastAdmission) {
    SKIP_IF_NO_CUDA();
    for (const bool admitted : {false, true}) {
        SCOPED_TRACE(admitted);
        KVCache cache(/*n_layers=*/2, /*n_kv_heads=*/4, /*head_dim=*/64, QType::F16, /*max_blocks=*/250,
                      /*block_size=*/16, /*alloc=*/nullptr, /*ceiling_blocks=*/1024);
        if (!cache.growable())
            GTEST_SKIP() << "no VMM backend on this device";
        const int have = cache.total_blocks();
        cache.start_grow_ahead();
        std::vector<int> held;
        while (cache.num_free_blocks() > 100)
            held.push_back(cache.allocate_block());
        if (admitted)
            cache.note_admitted(150);
        cache.grow_ahead_tick();
        for (int i = 0; i < 400 && cache.growths_ahead() == 0 && admitted; i++)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if (admitted) {
            ASSERT_EQ(cache.growths_ahead(), 1u) << "the step ignored the last admission";
            EXPECT_GE(cache.adopt_grown(), have + 150);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            EXPECT_EQ(cache.growths_ahead(), 0u) << "100 free is above max(64, 25 %): no wake";
        }
        for (int b : held)
            cache.free_block(b);
    }
}

}  // namespace
