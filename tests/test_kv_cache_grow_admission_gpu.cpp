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

// Idle release: a pool grown for a large request gives its free tail back. The cached prefix
// below stays where it is, the released blocks are no longer handed out, device memory drops,
// and a later growth maps them again, zeroed.
TEST(KVCacheIdleReleaseTest, FreeTailGoesBackAndTheCachedPrefixStays) {
    SKIP_IF_NO_CUDA();
    auto cache = std::make_unique<KVCache>(/*n_layers=*/4, /*n_kv_heads=*/8, /*head_dim=*/128, QType::F16,
                                           /*max_blocks=*/64, /*block_size=*/16, /*alloc=*/nullptr,
                                           /*ceiling_blocks=*/65536);
    if (!cache->growable())
        GTEST_SKIP() << "no VMM backend on this device";
    KVCache* kvc = cache.get();
    KVCacheManager mgr(std::move(cache));
    mgr.set_prefix_caching_enabled(true);
    kvc->start_grow_ahead();  // the start size (64 blocks) becomes the floor
    // 16001 blocks: the committed tail of a layer region does not end on a backend granule, as in
    // a serving pool (the backend rounds a mapping out and releases whole driver allocations only).
    const int grown = kvc->try_grow_to_now(16001);
    ASSERT_EQ(grown, 16001);
    make_cached_seq(&mgr, 0, 200, 1000);  // blocks 0..199 cached
    ASSERT_EQ(mgr.num_reclaimable_cached_blocks(), 200);
    // A pattern in a cached block: it has to survive the release.
    const size_t probe_bytes = 4096;
    ASSERT_EQ(cudaMemset(kvc->k_ptr(0, 199), 0x5A, probe_bytes), cudaSuccess);
    size_t free0 = 0, free1 = 0, total = 0;
    ASSERT_EQ(cudaMemGetInfo(&free0, &total), cudaSuccess);

    const int released = kvc->release_idle_tail();
    const int kept = grown - released;
    ASSERT_GT(released, 8192) << "the tail above the cached prefix and one grow-ahead step";
    EXPECT_GE(kept, 200);
    EXPECT_EQ(kvc->total_blocks(), kept);
    EXPECT_EQ(kvc->idle_releases(), 1u);
    EXPECT_EQ(mgr.num_reclaimable_cached_blocks(), 200);
    ASSERT_EQ(cudaMemGetInfo(&free1, &total), cudaSuccess);
    // 4 layers x K and V x 8 heads x 128 x 16 tokens x 2 bytes = 262144 bytes per block.
    const size_t per_block = size_t{4} * 2 * 8 * 128 * 16 * 2;
    EXPECT_GE(free1, free0 + static_cast<size_t>(released) * per_block * 9 / 10);
    std::vector<unsigned char> probe(probe_bytes);
    ASSERT_EQ(cudaMemcpy(probe.data(), kvc->k_ptr(0, 199), probe_bytes, cudaMemcpyDeviceToHost), cudaSuccess);
    for (unsigned char c : probe)
        ASSERT_EQ(c, 0x5A);
    // Every block the pool still hands out lies below the kept count.
    std::vector<int> ids;
    for (int id = kvc->allocate_block(); id >= 0; id = kvc->allocate_block())
        ids.push_back(id);
    ASSERT_EQ(static_cast<int>(ids.size()), kept - 200);
    for (int id : ids)
        ASSERT_LT(id, kept);
    for (int id : ids)
        kvc->free_block(id);
    // Growing again maps the tail and hands it out zeroed.
    ASSERT_EQ(kvc->try_grow_to_now(grown), grown);
    ASSERT_EQ(cudaMemcpy(probe.data(), kvc->k_ptr(3, grown - 1), probe_bytes, cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (unsigned char c : probe)
        ASSERT_EQ(c, 0);
    EXPECT_EQ(kvc->release_idle_tail(), released) << "a second idle period releases the same tail";
}

// Nothing to give back: a pool at its start size, and a pool whose last block is referenced.
TEST(KVCacheIdleReleaseTest, AReferencedTailAndTheStartSizeStay) {
    SKIP_IF_NO_CUDA();
    auto cache = std::make_unique<KVCache>(/*n_layers=*/2, /*n_kv_heads=*/4, /*head_dim=*/64, QType::F16,
                                           /*max_blocks=*/64, /*block_size=*/16, /*alloc=*/nullptr,
                                           /*ceiling_blocks=*/65536);
    if (!cache->growable())
        GTEST_SKIP() << "no VMM backend on this device";
    KVCache* kvc = cache.get();
    kvc->start_grow_ahead();
    EXPECT_EQ(kvc->release_idle_tail(), 0) << "start size is the floor";
    const int grown = kvc->try_grow_to_now(32768);
    ASSERT_EQ(grown, 32768);
    std::vector<int> ids;
    for (int id = kvc->allocate_block(); id >= 0; id = kvc->allocate_block())
        ids.push_back(id);
    ASSERT_EQ(static_cast<int>(ids.size()), grown);
    // Only the highest block stays referenced: nothing above it, so nothing is released.
    int top = -1;
    for (int id : ids)
        top = std::max(top, id);
    for (int id : ids)
        if (id != top)
            kvc->free_block(id);
    EXPECT_EQ(kvc->release_idle_tail(), 0);
    EXPECT_EQ(kvc->total_blocks(), grown);
    kvc->free_block(top);
    EXPECT_GT(kvc->release_idle_tail(), 0);
}
