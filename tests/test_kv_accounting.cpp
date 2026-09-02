// Prefix-cache and block-accounting behaviour, WITHOUT a GPU.
//
// Why this file exists, and why it is not more tests in test_kv_cache.cpp:
// that file's manager tests build a real KVCache, whose constructor allocates
// VRAM and throws without a device, so every one of them opens with
// SKIP_IF_NO_CUDA() and lives in `test-kv` - a binary CI never runs, because
// there is no GPU runner (docs/DESIGN_DECISIONS.md, "No GPU runner in CI").
//
// Mutation testing on 2026-09-02 measured what that costs. Of the 21 host-side
// mutants, the merge gate caught 15; all five survivors were in
// kv_cache_manager.cpp, and four of them are real faults with a plain
// failure mode:
//
//   M35  content_salt dropped from the chain: two prompts with identical token
//        ids but different images share KV.
//   M36  reuse no longer stops at the first miss: a hit after a hole is shared,
//        leaving uncomputed KV inside the range prefill was told to skip.
//   M38  the probe reports one block more than the chain reaches.
//   M40  reclaiming a cached block leaves its hash entry pointing at a block
//        that is back in the free list.
//
// KVCache::for_accounting() removes the only obstacle: the id space, the free
// list and the ref counts never needed the pool (BlockPool's open_slots mode
// is documented as "the caller owns the memory"). Each test below fails on one
// of those four mutants and runs in `ctest -L unit`.

#include "core/tensor.h"
#include "memory/kv_cache.h"
#include "memory/kv_cache_manager.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

using namespace imp;

namespace {

constexpr int kBlock = kKVBlockSize;  // 16 tokens per block

std::unique_ptr<KVCacheManager> MakeAccountingManager(int max_blocks) {
    auto cache = KVCache::for_accounting(/*n_layers=*/2, /*n_kv_heads=*/4, /*head_dim=*/64, QType::F16,
                                         max_blocks);
    auto mgr = std::make_unique<KVCacheManager>(std::move(cache));
    mgr->set_prefix_caching_enabled(true);
    return mgr;
}

// n full blocks of distinct ids, starting at `first`.
std::vector<int32_t> Tokens(int blocks, int32_t first) {
    std::vector<int32_t> t(static_cast<size_t>(blocks) * kBlock);
    std::iota(t.begin(), t.end(), first);
    return t;
}

// Fill the cache with `blocks` cached blocks from one sequence and return the
// tokens that produced them.
std::vector<int32_t> CacheOneSequence(KVCacheManager& mgr, int seq_id, int blocks, int32_t first,
                                      size_t salt = 0) {
    auto tokens = Tokens(blocks, first);
    EXPECT_GE(mgr.allocate_blocks_with_prefix(seq_id, tokens, /*max_reuse_blocks=*/-1, salt), 0);
    mgr.register_block_hashes(seq_id, tokens, salt);
    mgr.free_sequence(seq_id);
    return tokens;
}

}  // namespace

// The cache this file is built on is the real accounting path, not a stub: same
// geometry arithmetic, same id space, and it refuses to hand out memory.
TEST(KVAccounting, ForAccountingHasTheSameGeometryAndNoMemory) {
    auto cache = KVCache::for_accounting(2, 4, 64, QType::F16, 8);
    EXPECT_TRUE(cache->accounting_only());
    EXPECT_EQ(cache->n_layers(), 2);
    EXPECT_EQ(cache->n_kv_heads(), 4);
    EXPECT_EQ(cache->head_dim(), 64);
    EXPECT_EQ(cache->block_size(), kBlock);
    // 16 tokens * 4 heads * 64 dims * 2 bytes (F16)
    EXPECT_EQ(cache->block_bytes(), static_cast<size_t>(kBlock) * 4 * 64 * 2);
    EXPECT_EQ(cache->num_free_blocks(), 8);
}

// M35. content_salt seeds the hash chain, and allocate_blocks_with_prefix has
// to seed from the SAME value register_block_hashes used. Dropping it there
// (parent_hash = 0) is invisible to a text-only test, because 0 is the text
// salt: it only shows up when a salt is actually in play, which is every
// multimodal request (identical image-token ids, different pictures).
TEST(KVAccounting, SaltSeedsTheChainOnLookupToo) {
    auto mgr = MakeAccountingManager(16);
    constexpr size_t kSaltA = 0xA1A1A1A1ull;
    constexpr size_t kSaltB = 0xB2B2B2B2ull;

    auto tokens = CacheOneSequence(*mgr, 0, /*blocks=*/3, /*first=*/100, kSaltA);
    ASSERT_EQ(mgr->num_cached_blocks(), 3);

    // Same tokens, same picture: the whole prefix is reusable.
    EXPECT_EQ(mgr->allocate_blocks_with_prefix(1, tokens, -1, kSaltA), 3)
        << "lookup did not seed the chain with content_salt, so it missed its own blocks";
    mgr->free_sequence(1);

    // Same tokens, different picture: nothing may be shared.
    EXPECT_EQ(mgr->allocate_blocks_with_prefix(2, tokens, -1, kSaltB), 0)
        << "a different content_salt shared KV with the first request's picture";
    mgr->free_sequence(2);
}

// M38. The probe is what the hybrid snapshot path picks a restore boundary
// from, so an off-by-one there means restoring from a block whose KV was never
// computed. It must count the cached chain exactly, at every length.
TEST(KVAccounting, ProbeCountsTheCachedChainExactly) {
    auto mgr = MakeAccountingManager(16);
    std::vector<size_t> chain;

    auto tokens = Tokens(3, 200);
    EXPECT_EQ(mgr->longest_cached_prefix_blocks(tokens, chain), 0);
    EXPECT_EQ(chain.size(), 3u) << "the chain is a property of the tokens, not of the cache";

    // One, two, then three cached blocks: the probe tracks each step.
    for (int blocks = 1; blocks <= 3; ++blocks) {
        auto mgr_n = MakeAccountingManager(16);
        auto toks = CacheOneSequence(*mgr_n, 0, blocks, /*first=*/200);
        ASSERT_EQ(mgr_n->num_cached_blocks(), blocks);

        // Probe the same prefix plus one uncached block on top.
        auto longer = Tokens(blocks + 1, 200);
        EXPECT_EQ(mgr_n->longest_cached_prefix_blocks(longer, chain), blocks)
            << "probe disagrees with the cache at " << blocks << " cached block(s)";
    }
}

// M36. Reuse must stop at the first miss. A hole in the middle is not
// hypothetical: LRU reclaims the OLDEST cached block, which is the first block
// of the oldest sequence, so the surviving tail is exactly the shape that
// tempts a non-contiguous reuse. The caller skips prefill for
// reused * block_size tokens, so sharing block 1 after missing block 0 leaves
// uncomputed KV inside the skipped range.
TEST(KVAccounting, ReuseStopsAtTheFirstMiss) {
    auto mgr = MakeAccountingManager(16);
    auto tokens = CacheOneSequence(*mgr, 0, /*blocks=*/3, /*first=*/300);
    ASSERT_EQ(mgr->num_cached_blocks(), 3);

    // Reclaim the oldest cached block: block 0 of the chain.
    ASSERT_TRUE(mgr->evict_cached_block());
    ASSERT_EQ(mgr->num_cached_blocks(), 2);

    const int reused = mgr->allocate_blocks_with_prefix(1, tokens);
    EXPECT_EQ(reused, 0) << "reuse continued past a hole: blocks 1..2 were shared while block 0 was "
                            "reclaimed, so prefill would skip tokens whose KV was never written";
    EXPECT_EQ(static_cast<int>(mgr->block_table(1).size()), 3);
    mgr->free_sequence(1);
}

// M40. Reclaiming a cached block has to drop its hash entry with it. Left
// behind, the table points at a block that is back in the free list, and the
// probe answers for KV that no longer exists.
TEST(KVAccounting, ReclaimDropsTheHashEntryWithTheBlock) {
    auto mgr = MakeAccountingManager(16);
    auto tokens = CacheOneSequence(*mgr, 0, /*blocks=*/3, /*first=*/400);
    ASSERT_EQ(mgr->num_cached_blocks(), 3);

    std::vector<size_t> chain;
    ASSERT_EQ(mgr->longest_cached_prefix_blocks(tokens, chain), 3);

    ASSERT_TRUE(mgr->evict_cached_block());  // takes the chain's first block

    EXPECT_EQ(mgr->longest_cached_prefix_blocks(tokens, chain), 0)
        << "the probe still reports a cached prefix whose first block was reclaimed";
}
