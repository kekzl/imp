// Prefix-cache equivalence (TEST_AUDIT retired risk #7): existing KVCacheManager tests cover
// block-ID bookkeeping only. This covers correctness: (a) KV-DATA equivalence (reused block
// carries the SAME bytes), (b) eviction+refill never false-hits a stale block, (c) ref-count
// keep-alive survives one sharer freeing. Content oracle: bytes written directly to device
// memory against a zero-initialized pool, so a stale/wrong hit reads zeros and can't pass
// tautologically.

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "memory/kv_cache.h"
#include "memory/kv_cache_manager.h"
#include "core/tensor.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <cstdio>
#include <numeric>
#include <unordered_set>
#include <vector>

#include "test_cuda_skip.h"

namespace imp {
namespace {

// Real manager over a real (small) KV pool. F16 so k_ptr/v_ptr are plain
// contiguous device memory we can byte-copy in and out.
static std::unique_ptr<KVCacheManager> MakeManager(int max_blocks, int n_layers = 2,
                                                   int n_kv_heads = 2, int head_dim = 32) {
    auto cache = std::make_unique<KVCache>(n_layers, n_kv_heads, head_dim, QType::F16, max_blocks);
    return std::make_unique<KVCacheManager>(std::move(cache));
}

// Fill block `block_id`'s K and V regions (all layers) with a deterministic
// per-(layer, block) byte pattern. Returns nothing; pairs with ReadBlock.
static void WriteBlockPattern(KVCache* c, int block_id, uint8_t seed) {
    const size_t bb = c->block_bytes();
    std::vector<uint8_t> host(bb);
    for (int l = 0; l < c->n_layers(); ++l) {
        for (size_t i = 0; i < bb; ++i)
            host[i] = static_cast<uint8_t>(seed + l * 7 + (i & 0x3F));
        ASSERT_EQ(cudaMemcpy(c->k_ptr(l, block_id), host.data(), bb, cudaMemcpyHostToDevice),
                  cudaSuccess);
        for (size_t i = 0; i < bb; ++i)
            host[i] = static_cast<uint8_t>(seed + 100 + l * 7 + (i & 0x3F));
        ASSERT_EQ(cudaMemcpy(c->v_ptr(l, block_id), host.data(), bb, cudaMemcpyHostToDevice),
                  cudaSuccess);
    }
}

// Read block `block_id`'s K (k_or_v=0) or V (k_or_v=1) for one layer into host.
static std::vector<uint8_t> ReadBlock(KVCache* c, int block_id, int layer, int k_or_v) {
    const size_t bb = c->block_bytes();
    std::vector<uint8_t> host(bb);
    void* src = (k_or_v == 0) ? c->k_ptr(layer, block_id) : c->v_ptr(layer, block_id);
    EXPECT_EQ(cudaMemcpy(host.data(), src, bb, cudaMemcpyDeviceToHost), cudaSuccess);
    return host;
}

// The expected pattern bytes for a given (layer, k_or_v, seed) — recomputed
// independently of any device read.
static std::vector<uint8_t> ExpectedPattern(size_t bb, int layer, int k_or_v, uint8_t seed) {
    std::vector<uint8_t> e(bb);
    int base = (k_or_v == 0) ? 0 : 100;
    for (size_t i = 0; i < bb; ++i)
        e[i] = static_cast<uint8_t>(seed + base + layer * 7 + (i & 0x3F));
    return e;
}

// Two sequences with identical >=2-block prefixes: after seq A is cached, seq B's
// allocate_blocks_with_prefix must report exactly the FULL blocks as reused (FNV-1a chain
// through parent). 48 tokens / 16 = 3 full blocks -> 3 reused.
TEST(PrefixEquivTest, FullPrefixReuseCount) {
    SKIP_IF_NO_CUDA();
    auto mgr = MakeManager(16);
    mgr->set_prefix_caching_enabled(true);

    std::vector<int32_t> tokens(48);
    std::iota(tokens.begin(), tokens.end(), 1000);

    int reused_a = mgr->allocate_blocks_with_prefix(0, tokens);
    ASSERT_EQ(reused_a, 0) << "first sequence cannot reuse — cache is empty";
    mgr->register_block_hashes(0, tokens);
    mgr->free_sequence(0);
    ASSERT_EQ(mgr->num_cached_blocks(), 3);

    int reused_b = mgr->allocate_blocks_with_prefix(1, tokens);
    EXPECT_EQ(reused_b, 3) << "identical 3-full-block prefix must reuse all 3";
    mgr->free_sequence(1);
}

// ── (2) Exactly-one-block shared prefix ───────────────────────────────────
// Share the first block (16 identical tokens), differ in the second. The
// parent-chained hash breaks at block 2, so exactly ONE block reuses.
TEST(PrefixEquivTest, PartialOneBlockShared) {
    SKIP_IF_NO_CUDA();
    auto mgr = MakeManager(16);
    mgr->set_prefix_caching_enabled(true);

    std::vector<int32_t> a(32);
    std::iota(a.begin(), a.end(), 0);
    (void)mgr->allocate_blocks_with_prefix(0, a);  // seed: reuse count asserted below via a fresh seq
    mgr->register_block_hashes(0, a);
    mgr->free_sequence(0);
    ASSERT_EQ(mgr->num_cached_blocks(), 2);

    std::vector<int32_t> b(32);
    std::iota(b.begin(), b.begin() + 16, 0);     // same first block
    std::iota(b.begin() + 16, b.end(), 5000);    // different second block
    int reused = mgr->allocate_blocks_with_prefix(1, b);
    EXPECT_EQ(reused, 1) << "only the first block's hash matches; chain breaks after";
    mgr->free_sequence(1);
}

// ── (3) No common prefix → 0 reuse ────────────────────────────────────────
// Disjoint tokens from the very first block ⇒ no hash matches ⇒ 0 reuse.
TEST(PrefixEquivTest, NoCommonPrefixZeroReuse) {
    SKIP_IF_NO_CUDA();
    auto mgr = MakeManager(16);
    mgr->set_prefix_caching_enabled(true);

    std::vector<int32_t> a(48);
    std::iota(a.begin(), a.end(), 0);
    (void)mgr->allocate_blocks_with_prefix(0, a);  // seed: reuse count asserted below via a fresh seq
    mgr->register_block_hashes(0, a);
    mgr->free_sequence(0);

    std::vector<int32_t> b(48);
    std::iota(b.begin(), b.end(), 900000);  // completely different from block 0 on
    int reused = mgr->allocate_blocks_with_prefix(1, b);
    EXPECT_EQ(reused, 0);
    mgr->free_sequence(1);
}

// Heart of risk #7: write a known byte pattern into A's blocks, register+free A (cached),
// reuse the prefix for B, read B's blocks back. Pool is zero-initialized, so passing requires
// the SAME physical block with intact KV bytes, not just a matching reuse count.
TEST(PrefixEquivTest, ReusedBlockKVContentPreserved) {
    SKIP_IF_NO_CUDA();
    auto mgr = MakeManager(16);
    mgr->set_prefix_caching_enabled(true);
    KVCache* c = mgr->kv_cache();
    const size_t bb = c->block_bytes();

    std::vector<int32_t> tokens(32);  // 2 full blocks
    std::iota(tokens.begin(), tokens.end(), 7);
    ASSERT_EQ(mgr->allocate_blocks_with_prefix(0, tokens), 0);

    const std::vector<int> blocks_a = mgr->block_table(0);  // copy: A is about to be freed
    ASSERT_EQ(blocks_a.size(), 2u);
    WriteBlockPattern(c, blocks_a[0], /*seed=*/0x11);
    WriteBlockPattern(c, blocks_a[1], /*seed=*/0x22);

    mgr->register_block_hashes(0, tokens);
    mgr->free_sequence(0);

    int reused = mgr->allocate_blocks_with_prefix(1, tokens);
    ASSERT_EQ(reused, 2);
    const std::vector<int>& blocks_b = mgr->block_table(1);
    ASSERT_EQ(blocks_b.size(), 2u);

    // Same physical blocks (content-addressed reuse hands back the cached ids).
    EXPECT_EQ(blocks_b[0], blocks_a[0]);
    EXPECT_EQ(blocks_b[1], blocks_a[1]);

    // And the KV BYTES survived the free→cache→reuse round-trip, every layer.
    const uint8_t seeds[2] = {0x11, 0x22};
    for (int bi = 0; bi < 2; ++bi) {
        for (int l = 0; l < c->n_layers(); ++l) {
            EXPECT_EQ(ReadBlock(c, blocks_b[bi], l, /*K*/ 0),
                      ExpectedPattern(bb, l, 0, seeds[bi]))
                << "reused block " << bi << " layer " << l << " K bytes corrupted";
            EXPECT_EQ(ReadBlock(c, blocks_b[bi], l, /*V*/ 1),
                      ExpectedPattern(bb, l, 1, seeds[bi]))
                << "reused block " << bi << " layer " << l << " V bytes corrupted";
        }
    }
    mgr->free_sequence(1);
}

// A and B hold the shared prefix concurrently (ref_count=2). Freeing A must leave the block
// alive (ref_count back to 1, not returned to the pool) with KV bytes intact: the
// use-after-free / premature-free guard.
TEST(PrefixEquivTest, RefCountKeepsSharedBlockAliveForB) {
    SKIP_IF_NO_CUDA();
    auto mgr = MakeManager(16);
    mgr->set_prefix_caching_enabled(true);
    KVCache* c = mgr->kv_cache();
    const size_t bb = c->block_bytes();

    std::vector<int32_t> tokens(32);
    std::iota(tokens.begin(), tokens.end(), 50);
    ASSERT_EQ(mgr->allocate_blocks_with_prefix(0, tokens), 0);
    const std::vector<int> blocks_a = mgr->block_table(0);
    ASSERT_EQ(blocks_a.size(), 2u);
    WriteBlockPattern(c, blocks_a[0], 0x33);
    WriteBlockPattern(c, blocks_a[1], 0x44);
    mgr->register_block_hashes(0, tokens);

    // B reuses A's blocks WHILE A is still alive → ref_count == 2 (shared).
    int reused = mgr->allocate_blocks_with_prefix(1, tokens);
    ASSERT_EQ(reused, 2);
    const std::vector<int> blocks_b = mgr->block_table(1);
    ASSERT_EQ(blocks_b[0], blocks_a[0]);
    EXPECT_EQ(c->ref_count(blocks_a[0]), 2) << "block shared by A and B must be ref-counted twice";

    int free_before = mgr->num_free_blocks();
    mgr->free_sequence(0);  // A goes away

    // The shared blocks must NOT have returned to the pool (B still holds them).
    EXPECT_EQ(mgr->num_free_blocks(), free_before)
        << "shared blocks returned to pool while B still references them";
    EXPECT_GE(c->ref_count(blocks_b[0]), 1);
    EXPECT_GE(c->ref_count(blocks_b[1]), 1);

    // B's KV bytes are still the ones A wrote — no use-after-free overwrite.
    const uint8_t seeds[2] = {0x33, 0x44};
    for (int bi = 0; bi < 2; ++bi)
        for (int l = 0; l < c->n_layers(); ++l)
            EXPECT_EQ(ReadBlock(c, blocks_b[bi], l, 0), ExpectedPattern(bb, l, 0, seeds[bi]))
                << "B's shared block corrupted after A freed (use-after-free)";

    mgr->free_sequence(1);
}

// Cache a prefix, free it, force it out of the pool via new unrelated sequences, then
// re-allocate the ORIGINAL prefix: must report 0 reuse. Hash entries are removed on eviction,
// so it must not falsely hit a recycled/stale block.
TEST(PrefixEquivTest, EvictionThenRefillIsNewNotStaleHit) {
    SKIP_IF_NO_CUDA();
    auto mgr = MakeManager(4);  // tiny pool: 4 blocks total
    mgr->set_prefix_caching_enabled(true);

    // Cache a 2-block prefix, then free → 2 cached blocks, 2 free.
    std::vector<int32_t> p(32);
    std::iota(p.begin(), p.end(), 0);
    ASSERT_EQ(mgr->allocate_blocks_with_prefix(0, p), 0);
    mgr->register_block_hashes(0, p);
    mgr->free_sequence(0);
    ASSERT_EQ(mgr->num_cached_blocks(), 2);

    // A 4-block sequence needs the whole pool → the 2 cached prefix blocks must
    // be evicted (reclaimed) to satisfy it. Different tokens ⇒ 0 reuse here.
    std::vector<int32_t> big(64);
    std::iota(big.begin(), big.end(), 300000);
    int reused_big = mgr->allocate_blocks_with_prefix(1, big);
    ASSERT_EQ(reused_big, 0);
    ASSERT_EQ(static_cast<int>(mgr->block_table(1).size()), 4);
    EXPECT_EQ(mgr->num_cached_blocks(), 0) << "cached prefix must have been evicted to fit seq 1";
    mgr->register_block_hashes(1, big);  // register the new occupant
    mgr->free_sequence(1);

    // Now re-request the ORIGINAL prefix. Its hash entries were dropped on
    // eviction → must come back as NEW (0 reuse), never a stale hit on a block
    // that now belongs to a different token sequence.
    int reused_again = mgr->allocate_blocks_with_prefix(2, p);
    EXPECT_EQ(reused_again, 0)
        << "evicted prefix falsely re-hit a recycled block — STALE-BLOCK BUG";
    mgr->free_sequence(2);
}

// 40 tokens at block_size 16 = 2 full blocks + 1 partial (8 tokens). Only full blocks are
// cacheable, so re-requesting reuses exactly the 2 full blocks; the partial tail is always
// re-allocated fresh.
TEST(PrefixEquivTest, NonAlignedPrefixReusesFullBlocksOnly) {
    SKIP_IF_NO_CUDA();
    auto mgr = MakeManager(16);
    mgr->set_prefix_caching_enabled(true);

    std::vector<int32_t> tokens(40);
    std::iota(tokens.begin(), tokens.end(), 11);
    ASSERT_EQ(mgr->allocate_blocks_with_prefix(0, tokens), 0);
    ASSERT_EQ(static_cast<int>(mgr->block_table(0).size()), 3);  // 2 full + 1 partial
    mgr->register_block_hashes(0, tokens);
    mgr->free_sequence(0);
    // Only the 2 full blocks are cacheable; the partial tail is freed to pool.
    EXPECT_EQ(mgr->num_cached_blocks(), 2);

    int reused = mgr->allocate_blocks_with_prefix(1, tokens);
    EXPECT_EQ(reused, 2) << "2 full blocks reused; partial tail re-allocated fresh";
    EXPECT_EQ(static_cast<int>(mgr->block_table(1).size()), 3);
    mgr->free_sequence(1);
}

// LRU eviction removes cached blocks front-first, so an early chain block can be gone while
// later ones survive. Reuse must stop at the first miss, or the caller would skip prefill
// across a hole with uncomputed KV (silent garbage attention reads).
TEST(PrefixEquivTest, ChainHoleStopsReuse) {
    SKIP_IF_NO_CUDA();
    auto mgr = MakeManager(16);
    mgr->set_prefix_caching_enabled(true);

    std::vector<int32_t> tokens(48);  // 3 full blocks
    std::iota(tokens.begin(), tokens.end(), 42);
    ASSERT_EQ(mgr->allocate_blocks_with_prefix(0, tokens), 0);
    mgr->register_block_hashes(0, tokens);
    mgr->free_sequence(0);
    ASSERT_EQ(mgr->num_cached_blocks(), 3);

    // Evict exactly ONE cached block — LRU head = block 0 of the chain
    // (free_sequence pushes blocks to the cached-LRU in table order).
    ASSERT_TRUE(mgr->evict_cached_block());
    ASSERT_EQ(mgr->num_cached_blocks(), 2);

    // Blocks 1 and 2 of the chain still sit in the hash table, but the
    // prefix is broken at block 0 — nothing may be reused.
    int reused = mgr->allocate_blocks_with_prefix(1, tokens);
    EXPECT_EQ(reused, 0) << "hole in the chain counted as reused prefix — "
                            "caller would skip prefill over uncomputed KV";
    mgr->free_sequence(1);
}

// Hybrid models can only skip prefill up to the recurrent-snapshot position; blocks past the
// cap must be freshly allocated, never shared, because the continuation prefill will WRITE them.
TEST(PrefixEquivTest, MaxReuseBlocksCapsSharing) {
    SKIP_IF_NO_CUDA();
    auto mgr = MakeManager(16);
    mgr->set_prefix_caching_enabled(true);
    KVCache* c = mgr->kv_cache();

    std::vector<int32_t> tokens(48);  // 3 full blocks
    std::iota(tokens.begin(), tokens.end(), 7000);
    ASSERT_EQ(mgr->allocate_blocks_with_prefix(0, tokens), 0);
    const std::vector<int> blocks_a = mgr->block_table(0);
    mgr->register_block_hashes(0, tokens);
    mgr->free_sequence(0);

    int reused = mgr->allocate_blocks_with_prefix(1, tokens, /*max_reuse_blocks=*/1);
    EXPECT_EQ(reused, 1);
    const std::vector<int>& blocks_b = mgr->block_table(1);
    ASSERT_EQ(blocks_b.size(), 3u);
    EXPECT_EQ(blocks_b[0], blocks_a[0]) << "block below the cap must be the shared cached block";
    // Blocks past the cap are fresh allocations the continuation prefill may
    // write — they must not alias the cached chain blocks.
    EXPECT_NE(blocks_b[1], blocks_a[1]);
    EXPECT_NE(blocks_b[2], blocks_a[2]);
    EXPECT_EQ(c->ref_count(blocks_b[1]), 1);
    mgr->free_sequence(1);
}

// ── (9b) rollback of a partial allocation must drop the hashes it took ────
// `allocate_blocks_with_prefix` can move a cached block's reference into the
// new sequence and only THEN fail to allocate a fresh block, at which point
// rollback_partial_allocation() returns everything. Those moved-in blocks left
// cached_blocks_map_ but their entries are still in the prefix-hash table, so
// the rollback has to drop them (kv_cache_manager.cpp:542/562 →
// drop_stale_hash_if_last). If it does not, the next request for the same
// prefix could "hit" a block that is back in the free pool — the
// double-ownership bug the trim path's own comment names.
//
// Measured, so the comment does not overclaim: stubbing drop_stale_hash_if_last
// out entirely does NOT change the outcome. The lookup site (:509) independently
// rejects a hash whose block is ref-0-and-not-cached, logs
// "prefix cache: stale hash entry for free block N — dropping", and treats it as
// a miss. So the eager cleanup is defence in depth, not the load-bearing guard.
// This test exists because nothing else drove the rollback path at all —
// ManagerAllocateRollback never registers hashes first, and
// EvictionThenRefillIsNewNotStaleHit goes through a different cleanup.
TEST(PrefixEquivTest, RollbackOfPartialAllocationDropsItsHashes) {
    SKIP_IF_NO_CUDA();
    auto mgr = MakeManager(4);  // tiny pool: 4 blocks total
    mgr->set_prefix_caching_enabled(true);

    // Cache a 2-block prefix, then free → 2 cached blocks, 2 free.
    std::vector<int32_t> p(32);
    std::iota(p.begin(), p.end(), 0);
    ASSERT_EQ(mgr->allocate_blocks_with_prefix(0, p), 0);
    mgr->register_block_hashes(0, p);
    mgr->free_sequence(0);
    ASSERT_EQ(mgr->num_cached_blocks(), 2);

    // Ask for 5 blocks sharing that prefix: 2 come from the cache, 2 more fit,
    // the 5th cannot be allocated (pool is 4) → full rollback.
    std::vector<int32_t> too_big(80);
    std::iota(too_big.begin(), too_big.end(), 0);  // same first 32 tokens
    ASSERT_EQ(mgr->allocate_blocks_with_prefix(1, too_big), -1)
        << "a 5-block request must not fit a 4-block pool";
    ASSERT_TRUE(mgr->block_table(1).empty());
    ASSERT_EQ(mgr->num_free_blocks(), 4) << "rollback must return every block";

    // The rolled-back blocks are free again, so their hash entries must be gone.
    // A hit here would hand out a block nobody owns.
    const int reused = mgr->allocate_blocks_with_prefix(2, p);
    EXPECT_EQ(reused, 0)
        << "rolled-back prefix falsely re-hit a block that went back to the free pool "
           "— STALE-HASH BUG";
    mgr->free_sequence(2);
}

// A multimodal image is not in its token ids (placeholder id repeats), so two requests with
// the same text and different pictures produce identical token sequences. content_salt seeds
// the hash chain with image content (vision_content_hash) so the chains diverge at block 0;
// untested before, so dropping it would answer the second request about the wrong picture.
TEST(PrefixEquivTest, ContentSaltSeparatesIdenticalTokenPrefixes) {
    SKIP_IF_NO_CUDA();
    auto mgr = MakeManager(16);
    mgr->set_prefix_caching_enabled(true);

    constexpr size_t kImageA = 0xA11CE;
    constexpr size_t kImageB = 0xB0B;
    std::vector<int32_t> tokens(32);  // 2 full blocks, identical for both requests
    std::iota(tokens.begin(), tokens.end(), 500);

    // Request 1 carries image A. Cache it, then free so the blocks are cached.
    ASSERT_EQ(mgr->allocate_blocks_with_prefix(0, tokens, /*max_reuse_blocks=*/-1, kImageA), 0);
    mgr->register_block_hashes(0, tokens, kImageA);
    mgr->free_sequence(0);
    ASSERT_EQ(mgr->num_cached_blocks(), 2);

    // Same tokens, different image: must NOT hit.
    std::vector<size_t> chain_b;
    EXPECT_EQ(mgr->longest_cached_prefix_blocks(tokens, chain_b, kImageB), 0)
        << "probe matched a prefix cached under a different image";
    const int reused_b = mgr->allocate_blocks_with_prefix(1, tokens, /*max_reuse_blocks=*/-1, kImageB);
    ASSERT_GE(reused_b, 0);
    EXPECT_EQ(reused_b, 0) << "a different image reused the first image's KV blocks";
    mgr->free_sequence(1);

    // Same tokens, same image: must hit, or the salt has broken caching outright.
    std::vector<size_t> chain_a;
    EXPECT_EQ(mgr->longest_cached_prefix_blocks(tokens, chain_a, kImageA), 2);
    const int reused_a = mgr->allocate_blocks_with_prefix(2, tokens, /*max_reuse_blocks=*/-1, kImageA);
    EXPECT_EQ(reused_a, 2) << "same image failed to reuse its own cached prefix";
    mgr->free_sequence(2);

    // And a text prompt (salt 0) is a third, distinct chain.
    std::vector<size_t> chain_text;
    EXPECT_EQ(mgr->longest_cached_prefix_blocks(tokens, chain_text, /*content_salt=*/0), 0);
    EXPECT_NE(chain_a[0], chain_b[0]);
    EXPECT_NE(chain_a[0], chain_text[0]);
}

// Seeded pseudo-random workload over overlapping prefixes/frees/evictions, asserting three
// invariants: (a) probe==reuse (longest_cached_prefix_blocks vs allocate_blocks_with_prefix
// must agree, or the hybrid snapshot boundary is chosen for a prefix not actually reused);
// (b) no block appears twice within one sequence's table; (c) no leak (#1115 shape) after
// draining. Seeded and fixed-iteration: a failure reproduces exactly.
TEST(PrefixEquivTest, RandomisedWorkloadKeepsProbeAllocationAndPoolConsistent) {
    SKIP_IF_NO_CUDA();
    constexpr int kPool = 12;
    constexpr int kBlock = 16;  // kKVBlockSize
    auto mgr = MakeManager(kPool);
    mgr->set_prefix_caching_enabled(true);

    // Deterministic LCG — no <random> engine differences across toolchains.
    uint32_t rng = 0xC0FFEEu;
    auto next = [&rng](int n) {
        rng = rng * 1664525u + 1013904223u;
        return static_cast<int>((rng >> 16) % static_cast<uint32_t>(n));
    };

    // A handful of shared roots so prefixes actually collide, and lengths that
    // sit on, just below and just above block boundaries.
    const int kRoots[4] = {1000, 2000, 3000, 4000};
    const int kLens[6] = {kBlock - 1, kBlock, kBlock + 1, 2 * kBlock, 3 * kBlock - 1, 3 * kBlock + 1};

    std::vector<int> live;
    int next_seq = 0;

    for (int round = 0; round < 300; ++round) {
        if (!live.empty() && next(100) < 35) {
            const int idx = next(static_cast<int>(live.size()));
            mgr->free_sequence(live[idx]);
            live.erase(live.begin() + idx);
            continue;
        }
        if (next(100) < 10) {
            mgr->evict_cached_block();
            continue;
        }

        const int root = kRoots[next(4)];
        const int len = kLens[next(6)];
        std::vector<int32_t> tokens(static_cast<size_t>(len));
        std::iota(tokens.begin(), tokens.end(), root);

        std::vector<size_t> chain;
        const int probed = mgr->longest_cached_prefix_blocks(tokens, chain);
        ASSERT_EQ(static_cast<int>(chain.size()), len / kBlock)
            << "round " << round << ": probe must hash every FULL block";

        const int seq = next_seq++;
        const int reused = mgr->allocate_blocks_with_prefix(seq, tokens);
        if (reused < 0)
            continue;  // pool could not fit it; rollback is covered separately

        // (a)
        EXPECT_EQ(reused, probed)
            << "round " << round << ": probe said " << probed << " cached blocks, allocation reused "
            << reused << " — the snapshot restore boundary would be wrong";

        // (b)
        const std::vector<int>& table = mgr->block_table(seq);
        EXPECT_EQ(static_cast<int>(table.size()), (len + kBlock - 1) / kBlock) << "round " << round;
        std::unordered_set<int> seen_in_seq;
        for (int id : table) {
            ASSERT_GE(id, 0) << "round " << round;
            ASSERT_LT(id, kPool) << "round " << round;
            EXPECT_TRUE(seen_in_seq.insert(id).second)
                << "round " << round << ": block " << id << " appears twice in one sequence";
        }

        mgr->register_block_hashes(seq, tokens);
        live.push_back(seq);
    }

    // (c)
    for (int seq : live)
        mgr->free_sequence(seq);
    live.clear();
    while (mgr->evict_cached_block()) {
    }
    EXPECT_EQ(mgr->num_cached_blocks(), 0);
    EXPECT_EQ(mgr->num_free_blocks(), kPool)
        << "pool did not come back whole after every sequence was freed and the cache drained";
}

// Read-only probe used by the hybrid snapshot lookup: reports the contiguous cached chain
// length without allocating, filling per-block chain hashes for all full blocks.
TEST(PrefixEquivTest, LongestCachedPrefixProbe) {
    SKIP_IF_NO_CUDA();
    auto mgr = MakeManager(16);
    mgr->set_prefix_caching_enabled(true);

    std::vector<int32_t> tokens(40);  // 2 full blocks + partial
    std::iota(tokens.begin(), tokens.end(), 123);

    std::vector<size_t> hashes;
    EXPECT_EQ(mgr->longest_cached_prefix_blocks(tokens, hashes), 0);
    EXPECT_EQ(hashes.size(), 2u) << "hashes cover all FULL blocks regardless of cache state";

    ASSERT_EQ(mgr->allocate_blocks_with_prefix(0, tokens), 0);
    mgr->register_block_hashes(0, tokens);
    mgr->free_sequence(0);

    EXPECT_EQ(mgr->longest_cached_prefix_blocks(tokens, hashes), 2);
    // Probe must not have allocated anything.
    EXPECT_EQ(mgr->num_cached_blocks(), 2);

    // Break the chain at block 0 → probe reports 0 despite block 1 cached.
    ASSERT_TRUE(mgr->evict_cached_block());
    EXPECT_EQ(mgr->longest_cached_prefix_blocks(tokens, hashes), 0);
}

// A quantized KV block is meaningless without its scales (values are indices into a scale);
// restoring bytes alone decodes against whatever scales sit in the pool, with no error. NVFP4
// (like INT8/INT4/MXFP4_KV) uses a SEPARATE scale pool; FP16/FP8_E4M3 have none, which is why
// the gap stayed invisible on the default KV dtype.
TEST(PrefixPersistTest, QuantizedKvRestoresItsScales) {
    SKIP_IF_NO_CUDA();
    // head_dim must be a multiple of 16 for the NVFP4 scale geometry.
    auto cache = std::make_unique<KVCache>(2, /*n_kv_heads=*/2, /*head_dim=*/32, QType::NVFP4,
                                           /*max_blocks=*/16);
    ASSERT_GT(cache->scale_block_bytes(), 0u) << "NVFP4 must have a scale pool, or this proves nothing";
    const size_t sbb = cache->scale_block_bytes();
    auto mgr = std::make_unique<KVCacheManager>(std::move(cache));
    mgr->set_prefix_caching_enabled(true);
    KVCache* c = mgr->kv_cache();

    std::vector<int32_t> tokens(32);
    std::iota(tokens.begin(), tokens.end(), 4000);
    ASSERT_EQ(mgr->allocate_blocks_with_prefix(0, tokens), 0);
    const std::vector<int> table = mgr->block_table(0);
    ASSERT_GE(table.size(), 2u);

    // A pattern the zero-initialized pool cannot reproduce by accident.
    std::vector<uint8_t> scale_pattern(sbb);
    for (size_t i = 0; i < sbb; ++i)
        scale_pattern[i] = static_cast<uint8_t>(0xA0 + (i & 0x1F));
    for (int bi = 0; bi < 2; ++bi) {
        WriteBlockPattern(c, table[bi], static_cast<uint8_t>(11 + bi));
        for (int l = 0; l < c->n_layers(); ++l) {
            ASSERT_EQ(cudaMemcpy(c->k_scale_ptr(l, table[bi]), scale_pattern.data(), sbb,
                                 cudaMemcpyHostToDevice),
                      cudaSuccess);
            ASSERT_EQ(cudaMemcpy(c->v_scale_ptr(l, table[bi]), scale_pattern.data(), sbb,
                                 cudaMemcpyHostToDevice),
                      cudaSuccess);
        }
    }
    mgr->register_block_hashes(0, tokens);
    mgr->free_sequence(0);
    ASSERT_EQ(mgr->num_cached_blocks(), 2);

    const std::string path = "/tmp/imp_prefix_scale_roundtrip.bin";
    ::remove(path.c_str());
    ASSERT_GT(mgr->save_prefix_cache(path, /*fingerprint=*/0xABCDEF, nullptr), 0);

    // A FRESH pool: zero-initialized, so anything the restore does not write
    // reads back as zero rather than as the previous manager's leftovers.
    auto cache2 = std::make_unique<KVCache>(2, 2, 32, QType::NVFP4, 16);
    auto mgr2 = std::make_unique<KVCacheManager>(std::move(cache2));
    mgr2->set_prefix_caching_enabled(true);
    ASSERT_EQ(mgr2->load_prefix_cache(path, 0xABCDEF, nullptr), 2);
    KVCache* c2 = mgr2->kv_cache();

    // The restored blocks are the ones a matching prefix now hits.
    ASSERT_EQ(mgr2->allocate_blocks_with_prefix(1, tokens), 2);
    const std::vector<int> table2 = mgr2->block_table(1);
    ASSERT_GE(table2.size(), 2u);

    std::vector<uint8_t> got(sbb);
    for (int bi = 0; bi < 2; ++bi) {
        for (int l = 0; l < c2->n_layers(); ++l) {
            ASSERT_EQ(cudaMemcpy(got.data(), c2->k_scale_ptr(l, table2[bi]), sbb, cudaMemcpyDeviceToHost),
                      cudaSuccess);
            EXPECT_EQ(got, scale_pattern) << "K scales lost for block " << bi << " layer " << l;
            ASSERT_EQ(cudaMemcpy(got.data(), c2->v_scale_ptr(l, table2[bi]), sbb, cudaMemcpyDeviceToHost),
                      cudaSuccess);
            EXPECT_EQ(got, scale_pattern) << "V scales lost for block " << bi << " layer " << l;
        }
    }
    mgr2->free_sequence(1);
    ::remove(path.c_str());
}

}  // namespace
}  // namespace imp
