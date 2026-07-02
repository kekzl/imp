// Prefix-cache equivalence — TEST_AUDIT.md risk #7 (Phase 2.6).
//
// "Prefix cache ships off-by-default BECAUSE its determinism is unvalidated —
//  the test IS the enabler."
//
// The existing KVCacheManager suite (tests/test_kv_cache.cpp) already covers
// the BLOCK-ID BOOKKEEPING of prefix caching (reuse counts, partial match,
// cached-block eviction, pool integrity). What it does NOT cover — and what
// Risk #7 actually names — is the part that makes the feature *correct*:
//
//   (a) KV-DATA equivalence: a "reused" block must hand back the SAME physical
//       block carrying the SAME KV bytes that were computed for the prefix.
//       A correct reuse *count* with stale/wrong KV bytes is exactly the
//       silent-correctness failure that keeps the feature off-by-default.
//   (b) eviction+refill stability: once a cached prefix is EVICTED, re-allocating
//       the identical prefix must come back as NEW (0 reuse) — never a false
//       hit on a recycled/stale block (hash-collision / stale-block guard).
//   (c) ref-count keep-alive: a block shared by two live sequences must survive
//       one of them being freed, with its KV bytes intact (no use-after-free).
//
// These assert against the REAL KVCacheManager wrapping a REAL KVCache pool;
// the content checks use a fp32-independent oracle — bytes we wrote ourselves
// into device memory, read back via cudaMemcpy. The pool is zero-initialized
// on construction, so a stale/wrong-block hit reads zeros, not our pattern:
// the content assert cannot pass tautologically.
//
// block_size = kKVBlockSize = 16 throughout; expectations are derived from the
// documented hashing semantics (FNV-1a parent-chained per full block; partial
// blocks are NOT cacheable) and stated per case.

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "memory/kv_cache.h"
#include "memory/kv_cache_manager.h"
#include "core/tensor.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
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

// ── (1) Full-prefix reuse count ───────────────────────────────────────────
// Two sequences with IDENTICAL ≥2-block prefixes. After seq A is registered +
// freed (cached), seq B's allocate_blocks_with_prefix must report exactly the
// number of FULL blocks as reused. Derivation: each full block's FNV-1a hash
// chains through its parent; identical tokens ⇒ identical chain ⇒ every full
// block matches. 48 tokens / 16 = 3 full blocks ⇒ 3 reused.
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

// ── (4) KV-DATA equivalence: reused block carries the same bytes ──────────
// The heart of risk #7. Reuse is only correct if the reused physical block
// holds the KV data computed for that prefix. We:
//   1. allocate the prefix for seq A,
//   2. write a known byte pattern into A's blocks' K/V device memory
//      (standing in for "prefill computed these KV"),
//   3. register + free A (blocks cached, ref=1, data retained),
//   4. reuse the prefix for seq B,
//   5. read B's reused blocks back and assert byte-identical to what A wrote.
// The pool was zero-initialized, so a stale/wrong-block hit would read zeros:
// passing requires the SAME physical block with intact KV bytes.
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

// ── (5) ref-count keep-alive: freeing A must not corrupt B's shared block ──
// A and B both hold the shared prefix CONCURRENTLY (B reuses A's registered
// blocks; ref_count goes to 2). Freeing A must leave the shared block alive
// (ref_count back to 1, NOT returned to the pool) with KV bytes intact — the
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

// ── (6) eviction+refill: an evicted prefix must re-allocate as NEW ────────
// Cache a prefix, free it (cached, ref=1), then force the cached blocks out of
// the pool via NEW unrelated sequences that need the space. Re-allocating the
// ORIGINAL prefix must then report 0 reuse — its hash entries were removed on
// eviction, so it must not falsely hit a recycled/stale block. This is the
// hash-collision / stale-block guard that risk #7 names as the determinism
// blocker.
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

// ── (7) non-block-aligned prefix length ───────────────────────────────────
// 40 tokens at block_size 16 = 2 full blocks + 1 partial (8 tokens). Only full
// blocks are cacheable, so re-requesting the identical 40 tokens reuses exactly
// the 2 full blocks; the partial tail is always re-allocated fresh.
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

// ── (8) chain hole must not count as reused prefix ────────────────────────
// LRU eviction removes cached blocks front-first, so an EARLY block of a
// cached chain can be gone while LATER chain blocks survive. Reuse must stop
// at the first miss: counting later hits would make the caller skip prefill
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

// ── (9) max_reuse_blocks caps reuse (hybrid snapshot boundary) ────────────
// Hybrid models can only skip prefill up to the recurrent-snapshot position;
// blocks past the cap must be freshly allocated (never shared), because the
// continuation prefill will WRITE them.
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

// ── (10) longest_cached_prefix_blocks probe ───────────────────────────────
// Read-only probe used by the hybrid snapshot lookup: reports the contiguous
// cached chain length without allocating, and fills the per-block chain
// hashes for all full blocks.
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

}  // namespace
}  // namespace imp
