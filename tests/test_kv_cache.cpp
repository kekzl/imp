#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "memory/kv_cache.h"
#include "memory/kv_cache_manager.h"
#include "core/tensor.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <unordered_set>
#include <vector>

#include "test_cuda_skip.h"

namespace imp {
namespace {

// ============================================================================
// KVCache tests
// ============================================================================

// 7. KVCacheConstruction
TEST(KVCacheTest, KVCacheConstruction) {
    SKIP_IF_NO_CUDA();

    const int n_layers = 2;
    const int n_kv_heads = 4;
    const int head_dim = 64;
    const int max_blocks = 8;

    KVCache cache(n_layers, n_kv_heads, head_dim, QType::F16, max_blocks);

    EXPECT_EQ(cache.n_layers(), n_layers);
    EXPECT_EQ(cache.n_kv_heads(), n_kv_heads);
    EXPECT_EQ(cache.head_dim(), head_dim);
    EXPECT_EQ(cache.qtype(), QType::F16);
    EXPECT_EQ(cache.total_blocks(), max_blocks);
    EXPECT_EQ(cache.num_free_blocks(), max_blocks);

    // block_bytes = kKVBlockSize * n_kv_heads * head_dim * dtype_size(FP16)
    //            = 16 * 4 * 64 * 2 = 8192
    size_t expected_block_bytes = static_cast<size_t>(kKVBlockSize) * n_kv_heads * head_dim *
                                  dtype_size(QType::F16);
    EXPECT_EQ(cache.block_bytes(), expected_block_bytes);
    EXPECT_EQ(expected_block_bytes, 8192u);
}

// 8. KVCacheBlockAllocation
TEST(KVCacheTest, KVCacheBlockAllocation) {
    SKIP_IF_NO_CUDA();

    const int max_blocks = 8;
    KVCache cache(2, 4, 64, QType::F16, max_blocks);

    // Allocate all 8 blocks and verify IDs are 0..7 in order.
    std::vector<int> ids;
    for (int i = 0; i < max_blocks; ++i) {
        int id = cache.allocate_block();
        ASSERT_GE(id, 0) << "Failed to allocate block " << i;
        ids.push_back(id);
    }

    for (int i = 0; i < max_blocks; ++i) {
        EXPECT_EQ(ids[i], i);
    }

    EXPECT_EQ(cache.num_free_blocks(), 0);

    // 9th allocation should fail.
    EXPECT_EQ(cache.allocate_block(), -1);
}

// 9. KVCacheBlockFree
TEST(KVCacheTest, KVCacheBlockFree) {
    SKIP_IF_NO_CUDA();

    const int max_blocks = 8;
    KVCache cache(2, 4, 64, QType::F16, max_blocks);

    // Allocate 4 blocks.
    std::vector<int> ids;
    for (int i = 0; i < 4; ++i) {
        ids.push_back(cache.allocate_block());
    }
    EXPECT_EQ(cache.num_free_blocks(), 4);

    // Free 2 of them.
    cache.free_block(ids[0]);
    cache.free_block(ids[1]);
    EXPECT_EQ(cache.num_free_blocks(), 6);

    // Re-allocate 1 block.
    int new_id = cache.allocate_block();
    ASSERT_GE(new_id, 0);
    EXPECT_EQ(cache.num_free_blocks(), 5);
}

// 10. KVCacheRefCounting
TEST(KVCacheTest, KVCacheRefCounting) {
    SKIP_IF_NO_CUDA();

    const int max_blocks = 8;
    KVCache cache(2, 4, 64, QType::F16, max_blocks);

    int block = cache.allocate_block();
    ASSERT_GE(block, 0);
    EXPECT_EQ(cache.ref_count(block), 1);
    EXPECT_EQ(cache.num_free_blocks(), 7);

    // Increment reference count.
    cache.inc_ref(block);
    EXPECT_EQ(cache.ref_count(block), 2);

    // First free: ref_count drops to 1, block is NOT returned to free list.
    int free_before = cache.num_free_blocks();
    cache.free_block(block);
    EXPECT_EQ(cache.ref_count(block), 1);
    EXPECT_EQ(cache.num_free_blocks(), free_before);  // Unchanged.

    // Second free: ref_count drops to 0, block IS returned to free list.
    cache.free_block(block);
    EXPECT_EQ(cache.ref_count(block), 0);
    EXPECT_EQ(cache.num_free_blocks(), free_before + 1);
}

// 11. KVCachePointers
TEST(KVCacheTest, KVCachePointers) {
    SKIP_IF_NO_CUDA();

    const int n_layers = 2;
    const int n_kv_heads = 4;
    const int head_dim = 64;
    const int max_blocks = 8;

    KVCache cache(n_layers, n_kv_heads, head_dim, QType::F16, max_blocks);
    size_t bb = cache.block_bytes();

    int b0 = cache.allocate_block();
    int b1 = cache.allocate_block();
    ASSERT_GE(b0, 0);
    ASSERT_GE(b1, 0);

    // All pointers should be non-null.
    void* k0_l0 = cache.k_ptr(0, b0);
    void* v0_l0 = cache.v_ptr(0, b0);
    void* k1_l0 = cache.k_ptr(0, b1);
    void* v1_l0 = cache.v_ptr(0, b1);
    void* k0_l1 = cache.k_ptr(1, b0);

    ASSERT_NE(k0_l0, nullptr);
    ASSERT_NE(v0_l0, nullptr);
    ASSERT_NE(k1_l0, nullptr);
    ASSERT_NE(v1_l0, nullptr);
    ASSERT_NE(k0_l1, nullptr);

    // K and V pointers for the same (layer, block) should be distinct.
    // V blocks start max_blocks * bb after K blocks within a layer.
    EXPECT_NE(k0_l0, v0_l0);
    ptrdiff_t kv_diff = static_cast<char*>(v0_l0) - static_cast<char*>(k0_l0);
    EXPECT_EQ(static_cast<size_t>(kv_diff), static_cast<size_t>(max_blocks) * bb);

    // Expected offsets (K and V contiguous within layer):
    //   K(layer, block) = (layer * 2 * max_blocks + block) * bb
    //   V(layer, block) = (layer * 2 * max_blocks + max_blocks + block) * bb
    // Verify layer=1, block=0 K pointer is at the expected offset from
    // layer=0, block=0 K pointer.
    ptrdiff_t layer_diff = static_cast<char*>(k0_l1) - static_cast<char*>(k0_l0);
    size_t expected_layer_stride = static_cast<size_t>(max_blocks) * 2 * bb;
    EXPECT_EQ(static_cast<size_t>(layer_diff), expected_layer_stride);

    // Verify block=1 K is bb after block=0 K (K blocks contiguous within layer).
    ptrdiff_t block_diff = static_cast<char*>(k1_l0) - static_cast<char*>(k0_l0);
    EXPECT_EQ(static_cast<size_t>(block_diff), bb);

    // Write a known value via cudaMemcpy to k_ptr(0, b0), read back, verify.
    size_t num_elements = kKVBlockSize * n_kv_heads * head_dim;  // elements per block
    size_t buf_bytes = num_elements * sizeof(uint16_t);          // FP16 = 2 bytes
    ASSERT_EQ(buf_bytes, bb);

    std::vector<uint16_t> host_write(num_elements, 0);
    for (size_t i = 0; i < num_elements; ++i) {
        host_write[i] = static_cast<uint16_t>(i & 0xFFFF);
    }

    cudaError_t err;
    err = cudaMemcpy(k0_l0, host_write.data(), buf_bytes, cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess);

    std::vector<uint16_t> host_read(num_elements, 0);
    err = cudaMemcpy(host_read.data(), k0_l0, buf_bytes, cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess);

    EXPECT_EQ(host_write, host_read);
}

// 12. KVCacheReadWriteData
TEST(KVCacheTest, KVCacheReadWriteData) {
    SKIP_IF_NO_CUDA();

    const int n_layers = 2;
    const int n_kv_heads = 2;
    const int head_dim = 32;
    const int max_blocks = 4;

    KVCache cache(n_layers, n_kv_heads, head_dim, QType::F32, max_blocks);
    size_t bb = cache.block_bytes();

    int block = cache.allocate_block();
    ASSERT_GE(block, 0);

    size_t num_floats = bb / sizeof(float);
    ASSERT_EQ(bb, static_cast<size_t>(kKVBlockSize) * n_kv_heads * head_dim * sizeof(float));

    // Write a known pattern to K of layer 0.
    std::vector<float> k_pattern(num_floats);
    for (size_t i = 0; i < num_floats; ++i) {
        k_pattern[i] = static_cast<float>(i) * 0.5f;
    }

    cudaError_t err;
    err = cudaMemcpy(cache.k_ptr(0, block), k_pattern.data(), bb, cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess);

    // Write a different pattern to V of layer 0.
    std::vector<float> v_pattern(num_floats);
    for (size_t i = 0; i < num_floats; ++i) {
        v_pattern[i] = static_cast<float>(i) * -1.0f;
    }

    err = cudaMemcpy(cache.v_ptr(0, block), v_pattern.data(), bb, cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess);

    // Read back K and verify.
    std::vector<float> k_readback(num_floats, 0.0f);
    err = cudaMemcpy(k_readback.data(), cache.k_ptr(0, block), bb, cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess);
    EXPECT_EQ(k_pattern, k_readback);

    // Read back V and verify.
    std::vector<float> v_readback(num_floats, 0.0f);
    err = cudaMemcpy(v_readback.data(), cache.v_ptr(0, block), bb, cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess);
    EXPECT_EQ(v_pattern, v_readback);

    // Cross-check: K and V data should be different from each other.
    EXPECT_NE(k_readback, v_readback);

    // Also verify that layer 1 data is independent (should still be zeros
    // from the initial cudaMemset in the constructor).
    std::vector<float> l1_readback(num_floats, 999.0f);
    err = cudaMemcpy(l1_readback.data(), cache.k_ptr(1, block), bb, cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess);

    std::vector<float> zeros(num_floats, 0.0f);
    EXPECT_EQ(l1_readback, zeros);
}

// ============================================================================
// KVCacheManager tests
// ============================================================================

// 13. NVFP4 storage layout: block_bytes = block_size * n_kv_heads * head_dim / 2 (4-bit packed),
//     scale_block_bytes = block_size * n_kv_heads * (head_dim / 16) (UE4M3, 1 byte per group of 16).
TEST(KVCacheTest, KVCacheNVFP4Layout) {
    SKIP_IF_NO_CUDA();

    const int n_layers = 2;
    const int n_kv_heads = 8;
    const int head_dim = 128;
    const int max_blocks = 4;

    KVCache cache(n_layers, n_kv_heads, head_dim, QType::NVFP4, max_blocks);

    EXPECT_EQ(cache.qtype(), QType::NVFP4);
    EXPECT_EQ(cache.head_dim(), head_dim);

    // 4-bit packed pool sizing: 16 * 8 * 128 / 2 = 8192 bytes per block.
    size_t expected_block_bytes = static_cast<size_t>(kKVBlockSize) * n_kv_heads * head_dim / 2;
    EXPECT_EQ(cache.block_bytes(), expected_block_bytes);
    EXPECT_EQ(expected_block_bytes, 8192u);

    // UE4M3 scale layout: 16 * 8 * (128/16) = 1024 bytes per block.
    size_t expected_scale_bytes = static_cast<size_t>(kKVBlockSize) * n_kv_heads * (head_dim / kNVFP4Group);
    EXPECT_EQ(cache.scale_block_bytes(), expected_scale_bytes);
    EXPECT_EQ(expected_scale_bytes, 1024u);

    int b0 = cache.allocate_block();
    ASSERT_GE(b0, 0);

    // K + V data + scales must all be allocated and distinct.
    void* k_data = cache.k_ptr(0, b0);
    void* v_data = cache.v_ptr(0, b0);
    void* k_sc = cache.k_scale_ptr(0, b0);
    void* v_sc = cache.v_scale_ptr(0, b0);
    ASSERT_NE(k_data, nullptr);
    ASSERT_NE(v_data, nullptr);
    ASSERT_NE(k_sc, nullptr);
    ASSERT_NE(v_sc, nullptr);

    // K and V scale regions should be disjoint with the V scale region following K.
    ptrdiff_t scale_diff = static_cast<char*>(v_sc) - static_cast<char*>(k_sc);
    EXPECT_EQ(static_cast<size_t>(scale_diff), static_cast<size_t>(max_blocks) * cache.scale_block_bytes());
}

// 13b. NVFP4 head_dim that is not a multiple of 16 must fail with a clear error.
TEST(KVCacheTest, KVCacheNVFP4HeadDimReject) {
    SKIP_IF_NO_CUDA();
    EXPECT_THROW({ KVCache cache(1, 4, 24, QType::NVFP4, 2); }, std::runtime_error);
}

// 13c. NVFP4 per-layer constructor (Gemma 4 dual head_dim 256 SWA / 512 global).
TEST(KVCacheTest, KVCacheNVFP4PerLayer) {
    SKIP_IF_NO_CUDA();

    const int n_layers = 4;
    const int max_blocks = 2;
    std::vector<int> nkv = {8, 8, 8, 8};
    std::vector<int> hd = {128, 256, 128, 256};  // mixed head_dim

    KVCache cache(n_layers, nkv, hd, QType::NVFP4, max_blocks, kKVBlockSize, nullptr);
    EXPECT_EQ(cache.qtype(), QType::NVFP4);

    // Each layer's scale_block_bytes must match its own (nkv * hd / 16).
    EXPECT_EQ(cache.scale_block_bytes(0), static_cast<size_t>(kKVBlockSize) * 8 * (128 / 16));   // 1024
    EXPECT_EQ(cache.scale_block_bytes(1), static_cast<size_t>(kKVBlockSize) * 8 * (256 / 16));   // 2048
    EXPECT_EQ(cache.scale_block_bytes(2), static_cast<size_t>(kKVBlockSize) * 8 * (128 / 16));   // 1024
    EXPECT_EQ(cache.scale_block_bytes(3), static_cast<size_t>(kKVBlockSize) * 8 * (256 / 16));   // 2048

    int b0 = cache.allocate_block();
    ASSERT_GE(b0, 0);
    for (int l = 0; l < n_layers; ++l) {
        EXPECT_NE(cache.k_ptr(l, b0), nullptr) << "layer " << l;
        EXPECT_NE(cache.k_scale_ptr(l, b0), nullptr) << "layer " << l;
        EXPECT_NE(cache.v_scale_ptr(l, b0), nullptr) << "layer " << l;
    }
}

// Helper to create a KVCacheManager wrapping a fresh KVCache.
static std::unique_ptr<KVCacheManager> MakeManager(int max_blocks, int n_layers = 2, int n_kv_heads = 4,
                                                   int head_dim = 64, QType dtype = QType::F16) {
    auto cache = std::make_unique<KVCache>(n_layers, n_kv_heads, head_dim, dtype, max_blocks);
    return std::make_unique<KVCacheManager>(std::move(cache));
}

// 13. ManagerAllocateBlocks
TEST(KVCacheManagerTest, ManagerAllocateBlocks) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(16);

    bool ok = mgr->allocate_blocks(/*seq_id=*/0, /*num_blocks=*/4);
    ASSERT_TRUE(ok);

    const auto& table = mgr->block_table(0);
    EXPECT_EQ(static_cast<int>(table.size()), 4);
    EXPECT_EQ(mgr->num_active_sequences(), 1);
    EXPECT_EQ(mgr->total_allocated_blocks(), 4);
    EXPECT_EQ(mgr->num_free_blocks(), 12);
}

// 14. ManagerAllocateRollback
TEST(KVCacheManagerTest, ManagerAllocateRollback) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(16);

    // First sequence takes 10 blocks -- should succeed.
    bool ok = mgr->allocate_blocks(0, 10);
    ASSERT_TRUE(ok);
    EXPECT_EQ(mgr->num_free_blocks(), 6);

    // Second sequence asks for 10 blocks, but only 6 are left -- should fail
    // with full rollback.
    ok = mgr->allocate_blocks(1, 10);
    EXPECT_FALSE(ok);

    // Seq 1 should have an empty block table (or not exist).
    const auto& table1 = mgr->block_table(1);
    EXPECT_TRUE(table1.empty());

    // Free blocks should be restored to 6 (rollback of partial allocation).
    EXPECT_EQ(mgr->num_free_blocks(), 6);

    // Seq 0 should still be intact.
    EXPECT_EQ(static_cast<int>(mgr->block_table(0).size()), 10);
    EXPECT_EQ(mgr->num_active_sequences(), 1);
}

// 15. ManagerAppendBlock
TEST(KVCacheManagerTest, ManagerAppendBlock) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(16);

    (void)mgr->allocate_blocks(0, 2);
    EXPECT_EQ(static_cast<int>(mgr->block_table(0).size()), 2);

    int new_block = mgr->append_block(0);
    ASSERT_GE(new_block, 0);
    EXPECT_EQ(static_cast<int>(mgr->block_table(0).size()), 3);

    // Append to a non-existent sequence should return -1.
    EXPECT_EQ(mgr->append_block(99), -1);
}

// 16. ManagerFreeSequence
TEST(KVCacheManagerTest, ManagerFreeSequence) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(16);

    (void)mgr->allocate_blocks(0, 4);
    EXPECT_EQ(mgr->num_active_sequences(), 1);
    EXPECT_EQ(mgr->num_free_blocks(), 12);

    mgr->free_sequence(0);
    EXPECT_EQ(mgr->num_active_sequences(), 0);
    EXPECT_EQ(mgr->num_free_blocks(), 16);
    EXPECT_TRUE(mgr->block_table(0).empty());
}

// 17. ManagerLRUEviction
TEST(KVCacheManagerTest, ManagerLRUEviction) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(8);

    // Fill the entire pool across three sequences.
    (void)mgr->allocate_blocks(0, 3);  // seq 0: 3 blocks  (LRU order: 0)
    (void)mgr->allocate_blocks(1, 3);  // seq 1: 3 blocks  (LRU order: 0, 1)
    (void)mgr->allocate_blocks(2, 2);  // seq 2: 2 blocks  (LRU order: 0, 1, 2)
    EXPECT_EQ(mgr->num_free_blocks(), 0);

    // Touch seq 0, moving it to MRU. LRU order is now: 1, 2, 0.
    mgr->touch(0);

    // Evict the LRU sequence -- should evict seq 1 (the oldest untouched).
    int victim = mgr->evict_lru();
    EXPECT_EQ(victim, 1);

    // Seq 1 should be gone, its 3 blocks freed.
    EXPECT_TRUE(mgr->block_table(1).empty());
    EXPECT_EQ(mgr->num_free_blocks(), 3);
    EXPECT_EQ(mgr->num_active_sequences(), 2);

    // Seq 0 and seq 2 should still be intact.
    EXPECT_EQ(static_cast<int>(mgr->block_table(0).size()), 3);
    EXPECT_EQ(static_cast<int>(mgr->block_table(2).size()), 2);
}

// Regression (audit F-A1/F-A1b): allocation under full-KV pressure must NEVER
// evict a live sequence. Every lru_order_ entry is a live sequence and imp has
// no recompute-on-resume path, so freeing one would corrupt it (use-after-free
// once it runs). append_block/allocate_blocks reclaim *cached* (finished) blocks
// only, then fail — the engine reject-newests on that failure rather than
// preempting a live sequence. This locks the invariant the fix depends on.
TEST(KVCacheManagerTest, AllocationNeverEvictsLiveSequenceUnderPressure) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(8);
    (void)mgr->allocate_blocks(0, 5);  // seq 0: 5 live blocks
    (void)mgr->allocate_blocks(1, 3);  // seq 1: 3 live blocks -> pool full
    EXPECT_EQ(mgr->num_free_blocks(), 0);
    EXPECT_EQ(mgr->num_cached_blocks(), 0);  // nothing reclaimable — all live

    // No free and no reclaimable cached blocks => allocation must FAIL, not
    // evict a live sequence.
    EXPECT_EQ(mgr->append_block(0), -1);
    EXPECT_FALSE(mgr->allocate_blocks(1, 1));

    // Both sequences keep every block — neither was stripped.
    EXPECT_EQ(static_cast<int>(mgr->block_table(0).size()), 5);
    EXPECT_EQ(static_cast<int>(mgr->block_table(1).size()), 3);
}

// 18. ManagerCanAllocate
TEST(KVCacheManagerTest, ManagerCanAllocate) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(8);

    (void)mgr->allocate_blocks(0, 4);
    (void)mgr->allocate_blocks(1, 4);
    EXPECT_EQ(mgr->num_free_blocks(), 0);

    // We have 0 free blocks, but can evict 4+4 = 8 blocks total.
    // So can_allocate(4) should be true (eviction can recover enough).
    EXPECT_TRUE(mgr->can_allocate(4));

    // can_allocate(8) should also be true (evict everything).
    EXPECT_TRUE(mgr->can_allocate(8));

    // can_allocate(9) should be false -- even evicting all sequences only
    // frees 8 blocks, which is less than 9.
    EXPECT_FALSE(mgr->can_allocate(9));

    // Edge case: can_allocate(0) is trivially true.
    EXPECT_TRUE(mgr->can_allocate(0));
}

// 19. ManagerPrefixCaching -- legacy register/find/share_prefix removed.
// Content-addressed prefix caching (below) is the replacement.

// ============================================================================
// Content-addressed prefix caching tests
// ============================================================================

// 20. BlockHashDeterministic
TEST(KVCacheManagerTest, BlockHashDeterministic) {
    // Verify that compute_block_hash is deterministic.
    std::vector<int32_t> tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    size_t h1 = KVCacheManager::compute_block_hash(tokens, 0);
    size_t h2 = KVCacheManager::compute_block_hash(tokens, 0);
    EXPECT_EQ(h1, h2);

    // Different tokens produce different hashes.
    tokens[0] = 99;
    size_t h3 = KVCacheManager::compute_block_hash(tokens, 0);
    EXPECT_NE(h1, h3);
}

// 21. BlockHashChaining
TEST(KVCacheManagerTest, BlockHashChaining) {
    // Parent hash changes the result even for identical tokens.
    std::vector<int32_t> tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    size_t h_parent0 = KVCacheManager::compute_block_hash(tokens, 0);
    size_t h_parent1 = KVCacheManager::compute_block_hash(tokens, 42);
    EXPECT_NE(h_parent0, h_parent1);
}

// 22. ContentAddressedPrefixCaching
TEST(KVCacheManagerTest, ContentAddressedPrefixCaching) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(32);
    mgr->set_prefix_caching_enabled(true);
    EXPECT_TRUE(mgr->prefix_caching_enabled());

    // Sequence 0: 48 tokens = 3 full blocks.
    std::vector<int32_t> tokens(48);
    std::iota(tokens.begin(), tokens.end(), 100);

    // Allocate with prefix matching — no cache yet, so 0 reused.
    int reused = mgr->allocate_blocks_with_prefix(0, tokens);
    ASSERT_GE(reused, 0);
    EXPECT_EQ(reused, 0);  // No cache hits on first request.
    EXPECT_EQ(static_cast<int>(mgr->block_table(0).size()), 3);

    // Register hashes after "prefill."
    mgr->register_block_hashes(0, tokens);

    // Free sequence 0 — blocks should be cached (not returned to pool).
    int free_before = mgr->num_free_blocks();
    mgr->free_sequence(0);
    // Blocks are cached, not freed to pool — free count should NOT increase.
    EXPECT_EQ(mgr->num_free_blocks(), free_before);
    EXPECT_EQ(mgr->num_cached_blocks(), 3);

    // Sequence 1: same 48 tokens — should reuse all 3 blocks.
    reused = mgr->allocate_blocks_with_prefix(1, tokens);
    ASSERT_GE(reused, 0);
    EXPECT_EQ(reused, 3);
    EXPECT_EQ(static_cast<int>(mgr->block_table(1).size()), 3);
    // Cached blocks should have been consumed.
    EXPECT_EQ(mgr->num_cached_blocks(), 0);

    // Clean up.
    mgr->free_sequence(1);
}

// 22b. Persisted prefix cache is gated by a model fingerprint (N1). Block
// hashes are content-addressed over token IDs only, so a different model with
// identical KV geometry would otherwise match and serve the WRONG model's KV.
TEST(KVCacheManagerTest, PersistedCacheFingerprintGate) {
    SKIP_IF_NO_CUDA();

    const std::string path = "/tmp/imp_prefix_cache_fp_gate.bin";
    std::remove(path.c_str());
    constexpr uint64_t kFpA = 0xAAAAAAAAAAAAAAAAull;
    constexpr uint64_t kFpB = 0xBBBBBBBBBBBBBBBBull;  // same geometry, different model

    // Produce 3 cached blocks and persist them under fingerprint A.
    {
        auto mgr = MakeManager(32);
        mgr->set_prefix_caching_enabled(true);
        std::vector<int32_t> tokens(48);
        std::iota(tokens.begin(), tokens.end(), 100);
        ASSERT_GE(mgr->allocate_blocks_with_prefix(0, tokens), 0);
        mgr->register_block_hashes(0, tokens);
        mgr->free_sequence(0);
        ASSERT_EQ(mgr->num_cached_blocks(), 3);
        EXPECT_EQ(mgr->save_prefix_cache(path, kFpA), 3);
    }

    // Matching fingerprint → blocks restored.
    {
        auto mgr = MakeManager(32);
        mgr->set_prefix_caching_enabled(true);
        EXPECT_EQ(mgr->load_prefix_cache(path, kFpA), 3);
    }

    // Mismatched fingerprint (identical geometry) → rejected, nothing restored.
    {
        auto mgr = MakeManager(32);
        mgr->set_prefix_caching_enabled(true);
        EXPECT_LT(mgr->load_prefix_cache(path, kFpB), 0);
        EXPECT_EQ(mgr->num_cached_blocks(), 0);
    }

    std::remove(path.c_str());
}

// 23. PrefixCachingPartialMatch
TEST(KVCacheManagerTest, PrefixCachingPartialMatch) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(32);
    mgr->set_prefix_caching_enabled(true);

    // Seq 0: 32 tokens = 2 full blocks.
    std::vector<int32_t> tokens_a(32);
    std::iota(tokens_a.begin(), tokens_a.end(), 200);

    int reused = mgr->allocate_blocks_with_prefix(0, tokens_a);
    EXPECT_EQ(reused, 0);
    mgr->register_block_hashes(0, tokens_a);
    mgr->free_sequence(0);
    EXPECT_EQ(mgr->num_cached_blocks(), 2);

    // Seq 1: same first 16 tokens + different next 16 tokens.
    // Only the first block should be reused.
    std::vector<int32_t> tokens_b(32);
    std::iota(tokens_b.begin(), tokens_b.begin() + 16, 200);  // Same first block
    std::iota(tokens_b.begin() + 16, tokens_b.end(), 999);    // Different second block

    reused = mgr->allocate_blocks_with_prefix(1, tokens_b);
    ASSERT_GE(reused, 0);
    EXPECT_EQ(reused, 1);  // Only first block matched.
    EXPECT_EQ(static_cast<int>(mgr->block_table(1).size()), 2);

    // One cached block was consumed (first), one remains (second from seq 0
    // that didn't match due to parent hash chaining).
    EXPECT_EQ(mgr->num_cached_blocks(), 1);

    mgr->free_sequence(1);
}

// 24. CachedBlockEviction
TEST(KVCacheManagerTest, CachedBlockEviction) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(8);  // Small pool to force eviction.
    mgr->set_prefix_caching_enabled(true);

    // Seq 0: fill 4 blocks.
    std::vector<int32_t> tokens(64);
    std::iota(tokens.begin(), tokens.end(), 300);
    int reused = mgr->allocate_blocks_with_prefix(0, tokens);
    EXPECT_EQ(reused, 0);
    mgr->register_block_hashes(0, tokens);
    mgr->free_sequence(0);
    EXPECT_EQ(mgr->num_cached_blocks(), 4);
    EXPECT_EQ(mgr->num_free_blocks(), 4);  // 8 total - 4 cached (held at ref=1)

    // Seq 1: needs 5 blocks — must evict cached blocks to fit.
    std::vector<int32_t> tokens2(80);
    std::iota(tokens2.begin(), tokens2.end(), 500);
    reused = mgr->allocate_blocks_with_prefix(1, tokens2);
    ASSERT_GE(reused, 0);
    EXPECT_EQ(reused, 0);  // No matching prefix.
    EXPECT_EQ(static_cast<int>(mgr->block_table(1).size()), 5);

    // At least 1 cached block should have been evicted.
    EXPECT_LT(mgr->num_cached_blocks(), 4);

    mgr->free_sequence(1);
}

// 25. PrefixCachingDisabled
TEST(KVCacheManagerTest, PrefixCachingDisabled) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(16);
    EXPECT_FALSE(mgr->prefix_caching_enabled());  // Off by default.

    std::vector<int32_t> tokens(32);
    std::iota(tokens.begin(), tokens.end(), 400);

    // allocate_blocks_with_prefix with caching disabled — should still work
    // but never cache or reuse.
    int reused = mgr->allocate_blocks_with_prefix(0, tokens);
    EXPECT_EQ(reused, 0);
    EXPECT_EQ(static_cast<int>(mgr->block_table(0).size()), 2);

    mgr->free_sequence(0);
    // No cached blocks since prefix caching is disabled.
    EXPECT_EQ(mgr->num_cached_blocks(), 0);
    EXPECT_EQ(mgr->num_free_blocks(), 16);  // All returned to pool.
}

// 26. PrefixCachingWithPartialLastBlock
TEST(KVCacheManagerTest, PrefixCachingWithPartialLastBlock) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(16);
    mgr->set_prefix_caching_enabled(true);

    // 20 tokens = 1 full block (16 tokens) + 1 partial block (4 tokens).
    std::vector<int32_t> tokens(20);
    std::iota(tokens.begin(), tokens.end(), 500);

    int reused = mgr->allocate_blocks_with_prefix(0, tokens);
    EXPECT_EQ(reused, 0);
    EXPECT_EQ(static_cast<int>(mgr->block_table(0).size()), 2);

    mgr->register_block_hashes(0, tokens);
    mgr->free_sequence(0);

    // Only the first (full) block should be cached. The partial block
    // should be freed normally.
    EXPECT_EQ(mgr->num_cached_blocks(), 1);

    // Seq 1: same 20 tokens — first block reused, second allocated fresh.
    reused = mgr->allocate_blocks_with_prefix(1, tokens);
    EXPECT_EQ(reused, 1);
    EXPECT_EQ(static_cast<int>(mgr->block_table(1).size()), 2);

    mgr->free_sequence(1);
}

// ============================================================================
// Prefix block pinning tests
// ============================================================================

// 27. PinnedBlocksSurviveEviction
TEST(KVCacheManagerTest, PinnedBlocksSurviveEviction) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(8);

    // Seq 0: 3 blocks, seq 1: 3 blocks, seq 2: 2 blocks.
    (void)mgr->allocate_blocks(0, 3);
    (void)mgr->allocate_blocks(1, 3);
    (void)mgr->allocate_blocks(2, 2);
    EXPECT_EQ(mgr->num_free_blocks(), 0);

    // Pin seq 0's first 2 blocks.
    mgr->pin_prefix(0, 2);
    EXPECT_EQ(mgr->num_pinned_blocks(), 2);

    // Evict LRU — seq 0 is at the front of LRU but it's pinned.
    // So seq 1 should be evicted instead.
    int victim = mgr->evict_lru();
    EXPECT_EQ(victim, 1);
    EXPECT_EQ(mgr->num_free_blocks(), 3);

    // Seq 0 should still be intact.
    EXPECT_EQ(static_cast<int>(mgr->block_table(0).size()), 3);

    // Evict again — seq 2 should be evicted (seq 0 is still pinned).
    victim = mgr->evict_lru();
    EXPECT_EQ(victim, 2);
    EXPECT_EQ(mgr->num_free_blocks(), 5);

    // Evict again — only seq 0 remains but it's pinned. Should return -1.
    victim = mgr->evict_lru();
    EXPECT_EQ(victim, -1);

    mgr->unpin_prefix(0);
    mgr->free_sequence(0);
}

// 28. PinnedBlocksSurviveFreeSequence
TEST(KVCacheManagerTest, PinnedBlocksSurviveFreeSequence) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(16);

    (void)mgr->allocate_blocks(0, 4);
    const auto& table0 = mgr->block_table(0);
    ASSERT_EQ(static_cast<int>(table0.size()), 4);

    // Pin first 2 blocks.
    mgr->pin_prefix(0, 2);
    EXPECT_EQ(mgr->num_pinned_blocks(), 2);

    int free_before = mgr->num_free_blocks();
    mgr->free_sequence(0);

    // The sequence is gone from active tracking.
    EXPECT_TRUE(mgr->block_table(0).empty());
    EXPECT_EQ(mgr->num_active_sequences(), 0);

    // Only 2 unpinned blocks should have been freed to the pool.
    // The 2 pinned blocks stay in cached_blocks_lru_ with ref_count=1.
    EXPECT_EQ(mgr->num_free_blocks(), free_before + 2);

    // Pinned blocks should still be counted.
    EXPECT_EQ(mgr->num_pinned_blocks(), 2);

    // Unpin — now the cached blocks can be reclaimed.
    mgr->unpin_prefix(0);
    EXPECT_EQ(mgr->num_pinned_blocks(), 0);
}

// 29. UnpinAllowsEviction
TEST(KVCacheManagerTest, UnpinAllowsEviction) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(8);
    mgr->set_prefix_caching_enabled(true);

    // Seq 0: 4 blocks with prefix caching.
    std::vector<int32_t> tokens(64);
    std::iota(tokens.begin(), tokens.end(), 700);
    int reused = mgr->allocate_blocks_with_prefix(0, tokens);
    EXPECT_EQ(reused, 0);
    mgr->register_block_hashes(0, tokens);

    // Pin first 2 blocks.
    mgr->pin_prefix(0, 2);

    // Free seq 0 — all 4 blocks go to cached LRU, but 2 are pinned.
    mgr->free_sequence(0);
    EXPECT_EQ(mgr->num_cached_blocks(), 4);

    // Try to evict cached blocks — only 2 non-pinned should be reclaimable.
    EXPECT_TRUE(mgr->evict_cached_block());   // reclaims non-pinned
    EXPECT_TRUE(mgr->evict_cached_block());   // reclaims non-pinned
    EXPECT_FALSE(mgr->evict_cached_block());  // pinned blocks remain
    EXPECT_EQ(mgr->num_cached_blocks(), 2);   // 2 pinned remain in LRU

    // Unpin — now the remaining 2 can be evicted.
    mgr->unpin_prefix(0);
    EXPECT_EQ(mgr->num_pinned_blocks(), 0);
    EXPECT_TRUE(mgr->evict_cached_block());
    EXPECT_TRUE(mgr->evict_cached_block());
    EXPECT_EQ(mgr->num_cached_blocks(), 0);
    EXPECT_EQ(mgr->num_free_blocks(), 8);
}

// 30. PinPrefixCanAllocateAccuracy
TEST(KVCacheManagerTest, PinPrefixCanAllocateAccuracy) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(8);

    (void)mgr->allocate_blocks(0, 4);
    (void)mgr->allocate_blocks(1, 4);
    EXPECT_EQ(mgr->num_free_blocks(), 0);

    // Pin seq 0. Now only seq 1's 4 blocks are reclaimable via eviction.
    mgr->pin_prefix(0, 4);

    EXPECT_TRUE(mgr->can_allocate(4));   // Can evict seq 1.
    EXPECT_FALSE(mgr->can_allocate(5));  // Seq 0 is pinned, can't reclaim its blocks.

    mgr->unpin_prefix(0);
    EXPECT_TRUE(mgr->can_allocate(8));  // Both sequences reclaimable now.

    mgr->free_sequence(0);
    mgr->free_sequence(1);
}

// ============================================================================
// Prefix cache collision & eviction edge case tests
// ============================================================================

// 31. Hash collision resistance: distinct token sequences produce distinct hashes
TEST(KVCacheManagerTest, HashCollisionResistance) {
    // Generate many distinct 16-token blocks and verify no hash collisions.
    const int N = 1000;
    std::unordered_set<size_t> hashes;
    for (int i = 0; i < N; i++) {
        std::vector<int32_t> tokens(16);
        // Each block has a unique pattern
        std::iota(tokens.begin(), tokens.end(), i * 16);
        size_t h = KVCacheManager::compute_block_hash(tokens, 0);
        EXPECT_TRUE(hashes.insert(h).second) << "Hash collision at block " << i;
    }
}

// 32. Hash chaining: identical blocks at different positions produce different hashes
TEST(KVCacheManagerTest, HashChainingDistinguishesPosition) {
    std::vector<int32_t> tokens(16, 42);  // All same token
    // Same tokens, but different parent hashes (simulating different prefix positions)
    size_t h0 = KVCacheManager::compute_block_hash(tokens, 0);
    size_t h1 = KVCacheManager::compute_block_hash(tokens, h0);
    size_t h2 = KVCacheManager::compute_block_hash(tokens, h1);
    // All must differ despite identical token content
    EXPECT_NE(h0, h1);
    EXPECT_NE(h1, h2);
    EXPECT_NE(h0, h2);
}

// 33. Cached block LRU eviction order: earlier-freed blocks evicted first
TEST(KVCacheManagerTest, CachedBlockLRUEvictionOrder) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(6);  // 6 blocks total
    mgr->set_prefix_caching_enabled(true);

    // Seq 0: 2 blocks (tokens 0..31)
    std::vector<int32_t> tokens_a(32);
    std::iota(tokens_a.begin(), tokens_a.end(), 0);
    (void)mgr->allocate_blocks_with_prefix(0, tokens_a);
    mgr->register_block_hashes(0, tokens_a);

    // Seq 1: 2 blocks (tokens 100..131)
    std::vector<int32_t> tokens_b(32);
    std::iota(tokens_b.begin(), tokens_b.end(), 100);
    (void)mgr->allocate_blocks_with_prefix(1, tokens_b);
    mgr->register_block_hashes(1, tokens_b);

    // Free seq 0 first, then seq 1 — seq 0's blocks are older in LRU
    mgr->free_sequence(0);
    mgr->free_sequence(1);
    EXPECT_EQ(mgr->num_cached_blocks(), 4);
    EXPECT_EQ(mgr->num_free_blocks(), 2);  // 6 total - 4 cached

    // Allocate a new seq needing 4 blocks — must evict cached blocks
    // LRU order: seq 0's blocks should be evicted first (freed earlier)
    std::vector<int32_t> tokens_c(64);
    std::iota(tokens_c.begin(), tokens_c.end(), 500);
    int reused = mgr->allocate_blocks_with_prefix(2, tokens_c);
    EXPECT_EQ(reused, 0);  // Different tokens, no prefix match
    EXPECT_EQ(static_cast<int>(mgr->block_table(2).size()), 4);

    // Seq 0's blocks should have been evicted. Seq 1's tokens should
    // still be matchable if any cached blocks remain.
    // With 6 total blocks: 4 for seq 2 + up to 2 cached from seq 1
    EXPECT_LE(mgr->num_cached_blocks(), 2);

    mgr->free_sequence(2);
}

// 34. Three sequences with overlapping prefixes of different lengths
TEST(KVCacheManagerTest, ThreeSequencesOverlappingPrefixes) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(16);
    mgr->set_prefix_caching_enabled(true);

    // Shared prefix: 32 tokens (2 blocks)
    std::vector<int32_t> shared(32);
    std::iota(shared.begin(), shared.end(), 0);

    // Seq 0: shared + 16 unique = 48 tokens (3 blocks)
    std::vector<int32_t> tokens_0(48);
    std::copy(shared.begin(), shared.end(), tokens_0.begin());
    std::iota(tokens_0.begin() + 32, tokens_0.end(), 900);

    (void)mgr->allocate_blocks_with_prefix(0, tokens_0);
    mgr->register_block_hashes(0, tokens_0);
    mgr->free_sequence(0);
    EXPECT_EQ(mgr->num_cached_blocks(), 3);

    // Seq 1: shared + different 16 = 48 tokens (3 blocks)
    // Should reuse first 2 blocks (shared prefix), allocate 1 new
    std::vector<int32_t> tokens_1(48);
    std::copy(shared.begin(), shared.end(), tokens_1.begin());
    std::iota(tokens_1.begin() + 32, tokens_1.end(), 800);

    int reused = mgr->allocate_blocks_with_prefix(1, tokens_1);
    EXPECT_EQ(reused, 2);  // 2 shared prefix blocks reused
    mgr->register_block_hashes(1, tokens_1);
    mgr->free_sequence(1);

    // Seq 2: shared + same as seq 1 unique part = 48 tokens
    // Should reuse all 3 blocks from seq 1
    int reused2 = mgr->allocate_blocks_with_prefix(2, tokens_1);
    EXPECT_EQ(reused2, 3);  // All 3 blocks reused

    mgr->free_sequence(2);
}

// 35. Pool exhaustion during allocate_blocks_with_prefix
TEST(KVCacheManagerTest, PrefixAllocPoolExhaustion) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(3);  // Only 3 blocks total
    mgr->set_prefix_caching_enabled(true);

    // Try to allocate 4 blocks (64 tokens) — exceeds pool capacity
    std::vector<int32_t> tokens(64);
    std::iota(tokens.begin(), tokens.end(), 0);

    int reused = mgr->allocate_blocks_with_prefix(0, tokens);
    EXPECT_EQ(reused, -1);  // Allocation failure

    // Sequence should not have been partially allocated
    EXPECT_TRUE(mgr->block_table(0).empty());
    EXPECT_EQ(mgr->num_free_blocks(), 3);  // All blocks still free
}

// 36. Rollback then re-prefill reuses cached prefix
TEST(KVCacheManagerTest, RollbackThenReusePrefix) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(16);
    mgr->set_prefix_caching_enabled(true);

    // Seq 0: 48 tokens (3 blocks)
    std::vector<int32_t> tokens(48);
    std::iota(tokens.begin(), tokens.end(), 0);

    (void)mgr->allocate_blocks_with_prefix(0, tokens);
    mgr->register_block_hashes(0, tokens);

    // Rollback to 20 tokens — keeps 2 blocks (block 0: tokens 0-15, block 1: tokens 16-31)
    // Block 2 (tokens 32-47) is freed
    mgr->rollback(0, 20);
    EXPECT_EQ(static_cast<int>(mgr->block_table(0).size()), 2);

    // Free seq 0 completely
    mgr->free_sequence(0);

    // Seq 1: same first 32 tokens — should reuse 2 cached blocks
    std::vector<int32_t> tokens2(32);
    std::iota(tokens2.begin(), tokens2.end(), 0);
    int reused = mgr->allocate_blocks_with_prefix(1, tokens2);
    EXPECT_EQ(reused, 2);

    mgr->free_sequence(1);
}

// 37. Re-registering block hashes for same sequence is idempotent
TEST(KVCacheManagerTest, DoubleRegisterBlockHashes) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(16);
    mgr->set_prefix_caching_enabled(true);

    std::vector<int32_t> tokens(32);
    std::iota(tokens.begin(), tokens.end(), 0);

    (void)mgr->allocate_blocks_with_prefix(0, tokens);

    // Register twice — should not crash or corrupt state
    mgr->register_block_hashes(0, tokens);
    mgr->register_block_hashes(0, tokens);

    mgr->free_sequence(0);
    EXPECT_EQ(mgr->num_cached_blocks(), 2);

    // Reuse should still work correctly
    int reused = mgr->allocate_blocks_with_prefix(1, tokens);
    EXPECT_EQ(reused, 2);

    mgr->free_sequence(1);
}

// 38. Evict all cached blocks then verify pool is fully free
TEST(KVCacheManagerTest, EvictAllCachedBlocksPoolIntegrity) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(8);
    mgr->set_prefix_caching_enabled(true);

    // Fill entire pool with cached blocks from 2 sequences
    std::vector<int32_t> tokens_a(64);
    std::iota(tokens_a.begin(), tokens_a.end(), 0);
    (void)mgr->allocate_blocks_with_prefix(0, tokens_a);
    mgr->register_block_hashes(0, tokens_a);

    std::vector<int32_t> tokens_b(64);
    std::iota(tokens_b.begin(), tokens_b.end(), 200);
    (void)mgr->allocate_blocks_with_prefix(1, tokens_b);
    mgr->register_block_hashes(1, tokens_b);

    // Free both — all 8 blocks now cached
    mgr->free_sequence(0);
    mgr->free_sequence(1);
    EXPECT_EQ(mgr->num_cached_blocks(), 8);
    EXPECT_EQ(mgr->num_free_blocks(), 0);  // All held as cached

    // Evict all cached blocks one by one
    int evicted = 0;
    while (mgr->evict_cached_block())
        evicted++;
    EXPECT_EQ(evicted, 8);
    EXPECT_EQ(mgr->num_cached_blocks(), 0);
    EXPECT_EQ(mgr->num_free_blocks(), 8);  // All returned to free pool
}

// ============================================================================
// Edge case tests
// ============================================================================

// 39. AllocateAtCapacity — exhaust the pool, verify next allocate returns false
TEST(KVCacheManagerTest, AllocateAtCapacity) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(8);

    // Fill the entire pool.
    bool ok = mgr->allocate_blocks(0, 8);
    ASSERT_TRUE(ok);
    EXPECT_EQ(mgr->num_free_blocks(), 0);

    // Next allocation should fail gracefully (not crash).
    ok = mgr->allocate_blocks(1, 1);
    EXPECT_FALSE(ok);
    EXPECT_TRUE(mgr->block_table(1).empty());

    // Original sequence should be unaffected.
    EXPECT_EQ(static_cast<int>(mgr->block_table(0).size()), 8);

    mgr->free_sequence(0);
}

// 40. AllocateFreeAllocate — fill pool, free, reallocate (block recycling)
TEST(KVCacheManagerTest, AllocateFreeAllocate) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(8);

    // Fill the pool entirely.
    bool ok = mgr->allocate_blocks(0, 8);
    ASSERT_TRUE(ok);
    EXPECT_EQ(mgr->num_free_blocks(), 0);

    // Free the sequence — all blocks return to the pool.
    mgr->free_sequence(0);
    EXPECT_EQ(mgr->num_free_blocks(), 8);

    // Reallocate the same count — should succeed with recycled blocks.
    ok = mgr->allocate_blocks(1, 8);
    EXPECT_TRUE(ok);
    EXPECT_EQ(static_cast<int>(mgr->block_table(1).size()), 8);
    EXPECT_EQ(mgr->num_free_blocks(), 0);

    mgr->free_sequence(1);
}

// 41. EvictionUnderPressure — third sequence triggers eviction of oldest unused
TEST(KVCacheManagerTest, EvictionUnderPressure) {
    SKIP_IF_NO_CUDA();

    // Pool fits only 2 sequences worth of blocks (4 + 4 = 8).
    auto mgr = MakeManager(8);

    (void)mgr->allocate_blocks(0, 4);
    (void)mgr->allocate_blocks(1, 4);
    EXPECT_EQ(mgr->num_free_blocks(), 0);

    // Touch seq 1 so seq 0 remains LRU.
    mgr->touch(1);

    // Evict LRU to make room, then allocate third sequence.
    int victim = mgr->evict_lru();
    EXPECT_EQ(victim, 0);
    EXPECT_EQ(mgr->num_free_blocks(), 4);

    bool ok = mgr->allocate_blocks(2, 4);
    EXPECT_TRUE(ok);
    EXPECT_EQ(static_cast<int>(mgr->block_table(2).size()), 4);

    // Seq 0 should be gone, seq 1 and seq 2 should be active.
    EXPECT_TRUE(mgr->block_table(0).empty());
    EXPECT_EQ(mgr->num_active_sequences(), 2);

    mgr->free_sequence(1);
    mgr->free_sequence(2);
}

// 42. ZeroBlocks — allocating 0 blocks is a no-op, no crash
TEST(KVCacheManagerTest, ZeroBlocks) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(8);

    bool ok = mgr->allocate_blocks(0, 0);
    EXPECT_TRUE(ok);
    EXPECT_TRUE(mgr->block_table(0).empty());
    EXPECT_EQ(mgr->num_free_blocks(), 8);
}

// 43. SequenceIdReuse — free seq id=0, then reuse id=0 with fresh state
TEST(KVCacheManagerTest, SequenceIdReuse) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(16);

    // Allocate seq 0 with 4 blocks.
    bool ok = mgr->allocate_blocks(0, 4);
    ASSERT_TRUE(ok);
    EXPECT_EQ(static_cast<int>(mgr->block_table(0).size()), 4);

    // Free it.
    mgr->free_sequence(0);
    EXPECT_TRUE(mgr->block_table(0).empty());
    EXPECT_EQ(mgr->num_free_blocks(), 16);

    // Reuse id=0 with a different block count.
    ok = mgr->allocate_blocks(0, 2);
    EXPECT_TRUE(ok);
    EXPECT_EQ(static_cast<int>(mgr->block_table(0).size()), 2);
    EXPECT_EQ(mgr->num_active_sequences(), 1);
    EXPECT_EQ(mgr->num_free_blocks(), 14);

    mgr->free_sequence(0);
}

// 44. EvictMiddleBlocksKeepsSinksAndWindow — StreamingLLM smart KV cache.
TEST(KVCacheManagerTest, EvictMiddleBlocksKeepsSinksAndWindow) {
    SKIP_IF_NO_CUDA();

    // 32 total blocks, default block_size = 16 tokens => 512-token capacity.
    auto mgr = MakeManager(32);

    // Allocate 20 blocks for one sequence (320 tokens).
    ASSERT_TRUE(mgr->allocate_blocks(0, 20));
    EXPECT_EQ(static_cast<int>(mgr->block_table(0).size()), 20);
    EXPECT_EQ(mgr->num_free_blocks(), 12);

    // Snapshot the original block IDs so we can verify which survive.
    auto bt_before = mgr->block_table(0);

    // Keep first 4 sink tokens (=> 1 sink block) and last 64 window tokens
    // (=> 4 window blocks). Middle = 20 - 1 - 4 = 15 blocks should be freed.
    int freed = mgr->evict_middle_blocks(/*seq_id=*/0,
                                         /*n_sink_tokens=*/4,
                                         /*n_window_tokens=*/64);
    EXPECT_EQ(freed, 15);

    const auto& bt_after = mgr->block_table(0);
    ASSERT_EQ(static_cast<int>(bt_after.size()), 20);  // unchanged length
    EXPECT_EQ(bt_after[0], bt_before[0]);              // sink survives
    EXPECT_EQ(bt_after[1], -1);                        // first middle slot freed
    EXPECT_EQ(bt_after[15], -1);                       // last middle slot freed
    EXPECT_EQ(bt_after[16], bt_before[16]);            // window survives
    EXPECT_EQ(bt_after[19], bt_before[19]);            // last block survives

    // 15 freed back into the pool.
    EXPECT_EQ(mgr->num_free_blocks(), 12 + 15);
    EXPECT_EQ(mgr->num_pinned_blocks(), 1);  // sink block was pinned

    // Idempotent: calling again must not crash and not free more.
    int freed2 = mgr->evict_middle_blocks(0, 4, 64);
    EXPECT_EQ(freed2, 0);
    EXPECT_EQ(mgr->num_free_blocks(), 12 + 15);

    // free_sequence keeps the pinned sink block alive (pin_prefix semantics).
    mgr->free_sequence(0);
    EXPECT_EQ(mgr->num_pinned_blocks(), 1);
    mgr->unpin_prefix(0);  // explicitly drop the pin
}

// 45. EvictMiddleBlocksNoOpWhenSinksExceedSequence — short sequences untouched.
TEST(KVCacheManagerTest, EvictMiddleBlocksNoOpWhenShort) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(16);
    ASSERT_TRUE(mgr->allocate_blocks(0, 3));  // only 48 tokens
    int free_before = mgr->num_free_blocks();

    // Sinks (1 block) + Window (3 blocks) >= total 3 blocks → nothing to free.
    int freed = mgr->evict_middle_blocks(0, /*n_sink_tokens=*/4,
                                         /*n_window_tokens=*/48);
    EXPECT_EQ(freed, 0);
    EXPECT_EQ(mgr->num_free_blocks(), free_before);
    EXPECT_EQ(mgr->num_pinned_blocks(), 0);

    mgr->free_sequence(0);
}

// 46. EvictMiddleBlocksRejectsZeroOrNegativeArgs.
TEST(KVCacheManagerTest, EvictMiddleBlocksZeroArgsAreNoOp) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(16);
    ASSERT_TRUE(mgr->allocate_blocks(0, 8));

    EXPECT_EQ(mgr->evict_middle_blocks(0, /*sink=*/0, /*win=*/16), 0);
    EXPECT_EQ(mgr->evict_middle_blocks(0, /*sink=*/4, /*win=*/0), 0);
    EXPECT_EQ(mgr->evict_middle_blocks(0, /*sink=*/-1, /*win=*/-1), 0);
    EXPECT_EQ(mgr->evict_middle_blocks(/*missing_seq=*/99, 4, 16), 0);

    EXPECT_EQ(static_cast<int>(mgr->block_table(0).size()), 8);
    mgr->free_sequence(0);
}

// ============================================================================
// Prefix pinning v2 — owner bookkeeping, budget FIFO (Anthropic cache_control)
// ============================================================================

// Helper: allocate a prefix-cached sequence of `n_full_blocks` distinct full
// blocks, register hashes, optionally pin the whole prompt, then free it —
// the cache_control lifecycle (pin happens at finish, before free_sequence).
static void MakePinnedFreedSeq(KVCacheManager* mgr, int seq_id, int n_full_blocks, int token_base,
                               bool pin = true) {
    std::vector<int32_t> tokens(n_full_blocks * 16);
    std::iota(tokens.begin(), tokens.end(), token_base);
    ASSERT_EQ(mgr->allocate_blocks_with_prefix(seq_id, tokens), 0);
    mgr->register_block_hashes(seq_id, tokens);
    if (pin)
        mgr->pin_prefix(seq_id, n_full_blocks);
    mgr->free_sequence(seq_id);
}

// 47. UnpinFreedSeqKeepsOtherFreedSeqPins — unpinning one already-freed owner
// must not drop pins held by OTHER already-freed owners. (The old rebuild
// path reconstructed pinned_blocks_ from seq_blocks_, which free_sequence
// erases — so every freed owner's pins silently vanished.)
TEST(KVCacheManagerTest, UnpinFreedSeqKeepsOtherFreedSeqPins) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(16);
    mgr->set_prefix_caching_enabled(true);

    MakePinnedFreedSeq(mgr.get(), 0, 2, 100);
    MakePinnedFreedSeq(mgr.get(), 1, 2, 900);
    EXPECT_EQ(mgr->num_pinned_blocks(), 4);

    mgr->unpin_prefix(0);

    // Seq 1's pins must survive.
    EXPECT_EQ(mgr->num_pinned_blocks(), 2);
    // Exactly seq 0's two blocks became reclaimable; seq 1's stay protected.
    EXPECT_TRUE(mgr->evict_cached_block());
    EXPECT_TRUE(mgr->evict_cached_block());
    EXPECT_FALSE(mgr->evict_cached_block());
    EXPECT_EQ(mgr->num_pinned_blocks(), 2);
}

// 48. PinBudgetEvictsOldestPinFifo — pinning beyond the budget unpins the
// oldest owner first; its blocks degrade to normal (reclaimable) cached
// blocks instead of leaking pinned VRAM.
TEST(KVCacheManagerTest, PinBudgetEvictsOldestPinFifo) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(16);
    mgr->set_prefix_caching_enabled(true);
    mgr->set_pin_budget_blocks(4);

    MakePinnedFreedSeq(mgr.get(), 0, 2, 100);
    MakePinnedFreedSeq(mgr.get(), 1, 2, 900);
    EXPECT_EQ(mgr->num_pinned_blocks(), 4);

    // Third pin exceeds the budget — seq 0 (oldest) must be unpinned.
    MakePinnedFreedSeq(mgr.get(), 2, 2, 1700);
    EXPECT_EQ(mgr->num_pinned_blocks(), 4);

    // Seq 0's two blocks are reclaimable now; seq 1+2 remain pinned.
    EXPECT_TRUE(mgr->evict_cached_block());
    EXPECT_TRUE(mgr->evict_cached_block());
    EXPECT_FALSE(mgr->evict_cached_block());
    EXPECT_EQ(mgr->num_pinned_blocks(), 4);
}

// 49. PinLargerThanBudgetIsCappedAndEvictsAll — a single pin larger than the
// whole budget caps to the budget after evicting every older pin.
TEST(KVCacheManagerTest, PinLargerThanBudgetIsCapped) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(16);
    mgr->set_prefix_caching_enabled(true);
    mgr->set_pin_budget_blocks(3);

    MakePinnedFreedSeq(mgr.get(), 0, 2, 100);
    EXPECT_EQ(mgr->num_pinned_blocks(), 2);

    // 5-block pin: budget 3 → old pin evicted, new pin capped to 3.
    MakePinnedFreedSeq(mgr.get(), 1, 5, 900);
    EXPECT_EQ(mgr->num_pinned_blocks(), 3);
}

// 50. RePinSameSeqReplaces — re-pinning the same owner replaces its pin set
// (no accumulation); one unpin releases everything.
TEST(KVCacheManagerTest, RePinSameSeqReplaces) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(16);
    mgr->set_prefix_caching_enabled(true);

    std::vector<int32_t> tokens(3 * 16);
    std::iota(tokens.begin(), tokens.end(), 100);
    ASSERT_EQ(mgr->allocate_blocks_with_prefix(0, tokens), 0);
    mgr->register_block_hashes(0, tokens);

    mgr->pin_prefix(0, 1);
    EXPECT_EQ(mgr->num_pinned_blocks(), 1);
    mgr->pin_prefix(0, 3);
    EXPECT_EQ(mgr->num_pinned_blocks(), 3);

    mgr->unpin_prefix(0);
    EXPECT_EQ(mgr->num_pinned_blocks(), 0);
    mgr->free_sequence(0);
}

// 51. SharedPinnedBlockSurvivesUnpinOfOneOwner — two owners pinning the same
// physical prefix blocks (cache-hit reuse): unpinning one owner must keep
// the shared blocks pinned for the other.
TEST(KVCacheManagerTest, SharedPinnedBlockSurvivesUnpinOfOneOwner) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(16);
    mgr->set_prefix_caching_enabled(true);

    std::vector<int32_t> tokens(2 * 16);
    std::iota(tokens.begin(), tokens.end(), 100);

    MakePinnedFreedSeq(mgr.get(), 0, 2, 100);

    // Seq 1: identical tokens — full reuse of the pinned blocks.
    EXPECT_EQ(mgr->allocate_blocks_with_prefix(1, tokens), 2);
    mgr->register_block_hashes(1, tokens);
    mgr->pin_prefix(1, 2);
    mgr->free_sequence(1);
    EXPECT_EQ(mgr->num_pinned_blocks(), 2);

    mgr->unpin_prefix(0);
    // Still pinned by owner 1 — nothing reclaimable.
    EXPECT_EQ(mgr->num_pinned_blocks(), 2);
    EXPECT_FALSE(mgr->evict_cached_block());

    mgr->unpin_prefix(1);
    EXPECT_EQ(mgr->num_pinned_blocks(), 0);
    EXPECT_TRUE(mgr->evict_cached_block());
}

// 52. CacheHitOnCachedBlockKeepsReclaimableCountExact — a prefix HIT on an
// unreferenced cached block removes it from the cached LRU; the reclaimable
// counter must follow. An inflated counter makes can_allocate() drift
// optimistic and lets reclaim_cached_block() spin on a pinned-only LRU.
TEST(KVCacheManagerTest, CacheHitOnCachedBlockKeepsReclaimableCountExact) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(4);
    mgr->set_prefix_caching_enabled(true);

    // Seq 0: 2 full blocks, cached on free (2 reclaimable).
    MakePinnedFreedSeq(mgr.get(), 0, 2, 100, /*pin=*/false);
    EXPECT_EQ(mgr->num_reclaimable_cached_blocks(), 2);

    // Seq 1: identical prefix — HIT takes both blocks out of the cached LRU.
    std::vector<int32_t> tokens(2 * 16);
    std::iota(tokens.begin(), tokens.end(), 100);
    EXPECT_EQ(mgr->allocate_blocks_with_prefix(1, tokens), 2);
    EXPECT_EQ(mgr->num_reclaimable_cached_blocks(), 0);

    // Free again — both blocks return to the cached LRU exactly once.
    mgr->free_sequence(1);
    EXPECT_EQ(mgr->num_reclaimable_cached_blocks(), 2);
}

}  // namespace
}  // namespace imp
