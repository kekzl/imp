#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "memory/device_allocator.h"
#include "memory/pinned_allocator.h"
#include "memory/kv_cache.h"
#include "memory/kv_cache_manager.h"
#include "core/tensor.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

namespace imp {
namespace {

// ============================================================================
// Helper: skip the entire test if no CUDA device is available.
// ============================================================================
static bool HasCudaDevice() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

#define SKIP_IF_NO_CUDA()                                                     \
    do {                                                                       \
        if (!HasCudaDevice()) {                                                \
            GTEST_SKIP() << "No CUDA device available";                        \
        }                                                                      \
    } while (0)

// ============================================================================
// DeviceAllocator tests
// ============================================================================

// 1. DeviceAllocBasic
TEST(DeviceAllocatorTest, DeviceAllocBasic) {
    SKIP_IF_NO_CUDA();

    DeviceAllocator alloc;
    EXPECT_EQ(alloc.allocated(), 0u);

    void* ptr = alloc.allocate(4096);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(alloc.allocated(), 4096u);

    // Synchronize default stream so the async free completes deterministically.
    alloc.deallocate(ptr);
    cudaStreamSynchronize(nullptr);

    EXPECT_EQ(alloc.allocated(), 0u);
}

// 2. DeviceAllocPeakTracking
TEST(DeviceAllocatorTest, DeviceAllocPeakTracking) {
    SKIP_IF_NO_CUDA();

    DeviceAllocator alloc;

    void* a = alloc.allocate(1024);
    ASSERT_NE(a, nullptr);
    void* b = alloc.allocate(2048);
    ASSERT_NE(b, nullptr);

    // Peak should be 1024 + 2048 = 3072
    EXPECT_EQ(alloc.peak_allocated(), 3072u);
    EXPECT_EQ(alloc.allocated(), 3072u);

    // Free A -- allocated drops, but peak stays at high-water mark.
    alloc.deallocate(a);
    cudaStreamSynchronize(nullptr);

    EXPECT_EQ(alloc.allocated(), 2048u);
    EXPECT_EQ(alloc.peak_allocated(), 3072u);

    // Reset peak stats -- peak should now equal current allocated.
    alloc.reset_peak_stats();
    EXPECT_EQ(alloc.peak_allocated(), 2048u);

    // Clean up.
    alloc.deallocate(b);
    cudaStreamSynchronize(nullptr);
    EXPECT_EQ(alloc.allocated(), 0u);
}

// 3. DeviceAllocZeroBytes
TEST(DeviceAllocatorTest, DeviceAllocZeroBytes) {
    SKIP_IF_NO_CUDA();

    DeviceAllocator alloc;
    void* ptr = alloc.allocate(0);
    EXPECT_EQ(ptr, nullptr);
    EXPECT_EQ(alloc.allocated(), 0u);
}

// ============================================================================
// PinnedAllocator tests
// ============================================================================

// 4. PinnedAllocBasic
TEST(PinnedAllocatorTest, PinnedAllocBasic) {
    SKIP_IF_NO_CUDA();

    PinnedAllocator alloc(1 << 20); // 1 MiB pool

    void* ptr = alloc.allocate(512);
    ASSERT_NE(ptr, nullptr);
    EXPECT_GT(alloc.used(), 0u);

    // Write and read back data to verify the memory is actually usable.
    std::memset(ptr, 0xAB, 512);
    auto* bytes = static_cast<uint8_t*>(ptr);
    for (int i = 0; i < 512; ++i) {
        EXPECT_EQ(bytes[i], 0xAB) << "Byte mismatch at offset " << i;
    }

    alloc.deallocate(ptr);
    EXPECT_EQ(alloc.used(), 0u);
}

// 5. PinnedAllocPoolReuse
TEST(PinnedAllocatorTest, PinnedAllocPoolReuse) {
    SKIP_IF_NO_CUDA();

    PinnedAllocator alloc(1 << 20); // 1 MiB pool

    void* a = alloc.allocate(1024);
    ASSERT_NE(a, nullptr);
    size_t used_after_a = alloc.used();
    EXPECT_GT(used_after_a, 0u);

    alloc.deallocate(a);
    EXPECT_EQ(alloc.used(), 0u);

    // Allocate the same size again; should reuse from the free list.
    void* b = alloc.allocate(1024);
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(alloc.used(), used_after_a); // Same size class, same used.

    // The pool-based free-list should hand back the same pointer.
    EXPECT_EQ(a, b);

    alloc.deallocate(b);
    EXPECT_EQ(alloc.used(), 0u);
}

// 6. PinnedAllocZeroBytes
TEST(PinnedAllocatorTest, PinnedAllocZeroBytes) {
    SKIP_IF_NO_CUDA();

    PinnedAllocator alloc(1 << 20);
    void* ptr = alloc.allocate(0);
    EXPECT_EQ(ptr, nullptr);
    EXPECT_EQ(alloc.used(), 0u);
}

// ============================================================================
// KVCache tests
// ============================================================================

// 7. KVCacheConstruction
TEST(KVCacheTest, KVCacheConstruction) {
    SKIP_IF_NO_CUDA();

    const int n_layers   = 2;
    const int n_kv_heads = 4;
    const int head_dim   = 64;
    const int max_blocks = 8;

    KVCache cache(n_layers, n_kv_heads, head_dim, DType::FP16, max_blocks);

    EXPECT_EQ(cache.n_layers(), n_layers);
    EXPECT_EQ(cache.n_kv_heads(), n_kv_heads);
    EXPECT_EQ(cache.head_dim(), head_dim);
    EXPECT_EQ(cache.dtype(), DType::FP16);
    EXPECT_EQ(cache.total_blocks(), max_blocks);
    EXPECT_EQ(cache.num_free_blocks(), max_blocks);

    // block_bytes = kKVBlockSize * n_kv_heads * head_dim * dtype_size(FP16)
    //            = 16 * 4 * 64 * 2 = 8192
    size_t expected_block_bytes =
        static_cast<size_t>(kKVBlockSize) * n_kv_heads * head_dim * dtype_size(DType::FP16);
    EXPECT_EQ(cache.block_bytes(), expected_block_bytes);
    EXPECT_EQ(expected_block_bytes, 8192u);
}

// 8. KVCacheBlockAllocation
TEST(KVCacheTest, KVCacheBlockAllocation) {
    SKIP_IF_NO_CUDA();

    const int max_blocks = 8;
    KVCache cache(2, 4, 64, DType::FP16, max_blocks);

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
    KVCache cache(2, 4, 64, DType::FP16, max_blocks);

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
    KVCache cache(2, 4, 64, DType::FP16, max_blocks);

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
    EXPECT_EQ(cache.num_free_blocks(), free_before); // Unchanged.

    // Second free: ref_count drops to 0, block IS returned to free list.
    cache.free_block(block);
    EXPECT_EQ(cache.ref_count(block), 0);
    EXPECT_EQ(cache.num_free_blocks(), free_before + 1);
}

// 11. KVCachePointers
TEST(KVCacheTest, KVCachePointers) {
    SKIP_IF_NO_CUDA();

    const int n_layers   = 2;
    const int n_kv_heads = 4;
    const int head_dim   = 64;
    const int max_blocks = 8;

    KVCache cache(n_layers, n_kv_heads, head_dim, DType::FP16, max_blocks);
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

    // K and V pointers for the same (layer, block) should be distinct and
    // exactly block_bytes apart (V immediately follows K).
    EXPECT_NE(k0_l0, v0_l0);
    ptrdiff_t kv_diff = static_cast<char*>(v0_l0) - static_cast<char*>(k0_l0);
    EXPECT_EQ(static_cast<size_t>(kv_diff), bb);

    // Expected offsets:
    //   K(layer, block) = (layer * max_blocks * 2 + block * 2 + 0) * bb
    //   V(layer, block) = (layer * max_blocks * 2 + block * 2 + 1) * bb
    // Verify layer=1, block=0 K pointer is at the expected offset from
    // layer=0, block=0 K pointer.
    ptrdiff_t layer_diff = static_cast<char*>(k0_l1) - static_cast<char*>(k0_l0);
    size_t expected_layer_stride = static_cast<size_t>(max_blocks) * 2 * bb;
    EXPECT_EQ(static_cast<size_t>(layer_diff), expected_layer_stride);

    // Verify block=1 K is 2*bb after block=0 K (within same layer).
    ptrdiff_t block_diff = static_cast<char*>(k1_l0) - static_cast<char*>(k0_l0);
    EXPECT_EQ(static_cast<size_t>(block_diff), 2 * bb);

    // Write a known value via cudaMemcpy to k_ptr(0, b0), read back, verify.
    size_t num_elements = kKVBlockSize * n_kv_heads * head_dim; // elements per block
    size_t buf_bytes = num_elements * sizeof(uint16_t);         // FP16 = 2 bytes
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

    const int n_layers   = 2;
    const int n_kv_heads = 2;
    const int head_dim   = 32;
    const int max_blocks = 4;

    KVCache cache(n_layers, n_kv_heads, head_dim, DType::FP32, max_blocks);
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

// Helper to create a KVCacheManager wrapping a fresh KVCache.
static std::unique_ptr<KVCacheManager> MakeManager(int max_blocks,
                                                    int n_layers   = 2,
                                                    int n_kv_heads = 4,
                                                    int head_dim   = 64,
                                                    DType dtype    = DType::FP16) {
    auto cache = std::make_unique<KVCache>(n_layers, n_kv_heads, head_dim,
                                           dtype, max_blocks);
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

    mgr->allocate_blocks(0, 2);
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

    mgr->allocate_blocks(0, 4);
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
    mgr->allocate_blocks(0, 3); // seq 0: 3 blocks  (LRU order: 0)
    mgr->allocate_blocks(1, 3); // seq 1: 3 blocks  (LRU order: 0, 1)
    mgr->allocate_blocks(2, 2); // seq 2: 2 blocks  (LRU order: 0, 1, 2)
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

// 18. ManagerCanAllocate
TEST(KVCacheManagerTest, ManagerCanAllocate) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(8);

    mgr->allocate_blocks(0, 4);
    mgr->allocate_blocks(1, 4);
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

// 19. ManagerPrefixCaching
TEST(KVCacheManagerTest, ManagerPrefixCaching) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManager(16);

    // Allocate 3 blocks for seq 0.
    mgr->allocate_blocks(0, 3);
    const auto& table0 = mgr->block_table(0);
    ASSERT_EQ(static_cast<int>(table0.size()), 3);

    // Register prefix under hash 42.
    const size_t hash = 42;
    mgr->register_prefix(0, hash);

    // find_prefix should return the same 3 block IDs.
    std::vector<int> found = mgr->find_prefix(hash);
    ASSERT_EQ(found.size(), 3u);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(found[i], table0[i]);
    }

    // A non-existent hash should return empty.
    EXPECT_TRUE(mgr->find_prefix(99999).empty());

    // Share 2 blocks from seq 0 to seq 1.
    mgr->share_prefix(0, 1, 2);

    const auto& table1 = mgr->block_table(1);
    ASSERT_EQ(static_cast<int>(table1.size()), 2);

    // The first 2 blocks of seq 1 should be the same block IDs as seq 0's
    // first 2 blocks.
    EXPECT_EQ(table1[0], table0[0]);
    EXPECT_EQ(table1[1], table0[1]);

    // The shared blocks should now have ref_count = 2.  We need to peek into
    // the underlying KVCache; create one manually to access ref_count.
    // Since KVCacheManager owns the cache, we verify indirectly: after freeing
    // seq 0, the blocks should still be alive (ref_count 1) because seq 1
    // holds references.  Free seq 0 and verify free blocks only went up by 1
    // (the third, unshared block).

    int free_before_free0 = mgr->num_free_blocks();
    mgr->free_sequence(0);

    // Seq 0 had 3 blocks.  2 are shared (ref_count was 2, now 1 -- not freed).
    // 1 was unshared (ref_count was 1, now 0 -- freed).
    // So free count should increase by exactly 1.
    EXPECT_EQ(mgr->num_free_blocks(), free_before_free0 + 1);
    EXPECT_EQ(mgr->num_active_sequences(), 1); // Only seq 1 left.

    // Now free seq 1.  Its 2 blocks (ref_count goes from 1 to 0) are freed.
    int free_before_free1 = mgr->num_free_blocks();
    mgr->free_sequence(1);
    EXPECT_EQ(mgr->num_free_blocks(), free_before_free1 + 2);
    EXPECT_EQ(mgr->num_active_sequences(), 0);

    // All 16 blocks should now be free again.
    EXPECT_EQ(mgr->num_free_blocks(), 16);
}

} // namespace
} // namespace imp
