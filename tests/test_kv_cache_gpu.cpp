// KVCache / KVCacheManager tests that need a real pool: they read or write
// KV bytes, resolve device pointers, pack SWA snapshots or persist the cache.
// GPU lane (test-kv). The bookkeeping over KVCache::for_accounting() is in
// test_kv_cache.cpp and runs in `ctest -L unit`, which has no skips.

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

// Manager over a real pool.
static std::unique_ptr<KVCacheManager> MakeManagerWithMemory(int max_blocks, int n_layers = 2,
                                                             int n_kv_heads = 4, int head_dim = 64,
                                                             QType dtype = QType::F16) {
    auto cache = std::make_unique<KVCache>(n_layers, n_kv_heads, head_dim, dtype, max_blocks);
    return std::make_unique<KVCacheManager>(std::move(cache));
}

// Same helper as test_kv_cache.cpp: a prefix-cached sequence of distinct full
// blocks, hashes registered, optionally pinned, then freed.
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

// 13d. The same geometry with memory: every layer's K, K-scale and V-scale
// pointer resolves for an allocated block.
TEST(KVCacheTest, KVCacheNVFP4PerLayerPointers) {
    SKIP_IF_NO_CUDA();

    const int n_layers = 4;
    std::vector<int> nkv = {8, 8, 8, 8};
    std::vector<int> hd = {128, 256, 128, 256};
    KVCache cache(n_layers, nkv, hd, QType::NVFP4, /*max_blocks=*/2, kKVBlockSize, nullptr);

    int b0 = cache.allocate_block();
    ASSERT_GE(b0, 0);
    for (int l = 0; l < n_layers; ++l) {
        EXPECT_NE(cache.k_ptr(l, b0), nullptr) << "layer " << l;
        EXPECT_NE(cache.k_scale_ptr(l, b0), nullptr) << "layer " << l;
        EXPECT_NE(cache.v_scale_ptr(l, b0), nullptr) << "layer " << l;
    }
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
        auto mgr = MakeManagerWithMemory(32);
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
        auto mgr = MakeManagerWithMemory(32);
        mgr->set_prefix_caching_enabled(true);
        EXPECT_EQ(mgr->load_prefix_cache(path, kFpA), 3);
    }

    // Mismatched fingerprint (identical geometry) → rejected, nothing restored.
    {
        auto mgr = MakeManagerWithMemory(32);
        mgr->set_prefix_caching_enabled(true);
        EXPECT_LT(mgr->load_prefix_cache(path, kFpB), 0);
        EXPECT_EQ(mgr->num_cached_blocks(), 0);
    }

    std::remove(path.c_str());
}

// 52. CacheHitOnCachedBlockKeepsReclaimableCountExact — a prefix HIT on an
// unreferenced cached block removes it from the cached LRU; the reclaimable
// counter must follow. An inflated counter makes can_allocate() drift
// optimistic and lets reclaim_cached_block() spin on a pinned-only LRU.
TEST(KVCacheManagerTest, CacheHitOnCachedBlockKeepsReclaimableCountExact) {
    SKIP_IF_NO_CUDA();

    auto mgr = MakeManagerWithMemory(4);
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

// A SWA-space id resolves to memory in a windowed layer's own region.
TEST(KVCacheTest, SwaBlockPointerResolves) {
    SKIP_IF_NO_CUDA();

    const int n_layers = 2;
    std::vector<int> nkv(n_layers, 4);
    std::vector<int> hd(n_layers, 64);
    std::vector<char> is_swa = {1, 0};
    KVCache cache(n_layers, nkv, hd, QType::F16, /*global*/ 64, kKVBlockSize, nullptr, is_swa,
                  /*swa_max_blocks=*/6);

    const int id = cache.allocate_swa_block();
    ASSERT_GE(id, 0);
    EXPECT_NE(cache.k_ptr(0, id), nullptr);
    EXPECT_NE(cache.v_ptr(0, id), nullptr);
}

// rollback drops the SWA tail in lockstep with the global table.
TEST(KVCacheManagerTest, SwaRollback) {
    SKIP_IF_NO_CUDA();

    const int bs = 16;
    std::vector<int> nkv(2, 4), hd(2, 64);
    std::vector<char> is_swa = {1, 0};
    auto cache = std::make_unique<KVCache>(2, nkv, hd, QType::F16, 256, bs, nullptr, is_swa, 16);
    KVCacheManager mgr(std::move(cache));
    mgr.enable_swa_sizing(bs, bs);  // window=16, slack=16

    ASSERT_TRUE(mgr.allocate_blocks(0, 10));
    ASSERT_TRUE(mgr.swa_prepare(0, 160));
    int live_before = 16 - mgr.kv_cache()->num_free_swa_blocks();
    EXPECT_GT(live_before, 0);

    // Roll back to 5 blocks (80 tokens) — the SWA table shrinks to match and
    // frees the dropped tail.
    mgr.rollback(0, 80);
    EXPECT_EQ(static_cast<int>(mgr.swa_block_table(0).size()), 5);
    EXPECT_EQ(static_cast<int>(mgr.block_table(0).size()), 5);
}

// SWA window snapshots (kv_cache.swa_snapshot_mb): pack the live window of a
// sequence into a slab, restore it into a fresh sequence with a reused
// global prefix — the restored blocks must be byte-identical, private, and
// placed at the same positional slots with holes before the window.
TEST(KVCacheManagerTest, SwaSnapshotPackRestoreRoundtrip) {
    SKIP_IF_NO_CUDA();

    const int bs = 16;
    const int n_layers = 2;
    std::vector<int> nkv(n_layers, 4), hd(n_layers, 64);
    std::vector<char> is_swa = {1, 0};
    auto cache = std::make_unique<KVCache>(n_layers, nkv, hd, QType::F16, 256, bs, nullptr, is_swa,
                                           /*swa_max*/ 16);
    KVCache* raw = cache.get();
    KVCacheManager mgr(std::move(cache));
    mgr.enable_swa_sizing(/*window*/ 2 * bs, /*slack*/ bs);
    ASSERT_TRUE(mgr.enable_swa_snapshots());
    ASSERT_GT(mgr.swa_snapshot_bytes(), 0u);
    // first live block at 160 tokens: (160 - 32 - 16) / 16 = 7.
    EXPECT_EQ(mgr.swa_first_live_block(160), 7);

    // Sequence 0: 10 global blocks (160 tokens), live window blocks 7..9.
    ASSERT_TRUE(mgr.allocate_blocks(0, 10));
    ASSERT_TRUE(mgr.swa_prepare(0, 160));
    std::vector<int> swa0 = mgr.swa_block_table(0);
    const size_t kvb = raw->block_bytes(0);
    std::vector<char> pat(kvb);
    for (int b = 7; b < 10; ++b) {
        std::fill(pat.begin(), pat.end(), static_cast<char>(0x40 + b));
        ASSERT_EQ(cudaMemcpy(raw->k_ptr(0, swa0[b]), pat.data(), kvb, cudaMemcpyHostToDevice),
                  cudaSuccess);
        std::fill(pat.begin(), pat.end(), static_cast<char>(0x60 + b));
        ASSERT_EQ(cudaMemcpy(raw->v_ptr(0, swa0[b]), pat.data(), kvb, cudaMemcpyHostToDevice),
                  cudaSuccess);
    }
    void* slab = nullptr;
    ASSERT_EQ(cudaMalloc(&slab, mgr.swa_snapshot_bytes()), cudaSuccess);
    ASSERT_TRUE(mgr.swa_snapshot_pack(0, 160, slab, nullptr));
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);

    // Fresh sequence 1 whose global prefix [0, 160) was reused: restore.
    ASSERT_TRUE(mgr.allocate_blocks(1, 10));
    ASSERT_TRUE(mgr.swa_snapshot_restore(1, 160, slab, nullptr));
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
    const auto& swa1 = mgr.swa_block_table(1);
    ASSERT_EQ(static_cast<int>(swa1.size()), 10);
    for (int b = 0; b < 7; ++b)
        EXPECT_EQ(swa1[b], -1) << "pre-window block " << b << " must stay a hole";
    std::vector<char> got(kvb);
    for (int b = 7; b < 10; ++b) {
        ASSERT_GE(swa1[b], 0);
        EXPECT_NE(swa1[b], swa0[b]) << "restored block must be a private fresh allocation";
        ASSERT_EQ(cudaMemcpy(got.data(), raw->k_ptr(0, swa1[b]), kvb, cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(got.front(), static_cast<char>(0x40 + b));
        EXPECT_EQ(got.back(), static_cast<char>(0x40 + b));
        ASSERT_EQ(cudaMemcpy(got.data(), raw->v_ptr(0, swa1[b]), kvb, cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(got.front(), static_cast<char>(0x60 + b));
    }
    // Continuation prepare extends the restored table without touching it.
    ASSERT_TRUE(mgr.swa_prepare(1, 160, 176));
    cudaFree(slab);
    mgr.free_sequence(0);
    mgr.free_sequence(1);
    EXPECT_EQ(mgr.kv_cache()->num_free_swa_blocks(), 16);
}

// Generation-end saves pack at the block-FLOOR of the live context; with a
// non-block-aligned slack the lowest slack block can already be trimmed.
// The pack must tolerate holes below the read-relevant boundary (zero-fill)
// and the restore must reproduce the live blocks byte-identically.
TEST(KVCacheManagerTest, SwaSnapshotFinishPackToleratesTrimmedSlack) {
    SKIP_IF_NO_CUDA();

    const int bs = 16;
    std::vector<int> nkv(2, 4), hd(2, 64);
    std::vector<char> is_swa = {1, 0};
    auto cache = std::make_unique<KVCache>(2, nkv, hd, QType::F16, 256, bs, nullptr, is_swa,
                                           /*swa_max*/ 16);
    KVCache* raw = cache.get();
    KVCacheManager mgr(std::move(cache));
    mgr.enable_swa_sizing(/*window*/ 2 * bs, /*slack*/ 17);  // slack NOT block-aligned
    ASSERT_TRUE(mgr.enable_swa_snapshots());

    // Live context 170 tokens (11-block table): live span starts at block
    // (170-49)/16 = 7. The finish save packs at floor(170/16)*16 = 160,
    // where first_live(160) = (160-49)/16 = 6 — slot 6 is a trimmed hole
    // below the read boundary (160-32)/16 - 1 = 7.
    ASSERT_TRUE(mgr.allocate_blocks(0, 11));
    ASSERT_TRUE(mgr.swa_prepare(0, 170));
    std::vector<int> swa0 = mgr.swa_block_table(0);
    EXPECT_EQ(swa0[6], -1) << "slot 6 must be a hole for this scenario";
    const size_t kvb = raw->block_bytes(0);
    std::vector<char> pat(kvb);
    for (int b = 7; b < 10; ++b) {
        ASSERT_GE(swa0[b], 0);
        std::fill(pat.begin(), pat.end(), static_cast<char>(0x40 + b));
        ASSERT_EQ(cudaMemcpy(raw->k_ptr(0, swa0[b]), pat.data(), kvb, cudaMemcpyHostToDevice),
                  cudaSuccess);
    }
    void* slab = nullptr;
    ASSERT_EQ(cudaMalloc(&slab, mgr.swa_snapshot_bytes()), cudaSuccess);
    ASSERT_TRUE(mgr.swa_snapshot_pack(0, 160, slab, nullptr));
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);

    ASSERT_TRUE(mgr.allocate_blocks(1, 10));
    ASSERT_TRUE(mgr.swa_snapshot_restore(1, 160, slab, nullptr));
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
    const auto& swa1 = mgr.swa_block_table(1);
    std::vector<char> got(kvb);
    // Tolerated slot restores as zeros (never read: below the window of any
    // continuation query at >= 160).
    ASSERT_GE(swa1[6], 0);
    ASSERT_EQ(cudaMemcpy(got.data(), raw->k_ptr(0, swa1[6]), kvb, cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(got.front(), 0);
    EXPECT_EQ(got.back(), 0);
    for (int b = 7; b < 10; ++b) {
        ASSERT_GE(swa1[b], 0);
        ASSERT_EQ(cudaMemcpy(got.data(), raw->k_ptr(0, swa1[b]), kvb, cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(got.front(), static_cast<char>(0x40 + b));
        EXPECT_EQ(got.back(), static_cast<char>(0x40 + b));
    }
    cudaFree(slab);
    mgr.free_sequence(0);
    mgr.free_sequence(1);
}

// Restore on an exhausted SWA group fails cleanly: partial allocations are
// released and the caller can fall back to a full prefill.
TEST(KVCacheManagerTest, SwaSnapshotRestoreExhaustionRollsBack) {
    SKIP_IF_NO_CUDA();

    const int bs = 16;
    std::vector<int> nkv(2, 4), hd(2, 64);
    std::vector<char> is_swa = {1, 0};
    // Group of 4: seq 0's window takes 3, leaving only 1 free.
    auto cache = std::make_unique<KVCache>(2, nkv, hd, QType::F16, 256, bs, nullptr, is_swa,
                                           /*swa_max*/ 4);
    KVCacheManager mgr(std::move(cache));
    mgr.enable_swa_sizing(2 * bs, bs);
    ASSERT_TRUE(mgr.enable_swa_snapshots());

    ASSERT_TRUE(mgr.allocate_blocks(0, 10));
    ASSERT_TRUE(mgr.swa_prepare(0, 160));
    void* slab = nullptr;
    ASSERT_EQ(cudaMalloc(&slab, mgr.swa_snapshot_bytes()), cudaSuccess);
    ASSERT_TRUE(mgr.swa_snapshot_pack(0, 160, slab, nullptr));
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);

    const int free_before = mgr.kv_cache()->num_free_swa_blocks();
    ASSERT_EQ(free_before, 1);
    ASSERT_TRUE(mgr.allocate_blocks(1, 10));
    EXPECT_FALSE(mgr.swa_snapshot_restore(1, 160, slab, nullptr));
    EXPECT_EQ(mgr.kv_cache()->num_free_swa_blocks(), free_before)
        << "failed restore must release its partial allocations";
    cudaFree(slab);
    mgr.free_sequence(0);
    mgr.free_sequence(1);
}

}  // namespace
}  // namespace imp
