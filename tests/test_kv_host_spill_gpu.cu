// KV host spill tier (#2203) on a device KVCache: a prefix block reclaimed under pool pressure goes
// to host RAM and a later prefix hit restores it bit-exact, counted as reused.
#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "memory/kv_cache_manager.h"

#include <cstdint>
#include <numeric>
#include <vector>

namespace imp {
namespace {

std::vector<uint8_t> read_block(KVCache& c, int block) {
    std::vector<uint8_t> out(kv_host_block_bytes(c));
    EXPECT_TRUE(kv_block_to_host(c, block, out.data(), nullptr));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    return out;
}

void fill_block(KVCache& c, int block, uint8_t seed) {
    std::vector<uint8_t> pat(kv_host_block_bytes(c));
    for (size_t i = 0; i < pat.size(); ++i)
        pat[i] = static_cast<uint8_t>(seed + i * 7);
    EXPECT_TRUE(kv_block_from_host(c, pat.data(), block, nullptr));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

std::vector<int32_t> tokens(int n, int32_t base) {
    std::vector<int32_t> t(n);
    std::iota(t.begin(), t.end(), base);
    return t;
}

class KVHostSpillGpuTest : public ::testing::TestWithParam<QType> {
protected:
    void SetUp() override {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0)
            GTEST_SKIP() << "no CUDA device";
    }
};

// 4-block pool: A (2 full blocks) is cached, B (4 blocks) reclaims both A blocks, A again restores
// them from the host tier with the bytes it had.
TEST_P(KVHostSpillGpuTest, ReclaimedPrefixRestoresBitExact) {
    auto cache = std::make_unique<KVCache>(2, 2, 64, GetParam(), 4);
    KVCache* c = cache.get();
    KVCacheManager mgr(std::move(cache));
    mgr.set_prefix_caching_enabled(true);
    ASSERT_TRUE(mgr.enable_host_spill(8 * kv_host_block_bytes(*c)));
    const int bs = c->block_size();

    const auto a = tokens(2 * bs, 1000);
    ASSERT_EQ(mgr.allocate_blocks_with_prefix(0, a), 0);
    const std::vector<int> a_blocks = mgr.block_table(0);
    ASSERT_EQ(a_blocks.size(), 2u);
    fill_block(*c, a_blocks[0], 11);
    fill_block(*c, a_blocks[1], 22);
    const auto want0 = read_block(*c, a_blocks[0]), want1 = read_block(*c, a_blocks[1]);
    mgr.register_block_hashes(0, a);
    mgr.free_sequence(0);
    ASSERT_EQ(mgr.num_cached_blocks(), 2);

    ASSERT_EQ(mgr.allocate_blocks_with_prefix(1, tokens(4 * bs, 5000)), 0);
    EXPECT_EQ(mgr.host_spill()->saves(), 2u) << "both A blocks go to the host tier on reclaim";
    for (int b : mgr.block_table(1))
        fill_block(*c, b, 99);  // overwrite every device block A used
    mgr.free_sequence(1);

    ASSERT_EQ(mgr.allocate_blocks_with_prefix(2, a), 2) << "A's 2 blocks come back as reused";
    EXPECT_EQ(mgr.host_spill()->restores(), 2u);
    const std::vector<int> back = mgr.block_table(2);
    EXPECT_EQ(read_block(*c, back[0]), want0);
    EXPECT_EQ(read_block(*c, back[1]), want1);
}

std::vector<uint8_t> read_minmax(KVCache& c, int block) {
    const size_t bytes = KVCache::minmax_block_bytes(c.n_kv_heads(), c.head_dim());
    std::vector<uint8_t> out(bytes * static_cast<size_t>(c.n_layers()));
    for (int l = 0; l < c.n_layers(); ++l)
        EXPECT_EQ(cudaMemcpy(out.data() + (bytes * l), c.key_minmax_ptr(l, block), bytes,
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
    return out;
}

// Writes the pages directly, not through the host layout under test.
void write_minmax(KVCache& c, int block, uint8_t seed) {
    const size_t bytes = KVCache::minmax_block_bytes(c.n_kv_heads(), c.head_dim());
    std::vector<uint8_t> pat(bytes);
    for (int l = 0; l < c.n_layers(); ++l) {
        for (size_t i = 0; i < bytes; ++i)
            pat[i] = static_cast<uint8_t>(seed + l + i * 3);
        EXPECT_EQ(cudaMemcpy(c.key_minmax_ptr(l, block), pat.data(), bytes, cudaMemcpyHostToDevice),
                  cudaSuccess);
    }
}

// Sparse attention's key min/max pages ride along: a restored block selects pages from the
// metadata its own rows produced, not from whatever the reclaiming sequence left there.
TEST_P(KVHostSpillGpuTest, ReclaimedPrefixRestoresItsKeyMinMax) {
    auto cache = std::make_unique<KVCache>(2, 2, 64, GetParam(), 4);
    KVCache* c = cache.get();
    ASSERT_TRUE(c->enable_key_minmax());
    KVCacheManager mgr(std::move(cache));
    mgr.set_prefix_caching_enabled(true);
    ASSERT_TRUE(mgr.enable_host_spill(8 * kv_host_block_bytes(*c)));
    const int bs = c->block_size();

    const auto a = tokens(2 * bs, 1000);
    ASSERT_EQ(mgr.allocate_blocks_with_prefix(0, a), 0);
    const std::vector<int> a_blocks = mgr.block_table(0);
    write_minmax(*c, a_blocks[0], 11);
    write_minmax(*c, a_blocks[1], 22);
    const auto want0 = read_minmax(*c, a_blocks[0]), want1 = read_minmax(*c, a_blocks[1]);
    mgr.register_block_hashes(0, a);
    mgr.free_sequence(0);

    ASSERT_EQ(mgr.allocate_blocks_with_prefix(1, tokens(4 * bs, 5000)), 0);
    for (int b : mgr.block_table(1))
        write_minmax(*c, b, 99);
    mgr.free_sequence(1);

    ASSERT_EQ(mgr.allocate_blocks_with_prefix(2, a), 2);
    const std::vector<int> back = mgr.block_table(2);
    EXPECT_EQ(read_minmax(*c, back[0]), want0);
    EXPECT_EQ(read_minmax(*c, back[1]), want1);
}

// Hybrid restore (#2174): a recurrent snapshot's KV chain survives its blocks going to the host
// tier, the restore follows the chain, and the restored blocks still carry the binding it named.
TEST_P(KVHostSpillGpuTest, SnapshotChainRestoresThroughTheHostTier) {
    auto cache = std::make_unique<KVCache>(2, 2, 64, GetParam(), 4);
    KVCache* c = cache.get();
    KVCacheManager mgr(std::move(cache));
    mgr.set_prefix_caching_enabled(true);
    ASSERT_TRUE(mgr.enable_host_spill(8 * kv_host_block_bytes(*c)));
    const int bs = c->block_size();

    const auto a = tokens(2 * bs, 1000);
    ASSERT_EQ(mgr.allocate_blocks_with_prefix(0, a), 0);
    const std::vector<int> a_blocks = mgr.block_table(0);
    fill_block(*c, a_blocks[0], 11);
    fill_block(*c, a_blocks[1], 22);
    const auto want0 = read_block(*c, a_blocks[0]), want1 = read_block(*c, a_blocks[1]);
    const auto chain = mgr.adopt_chain(0, a, 2, 0, 0);
    ASSERT_EQ(chain.size(), 2u);
    mgr.free_sequence(0);

    for (int round = 0; round < 2; ++round) {
        ASSERT_EQ(mgr.allocate_blocks_with_prefix(1, tokens(4 * bs, 5000 + round)), 0);
        for (int b : mgr.block_table(1))
            fill_block(*c, b, 99);
        mgr.free_sequence(1);
        EXPECT_TRUE(mgr.chain_intact(chain)) << "round " << round << ": both links sit in the host tier";

        ASSERT_EQ(mgr.allocate_blocks_with_prefix(2, a, 2, 0, chain), 2) << "round " << round;
        const std::vector<int> back = mgr.block_table(2);
        EXPECT_EQ(read_block(*c, back[0]), want0);
        EXPECT_EQ(read_block(*c, back[1]), want1);
        EXPECT_TRUE(mgr.chain_intact(chain))
            << "round " << round << ": restored blocks keep the snapshot's binding";
        mgr.free_sequence(2);
    }
    EXPECT_EQ(mgr.host_spill()->restores(), 4u);
}

// Another forward of the same tokens binds the hash: its KV is not the snapshot's, so the chain
// is broken and the restore re-prefills instead of pairing foreign KV.
TEST_P(KVHostSpillGpuTest, ForeignForwardOfTheHashBreaksTheChain) {
    auto cache = std::make_unique<KVCache>(2, 2, 64, GetParam(), 4);
    KVCache* c = cache.get();
    KVCacheManager mgr(std::move(cache));
    mgr.set_prefix_caching_enabled(true);
    ASSERT_TRUE(mgr.enable_host_spill(8 * kv_host_block_bytes(*c)));
    const int bs = c->block_size();

    const auto a = tokens(2 * bs, 1000);
    ASSERT_EQ(mgr.allocate_blocks_with_prefix(0, a), 0);
    const auto chain = mgr.adopt_chain(0, a, 2, 0, 0);
    ASSERT_EQ(chain.size(), 2u);
    mgr.free_sequence(0);
    ASSERT_EQ(mgr.allocate_blocks_with_prefix(1, tokens(4 * bs, 5000)), 0);
    mgr.free_sequence(1);
    ASSERT_TRUE(mgr.chain_intact(chain));

    ASSERT_EQ(mgr.allocate_blocks_with_prefix(3, a, 0), 0);  // a fresh forward of the same tokens
    mgr.register_block_hashes(3, a);
    EXPECT_FALSE(mgr.chain_intact(chain));
    EXPECT_EQ(mgr.allocate_blocks_with_prefix(4, a, 2, 0, chain), 0);
}

// Without the tier the same sequence re-prefills: the control that the restore is what reuses.
TEST_P(KVHostSpillGpuTest, WithoutTheTierTheReclaimedPrefixMisses) {
    auto cache = std::make_unique<KVCache>(2, 2, 64, GetParam(), 4);
    const int bs = cache->block_size();
    KVCacheManager mgr(std::move(cache));
    mgr.set_prefix_caching_enabled(true);
    const auto a = tokens(2 * bs, 1000);
    ASSERT_EQ(mgr.allocate_blocks_with_prefix(0, a), 0);
    mgr.register_block_hashes(0, a);
    mgr.free_sequence(0);
    ASSERT_EQ(mgr.allocate_blocks_with_prefix(1, tokens(4 * bs, 5000)), 0);
    mgr.free_sequence(1);
    EXPECT_EQ(mgr.allocate_blocks_with_prefix(2, a), 0);
}

INSTANTIATE_TEST_SUITE_P(Dtypes, KVHostSpillGpuTest, ::testing::Values(QType::F16, QType::INT8, QType::NVFP4),
                         [](const auto& info) { return std::string(qtype_name(info.param)); });

}  // namespace
}  // namespace imp
