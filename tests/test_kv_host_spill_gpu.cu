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
    kv_block_to_host(c, block, out.data(), nullptr);
    cudaDeviceSynchronize();
    return out;
}

void fill_block(KVCache& c, int block, uint8_t seed) {
    std::vector<uint8_t> pat(kv_host_block_bytes(c));
    for (size_t i = 0; i < pat.size(); ++i)
        pat[i] = static_cast<uint8_t>(seed + i * 7);
    kv_block_from_host(c, pat.data(), block, nullptr);
    cudaDeviceSynchronize();
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
