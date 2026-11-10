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

// Another forward of the same tokens holds the hash on the device: the snapshot's chain still
// restores its OWN KV from the host tier, as shadows beside that primary, and again after the
// shadows themselves were reclaimed (each concurrent chat forwards the last shared block, #2409).
TEST_P(KVHostSpillGpuTest, ForeignForwardOfTheHashKeepsTheSnapshotsOwnKv) {
    auto cache = std::make_unique<KVCache>(2, 2, 64, GetParam(), 6);
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
    ASSERT_EQ(mgr.allocate_blocks_with_prefix(1, tokens(6 * bs, 5000)), 0);  // reclaims both A blocks
    mgr.free_sequence(1);

    ASSERT_EQ(mgr.allocate_blocks_with_prefix(3, a, 0), 0);  // a fresh forward of the same tokens
    for (int b : mgr.block_table(3))
        fill_block(*c, b, 77);
    mgr.register_block_hashes(3, a);
    const auto chain3 = mgr.adopt_chain(3, a, 2, 0, 0);
    EXPECT_TRUE(mgr.chain_intact(chain)) << "the host tier holds the snapshot's own forward";

    ASSERT_EQ(mgr.allocate_blocks_with_prefix(4, a, 2, 0, chain), 2);
    std::vector<int> back = mgr.block_table(4);
    EXPECT_EQ(read_block(*c, back[0]), want0) << "A's KV, not the other forward's";
    EXPECT_EQ(read_block(*c, back[1]), want1);
    EXPECT_TRUE(mgr.chain_intact(chain));
    EXPECT_TRUE(mgr.chain_intact(chain3)) << "the other forward keeps its primary binding";

    mgr.free_sequence(4);
    ASSERT_EQ(mgr.allocate_blocks_with_prefix(5, tokens(4 * bs, 9000)), 0);  // reclaims the two shadows
    for (int b : mgr.block_table(5))
        fill_block(*c, b, 99);
    mgr.free_sequence(5);
    EXPECT_TRUE(mgr.chain_intact(chain)) << "a reclaimed shadow goes to the host tier too";
    ASSERT_EQ(mgr.allocate_blocks_with_prefix(6, a, 2, 0, chain), 2);
    back = mgr.block_table(6);
    EXPECT_EQ(read_block(*c, back[0]), want0);
    EXPECT_EQ(read_block(*c, back[1]), want1);
}

struct ChainFixture {
    std::unique_ptr<KVCacheManager> mgr;
    KVCache* c = nullptr;
    int bs = 0;
    std::vector<int32_t> a;
    std::vector<KvChainLink> chain;
    std::vector<uint8_t> want0, want1;
};

// A (2 blocks, bytes 11/22) adopted as a snapshot chain, then reclaimed into a host tier of
// host_slots slots by a sequence taking the whole pool.
ChainFixture spilled_chain(QType dtype, int blocks, int host_slots) {
    ChainFixture f;
    auto cache = std::make_unique<KVCache>(2, 2, 64, dtype, blocks);
    f.c = cache.get();
    f.mgr = std::make_unique<KVCacheManager>(std::move(cache));
    f.mgr->set_prefix_caching_enabled(true);
    EXPECT_TRUE(f.mgr->enable_host_spill(host_slots * kv_host_block_bytes(*f.c)));
    f.bs = f.c->block_size();
    f.a = tokens(2 * f.bs, 1000);
    EXPECT_EQ(f.mgr->allocate_blocks_with_prefix(0, f.a), 0);
    const std::vector<int> ab = f.mgr->block_table(0);
    fill_block(*f.c, ab[0], 11);
    fill_block(*f.c, ab[1], 22);
    f.want0 = read_block(*f.c, ab[0]);
    f.want1 = read_block(*f.c, ab[1]);
    f.chain = f.mgr->adopt_chain(0, f.a, 2, 0, 0);
    f.mgr->free_sequence(0);
    EXPECT_EQ(f.mgr->allocate_blocks_with_prefix(1, tokens(blocks * f.bs, 5000)), 0);  // the whole pool
    f.mgr->free_sequence(1);
    return f;
}

// A registered 2-block prefix freed into the cache, then reclaimed by `fill_blocks` fresh blocks.
void push_through_host(KVCacheManager& mgr, int bs, int seq, int32_t base, int fill_blocks) {
    const auto x = tokens(2 * bs, base);
    ASSERT_EQ(mgr.allocate_blocks_with_prefix(seq, x), 0);
    mgr.register_block_hashes(seq, x);
    mgr.free_sequence(seq);
    ASSERT_EQ(mgr.allocate_blocks_with_prefix(seq + 1, tokens(fill_blocks * bs, base + 50000)), 0);
    mgr.free_sequence(seq + 1);
}

// The restored blocks carry the snapshot's binding: the chain holds after the host slots it came
// from were replaced (2-slot tier).
TEST_P(KVHostSpillGpuTest, RestoredBlocksKeepTheChainAfterTheirSlotsAreReplaced) {
    auto f = spilled_chain(GetParam(), 6, 2);
    ASSERT_EQ(f.mgr->allocate_blocks_with_prefix(2, f.a, 2, 0, f.chain), 2);  // stays live
    push_through_host(*f.mgr, f.bs, 3, 7000, 4);
    ASSERT_FALSE(f.mgr->host_spill()->holds(f.chain[0])) << "precondition: A's slot was replaced";
    EXPECT_TRUE(f.mgr->chain_intact(f.chain));
}

// The chain restores its own forward's slot, not the hash's newest one (another forward spilled later).
TEST_P(KVHostSpillGpuTest, ChainRestoresItsOwnSlotNotTheNewest) {
    auto f = spilled_chain(GetParam(), 6, 4);
    ASSERT_EQ(f.mgr->allocate_blocks_with_prefix(3, f.a, 0), 0);  // another forward of A's tokens
    for (int b : f.mgr->block_table(3))
        fill_block(*f.c, b, 77);
    f.mgr->register_block_hashes(3, f.a);
    f.mgr->free_sequence(3);
    ASSERT_EQ(f.mgr->allocate_blocks_with_prefix(4, tokens(6 * f.bs, 8000)), 0);  // spills it
    f.mgr->free_sequence(4);
    ASSERT_NE(f.mgr->host_spill()->origin(f.chain[0].hash).serial, f.chain[0].serial)
        << "precondition: the newest slot of the hash is the other forward's";
    ASSERT_EQ(f.mgr->allocate_blocks_with_prefix(5, f.a, 2, 0, f.chain), 2);
    const std::vector<int> back = f.mgr->block_table(5);
    EXPECT_EQ(read_block(*f.c, back[0]), f.want0);
    EXPECT_EQ(read_block(*f.c, back[1]), f.want1);
}

// A restored shadow that is reclaimed goes back to the host tier (2-slot tier, A's slots replaced).
TEST_P(KVHostSpillGpuTest, ReclaimedShadowGoesBackToTheHostTier) {
    auto f = spilled_chain(GetParam(), 8, 2);
    ASSERT_EQ(f.mgr->allocate_blocks_with_prefix(3, f.a, 0), 0);  // another forward holds the primary
    f.mgr->register_block_hashes(3, f.a);
    ASSERT_EQ(f.mgr->allocate_blocks_with_prefix(4, f.a, 2, 0, f.chain), 2);  // A's KV as shadows
    EXPECT_GT(f.mgr->host_spill()->restores(), 0u) << "precondition: A came from the host tier";
    push_through_host(*f.mgr, f.bs, 10, 7000, 4);
    ASSERT_FALSE(f.mgr->host_spill()->holds(f.chain[0])) << "precondition: A's slot was replaced";
    EXPECT_TRUE(f.mgr->chain_intact(f.chain)) << "the restored shadows carry the snapshot's binding";
    f.mgr->free_sequence(4);
    ASSERT_EQ(f.mgr->allocate_blocks_with_prefix(6, tokens(6 * f.bs, 9000)), 0);  // reclaims the shadows
    f.mgr->free_sequence(6);
    EXPECT_TRUE(f.mgr->chain_intact(f.chain));
    ASSERT_EQ(f.mgr->allocate_blocks_with_prefix(7, f.a, 2, 0, f.chain), 2);
    const std::vector<int> back = f.mgr->block_table(7);
    EXPECT_EQ(read_block(*f.c, back[0]), f.want0);
    EXPECT_EQ(read_block(*f.c, back[1]), f.want1);
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

// A growable pool (layer pitch = the ceiling) moves a block as one cudaMemcpyBatchAsync on a real
// stream; the NULL-stream path (one copy per segment) reads back the same bytes, both directions.
TEST_P(KVHostSpillGpuTest, GrowablePoolBlockRoundTripsThroughTheBatchCopy) {
    KVCache c(4, 2, 64, GetParam(), 8, 16, nullptr, 4096);
    ASSERT_TRUE(c.growable());
    cudaStream_t s = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), cudaSuccess);
    const size_t bytes = kv_host_block_bytes(c);
    uint8_t* pinned = nullptr;
    ASSERT_EQ(cudaHostAlloc(reinterpret_cast<void**>(&pinned), 2 * bytes, cudaHostAllocDefault), cudaSuccess);
    for (size_t i = 0; i < bytes; ++i)
        pinned[i] = static_cast<uint8_t>(31 + i * 13);
    const std::vector<uint8_t> want(pinned, pinned + bytes);
    EXPECT_TRUE(kv_block_from_host(c, pinned, 5, s));
    EXPECT_EQ(cudaStreamSynchronize(s), cudaSuccess);
    EXPECT_EQ(read_block(c, 5), want);

    fill_block(c, 6, 77);
    EXPECT_TRUE(kv_block_to_host(c, 6, pinned + bytes, s));
    EXPECT_EQ(cudaStreamSynchronize(s), cudaSuccess);
    EXPECT_EQ(std::vector<uint8_t>(pinned + bytes, pinned + (2 * bytes)), read_block(c, 6));
    cudaFreeHost(pinned);
    cudaStreamDestroy(s);
}

INSTANTIATE_TEST_SUITE_P(Dtypes, KVHostSpillGpuTest, ::testing::Values(QType::F16, QType::INT8, QType::NVFP4),
                         [](const auto& info) { return std::string(qtype_name(info.param)); });

}  // namespace
}  // namespace imp
