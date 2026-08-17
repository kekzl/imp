// The growable backend against the real driver.
//
// tools/analysis/vmm_wsl2_probe.cu established that the mechanism works on this
// box; this asserts that imp's use of it does. The properties are the ones the
// KV pool depends on and cannot check for itself at runtime:
//
//   the base address survives growth, because a captured CUDA graph has baked
//   pointers into the pool and re-instantiating graphs would cost 2-3x decode;
//
//   committing one interior range does not disturb another, because the pool
//   is laid out per layer and grows every layer's region at once;
//
//   data written before a growth is still there afterwards.
//
// Skips rather than fails where there is no device or no virtual memory
// management: that is a supported configuration, it just has a fixed pool.

#include "memory/backend.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <vector>

using namespace imp;

namespace {

constexpr size_t kMiB = 1024 * 1024;

bool have_device() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

// Fill `bytes` at `p` with a byte pattern, and read it back.
void write_pattern(void* p, size_t bytes, unsigned char v) {
    ASSERT_EQ(cudaMemset(p, v, bytes), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

size_t count_mismatches(const void* p, size_t bytes, unsigned char v) {
    std::vector<unsigned char> host(bytes);
    EXPECT_EQ(cudaMemcpy(host.data(), p, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
    size_t bad = 0;
    for (unsigned char b : host)
        bad += (b != v);
    return bad;
}

class VmmBackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!have_device())
            GTEST_SKIP() << "no CUDA device";
        be_ = vmm_backend();
        if (be_ == nullptr)
            GTEST_SKIP() << "device has no virtual memory management; the pool stays fixed";
    }
    Backend* be_ = nullptr;
};

TEST_F(VmmBackendTest, ReservationCostsNoMemoryAndGrowthKeepsTheBase) {
    size_t free_before = 0, total = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_before, &total), cudaSuccess);

    // Reserve far more than is committed: this is the whole point, a ceiling
    // that costs address space rather than VRAM.
    auto res = be_->acquire_growable(512 * kMiB, 0, 256, RegionTag::KvBlockPool);
    ASSERT_TRUE(res) << "acquire_growable failed: " << mem_error_name(res.error);
    Region region = std::move(res.region);
    void* const base = region.base();
    ASSERT_NE(base, nullptr);
    EXPECT_EQ(region.committed(), 0u);

    size_t free_after_reserve = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_after_reserve, &total), cudaSuccess);
    EXPECT_LT(free_before - free_after_reserve, 8 * kMiB)
        << "reserving address space must not cost physical memory";

    ASSERT_EQ(be_->commit_range(region, 0, 8 * kMiB), MemError::Ok);
    write_pattern(base, 8 * kMiB, 0xA5);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    ASSERT_EQ(be_->commit_range(region, 64 * kMiB, 8 * kMiB), MemError::Ok);
    EXPECT_EQ(region.base(), base) << "growth must not move the base (I3)";
    EXPECT_EQ(count_mismatches(base, 8 * kMiB, 0xA5), 0u)
        << "committing elsewhere must not disturb data already written";
}

// The KV pool commits one range per layer, and those ranges are interior and
// not adjacent. A backend that only understood a growing prefix would have to
// commit everything up to the last layer, i.e. the whole reservation.
TEST_F(VmmBackendTest, InteriorRangesAreIndependent) {
    auto res = be_->acquire_growable(256 * kMiB, 0, 256, RegionTag::KvBlockPool);
    ASSERT_TRUE(res);
    Region region = std::move(res.region);
    char* const base = static_cast<char*>(region.base());

    struct Range {
        size_t offset;
        unsigned char value;
    };
    const Range ranges[] = {{0, 0x11}, {32 * kMiB, 0x22}, {96 * kMiB, 0x33}};
    for (const Range& r : ranges) {
        ASSERT_EQ(be_->commit_range(region, r.offset, 4 * kMiB), MemError::Ok) << r.offset;
        write_pattern(base + r.offset, 4 * kMiB, r.value);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
    }
    for (const Range& r : ranges)
        EXPECT_EQ(count_mismatches(base + r.offset, 4 * kMiB, r.value), 0u) << r.offset;

    // Releasing the middle one leaves its neighbours alone. This is what makes
    // shrinking safe: a pool gives back the layers it decided to give back.
    ASSERT_EQ(be_->decommit_range(region, 32 * kMiB, 4 * kMiB), MemError::Ok);
    EXPECT_EQ(count_mismatches(base, 4 * kMiB, 0x11), 0u);
    EXPECT_EQ(count_mismatches(base + 96 * kMiB, 4 * kMiB, 0x33), 0u);
}

// cudaFree does not return a process's peak commitment on WSL2/WDDM, which is
// the reason a clamped server stays clamped. Decommit is the one path that
// does, and it is the premise of shrinking at all.
TEST_F(VmmBackendTest, DecommitReturnsMemoryToTheDriver) {
    auto res = be_->acquire_growable(512 * kMiB, 0, 256, RegionTag::KvBlockPool);
    ASSERT_TRUE(res);
    Region region = std::move(res.region);

    size_t free_reserved = 0, total = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_reserved, &total), cudaSuccess);

    ASSERT_EQ(be_->commit_range(region, 0, 256 * kMiB), MemError::Ok);
    size_t free_committed = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_committed, &total), cudaSuccess);
    EXPECT_GT(free_reserved, free_committed + 200 * kMiB) << "committing must actually take memory";

    ASSERT_EQ(be_->decommit_range(region, 0, 256 * kMiB), MemError::Ok);
    size_t free_back = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_back, &total), cudaSuccess);
    EXPECT_GE(free_back + 8 * kMiB, free_reserved) << "decommit must return the memory, not merely unmap it";
    EXPECT_EQ(region.committed(), 0u);
}

}  // namespace

// ── The KV pool itself ──────────────────────────────────────────────────────

#include "memory/kv_cache.h"

namespace {

// A pool built with a ceiling starts at what it could afford and grows into
// what it asked for. The scenario is a server started while another process
// still held the card: it lands on a fraction of its pool and, before this,
// stayed there for the rest of its life.
TEST_F(VmmBackendTest, PoolStartsSmallAndGrowsIntoItsCeiling) {
    constexpr int kStart = 64, kCeiling = 512;
    KVCache kv(/*n_layers=*/4, /*n_kv_heads=*/8, /*head_dim=*/128, QType::F16, kStart,
               /*block_size=*/16, /*alloc=*/nullptr, kCeiling);

    ASSERT_EQ(kv.total_blocks(), kStart) << "only what is committed may be handed out";
    ASSERT_EQ(kv.ceiling_blocks(), kCeiling);

    // Every block the pool admits to having must be usable in every layer, or
    // admission hands out memory that faults on the first write.
    auto write_all = [&](int block, unsigned char v) {
        for (int l = 0; l < 4; l++) {
            write_pattern(kv.k_ptr(l, block), kv.block_bytes(), v);
            write_pattern(kv.v_ptr(l, block), kv.block_bytes(), v);
        }
    };
    write_all(kStart - 1, 0x5A);
    ASSERT_FALSE(::testing::Test::HasFatalFailure()) << "the last committed block must be backed";

    const int grown = kv.try_grow_to(300);
    EXPECT_GE(grown, 300);
    EXPECT_EQ(kv.total_blocks(), grown);
    write_all(299, 0x3C);
    ASSERT_FALSE(::testing::Test::HasFatalFailure()) << "a block that growth added must be backed";

    // Data written before the growth is still there: the base did not move and
    // the old ranges were not remapped.
    for (int l = 0; l < 4; l++)
        EXPECT_EQ(count_mismatches(kv.k_ptr(l, kStart - 1), kv.block_bytes(), 0x5A), 0u) << "layer " << l;

    // The ceiling is a ceiling.
    EXPECT_EQ(kv.try_grow_to(kCeiling * 4), kCeiling);
    EXPECT_EQ(kv.total_blocks(), kCeiling);
}

// The fixed pool zeroes itself with one big memset at construction, and blocks
// are handed out on the strength of that. Driver-committed pages make no such
// promise, so a grown block could arrive with stale bytes in the slots the
// sequence does not fill — which reads as rare, unreproducible output rather
// than as an error.
TEST_F(VmmBackendTest, GrownBlocksArriveZeroed) {
    KVCache kv(/*n_layers=*/2, /*n_kv_heads=*/8, /*head_dim=*/128, QType::F16, /*max_blocks=*/32,
               /*block_size=*/16, /*alloc=*/nullptr, /*ceiling=*/256);
    // Dirty the ceiling's worth of address space through the pool's own view,
    // so a later commit that reused these pages would be visible.
    ASSERT_EQ(kv.try_grow_to(128), 128);
    for (int l = 0; l < 2; l++)
        write_pattern(kv.k_ptr(l, 100), kv.block_bytes(), 0xEE);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    ASSERT_EQ(kv.try_grow_to(200), 200);
    for (int l = 0; l < 2; l++) {
        EXPECT_EQ(count_mismatches(kv.k_ptr(l, 199), kv.block_bytes(), 0x00), 0u)
            << "a grown block must start clean, layer " << l;
        EXPECT_EQ(count_mismatches(kv.v_ptr(l, 199), kv.block_bytes(), 0x00), 0u) << "layer " << l;
        // And the growth must not have zeroed what was already live.
        EXPECT_EQ(count_mismatches(kv.k_ptr(l, 100), kv.block_bytes(), 0xEE), 0u)
            << "growth erased live KV, layer " << l;
    }
}

// Without a ceiling nothing changes, which is what keeps every existing model
// on exactly the path it had before.
TEST_F(VmmBackendTest, PoolWithoutACeilingIsFixed) {
    KVCache kv(/*n_layers=*/2, /*n_kv_heads=*/8, /*head_dim=*/128, QType::F16, /*max_blocks=*/32,
               /*block_size=*/16);
    EXPECT_EQ(kv.total_blocks(), 32);
    EXPECT_EQ(kv.ceiling_blocks(), 32);
    EXPECT_EQ(kv.try_grow_to(64), 32) << "a fixed pool must refuse rather than pretend";
}

}  // namespace

namespace {

// The per-layer constructor is the one hybrid and dual-geometry models take,
// and it opened its id space against the ceiling while only the initial blocks
// had memory. Admission would then hand out a block that faults on first write.
TEST_F(VmmBackendTest, PerLayerPoolAlsoStartsAtWhatIsCommitted) {
    const std::vector<int> nkv(4, 8), hd(4, 128);
    KVCache kv(/*n_layers=*/4, nkv, hd, QType::F16, /*max_blocks=*/48, /*block_size=*/16,
               /*alloc=*/nullptr, /*layer_is_swa=*/{}, /*swa_max_blocks=*/0, /*ceiling=*/256);
    ASSERT_EQ(kv.total_blocks(), 48) << "the id space must not exceed the committed blocks";
    ASSERT_EQ(kv.ceiling_blocks(), 256);

    // Every id the pool will hand out has to be backed in every layer.
    for (int b = 0; b < kv.total_blocks(); b += 16)
        for (int l = 0; l < 4; l++) {
            write_pattern(kv.k_ptr(l, b), kv.block_bytes(), 0x77);
            ASSERT_FALSE(::testing::Test::HasFatalFailure()) << "block " << b << " layer " << l;
        }

    EXPECT_GE(kv.try_grow_to(200), 200);
    for (int l = 0; l < 4; l++) {
        write_pattern(kv.k_ptr(l, 199), kv.block_bytes(), 0x88);
        ASSERT_FALSE(::testing::Test::HasFatalFailure()) << "grown block, layer " << l;
    }
}

}  // namespace

namespace {

// A pool whose layers are half sliding-window costs less to grow, because a
// windowed layer's region holds only its window. Measured rather than assumed:
// capping the per-layer commit at that window changes nothing (the same 34 MiB
// either way), so this asserts the layout property that is real and not a guard
// that is not.
TEST_F(VmmBackendTest, WindowedLayersMakeAGrownPoolCheaper) {
    const std::vector<int> nkv(4, 8), hd(4, 128);
    auto grown_bytes = [&](const std::vector<char>& is_swa, int swa_blocks) {
        KVCache kv(/*n_layers=*/4, nkv, hd, QType::F16, /*max_blocks=*/32, /*block_size=*/16,
                   /*alloc=*/nullptr, is_swa, swa_blocks, /*ceiling=*/256);
        EXPECT_EQ(kv.try_grow_to(200), 200);
        return kv.committed_bytes();
    };
    const size_t all_full = grown_bytes({}, 0);
    const size_t half_windowed = grown_bytes({1, 0, 1, 0}, 24);
    ASSERT_GT(all_full, 0u);
    EXPECT_LT(half_windowed, all_full);

    // Blocks in both kinds of layer are backed after the growth.
    const std::vector<char> is_swa = {1, 0, 1, 0};
    KVCache kv(/*n_layers=*/4, nkv, hd, QType::F16, /*max_blocks=*/32, /*block_size=*/16,
               /*alloc=*/nullptr, is_swa, /*swa_max_blocks=*/24, /*ceiling=*/256);
    ASSERT_EQ(kv.try_grow_to(200), 200);
    write_pattern(kv.k_ptr(1, 199), kv.block_bytes(), 0x5E);
    write_pattern(kv.k_ptr(0, 23), kv.block_bytes(), 0x2D);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    EXPECT_EQ(count_mismatches(kv.k_ptr(1, 199), kv.block_bytes(), 0x5E), 0u);
    EXPECT_EQ(count_mismatches(kv.k_ptr(0, 23), kv.block_bytes(), 0x2D), 0u);
}

}  // namespace
