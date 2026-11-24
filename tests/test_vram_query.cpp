// vram_query (behind EngineConfig.vram_budget_mb): my_used = free_at_install - free_now;
// free' = min(free_now, budget - my_used); total' = budget. A co-tenant's pre-existing usage
// never counts against this process; this process's own allocations shrink the budgeted view 1:1.

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "memory/device_alloc_all.h"
#include "memory/vram_query.h"

#include "test_cuda_skip.h"

namespace imp {
namespace {

class VramQueryTest : public ::testing::Test {
protected:
    void TearDown() override { vram_budget_install(0); }  // never leak the cap
};

TEST_F(VramQueryTest, UncappedIsPassthrough) {
    SKIP_IF_NO_CUDA();
    vram_budget_install(0);
    size_t raw_free = 0, raw_total = 0;
    ASSERT_EQ(cudaMemGetInfo(&raw_free, &raw_total), cudaSuccess);
    size_t f = 0, t = 0;
    ASSERT_TRUE(vram_budget_mem_get_info(&f, &t));
    EXPECT_EQ(t, raw_total);
    // free can jitter between the two calls — allow small drift.
    EXPECT_NEAR(static_cast<double>(f), static_cast<double>(raw_free), 64.0 * 1024 * 1024);
}

TEST_F(VramQueryTest, BudgetCapsTotalAndTracksOwnUsage) {
    SKIP_IF_NO_CUDA();
    constexpr size_t kBudgetMb = 1024;
    vram_budget_install(kBudgetMb);
    ASSERT_EQ(vram_budget_bytes(), kBudgetMb << 20);

    size_t f0 = 0, t0 = 0;
    ASSERT_TRUE(vram_budget_mem_get_info(&f0, &t0));
    EXPECT_EQ(t0, kBudgetMb << 20) << "total' must be the virtual GPU size";
    EXPECT_LE(f0, kBudgetMb << 20);
    // Nothing allocated since install → the full budget is visible
    // (device free far exceeds 1 GiB on the test box).
    EXPECT_GT(f0, (kBudgetMb - 64) << 20);

    // Allocate 256 MiB — the budgeted view must shrink by ~that amount.
    void* p = nullptr;
    ASSERT_EQ(cudaMalloc(&p, 256ULL << 20), cudaSuccess);
    size_t f1 = 0;
    ASSERT_TRUE(vram_budget_mem_get_info(&f1, nullptr));
    EXPECT_LT(f1, f0 - (200ULL << 20)) << "own allocation must count against the budget";
    EXPECT_GT(f1 + (320ULL << 20), f0) << "…but not by much more than its size";

    // Free it — the budget must come back.
    ASSERT_EQ(cudaFree(p), cudaSuccess);
    size_t f2 = 0;
    ASSERT_TRUE(vram_budget_mem_get_info(&f2, nullptr));
    EXPECT_GT(f2 + (64ULL << 20), f0) << "freed memory must return to the budget";
}

TEST_F(VramQueryTest, BudgetExhaustionClampsToZero) {
    SKIP_IF_NO_CUDA();
    vram_budget_install(128);  // tiny budget
    void* p = nullptr;
    ASSERT_EQ(cudaMalloc(&p, 256ULL << 20), cudaSuccess);  // overshoot it
    size_t f = 0, t = 0;
    ASSERT_TRUE(vram_budget_mem_get_info(&f, &t));
    EXPECT_EQ(f, 0u) << "over-budget usage must clamp free' to 0, not underflow";
    EXPECT_EQ(t, 128ULL << 20);
    ASSERT_EQ(cudaFree(p), cudaSuccess);
}

TEST_F(VramQueryTest, InstallClampsToDeviceTotal) {
    SKIP_IF_NO_CUDA();
    size_t raw_total = 0, raw_free = 0;
    ASSERT_EQ(cudaMemGetInfo(&raw_free, &raw_total), cudaSuccess);
    vram_budget_install((raw_total >> 20) * 4);  // absurd budget
    size_t f = 0, t = 0;
    ASSERT_TRUE(vram_budget_mem_get_info(&f, &t));
    EXPECT_EQ(t, raw_total) << "budget beyond the card clamps to device total";
}

}  // namespace
}  // namespace imp

// vram_fits: a fixed size against the budget-view free bytes, margin on top, the caller's pending
// reservation credited. The guard never chooses a size; it only answers yes or no.
TEST(VramFits, ComparesTheSizePlusMarginWithFreePlusCredit) {
    size_t free_b = 0, total_b = 0;
    if (!imp::vram_budget_mem_get_info(&free_b, &total_b) || total_b == 0)
        GTEST_SKIP() << "no device";
    size_t seen = 0;
    EXPECT_TRUE(imp::vram_fits(0, 0, 0, &seen));
    EXPECT_GT(seen, 0u);
    EXPECT_FALSE(imp::vram_fits(total_b, 0)) << "the whole card is never free";
    EXPECT_FALSE(imp::vram_fits(0, total_b)) << "the margin counts like the size";
    // A credit as large as the card makes any size up to the card fit.
    EXPECT_TRUE(imp::vram_fits(total_b, 0, total_b));
    EXPECT_FALSE(imp::vram_fits(2 * total_b + 1, 0, total_b));
}

// The pre-upload capacity is the install snapshot capped at the budget, not a later reading:
// an allocation after install leaves it where it was.
TEST(VramQuerySnapshot, FreeAtInstallIsASnapshotCappedAtTheBudget) {
    size_t raw_free = 0, raw_total = 0;
    if (!imp::vram_diag_mem_info(&raw_free, &raw_total) || raw_free < (size_t{512} << 20))
        GTEST_SKIP() << "no device or under 512 MiB free";
    imp::vram_budget_install(0);
    const size_t uncapped = imp::vram_free_at_install_bytes();
    EXPECT_GT(uncapped, 0u);
    EXPECT_LE(uncapped, raw_total);
    void* p = nullptr;
    ASSERT_EQ(cudaMalloc(&p, size_t{128} << 20), cudaSuccess);
    EXPECT_EQ(imp::vram_free_at_install_bytes(), uncapped) << "a snapshot does not follow the device";
    cudaFree(p);
    imp::vram_budget_install(256);
    EXPECT_EQ(imp::vram_free_at_install_bytes(), size_t{256} << 20);
    imp::vram_budget_install(0);  // never leak the cap
}

// The own-bytes ledger: an allocation through device_alloc_all moves the live count and the ledger's
// free bytes by exactly its size, and the free moves both back. No device reading is involved.
TEST(VramLedger, PlainAllocationMovesLiveBytesAndLedgerFreeByItsSize) {
    constexpr size_t kBytes = size_t{64} << 20;
    size_t raw_free = 0, raw_total = 0;
    if (!imp::vram_diag_mem_info(&raw_free, &raw_total) || raw_free < (size_t{512} << 20))
        GTEST_SKIP() << "no device or under 512 MiB free";
    imp::vram_budget_install(0);
    const size_t live0 = imp::device_alloc_live_bytes();
    const size_t free0 = imp::vram_ledger_free_bytes();
    ASSERT_GE(free0, kBytes) << "the ledger must have room to fall by the allocation";
    void* p = nullptr;
    ASSERT_EQ(imp::device_alloc_all(imp::dev_req(p, kBytes)), cudaSuccess);
    EXPECT_EQ(imp::device_alloc_live_bytes(), live0 + kBytes);
    EXPECT_EQ(imp::vram_ledger_free_bytes(), free0 - kBytes);
    // The sizing figure keeps kLedgerUnbookedBytes back and follows the ledger byte for byte.
    EXPECT_EQ(imp::vram_planned_free_bytes(), imp::vram_ledger_free_bytes() - imp::kLedgerUnbookedBytes);
    imp::device_free_all(p);
    EXPECT_EQ(imp::device_alloc_live_bytes(), live0);
    EXPECT_EQ(imp::vram_ledger_free_bytes(), free0);
}

// A pointer the wrapper did not hand out is freed and leaves the count alone, also while a
// counted allocation is live.
TEST(VramLedger, FreeOfAnUnknownPointerLeavesTheCountUnchanged) {
    size_t raw_free = 0, raw_total = 0;
    if (!imp::vram_diag_mem_info(&raw_free, &raw_total) || raw_free < (size_t{512} << 20))
        GTEST_SKIP() << "no device or under 512 MiB free";
    void* counted = nullptr;
    ASSERT_EQ(imp::device_alloc_all(imp::dev_req(counted, size_t{8} << 20)), cudaSuccess);
    const size_t live = imp::device_alloc_live_bytes();
    void* foreign = nullptr;
    ASSERT_EQ(cudaMalloc(&foreign, size_t{4} << 20), cudaSuccess);
    EXPECT_EQ(imp::device_alloc_live_bytes(), live) << "a raw cudaMalloc is not counted";
    imp::device_free_all(foreign);
    EXPECT_EQ(imp::device_alloc_live_bytes(), live);
    imp::device_free_all(counted);
    EXPECT_EQ(imp::device_alloc_live_bytes(), live - (size_t{8} << 20));
}

// The accounting view: used = total - free with the lazy pools' pending charge left out of "used".
TEST(VramQuerySnapshot, ViewUsedDoesNotCountPendingReservations) {
    size_t free_b = 0, total_b = 0;
    if (!imp::vram_budget_mem_get_info(&free_b, &total_b) || total_b == 0)
        GTEST_SKIP() << "no device";
    imp::vram_budget_install(0);
    size_t before = 0, pending = 0;
    ASSERT_TRUE(imp::vram_view_used_bytes(&before));
    imp::vram_reserved_uncommitted_add(std::ptrdiff_t{64} << 20);
    ASSERT_TRUE(imp::vram_view_used_bytes(&pending));
    imp::vram_reserved_uncommitted_add(-(std::ptrdiff_t{64} << 20));
    // Another tenant may move the device by a few MiB between the two reads; 64 MiB would show.
    const size_t diff = pending > before ? pending - before : before - pending;
    EXPECT_LT(diff, size_t{32} << 20);
}
