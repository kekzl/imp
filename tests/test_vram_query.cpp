// vram_query — the budget-aware cudaMemGetInfo view behind
// EngineConfig.vram_budget_mb ("pretend the GPU is only X MiB").
//
// Semantics under test:
//   my_used = free_at_install − free_now   (baseline delta)
//   free'   = min(free_now, budget − my_used)
//   total'  = budget
// so a co-tenant's pre-existing usage never counts against this process,
// while this process's own allocations shrink the budgeted view 1:1.

#include <gtest/gtest.h>
#include <cuda_runtime.h>

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
