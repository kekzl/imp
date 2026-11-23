// GrowRows (memory/grow_rows.h) on the fake backend: what is committed, what is booked as planned.

#include <gtest/gtest.h>

#include "memory/fake_backend.h"
#include "memory/grow_rows.h"
#include "memory/vram_allocator.h"

using namespace imp;

// 2 arrays of 64 rows of 4096 bytes, 8 rows per step: the first step is committed, the rest is planned;
// ensure() moves whole steps from planned to committed for both arrays.
TEST(GrowRows, CommitsInStepsAndBooksTheRest) {
    const size_t row = FakeBackend::kGranularity, max_rows = 64, step = 8;
    FakeBackend be;
    VRAMAllocator va;
    {
        GrowRows g;
        g.init(va, &be, "grow_rows_test", row, max_rows, step);
        ASSERT_TRUE(g.growable());
        void* a = g.add();
        void* b = g.add();
        ASSERT_NE(a, nullptr);
        ASSERT_NE(b, nullptr);
        EXPECT_EQ(g.rows(), step);
        EXPECT_EQ(be.stats().live_bytes, 2 * step * row);
        EXPECT_EQ(va.planned(), 2 * (max_rows - step) * row);
        EXPECT_TRUE(g.ensure(step));  // nothing to do
        EXPECT_EQ(be.stats().live_bytes, 2 * step * row);
        EXPECT_TRUE(g.ensure(step + 1));  // one more step
        EXPECT_EQ(g.rows(), 2 * step);
        EXPECT_EQ(be.stats().live_bytes, 2 * 2 * step * row);
        EXPECT_EQ(va.planned(), 2 * (max_rows - 2 * step) * row);
        EXPECT_TRUE(g.ensure(max_rows));
        EXPECT_EQ(g.rows(), max_rows);
        EXPECT_EQ(va.planned(), 0u);
        EXPECT_FALSE(g.ensure(max_rows + 1));  // past the reservation
        EXPECT_EQ(be.stats().live_bytes, 2 * max_rows * row);
    }
    EXPECT_EQ(be.stats().live_bytes, 0u);
    EXPECT_EQ(va.planned(), 0u);
}

// A device that refuses the commit: ensure() is false, the row count stays, the books match the regions.
TEST(GrowRows, ARefusedCommitLeavesTheRowCount) {
    const size_t row = FakeBackend::kGranularity, max_rows = 64, step = 8;
    FakeBackend be(/*capacity_bytes=*/3 * step * row);
    VRAMAllocator va;
    {
        GrowRows g;
        g.init(va, &be, "grow_rows_test", row, max_rows, step);
        ASSERT_NE(g.add(), nullptr);
        ASSERT_NE(g.add(), nullptr);
        EXPECT_FALSE(g.ensure(2 * step));  // 4 steps needed, 3 fit
        EXPECT_EQ(g.rows(), step);
        EXPECT_EQ(va.planned() + be.stats().live_bytes, 2 * max_rows * row);
        EXPECT_GT(va.planned(), 0u);
    }
    // Released with rows still planned: the plan is empty again.
    EXPECT_EQ(va.planned(), 0u);
    EXPECT_EQ(be.stats().live_bytes, 0u);
}

// Without a growable backend the arrays are whole from the start and nothing is planned.
TEST(GrowRows, FixedWithoutABackend) {
    VRAMAllocator va;
    GrowRows g;
    g.init(va, nullptr, "grow_rows_test", 256, 100, 8);
    EXPECT_FALSE(g.growable());
    EXPECT_EQ(g.rows(), 100u);
    EXPECT_TRUE(g.ensure(100));
    EXPECT_FALSE(g.ensure(101));
    EXPECT_EQ(va.planned(), 0u);
}
