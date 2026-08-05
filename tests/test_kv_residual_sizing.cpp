// kv_blocks_from_residual — the arithmetic that sizes the KV pool from what is
// left after the weight caches are built.
//
// This is a CPU-lane test on purpose. The decision it covers is pure integer
// arithmetic, but until #1251 it lived inline in engine_kv_cache_init.cpp,
// where nothing could reach it without a GPU and a 22 GB checkpoint — so the
// one case that matters was never asserted anywhere.
//
// The case that matters: the caller subtracts the allocator headroom from the
// measured residual. If whoever reserved room *for* this pool did not also
// reserve the headroom, the residual is entirely headroom, `room` is 0, and the
// pool silently falls to the floor. That is a rescue, not a size, and
// `floored` is what tells the two apart.

#include <gtest/gtest.h>

#include "memory/vram_query.h"

namespace imp {
namespace {

constexpr size_t kMiB = 1024ULL * 1024ULL;

// Qwen3.6-35B-A3B-UD-Q4_K_M on a 32 GiB 5090: 10 attention layers, block_size
// 32, 2 KV heads, head_dim 256, FP16 -> 640 KiB per block.
constexpr size_t k35bPerBlock = 640ULL * 1024ULL;

TEST(KvResidualSizing, PlanFitsSoNothingIsClamped) {
    const auto s = kv_blocks_from_residual(8192 * kMiB, 1630 * kMiB, k35bPerBlock, 4096, 16);
    EXPECT_EQ(s.blocks, 4096);
    EXPECT_FALSE(s.clamped);
    EXPECT_FALSE(s.floored);
}

TEST(KvResidualSizing, ResidualSmallerThanPlanClampsButDoesNotFloor) {
    // 2416 MiB free - 1630 MiB headroom = 786 MiB -> 1257 blocks. These are the
    // post-fix numbers measured on the #1251 repro.
    const auto s = kv_blocks_from_residual(2416 * kMiB, 1630 * kMiB, k35bPerBlock, 4096, 16);
    EXPECT_EQ(s.blocks, 1257);
    EXPECT_TRUE(s.clamped);
    EXPECT_FALSE(s.floored) << "a pool smaller than the plan is normal — the plan is a "
                               "projection, the residual is the truth";
}

// The #1251 regression itself, with the numbers straight out of the bug report.
TEST(KvResidualSizing, HeadroomExceedingResidualFloorsAndSaysSo) {
    const auto s = kv_blocks_from_residual(1264 * kMiB, 1630 * kMiB, k35bPerBlock, 4096, 16);
    EXPECT_EQ(s.blocks, 16) << "the floor, not a computed size";
    EXPECT_TRUE(s.clamped);
    EXPECT_TRUE(s.floored) << "#1251: nothing was left to size the pool from, and the load "
                              "reported success anyway";
}

TEST(KvResidualSizing, ResidualExactlyHeadroomLeavesNothing) {
    const auto s = kv_blocks_from_residual(1630 * kMiB, 1630 * kMiB, k35bPerBlock, 4096, 16);
    EXPECT_EQ(s.blocks, 16);
    EXPECT_TRUE(s.floored);
}

// One block short of the floor still floors; exactly the floor does not.
TEST(KvResidualSizing, FloorBoundaryIsExact) {
    const size_t headroom = 1000 * kMiB;
    const auto below = kv_blocks_from_residual(headroom + 15 * k35bPerBlock, headroom,
                                               k35bPerBlock, 4096, 16);
    EXPECT_EQ(below.blocks, 16);
    EXPECT_TRUE(below.floored);

    const auto at = kv_blocks_from_residual(headroom + 16 * k35bPerBlock, headroom, k35bPerBlock,
                                            4096, 16);
    EXPECT_EQ(at.blocks, 16);
    EXPECT_TRUE(at.clamped);
    EXPECT_FALSE(at.floored) << "16 blocks that were actually computed is not the rescue path";
}

TEST(KvResidualSizing, MoreRoomThanPlannedNeverGrowsThePool) {
    const auto s = kv_blocks_from_residual(30000 * kMiB, 1630 * kMiB, k35bPerBlock, 64, 16);
    EXPECT_EQ(s.blocks, 64) << "the residual can only shrink the plan, never grow it";
    EXPECT_FALSE(s.clamped);
}

TEST(KvResidualSizing, DegenerateInputsAreInert) {
    const auto zero_block = kv_blocks_from_residual(8192 * kMiB, 1630 * kMiB, 0, 4096, 16);
    EXPECT_EQ(zero_block.blocks, 4096) << "per_block==0 must not divide";
    EXPECT_FALSE(zero_block.clamped);

    const auto no_plan = kv_blocks_from_residual(8192 * kMiB, 1630 * kMiB, k35bPerBlock, 0, 16);
    EXPECT_EQ(no_plan.blocks, 0);
    EXPECT_FALSE(no_plan.floored);
}

TEST(KvResidualSizing, FreeBelowHeadroomDoesNotUnderflow) {
    // free < headroom must saturate at 0 room, not wrap around size_t.
    const auto s = kv_blocks_from_residual(100 * kMiB, 1630 * kMiB, k35bPerBlock, 4096, 16);
    EXPECT_EQ(s.blocks, 16);
    EXPECT_TRUE(s.floored);
}

}  // namespace
}  // namespace imp
