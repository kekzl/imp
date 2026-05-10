// tests/test_gemm_grouped_nvfp4_smallM.cu
#include <gtest/gtest.h>
#include "compute/gemm_grouped_nvfp4_smallM.h"
#include <vector>

namespace {

TEST(SmallMScheduler, PicksMinimalTile) {
    using imp::detail::pick_m_tile;
    EXPECT_EQ(pick_m_tile(1),   16);
    EXPECT_EQ(pick_m_tile(16),  16);
    EXPECT_EQ(pick_m_tile(17),  32);
    EXPECT_EQ(pick_m_tile(32),  32);
    EXPECT_EQ(pick_m_tile(40),  64);
    EXPECT_EQ(pick_m_tile(64),  64);
    EXPECT_EQ(pick_m_tile(128), 128);
    EXPECT_EQ(pick_m_tile(200), 128);
}

TEST(SmallMScheduler, WorkQueueOrderedByTileSize) {
    using imp::detail::build_work_queue;
    int M_per[] = {32, 100, 8, 0, 200};   // 5 experts; e=3 inactive
    auto q = build_work_queue(5, M_per, 256);
    ASSERT_FALSE(q.empty());

    // First items must be tile_M=128 (from e=4 with M=200, two M-tiles needed)
    EXPECT_EQ(q[0].m_tile_size, 128);
    // Last items must be tile_M=16 (from e=2 with M=8)
    EXPECT_EQ(q.back().m_tile_size, 16);
    // No work for inactive expert e=3
    for (auto& wi : q) EXPECT_NE(wi.expert_id, 3);
}

}  // anonymous namespace
