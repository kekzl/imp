// --gpu-layers turns token recycling on (width 4) unless the operator set the key (offload_spec_defaults.h).

#include "runtime/offload_spec_defaults.h"

#include <gtest/gtest.h>

namespace imp {
namespace {

TEST(OffloadSpecDefaults, OffloadTurnsRecyclingOnAtWidthEight) {
    RuntimeConfig rc;
    ASSERT_FALSE(rc.speculative.token_recycling);
    apply_offload_recycling_default(rc, /*gpu_layers=*/0);
    EXPECT_TRUE(rc.speculative.token_recycling);
    EXPECT_EQ(rc.speculative.recycle_width, kOffloadRecycleWidth);
    EXPECT_EQ(kOffloadRecycleWidth, 8);
}

TEST(OffloadSpecDefaults, ResidentModelKeepsTheDefaults) {
    RuntimeConfig rc;
    const int width = rc.speculative.recycle_width;
    apply_offload_recycling_default(rc, /*gpu_layers=*/-1);
    EXPECT_FALSE(rc.speculative.token_recycling);
    EXPECT_EQ(rc.speculative.recycle_width, width);
}

TEST(OffloadSpecDefaults, OperatorKeysWin) {
    RuntimeConfig off;  // speculative.token_recycling=false set explicitly
    off.explicit_keys.push_back("speculative.token_recycling");
    apply_offload_recycling_default(off, 20);
    EXPECT_FALSE(off.speculative.token_recycling);

    RuntimeConfig wide;  // speculative.recycle_width=2 set explicitly
    wide.speculative.recycle_width = 2;
    wide.explicit_keys.push_back("speculative.recycle_width");
    apply_offload_recycling_default(wide, 20);
    EXPECT_TRUE(wide.speculative.token_recycling);
    EXPECT_EQ(wide.speculative.recycle_width, 2);
}

// Resident: width * (1 + depth) <= 17; offload: <= 64 rows.
TEST(OffloadSpecDefaults, McShapeResidentKeepsTheSeventeenRowBudget) {
    const RecycleMcShape w4 = recycle_mc_shape(4, 3, 16, false);
    EXPECT_EQ(w4.width, 4);
    EXPECT_EQ(w4.depth, 3);
    const RecycleMcShape w8 = recycle_mc_shape(8, 15, 16, false);
    EXPECT_EQ(w8.width, 8);
    EXPECT_EQ(w8.depth, 1);  // 8 x 2 = 16 rows
}

TEST(OffloadSpecDefaults, McShapeOffloadUsesSixtyFourRows) {
    const RecycleMcShape w8 = recycle_mc_shape(8, 15, 16, true);
    EXPECT_EQ(w8.depth, 7);  // 8 x 8 = 64 rows
    const RecycleMcShape w4 = recycle_mc_shape(4, 15, 16, true);
    EXPECT_EQ(w4.depth, 15);  // 4 x 16 = 64 rows
    EXPECT_EQ(recycle_mc_shape(4, 15, 6, true).depth, 6) << "k caps the depth";
    EXPECT_EQ(recycle_mc_shape(16, 3, 16, true).width, 8) << "8 recycle slots";
    EXPECT_EQ(recycle_mc_shape(4, 3, 16, true).depth, 3) << "recycle_depth still caps";
}

}  // namespace
}  // namespace imp
