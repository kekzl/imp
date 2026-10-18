// --gpu-layers turns token recycling on (width 4) unless the operator set the key (offload_spec_defaults.h).

#include "runtime/offload_spec_defaults.h"

#include <gtest/gtest.h>

namespace imp {
namespace {

TEST(OffloadSpecDefaults, OffloadTurnsRecyclingOnAtWidthFour) {
    RuntimeConfig rc;
    ASSERT_FALSE(rc.speculative.token_recycling);
    apply_offload_recycling_default(rc, /*gpu_layers=*/0);
    EXPECT_TRUE(rc.speculative.token_recycling);
    EXPECT_EQ(rc.speculative.recycle_width, kOffloadRecycleWidth);
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

    RuntimeConfig wide;  // speculative.recycle_width=8 set explicitly
    wide.speculative.recycle_width = 8;
    wide.explicit_keys.push_back("speculative.recycle_width");
    apply_offload_recycling_default(wide, 20);
    EXPECT_TRUE(wide.speculative.token_recycling);
    EXPECT_EQ(wide.speculative.recycle_width, 8);
}

}  // namespace
}  // namespace imp
