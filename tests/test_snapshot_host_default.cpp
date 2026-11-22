// The recurrent snapshot host tier follows kv_cache.host_spill_mb / 2 unless the operator set it
// (snapshot_host_default.h).

#include "runtime/snapshot_host_default.h"

#include <gtest/gtest.h>

namespace imp {
namespace {

TEST(SnapshotHostDefault, SpillOffKeepsTheDefault) {
    RuntimeConfig rc;
    const int def = rc.server.recurrent_snapshot_host_mb;
    EXPECT_EQ(def, 2048);
    apply_snapshot_host_spill_default(rc);
    EXPECT_EQ(rc.server.recurrent_snapshot_host_mb, def);
}

TEST(SnapshotHostDefault, SpillRaisesTheHostTierToHalfTheSpillBudget) {
    RuntimeConfig rc;
    rc.kv_cache.host_spill_mb = 16384;
    apply_snapshot_host_spill_default(rc);
    EXPECT_EQ(rc.server.recurrent_snapshot_host_mb, 8192);

    RuntimeConfig init;  // the store's byte budget path applies the same default
    init.kv_cache.host_spill_mb = 16384;
    EXPECT_EQ(snapshot_host_budget_bytes(init), size_t{8192} << 20);
}

TEST(SnapshotHostDefault, SmallSpillNeverLowersTheDefault) {
    RuntimeConfig rc;
    rc.kv_cache.host_spill_mb = 2048;
    apply_snapshot_host_spill_default(rc);
    EXPECT_EQ(rc.server.recurrent_snapshot_host_mb, 2048);
}

TEST(SnapshotHostDefault, OperatorValueWins) {
    RuntimeConfig rc;
    rc.kv_cache.host_spill_mb = 16384;
    rc.server.recurrent_snapshot_host_mb = 1024;
    rc.explicit_keys.push_back("server.recurrent_snapshot_host_mb");
    apply_snapshot_host_spill_default(rc);
    EXPECT_EQ(rc.server.recurrent_snapshot_host_mb, 1024);
}

}  // namespace
TEST(HostSpillAuto, TwoThirdsOfTheSpareRamInWholeGib) {
    const size_t gib = size_t{1} << 30;
    EXPECT_EQ(host_spill_auto_mb(59 * gib, 8 * gib), 16384) << "51 GiB spare: the cap";
    EXPECT_EQ(host_spill_auto_mb(20 * gib, 6 * gib), 9 * 1024) << "14 GiB spare";
    EXPECT_EQ(host_spill_auto_mb(8 * gib, 6 * gib), 1024) << "2 GiB spare";
    EXPECT_EQ(host_spill_auto_mb(7 * gib, 6 * gib), 0) << "1 GiB spare is no tier";
    EXPECT_EQ(host_spill_auto_mb(4 * gib, 6 * gib), 0);
    EXPECT_EQ(host_spill_auto_mb(0, 6 * gib), 0) << "unknown host figure";
}

TEST(HostSpillAuto, ResolvesOnlyAnUnsetKeyAndFeedsTheSnapshotTier) {
    const size_t gib = size_t{1} << 30;
    RuntimeConfig rc;
    ASSERT_LT(rc.kv_cache.host_spill_mb, 0) << "the default is auto";
    EXPECT_TRUE(apply_host_spill_auto(rc, /*prefix_caching=*/true, 59 * gib, 8 * gib));
    EXPECT_EQ(rc.kv_cache.host_spill_mb, 16384);
    EXPECT_EQ(snapshot_host_budget_bytes(rc), size_t{8192} << 20);

    RuntimeConfig off;
    off.kv_cache.host_spill_mb = 0;
    EXPECT_FALSE(apply_host_spill_auto(off, true, 59 * gib, 8 * gib));
    EXPECT_EQ(off.kv_cache.host_spill_mb, 0);

    RuntimeConfig no_cache;
    EXPECT_TRUE(apply_host_spill_auto(no_cache, /*prefix_caching=*/false, 59 * gib, 8 * gib));
    EXPECT_EQ(no_cache.kv_cache.host_spill_mb, 0);
}

}  // namespace imp
