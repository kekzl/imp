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
}  // namespace imp
