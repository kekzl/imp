// Audit C-9: Engine::init bridged imp.conf into EngineConfig by sentinel-equals-default
// (`== 25`, `== 1.0f`, `== 0`), so an embedder passing the default explicitly lost to imp.conf.
// The four fields whose default is also a meaningful value are std::optional now.

#include "runtime/config.h"
#include "runtime/engine_config.h"

#include <gtest/gtest.h>

using imp::EngineConfig;
using imp::RuntimeConfig;

namespace {

RuntimeConfig tuned_conf() {
    RuntimeConfig rc;
    rc.server.prefix_pin_budget_pct = 10;
    rc.vram.kv_fraction = 0.5f;
    rc.vram.reserve_floor_pct = 12;
    rc.runtime.vram_budget_mb = 4096;
    return rc;
}

}  // namespace

TEST(EngineConfigBridge, ExplicitDefaultWinsOverImpConf) {
    EngineConfig ec;
    ec.prefix_pin_budget_pct = 25;
    ec.kv_fraction = 1.0f;
    ec.vram_reserve_floor_pct = 0;
    ec.vram_budget_mb = 0;
    imp::bridge_runtime_config(ec, tuned_conf());
    EXPECT_EQ(ec.prefix_pin_budget_pct, 25);
    EXPECT_EQ(ec.kv_fraction, 1.0f);
    EXPECT_EQ(ec.vram_reserve_floor_pct, 0);
    EXPECT_EQ(ec.vram_budget_mb, 0u);
}

TEST(EngineConfigBridge, UnsetFallsThroughToImpConf) {
    EngineConfig ec;
    imp::bridge_runtime_config(ec, tuned_conf());
    EXPECT_EQ(ec.prefix_pin_budget_pct, 10);
    EXPECT_EQ(ec.kv_fraction, 0.5f);
    EXPECT_EQ(ec.vram_reserve_floor_pct, 12);
    EXPECT_EQ(ec.vram_budget_mb, 4096u);
}

// Unset with an untouched imp.conf lands on the values the sentinels used to stand for.
TEST(EngineConfigBridge, UnsetWithDefaultConfKeepsTheOldDefaults) {
    EngineConfig ec;
    imp::bridge_runtime_config(ec, RuntimeConfig{});
    EXPECT_EQ(ec.prefix_pin_budget_pct, imp::kPrefixPinBudgetPctDefault);
    EXPECT_EQ(RuntimeConfig{}.server.prefix_pin_budget_pct, imp::kPrefixPinBudgetPctDefault);
    EXPECT_EQ(ec.kv_fraction, 1.0f);
    EXPECT_EQ(ec.vram_reserve_floor_pct, 0);
    EXPECT_EQ(ec.vram_budget_mb, 0u);
}
