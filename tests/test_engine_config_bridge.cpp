// Audit C-9: Engine::init bridged imp.conf into EngineConfig by sentinel-equals-default
// (`== 25`, `== 1.0f`, `== 0`), so an embedder passing the default explicitly lost to imp.conf.
// All 8 bridged fields are std::optional: unset takes imp.conf, any set value wins.

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
    rc.paths.mmproj = "/models/mmproj.gguf";
    rc.vram.library_reserve_mb = 900;
    rc.kv_cache.max_blocks = 512;
    rc.kv_cache.block_size = 32;
    return rc;
}

}  // namespace

TEST(EngineConfigBridge, ExplicitDefaultWinsOverImpConf) {
    EngineConfig ec;
    ec.prefix_pin_budget_pct = 25;
    ec.kv_fraction = 1.0f;
    ec.vram_reserve_floor_pct = 0;
    ec.vram_budget_mb = 0;
    ec.mmproj_path = "";
    ec.library_reserve_mb = -1;
    ec.kv_cache_max_blocks = 0;
    ec.kv_block_size = 0;
    imp::bridge_runtime_config(ec, tuned_conf());
    EXPECT_EQ(ec.prefix_pin_budget_pct, 25);
    EXPECT_EQ(ec.kv_fraction, 1.0f);
    EXPECT_EQ(ec.vram_reserve_floor_pct, 0);
    EXPECT_EQ(ec.vram_budget_mb, 0u);
    EXPECT_EQ(ec.mmproj_path, "");
    EXPECT_EQ(ec.library_reserve_mb, -1);
    EXPECT_EQ(ec.kv_cache_max_blocks, 0);
    EXPECT_EQ(ec.kv_block_size, 0);
}

TEST(EngineConfigBridge, UnsetFallsThroughToImpConf) {
    EngineConfig ec;
    imp::bridge_runtime_config(ec, tuned_conf());
    EXPECT_EQ(ec.prefix_pin_budget_pct, 10);
    EXPECT_EQ(ec.kv_fraction, 0.5f);
    EXPECT_EQ(ec.vram_reserve_floor_pct, 12);
    EXPECT_EQ(ec.vram_budget_mb, 4096u);
    EXPECT_EQ(ec.mmproj_path, "/models/mmproj.gguf");
    EXPECT_EQ(ec.library_reserve_mb, 900);
    EXPECT_EQ(ec.kv_cache_max_blocks, 512);
    EXPECT_EQ(ec.kv_block_size, 32);
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
    EXPECT_EQ(ec.mmproj_path, "");
    EXPECT_EQ(ec.library_reserve_mb, -1);
    EXPECT_EQ(ec.kv_cache_max_blocks, 0);
    EXPECT_EQ(ec.kv_block_size, 0);
}

// Batch, context, KV dtype and StreamingLLM auto resolve into EngineRuntimeState (C-9, Inkling's
// limits among the writers): the seed copies the operator's values, the sizing view carries the
// resolved ones and every other field of the operator's config.
TEST(EngineConfigBridge, ResolvedFieldsLiveInTheRuntimeState) {
    EngineConfig ec;
    ec.max_batch_size = 0;  // auto
    ec.max_seq_len = 0;
    ec.kv_cache_dtype = imp::QType::FP8_E4M3;
    ec.streaming_kv_auto = true;
    ec.use_cuda_graphs = false;
    ec.prefill_chunk_size = 777;
    imp::EngineRuntimeState rt = imp::seed_runtime_state(ec);
    EXPECT_EQ(rt.max_batch_size, 0);
    EXPECT_EQ(rt.max_seq_len, 0);
    EXPECT_EQ(rt.kv_cache_dtype, imp::QType::FP8_E4M3);
    EXPECT_TRUE(rt.streaming_kv_auto);
    EXPECT_FALSE(rt.cuda_graphs);
    // What init_inkling_limits_ writes: 2 slots, 81920 positions, fp16 K/V, no StreamingLLM auto.
    rt.max_batch_size = 2;
    rt.max_seq_len = 81920;
    rt.kv_cache_dtype = imp::QType::F16;
    rt.streaming_kv_auto = false;
    const EngineConfig sized = imp::resolved_engine_config(ec, rt);
    EXPECT_EQ(sized.max_batch_size, 2);
    EXPECT_EQ(sized.max_seq_len, 81920);
    EXPECT_EQ(sized.kv_cache_dtype, imp::QType::F16);
    EXPECT_FALSE(sized.streaming_kv_auto);
    EXPECT_EQ(sized.prefill_chunk_size, 777);
    EXPECT_FALSE(sized.use_cuda_graphs);
    // The operator's config is untouched.
    EXPECT_EQ(ec.max_batch_size, 0);
    EXPECT_EQ(ec.max_seq_len, 0);
    EXPECT_EQ(ec.kv_cache_dtype, imp::QType::FP8_E4M3);
    EXPECT_TRUE(ec.streaming_kv_auto);
}
