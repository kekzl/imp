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

// B-129 rest: the quant stack, SSM state dtype, KV block size, library reserve, prefix caching and
// the planner inputs init derives resolve into EngineRuntimeState too (27 writes left config_).
TEST(EngineConfigBridge, QuantAndPlannerFieldsLiveInTheRuntimeState) {
    EngineConfig ec;
    ec.use_fp8_prefill = true;
    ec.use_nvfp4_decode = -1;  // auto
    ec.dual_path_quant = true;
    ec.use_prefix_caching = true;
    imp::bridge_runtime_config(ec, RuntimeConfig{});  // kv_block_size 0, library_reserve_mb -1
    imp::EngineRuntimeState rt = imp::seed_runtime_state(ec);
    EXPECT_TRUE(rt.use_fp8_prefill);
    EXPECT_EQ(rt.use_nvfp4_decode, -1);
    EXPECT_TRUE(rt.dual_path_quant);
    EXPECT_EQ(rt.ssm_state_dtype, imp::QType::F32);
    EXPECT_EQ(rt.kv_block_size, 0);
    EXPECT_EQ(rt.library_reserve_mb, -1);
    EXPECT_TRUE(rt.use_prefix_caching);
    EXPECT_FALSE(rt.q8_drop_source);
    // What the init resolvers and init_kv_cache write: GDN drops FP8 prefill, auto picks NVFP4
    // mode 2, Gemma-4 drops dual-path, BF16 GDN state, block 16, a remembered 900 MiB reserve,
    // no SWA snapshot store, the planner flags.
    rt.use_fp8_prefill = false;
    rt.use_nvfp4_decode = 2;
    rt.dual_path_quant = false;
    rt.ssm_state_dtype = imp::QType::BF16;
    rt.kv_block_size = 16;
    rt.library_reserve_mb = 900;
    rt.use_prefix_caching = false;
    rt.swa_shared_tokens = 2048;
    rt.kv_ceiling_seqs = 3;
    rt.nvfp4_decode_all = true;
    rt.fp8_lm_head = true;
    rt.lm_head_source_only = true;
    rt.q8_drop_source = true;
    rt.q4k_in_place = true;
    const EngineConfig sized = imp::resolved_engine_config(ec, rt);
    EXPECT_FALSE(sized.use_fp8_prefill);
    EXPECT_EQ(sized.use_nvfp4_decode, 2);
    EXPECT_FALSE(sized.dual_path_quant);
    EXPECT_EQ(sized.ssm_state_dtype, imp::QType::BF16);
    EXPECT_EQ(sized.kv_block_size, 16);
    EXPECT_EQ(sized.library_reserve_mb, 900);
    EXPECT_FALSE(sized.use_prefix_caching);
    EXPECT_EQ(sized.swa_shared_tokens, 2048);
    EXPECT_EQ(sized.kv_ceiling_seqs, 3);
    EXPECT_TRUE(sized.nvfp4_decode_all);
    EXPECT_TRUE(sized.fp8_lm_head);
    EXPECT_TRUE(sized.lm_head_source_only);
    EXPECT_TRUE(sized.q8_drop_source);
    EXPECT_TRUE(sized.q4k_in_place);
    // The operator's config is untouched.
    EXPECT_TRUE(ec.use_fp8_prefill);
    EXPECT_EQ(ec.use_nvfp4_decode, -1);
    EXPECT_TRUE(ec.dual_path_quant);
    EXPECT_EQ(ec.ssm_state_dtype, imp::QType::F32);
    EXPECT_EQ(ec.kv_block_size, 0);
    EXPECT_EQ(ec.library_reserve_mb, -1);
    EXPECT_TRUE(ec.use_prefix_caching);
    EXPECT_EQ(ec.swa_shared_tokens, 0);
    EXPECT_EQ(ec.kv_ceiling_seqs, 0);
    EXPECT_FALSE(ec.q8_drop_source);
}
