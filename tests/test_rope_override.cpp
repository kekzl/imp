// Runtime RoPE-scaling override (imp.conf [rope]) — host-only semantics.
//
// apply_rope_override must set exactly the ModelConfig fields the GGUF/HF
// loaders set from model-declared rope_scaling (rope_freq_scale stores the
// FACTOR — the kernel applies 1/factor and the paper mscale itself), bump
// max_seq_len to factor × orig_ctx, and refuse model classes where a scalar
// factor is silently wrong (per-dim tables, MLA, NoPE).

#include <gtest/gtest.h>

#include "model/model_config.h"
#include "runtime/config.h"
#include "runtime/engine.h"

namespace imp {
namespace {

ModelConfig base_model(int ctx = 32768) {
    ModelConfig m;
    m.max_seq_len = ctx;
    m.n_heads = 32;
    m.n_kv_heads = 8;
    m.d_model = 4096;
    m.head_dim = 128;
    return m;
}

TEST(RopeOverrideTest, OffByDefault) {
    ModelConfig m = base_model();
    RuntimeConfig cfg;
    EXPECT_FALSE(apply_rope_override(m, cfg.rope));
    EXPECT_FLOAT_EQ(m.rope_freq_scale, 1.0f);
    EXPECT_FLOAT_EQ(m.yarn_ext_factor, 0.0f);
    EXPECT_EQ(m.max_seq_len, 32768);
}

TEST(RopeOverrideTest, YarnInjectionSetsLoaderFields) {
    ModelConfig m = base_model(32768);
    RuntimeConfig::Rope rope;
    rope.scaling = "yarn";
    rope.factor = 4.0f;

    ASSERT_TRUE(apply_rope_override(m, rope));
    // Same convention as hf_config_loader rope_scaling type=yarn.
    EXPECT_FLOAT_EQ(m.rope_freq_scale, 4.0f);
    EXPECT_FLOAT_EQ(m.yarn_ext_factor, 1.0f);
    EXPECT_FLOAT_EQ(m.yarn_attn_factor, 1.0f);
    EXPECT_FLOAT_EQ(m.yarn_beta_fast, 32.0f);
    EXPECT_FLOAT_EQ(m.yarn_beta_slow, 1.0f);
    // orig_ctx resolved from the model's declared context (pre-bump)…
    EXPECT_EQ(m.rope_n_ctx_orig, 32768);
    // …and the declared context is extended to factor × orig_ctx.
    EXPECT_EQ(m.max_seq_len, 131072);
}

TEST(RopeOverrideTest, LinearInjectionKeepsYarnOff) {
    ModelConfig m = base_model(8192);
    RuntimeConfig::Rope rope;
    rope.scaling = "linear";
    rope.factor = 2.0f;

    ASSERT_TRUE(apply_rope_override(m, rope));
    EXPECT_FLOAT_EQ(m.rope_freq_scale, 2.0f);
    EXPECT_FLOAT_EQ(m.yarn_ext_factor, 0.0f);  // pure interpolation, no blending
    EXPECT_EQ(m.max_seq_len, 16384);
}

TEST(RopeOverrideTest, DeclaredYarnModelUsesItsOrigCtx) {
    // Model already ships YaRN (gpt-oss style): declared ctx is the EXTENDED
    // window, rope_n_ctx_orig carries the native one. A replacement override
    // must scale from the native window, not the extended one.
    ModelConfig m = base_model(131072);
    m.rope_freq_scale = 32.0f;
    m.yarn_ext_factor = 1.0f;
    m.rope_n_ctx_orig = 4096;

    RuntimeConfig::Rope rope;
    rope.scaling = "yarn";
    rope.factor = 64.0f;

    ASSERT_TRUE(apply_rope_override(m, rope));
    EXPECT_FLOAT_EQ(m.rope_freq_scale, 64.0f);
    EXPECT_EQ(m.rope_n_ctx_orig, 4096);
    EXPECT_EQ(m.max_seq_len, 262144);  // 64 × 4096
}

TEST(RopeOverrideTest, ExplicitOrigCtxWins) {
    ModelConfig m = base_model(32768);
    RuntimeConfig::Rope rope;
    rope.scaling = "yarn";
    rope.factor = 4.0f;
    rope.orig_ctx = 8192;

    ASSERT_TRUE(apply_rope_override(m, rope));
    EXPECT_EQ(m.rope_n_ctx_orig, 8192);
    // 4 × 8192 = 32768 does not exceed the declared ctx — no bump down.
    EXPECT_EQ(m.max_seq_len, 32768);
}

TEST(RopeOverrideTest, CustomYarnParams) {
    ModelConfig m = base_model();
    RuntimeConfig::Rope rope;
    rope.scaling = "yarn";
    rope.factor = 2.0f;
    rope.attn_factor = 1.5f;
    rope.beta_fast = 24.0f;
    rope.beta_slow = 2.0f;

    ASSERT_TRUE(apply_rope_override(m, rope));
    EXPECT_FLOAT_EQ(m.yarn_attn_factor, 1.5f);
    EXPECT_FLOAT_EQ(m.yarn_beta_fast, 24.0f);
    EXPECT_FLOAT_EQ(m.yarn_beta_slow, 2.0f);
}

TEST(RopeOverrideTest, RefusesPerDimTables) {
    // LongRoPE / llama3 precompute per-pair factors — a scalar override on
    // top would be silently wrong.
    ModelConfig m = base_model();
    m.rope_short_factor = {1.0f, 2.0f};
    m.rope_long_factor = {1.0f, 2.0f};

    RuntimeConfig::Rope rope;
    rope.scaling = "yarn";
    rope.factor = 4.0f;

    EXPECT_FALSE(apply_rope_override(m, rope));
    EXPECT_FLOAT_EQ(m.rope_freq_scale, 1.0f);
    EXPECT_EQ(m.max_seq_len, 32768);
}

TEST(RopeOverrideTest, RefusesMla) {
    ModelConfig m = base_model();
    m.kv_lora_rank = 512;  // is_mla()

    RuntimeConfig::Rope rope;
    rope.scaling = "yarn";
    rope.factor = 4.0f;

    EXPECT_FALSE(apply_rope_override(m, rope));
    EXPECT_FLOAT_EQ(m.rope_freq_scale, 1.0f);
}

TEST(RopeOverrideTest, RefusesNope) {
    ModelConfig m = base_model();
    m.rope_attn_disabled = true;

    RuntimeConfig::Rope rope;
    rope.scaling = "yarn";
    rope.factor = 4.0f;

    EXPECT_FALSE(apply_rope_override(m, rope));
}

TEST(RopeOverrideTest, RejectsBadFactorAndUnknownMode) {
    ModelConfig m = base_model();

    RuntimeConfig::Rope rope;
    rope.scaling = "yarn";
    rope.factor = 1.0f;  // must be > 1.0
    EXPECT_FALSE(apply_rope_override(m, rope));

    rope.factor = 4.0f;
    rope.scaling = "ntk";  // not supported
    EXPECT_FALSE(apply_rope_override(m, rope));

    EXPECT_FLOAT_EQ(m.rope_freq_scale, 1.0f);
    EXPECT_EQ(m.max_seq_len, 32768);
}

}  // namespace
}  // namespace imp
