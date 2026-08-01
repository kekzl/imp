#include <gtest/gtest.h>
#include "runtime/config.h"
#include "model/model_arch.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

namespace imp {
namespace {

// Helper to write a temporary config file and clean it up.
struct TempFile {
    std::string path;
    explicit TempFile(const std::string& body) {
        path = "/tmp/imp_test_config_" + std::to_string(::getpid()) + ".conf";
        std::ofstream ofs(path);
        ofs << body;
    }
    ~TempFile() { std::remove(path.c_str()); }
};

// Apply overrides and fail if any of them bound to nothing — the contract the
// tool mains enforce. Going through this helper means a typo in a test is a red
// test rather than an assertion that quietly stops testing anything.
void set_(RuntimeConfig& cfg, const std::vector<std::string>& kvs) {
    const std::vector<std::string> rejected = cfg.apply_overrides(kvs);
    ASSERT_TRUE(rejected.empty()) << "unbound override: " << rejected.front();
}

TEST(RuntimeConfigTest, DefaultsAreSane) {
    RuntimeConfig cfg;
    EXPECT_FALSE(cfg.runtime.deterministic_gemm);
    EXPECT_EQ(cfg.runtime.cuda_graphs, "auto");
    EXPECT_TRUE(cfg.runtime.warmup);  // default ON: greedy request-order independence (docs/determinism.md)
    EXPECT_EQ(cfg.kv_cache.dtype, "auto");
    EXPECT_EQ(cfg.kv_cache.swa_sizing, "auto");
    EXPECT_EQ(cfg.kv_cache.swa_sizing_mode(), SwaSizingMode::Auto);
    EXPECT_EQ(cfg.moe.expert_overhead_pct, 10);
    EXPECT_FALSE(cfg.gdn.fp32_scan);
    EXPECT_EQ(cfg.diagnostics.exit_layer, -1);
}

// kv_cache.swa_sizing tri-state: "auto"/"on"/"off" plus legacy bool literals
// (the key was a bool until 2026-07-24 — existing imp.conf files keep parsing).
TEST(RuntimeConfigTest, SwaSizingTriState) {
    RuntimeConfig cfg;
    set_(cfg, {"kv_cache.swa_sizing=on"});
    EXPECT_EQ(cfg.kv_cache.swa_sizing_mode(), SwaSizingMode::On);
    set_(cfg, {"kv_cache.swa_sizing=off"});
    EXPECT_EQ(cfg.kv_cache.swa_sizing_mode(), SwaSizingMode::Off);
    set_(cfg, {"kv_cache.swa_sizing=auto"});
    EXPECT_EQ(cfg.kv_cache.swa_sizing_mode(), SwaSizingMode::Auto);
    // Legacy bool spellings map to On/Off.
    set_(cfg, {"kv_cache.swa_sizing=true"});
    EXPECT_EQ(cfg.kv_cache.swa_sizing_mode(), SwaSizingMode::On);
    set_(cfg, {"kv_cache.swa_sizing=false"});
    EXPECT_EQ(cfg.kv_cache.swa_sizing_mode(), SwaSizingMode::Off);
    // Unknown value falls back to Off (never silently forces sizing on).
    set_(cfg, {"kv_cache.swa_sizing=banana"});
    EXPECT_EQ(cfg.kv_cache.swa_sizing_mode(), SwaSizingMode::Off);
    // SWA snapshot budget: off by default, plain int binder.
    EXPECT_EQ(RuntimeConfig{}.kv_cache.swa_snapshot_mb, 0);
    set_(cfg, {"kv_cache.swa_snapshot_mb=512"});
    EXPECT_EQ(cfg.kv_cache.swa_snapshot_mb, 512);
}

// The long-context FP8-KV quality gate (kv_cache.dtype=auto): only arch families
// empirically verified safe honor the model author's kv_cache_quant_algo=FP8 hint
// by default. Keep this allowlist conservative — see model.cpp for the evidence.
TEST(RuntimeConfigTest, KvFp8HintDefaultSafeAllowlist) {
    EXPECT_TRUE(kv_fp8_hint_default_safe(ModelArch::QWEN3));
    EXPECT_TRUE(kv_fp8_hint_default_safe(ModelArch::QWEN3_MOE));
    // LLAMA: verified via Phi-4-reasoning-plus-NVFP4 (+0.25% PPL, PR #749).
    EXPECT_TRUE(kv_fp8_hint_default_safe(ModelArch::LLAMA));
    // NEMOTRON_H_MOE: verified via Nemotron-3-Nano-30B — FP8 vs FP16 mean PPL
    // ~0.00% over a 26.5k-token corpus, 5 runs each (the 2026-06-23 single A/B
    // was run-to-run noise on this MoE+Mamba2 hybrid).
    EXPECT_TRUE(kv_fp8_hint_default_safe(ModelArch::NEMOTRON_H_MOE));
    // Not yet verified on this box → must stay FP16 by default. GEMMA4 baseline
    // PPL broken on the gate corpus; QWEN36_MOE quality fine but no hint declared.
    EXPECT_FALSE(kv_fp8_hint_default_safe(ModelArch::GEMMA4));
    EXPECT_FALSE(kv_fp8_hint_default_safe(ModelArch::QWEN35));
    EXPECT_FALSE(kv_fp8_hint_default_safe(ModelArch::QWEN36_MOE));
    EXPECT_FALSE(kv_fp8_hint_default_safe(ModelArch::GENERIC));
}

// The NO-HINT FP8-KV gate (kv_cache.dtype=auto on checkpoints without a
// kv_cache_quant_algo hint — i.e. every GGUF). Stricter bar than the hint list:
// the family must gate ~PPL-neutral at 16k context. See model.cpp for the
// 2026-07-12 evidence and the measured exclusions (QWEN36_MOE: +1.47% real on
// the NVFP4 variant; LLAMA: gate-corpus baseline broken).
TEST(RuntimeConfigTest, KvFp8NoHintDefaultSafeAllowlist) {
    EXPECT_TRUE(kv_fp8_no_hint_default_safe(ModelArch::QWEN3));
    EXPECT_TRUE(kv_fp8_no_hint_default_safe(ModelArch::QWEN3_MOE));
    EXPECT_FALSE(kv_fp8_no_hint_default_safe(ModelArch::LLAMA));
    EXPECT_FALSE(kv_fp8_no_hint_default_safe(ModelArch::QWEN36_MOE));
    EXPECT_FALSE(kv_fp8_no_hint_default_safe(ModelArch::NEMOTRON_H_MOE));
    EXPECT_FALSE(kv_fp8_no_hint_default_safe(ModelArch::GEMMA4));
    EXPECT_FALSE(kv_fp8_no_hint_default_safe(ModelArch::GENERIC));
}

TEST(RuntimeConfigTest, ParsesBasicSections) {
    TempFile f(R"(
[runtime]
deterministic_gemm = true
cuda_graphs = "never"

[kv_cache]
dtype = "fp8"

[moe]
expert_overhead_pct = 30
)");

    RuntimeConfig cfg;
    ASSERT_TRUE(cfg.load_from_file(f.path));
    EXPECT_TRUE(cfg.runtime.deterministic_gemm);
    EXPECT_EQ(cfg.runtime.cuda_graphs, "never");
    EXPECT_EQ(cfg.kv_cache.dtype, "fp8");
    EXPECT_EQ(cfg.moe.expert_overhead_pct, 30);
}

TEST(RuntimeConfigTest, ParsesRopeAndVramSections) {
    // Defaults: override off, planner at the historical envelope.
    RuntimeConfig defaults;
    EXPECT_TRUE(defaults.rope.scaling.empty());
    EXPECT_FLOAT_EQ(defaults.rope.factor, 1.0f);
    EXPECT_EQ(defaults.rope.orig_ctx, 0);
    EXPECT_FLOAT_EQ(defaults.rope.attn_factor, 1.0f);
    EXPECT_FLOAT_EQ(defaults.vram.kv_fraction, 0.8f);
    EXPECT_EQ(defaults.vram.reserve_floor_pct, 10);

    TempFile f(R"(
[rope]
scaling = "yarn"
factor = 4.0
orig_ctx = 32768
attn_factor = 1.5
beta_fast = 24
beta_slow = 2

[vram]
kv_fraction = 0.5
reserve_floor_pct = 5
)");

    RuntimeConfig cfg;
    ASSERT_TRUE(cfg.load_from_file(f.path));
    EXPECT_EQ(cfg.rope.scaling, "yarn");
    EXPECT_FLOAT_EQ(cfg.rope.factor, 4.0f);
    EXPECT_EQ(cfg.rope.orig_ctx, 32768);
    EXPECT_FLOAT_EQ(cfg.rope.attn_factor, 1.5f);
    EXPECT_FLOAT_EQ(cfg.rope.beta_fast, 24.0f);
    EXPECT_FLOAT_EQ(cfg.rope.beta_slow, 2.0f);
    EXPECT_FLOAT_EQ(cfg.vram.kv_fraction, 0.5f);
    EXPECT_EQ(cfg.vram.reserve_floor_pct, 5);

    // --set style overrides reach the same fields.
    set_(cfg, {"rope.scaling=linear", "rope.factor=2", "vram.kv_fraction=0.9"});
    EXPECT_EQ(cfg.rope.scaling, "linear");
    EXPECT_FLOAT_EQ(cfg.rope.factor, 2.0f);
    EXPECT_FLOAT_EQ(cfg.vram.kv_fraction, 0.9f);
}

TEST(RuntimeConfigTest, IgnoresCommentsAndBlankLines) {
    TempFile f(R"(
# top comment
[runtime]
# inline comment

deterministic_gemm = true   # trailing comment
warmup              = false
)");

    RuntimeConfig cfg;
    ASSERT_TRUE(cfg.load_from_file(f.path));
    EXPECT_TRUE(cfg.runtime.deterministic_gemm);
    EXPECT_FALSE(cfg.runtime.warmup);
}

TEST(RuntimeConfigTest, ApplyOverrides) {
    RuntimeConfig cfg;
    set_(cfg, {
        "kv_cache.dtype=fp8",
        "runtime.cuda_graphs=never",
        "moe.expert_overhead_pct=30",
        "gdn.fp32_scan=true",
    });
    EXPECT_EQ(cfg.kv_cache.dtype, "fp8");
    EXPECT_EQ(cfg.runtime.cuda_graphs, "never");
    EXPECT_EQ(cfg.moe.expert_overhead_pct, 30);
    EXPECT_TRUE(cfg.gdn.fp32_scan);
}

TEST(RuntimeConfigTest, OverrideWinsOverFile) {
    TempFile f(R"(
[kv_cache]
dtype = "fp16"
)");

    RuntimeConfig cfg;
    cfg.load_from_file(f.path);
    EXPECT_EQ(cfg.kv_cache.dtype, "fp16");
    set_(cfg, {"kv_cache.dtype=fp8"});
    EXPECT_EQ(cfg.kv_cache.dtype, "fp8");
}

TEST(RuntimeConfigTest, BoolParsingIsLenient) {
    RuntimeConfig cfg;
    set_(cfg, {"runtime.warmup=false"});
    EXPECT_FALSE(cfg.runtime.warmup);
    set_(cfg, {"runtime.warmup=on"});
    EXPECT_TRUE(cfg.runtime.warmup);
    set_(cfg, {"runtime.warmup=0"});
    EXPECT_FALSE(cfg.runtime.warmup);
    set_(cfg, {"runtime.warmup=1"});
    EXPECT_TRUE(cfg.runtime.warmup);
}

TEST(RuntimeConfigTest, UnknownOverrideIsReportedNotSwallowed) {
    // An override that binds to nothing comes back named, so `--set` can refuse
    // instead of running a measurement the flag never configured. This was not
    // hypothetical: `--set gemm.deterministic=true` (the key is
    // runtime.deterministic_gemm) sat in the AWQ reproduction harness and did
    // nothing at all.
    RuntimeConfig cfg;
    const std::vector<std::string> rejected =
        cfg.apply_overrides({"runtime.does_not_exist=42", "runtime.warmup=false", "gemm.deterministic=true"});
    ASSERT_EQ(rejected.size(), 2u);
    EXPECT_NE(rejected[0].find("runtime.does_not_exist=42"), std::string::npos);
    EXPECT_NE(rejected[1].find("gemm.deterministic=true"), std::string::npos);
    // The good key in the same batch still applied, and no field moved for the
    // bad ones.
    EXPECT_FALSE(cfg.runtime.warmup);
    EXPECT_FALSE(cfg.runtime.deterministic_gemm);
}

TEST(RuntimeConfigTest, MalformedOverrideIsReported) {
    RuntimeConfig cfg;
    const std::vector<std::string> rejected = cfg.apply_overrides({"runtime.warmup"});
    ASSERT_EQ(rejected.size(), 1u);
    EXPECT_NE(rejected[0].find("expected key=value"), std::string::npos);
}

TEST(RuntimeConfigTest, UnknownKeyInAFileStaysAWarning) {
    // The other half of the contract: a config file may outlive the build that
    // understood every key in it, so the file path must not become fatal.
    TempFile tf("[runtime]\ndoes_not_exist = 42\nwarmup = false\n");
    RuntimeConfig cfg;
    ASSERT_TRUE(cfg.load_from_file(tf.path));
    EXPECT_FALSE(cfg.runtime.warmup);
}

TEST(RuntimeConfigTest, MissingFileFallsBackToDefaults) {
    RuntimeConfig cfg;
    EXPECT_FALSE(cfg.load_from_file("/nonexistent/path/imp.conf"));
    // Defaults still in place.
    EXPECT_EQ(cfg.kv_cache.dtype, "auto");
}

}  // namespace
}  // namespace imp
