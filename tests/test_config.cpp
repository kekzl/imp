#include <gtest/gtest.h>
#include "runtime/config.h"

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

TEST(RuntimeConfigTest, DefaultsAreSane) {
    RuntimeConfig cfg;
    EXPECT_FALSE(cfg.runtime.deterministic_gemm);
    EXPECT_EQ(cfg.runtime.cuda_graphs, "auto");
    EXPECT_FALSE(cfg.runtime.warmup);  // off by default; opt-in for prod rollout
    EXPECT_EQ(cfg.kv_cache.dtype, "fp16");
    EXPECT_EQ(cfg.moe.expert_overhead_pct, 10);
    EXPECT_FALSE(cfg.gdn.fp32_scan);
    EXPECT_EQ(cfg.diagnostics.exit_layer, -1);
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
    cfg.apply_overrides({
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
    cfg.apply_overrides({"kv_cache.dtype=fp8"});
    EXPECT_EQ(cfg.kv_cache.dtype, "fp8");
}

TEST(RuntimeConfigTest, BoolParsingIsLenient) {
    RuntimeConfig cfg;
    cfg.apply_overrides({"runtime.warmup=false"});
    EXPECT_FALSE(cfg.runtime.warmup);
    cfg.apply_overrides({"runtime.warmup=on"});
    EXPECT_TRUE(cfg.runtime.warmup);
    cfg.apply_overrides({"runtime.warmup=0"});
    EXPECT_FALSE(cfg.runtime.warmup);
    cfg.apply_overrides({"runtime.warmup=1"});
    EXPECT_TRUE(cfg.runtime.warmup);
}

TEST(RuntimeConfigTest, UnknownKeyIsIgnored) {
    // Should not crash; just log a warning.
    RuntimeConfig cfg;
    cfg.apply_overrides({"runtime.does_not_exist=42"});
    // Default is unchanged.
    EXPECT_FALSE(cfg.runtime.deterministic_gemm);
}

TEST(RuntimeConfigTest, MissingFileFallsBackToDefaults) {
    RuntimeConfig cfg;
    EXPECT_FALSE(cfg.load_from_file("/nonexistent/path/imp.conf"));
    // Defaults still in place.
    EXPECT_EQ(cfg.kv_cache.dtype, "fp16");
}

} // namespace
} // namespace imp
