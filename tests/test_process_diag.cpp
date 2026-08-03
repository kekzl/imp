// process_diag ↔ RuntimeConfig coupling (#1205).
//
// process_diag is a process-wide snapshot of the flags that leaf kernels read
// when they have no RuntimeConfig to hand. Until #1205 process_diag_install()
// ran ONLY in the tool mains, so a C-API embedding got the built-in defaults
// for all mirrored flags while exec/ read the engine's own RuntimeConfig — the
// same config producing different kernels depending on who started the engine.
// Engine::init() now installs, which makes these two properties load-bearing:
//
//   1. install() must transfer EVERY mirrored field. A field added to
//      ProcessDiag but forgotten in install() silently keeps its default
//      forever; this test fails when that happens for any field it covers.
//   2. The built-in defaults must equal the RuntimeConfig defaults, so the
//      engine-less paths that still read process_diag before any install
//      (standalone tools, unit tests) behave like a default engine.
//
// CPU-only: no CUDA, no Model, no Engine — this is the whole point of the
// process_diag indirection and it keeps the check in the CI unit lane.

#include "runtime/config.h"
#include "core/logging.h"

#include <utility>
#include "runtime/process_diag.h"

#include <gtest/gtest.h>

#include <string>

using namespace imp;

namespace {

// Every mirrored field flipped away from its default. Keep in lock-step with
// process_diag_install(); a field added there and not here is not covered.
RuntimeConfig make_non_default() {
    RuntimeConfig c;
    c.diagnostics.debug_forward = true;
    c.diagnostics.debug_template = true;
    c.diagnostics.graph_diag = true;
    c.diagnostics.nvfp4_force_dequant = true;
    c.diagnostics.log_gemm_algo = true;
    c.diagnostics.audit_nvfp4_scales = true;
    c.diagnostics.dump_hidden_dir = "/var/tmp/imp-hidden";
    c.diagnostics.graph_dump_dir = "/var/tmp/imp-graphs";

    c.runtime.no_pdl = true;
    c.runtime.no_vision_graph = true;
    c.runtime.graph_capture_mode = "global";
    c.runtime.prefill_graph = false;
    c.runtime.deterministic_gemm = true;

    c.gemm.cublas_fp16_acc = "on";

    c.attention.splitk_pipe = false;
    c.attention.fp8_tile = false;
    c.attention.fp8_tile_gqa = false;
    c.attention.fa2_f16acc = false;
    c.attention.fa2_pv_f16acc = false;
    c.attention.fa2_hd256 = false;
    c.attention.fp8_qk_scaled = true;
    c.attention.mxfp4 = "always";
    c.attention.mxfp4_blockscale = true;
    c.attention.mxfp4_ksmooth = true;
    c.attention.mxfp4_pv_fp4 = true;
    c.attention.mxfp4_promote_budget = 0.25f;

    c.ffn.sparsity_probe = true;

    c.moe.mr_nr = 32;
    c.moe.expert_overhead_pct = 42;
    c.moe.force_host_experts = 7;

    c.gdn.layout_override = "blocked";
    return c;
}

// Restores the process-wide slot so test ordering inside the binary cannot
// leak a non-default snapshot into an unrelated test.
struct ProcessDiagGuard {
    // install() now also sets the process log level, so restoring the defaults
    // has to put that back too — otherwise one test leaks DEBUG into every
    // later test in this binary.
    LogLevel saved_level = log_get_level();
    ~ProcessDiagGuard() {
        process_diag_install(RuntimeConfig{});
        log_set_level(saved_level);
    }
};

}  // namespace

// Every mirrored field survives the RuntimeConfig → ProcessDiag transfer.
// This is the regression guard for "added a flag, forgot the install() line".
TEST(ProcessDiag, InstallTransfersEveryMirroredField) {
    ProcessDiagGuard restore;
    const RuntimeConfig c = make_non_default();
    process_diag_install(c);

    EXPECT_TRUE(process_diag_debug_forward());
    EXPECT_TRUE(process_diag_debug_template());
    EXPECT_TRUE(process_diag_graph_diag());
    EXPECT_TRUE(process_diag_nvfp4_force_dequant());
    EXPECT_TRUE(process_diag_log_gemm_algo());
    EXPECT_TRUE(process_diag_audit_nvfp4_scales());
    ASSERT_NE(process_diag_dump_hidden_dir(), nullptr);
    EXPECT_EQ(std::string(process_diag_dump_hidden_dir()), "/var/tmp/imp-hidden");
    ASSERT_NE(process_diag_graph_dump_dir(), nullptr);
    EXPECT_EQ(std::string(process_diag_graph_dump_dir()), "/var/tmp/imp-graphs");

    EXPECT_TRUE(process_diag_no_pdl());
    EXPECT_TRUE(process_diag_no_vision_graph());
    EXPECT_EQ(process_diag_graph_capture_mode(), "global");
    EXPECT_FALSE(process_diag_prefill_graph_enabled());

    EXPECT_TRUE(process_diag_deterministic_gemm());
    EXPECT_TRUE(process_diag_cublas_fp16_acc());

    EXPECT_FALSE(process_diag_attention_splitk_pipe());
    EXPECT_FALSE(process_diag_attention_fp8_tile());
    EXPECT_FALSE(process_diag_attention_fp8_tile_gqa());
    EXPECT_FALSE(process_diag_fa2_f16acc());
    EXPECT_FALSE(process_diag_fa2_pv_f16acc());
    EXPECT_FALSE(process_diag_fa2_hd256());
    EXPECT_TRUE(process_diag_fp8_qk_scaled());
    EXPECT_EQ(process_diag_attention_mxfp4_mode(), "always");
    EXPECT_TRUE(process_diag_mxfp4_blockscale());
    EXPECT_TRUE(process_diag_mxfp4_ksmooth());
    EXPECT_TRUE(process_diag_mxfp4_pv_fp4());
    EXPECT_FLOAT_EQ(process_diag_mxfp4_promote_budget(), 0.25f);

    EXPECT_TRUE(process_diag_ffn_sparsity_probe());

    EXPECT_EQ(process_diag_moe_mr_nr(), 32);
    EXPECT_EQ(process_diag_moe_expert_overhead_pct(), 42);
    EXPECT_EQ(process_diag_moe_force_host_experts(), 7);

    EXPECT_EQ(process_diag_gdn_layout_override(), "blocked");
}

// Installing a default RuntimeConfig must reproduce the built-in defaults.
// If these drift apart, an engine-less reader (standalone tool, unit test)
// sees different kernel behaviour than a default engine does.
TEST(ProcessDiag, DefaultConfigMatchesBuiltInDefaults) {
    ProcessDiagGuard restore;
    // Dirty the slot first so a no-op install() cannot pass this by accident.
    process_diag_install(make_non_default());
    process_diag_install(RuntimeConfig{});

    const RuntimeConfig d;
    EXPECT_EQ(process_diag_debug_forward(), d.diagnostics.debug_forward);
    EXPECT_EQ(process_diag_graph_diag(), d.diagnostics.graph_diag);
    EXPECT_EQ(process_diag_no_pdl(), d.runtime.no_pdl);
    EXPECT_EQ(process_diag_no_vision_graph(), d.runtime.no_vision_graph);
    EXPECT_EQ(process_diag_graph_capture_mode(), d.runtime.graph_capture_mode);
    EXPECT_EQ(process_diag_prefill_graph_enabled(), d.runtime.prefill_graph);
    EXPECT_EQ(process_diag_deterministic_gemm(), d.runtime.deterministic_gemm);
    EXPECT_EQ(process_diag_attention_splitk_pipe(), d.attention.splitk_pipe);
    EXPECT_EQ(process_diag_attention_fp8_tile(), d.attention.fp8_tile);
    EXPECT_EQ(process_diag_attention_fp8_tile_gqa(), d.attention.fp8_tile_gqa);
    EXPECT_EQ(process_diag_fa2_f16acc(), d.attention.fa2_f16acc);
    EXPECT_EQ(process_diag_fa2_pv_f16acc(), d.attention.fa2_pv_f16acc);
    EXPECT_EQ(process_diag_fa2_hd256(), d.attention.fa2_hd256);
    EXPECT_EQ(process_diag_fp8_qk_scaled(), d.attention.fp8_qk_scaled);
    EXPECT_EQ(process_diag_attention_mxfp4_mode(), d.attention.mxfp4);
    EXPECT_EQ(process_diag_ffn_sparsity_probe(), d.ffn.sparsity_probe);
    EXPECT_EQ(process_diag_moe_mr_nr(), d.moe.mr_nr);
    EXPECT_EQ(process_diag_moe_expert_overhead_pct(), d.moe.expert_overhead_pct);
    EXPECT_EQ(process_diag_moe_force_host_experts(), d.moe.force_host_experts);
    EXPECT_EQ(process_diag_gdn_layout_override(), d.gdn.layout_override);

    // "auto" (the default) maps to OFF — the arch resolvers promote it later.
    EXPECT_EQ(d.gemm.cublas_fp16_acc, "auto");
    EXPECT_FALSE(process_diag_cublas_fp16_acc());
}

// dump_hidden_dir carries a documented shorthand: "1"/"all" resolve to /tmp.
TEST(ProcessDiag, DumpHiddenDirShorthandResolves) {
    ProcessDiagGuard restore;
    RuntimeConfig c;
    c.diagnostics.dump_hidden_dir = "1";
    process_diag_install(c);
    ASSERT_NE(process_diag_dump_hidden_dir(), nullptr);
    EXPECT_EQ(std::string(process_diag_dump_hidden_dir()), "/tmp");

    c.diagnostics.dump_hidden_dir = "all";
    process_diag_install(c);
    ASSERT_NE(process_diag_dump_hidden_dir(), nullptr);
    EXPECT_EQ(std::string(process_diag_dump_hidden_dir()), "/tmp");
}

// --------------------------------------------------------------------------
// Log level (2026-08-03)
// --------------------------------------------------------------------------
//
// `log_set_level` was the only writer of g_log_level and NOTHING called it, so
// the level was pinned at INFO and all 76 IMP_LOG_DEBUG sites in the engine
// were unreachable — a debug facility that could not be switched on. The fix is
// a config key applied here, in the one function that runs from both tool mains
// AND Engine::init. These tests pin both halves: the parse and the transfer.

TEST(LogLevel, FromStringMapsEveryWordCaseInsensitively) {
    const std::pair<const char*, LogLevel> cases[] = {
        {"debug", LogLevel::DEBUG}, {"DEBUG", LogLevel::DEBUG}, {"Info", LogLevel::INFO},
        {"warn", LogLevel::WARN},   {"error", LogLevel::ERROR}, {"FATAL", LogLevel::FATAL},
    };
    for (const auto& [word, expected] : cases) {
        LogLevel got = LogLevel::FATAL;
        ASSERT_TRUE(log_level_from_string(word, got)) << word;
        EXPECT_EQ(got, expected) << word;
    }
}

// Rejection has to be explicit, not "falls back to INFO": a typo that silently
// resolves to the default would restore exactly the state this key fixed.
TEST(LogLevel, FromStringRejectsAnythingElseAndLeavesOutAlone) {
    for (const char* bad : {"", "verbose", "dbg", "informational", "debug ", "0"}) {
        LogLevel got = LogLevel::WARN;  // sentinel: must survive untouched
        EXPECT_FALSE(log_level_from_string(bad, got)) << "'" << bad << "'";
        EXPECT_EQ(got, LogLevel::WARN) << "'" << bad << "' modified out";
    }
    LogLevel got = LogLevel::WARN;
    EXPECT_FALSE(log_level_from_string(nullptr, got));
}

TEST(ProcessDiag, InstallAppliesTheConfiguredLogLevel) {
    ProcessDiagGuard restore;
    RuntimeConfig c;
    c.diagnostics.log_level = "debug";
    process_diag_install(c);
    EXPECT_EQ(log_get_level(), LogLevel::DEBUG);

    c.diagnostics.log_level = "error";
    process_diag_install(c);
    EXPECT_EQ(log_get_level(), LogLevel::ERROR);
}

TEST(ProcessDiag, InstallKeepsTheCurrentLevelOnAnUnknownWord) {
    ProcessDiagGuard restore;
    RuntimeConfig c;
    c.diagnostics.log_level = "warn";
    process_diag_install(c);
    ASSERT_EQ(log_get_level(), LogLevel::WARN);

    c.diagnostics.log_level = "louder-please";
    process_diag_install(c);
    EXPECT_EQ(log_get_level(), LogLevel::WARN) << "an unknown word must not silently reset the level";
}

// The default must be the level the engine had before the key existed, so
// adding it changed nothing for anyone who does not set it.
TEST(ProcessDiag, DefaultConfigLeavesTheLevelAtInfo) {
    ProcessDiagGuard restore;
    log_set_level(LogLevel::ERROR);
    process_diag_install(RuntimeConfig{});
    EXPECT_EQ(log_get_level(), LogLevel::INFO);
}
