// CPU unit tests for the host-side routing decisions extracted from the
// attention-prefill dispatch (attention_dispatch.cu) and the MoE-prefill GEMM
// dispatch (executor_forward_moe_cutlass.cu). R2 / P1.4 — the #493 regression
// was a routing change covered only by E2E; the grouped-GEMM-vs-fallback path
// (#574) was likewise E2E-only. These pin the tables as a cheap CPU diff.

#include "compute/attention_dispatch_decision.h"
#include "exec/moe_prefill_decision.h"

#include <gtest/gtest.h>

using namespace imp;

// --------------------------------------------------------------------------
// Attention prefill dispatch table
// --------------------------------------------------------------------------

namespace {

// Default config: fmha_fa2="on", fp8_fmha="never" (opt-in, #511), fmha_sm120="auto".
RuntimeConfig default_cfg() { return RuntimeConfig{}; }

// Typical hd=128 F16 prefill: every specialized kernel would accept.
AttnKernelSupport all_accept() {
    AttnKernelSupport s;
    s.mxfp4_available = false;  // mxfp4 is opt-in, off by default
    s.mxfp4_accepts = true;
    s.fa2_accepts = true;
    s.fp8_accepts = true;
    s.fmha_sm120_accepts = true;
    return s;
}

}  // namespace

TEST(AttnDispatchTable, DefaultConfigPicksFA2) {
    // fmha_fa2="on" (default) and FA2 accepts hd=128 → FA2 wins over FP8.
    EXPECT_EQ(select_attn_prefill_path(default_cfg(), all_accept()), AttnPrefillPath::FA2);
}

TEST(AttnDispatchTable, Mxfp4WinsWhenAvailableAndAccepts) {
    auto cfg = default_cfg();
    auto sup = all_accept();
    sup.mxfp4_available = true;
    EXPECT_EQ(select_attn_prefill_path(cfg, sup), AttnPrefillPath::MXFP4);
}

TEST(AttnDispatchTable, Mxfp4AvailableButDeclinesFallsToFA2) {
    auto cfg = default_cfg();
    auto sup = all_accept();
    sup.mxfp4_available = true;
    sup.mxfp4_accepts = false;  // e.g. head_dim < 32
    EXPECT_EQ(select_attn_prefill_path(cfg, sup), AttnPrefillPath::FA2);
}

TEST(AttnDispatchTable, FA2NeverFallsToFMHANotFP8) {
    // fp8-QK is opt-in (#511): disabling FA2 must NOT resurrect the e4m3 path.
    auto cfg = default_cfg();
    cfg.attention.fmha_fa2 = "never";
    EXPECT_EQ(select_attn_prefill_path(cfg, all_accept()), AttnPrefillPath::FMHA_SM120);
}

TEST(AttnDispatchTable, FA2OnButDeclinesFallsToFMHA) {
    // hd != 128 (gemma-3 hd=256, the #511 production victim): FA2 declines
    // even with fmha_fa2="on" → FP16 WMMA serves, never the fp8-QK kernel.
    auto cfg = default_cfg();
    auto sup = all_accept();
    sup.fa2_accepts = false;
    EXPECT_EQ(select_attn_prefill_path(cfg, sup), AttnPrefillPath::FMHA_SM120);
}

TEST(AttnDispatchTable, FP8AutoIsOff) {
    // Legacy "auto" (pre-#511-fix default) must behave as OFF, not ON.
    auto cfg = default_cfg();
    cfg.attention.fmha_fa2 = "never";
    cfg.attention.fp8_fmha = "auto";
    EXPECT_EQ(select_attn_prefill_path(cfg, all_accept()), AttnPrefillPath::FMHA_SM120);
}

TEST(AttnDispatchTable, FP8OptInServes) {
    // Explicit fp8_fmha="on" (experiments) restores the fp8 tier behind FA2.
    auto cfg = default_cfg();
    cfg.attention.fmha_fa2 = "never";
    cfg.attention.fp8_fmha = "on";
    EXPECT_EQ(select_attn_prefill_path(cfg, all_accept()), AttnPrefillPath::FP8);
}

TEST(AttnDispatchTable, FP8NeverWithFA2NeverFallsToFMHA) {
    auto cfg = default_cfg();
    cfg.attention.fmha_fa2 = "never";
    cfg.attention.fp8_fmha = "never";
    EXPECT_EQ(select_attn_prefill_path(cfg, all_accept()), AttnPrefillPath::FMHA_SM120);
}

TEST(AttnDispatchTable, FP8DeclinesFallsToFMHA) {
    auto cfg = default_cfg();
    cfg.attention.fmha_fa2 = "never";
    auto sup = all_accept();
    sup.fp8_accepts = false;
    EXPECT_EQ(select_attn_prefill_path(cfg, sup), AttnPrefillPath::FMHA_SM120);
}

TEST(AttnDispatchTable, AllSpecializedDisabledFallsToBlackwell) {
    auto cfg = default_cfg();
    cfg.attention.fmha_fa2 = "never";
    cfg.attention.fp8_fmha = "never";
    cfg.attention.fmha_sm120 = "never";
    EXPECT_EQ(select_attn_prefill_path(cfg, all_accept()), AttnPrefillPath::BLACKWELL);
}

TEST(AttnDispatchTable, AllSpecializedDeclineFallsToBlackwell) {
    // Unsupported config for every specialized kernel → final WMMA fallback.
    auto cfg = default_cfg();
    AttnKernelSupport sup;  // all false
    EXPECT_EQ(select_attn_prefill_path(cfg, sup), AttnPrefillPath::BLACKWELL);
}

TEST(AttnDispatchTable, FMHASm120AutoIsNotNever) {
    // "auto" (default) must not be treated as "never" for the WMMA tier.
    auto cfg = default_cfg();
    cfg.attention.fmha_fa2 = "never";
    cfg.attention.fp8_fmha = "never";
    auto sup = all_accept();
    EXPECT_EQ(cfg.attention.fmha_sm120, "auto");
    EXPECT_EQ(select_attn_prefill_path(cfg, sup), AttnPrefillPath::FMHA_SM120);
}

// --------------------------------------------------------------------------
// MoE prefill GEMM dispatch table
// --------------------------------------------------------------------------

namespace {

// Workspace with every tier's preconditions satisfied — the path is then
// decided purely by the config + arch gates.
MoePrefillWorkspace all_ready() {
    MoePrefillWorkspace ws;
    ws.device_args_ready = true;
    ws.grouped_available = true;
    ws.smallM_available = true;
    ws.smallM_under_threshold = true;
    ws.grouped_ready = true;
    return ws;
}

}  // namespace

TEST(MoePrefillTable, DefaultPicksDeviceArgs) {
    // nvfp4_device_args=true (default), non-gpt-oss, workspace ready.
    EXPECT_EQ(select_moe_prefill_path(ModelArch::QWEN3_MOE, default_cfg(), all_ready()),
              MoePrefillPath::DEVICE_ARGS);
}

TEST(MoePrefillTable, GptOssSkipsDeviceArgsTakesGrouped) {
    // gpt-oss is arch-gated off device-args AND smallM → host-args grouped
    // (with bias seams), even when smallM would otherwise apply.
    auto ws = all_ready();
    auto cfg = default_cfg();
    cfg.moe.nvfp4_smallM = true;  // would route others to smallM
    EXPECT_EQ(select_moe_prefill_path(ModelArch::GPT_OSS, cfg, ws), MoePrefillPath::GROUPED);
}

TEST(MoePrefillTable, DeviceArgsDisabledFallsToGrouped) {
    auto cfg = default_cfg();
    cfg.moe.nvfp4_device_args = false;
    EXPECT_EQ(select_moe_prefill_path(ModelArch::QWEN3_MOE, cfg, all_ready()),
              MoePrefillPath::GROUPED);
}

TEST(MoePrefillTable, DeviceArgsWorkspaceNotReadyFallsToGrouped) {
    auto ws = all_ready();
    ws.device_args_ready = false;
    EXPECT_EQ(select_moe_prefill_path(ModelArch::QWEN3_MOE, default_cfg(), ws),
              MoePrefillPath::GROUPED);
}

TEST(MoePrefillTable, SmallMWhenOptedInAndUnderThreshold) {
    auto cfg = default_cfg();
    cfg.moe.nvfp4_device_args = false;  // so device-args doesn't win first
    cfg.moe.nvfp4_smallM = true;
    EXPECT_EQ(select_moe_prefill_path(ModelArch::QWEN3_MOE, cfg, all_ready()),
              MoePrefillPath::SMALL_M);
}

TEST(MoePrefillTable, SmallMOverThresholdFallsToGrouped) {
    auto cfg = default_cfg();
    cfg.moe.nvfp4_device_args = false;
    cfg.moe.nvfp4_smallM = true;
    auto ws = all_ready();
    ws.smallM_under_threshold = false;  // too many tokens
    EXPECT_EQ(select_moe_prefill_path(ModelArch::QWEN3_MOE, cfg, ws), MoePrefillPath::GROUPED);
}

TEST(MoePrefillTable, NoCutlass3xFallsToLegacy) {
    auto cfg = default_cfg();
    cfg.moe.no_cutlass3x = true;
    EXPECT_EQ(select_moe_prefill_path(ModelArch::QWEN3_MOE, cfg, all_ready()),
              MoePrefillPath::LEGACY);
}

TEST(MoePrefillTable, GroupedUnavailableFallsToLegacy) {
    auto cfg = default_cfg();
    cfg.moe.nvfp4_device_args = false;
    auto ws = all_ready();
    ws.grouped_available = false;
    EXPECT_EQ(select_moe_prefill_path(ModelArch::QWEN3_MOE, cfg, ws), MoePrefillPath::LEGACY);
}

TEST(MoePrefillTable, GroupedNotReadyFallsToLegacy) {
    auto cfg = default_cfg();
    cfg.moe.nvfp4_device_args = false;
    auto ws = all_ready();
    ws.grouped_ready = false;
    EXPECT_EQ(select_moe_prefill_path(ModelArch::QWEN3_MOE, cfg, ws), MoePrefillPath::LEGACY);
}

TEST(MoePrefillTable, GptOssNoCutlass3xFallsToLegacy) {
    // Even gpt-oss drops to legacy when the grouped kernel is unavailable.
    auto cfg = default_cfg();
    cfg.moe.no_cutlass3x = true;
    EXPECT_EQ(select_moe_prefill_path(ModelArch::GPT_OSS, cfg, all_ready()),
              MoePrefillPath::LEGACY);
}
