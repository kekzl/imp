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
    s.blackwell_accepts = true;
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

TEST(AttnDispatchTable, AllDeclineIsNoneAndThrows) {
    // Unsupported config for every kernel incl. the final blackwell tier
    // (e.g. hd=256 with fmha_sm120=never: blackwell needs ~176 KB smem) →
    // NONE; the dispatcher throws instead of silently corrupting O (#654).
    auto cfg = default_cfg();
    AttnKernelSupport sup;  // all false
    EXPECT_EQ(select_attn_prefill_path(cfg, sup), AttnPrefillPath::NONE);
}

TEST(AttnDispatchTable, BlackwellDeclineWithFMHADisabledIsNone) {
    // The exact #654 production-forced config: hd=256, fmha_sm120=never.
    // FA2 declines (hd!=128), fp8 opt-in/off, blackwell declines (smem).
    auto cfg = default_cfg();
    cfg.attention.fmha_sm120 = "never";
    auto sup = all_accept();
    sup.fa2_accepts = false;
    sup.blackwell_accepts = false;
    EXPECT_EQ(select_attn_prefill_path(cfg, sup), AttnPrefillPath::NONE);
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
    EXPECT_EQ(select_moe_prefill_path(ModelArch::QWEN3_MOE, cfg, all_ready()), MoePrefillPath::GROUPED);
}

TEST(MoePrefillTable, DeviceArgsWorkspaceNotReadyFallsToGrouped) {
    auto ws = all_ready();
    ws.device_args_ready = false;
    EXPECT_EQ(select_moe_prefill_path(ModelArch::QWEN3_MOE, default_cfg(), ws), MoePrefillPath::GROUPED);
}

TEST(MoePrefillTable, SmallMWhenOptedInAndUnderThreshold) {
    auto cfg = default_cfg();
    cfg.moe.nvfp4_device_args = false;  // so device-args doesn't win first
    cfg.moe.nvfp4_smallM = true;
    EXPECT_EQ(select_moe_prefill_path(ModelArch::QWEN3_MOE, cfg, all_ready()), MoePrefillPath::SMALL_M);
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
    EXPECT_EQ(select_moe_prefill_path(ModelArch::QWEN3_MOE, cfg, all_ready()), MoePrefillPath::LEGACY);
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
    EXPECT_EQ(select_moe_prefill_path(ModelArch::GPT_OSS, cfg, all_ready()), MoePrefillPath::LEGACY);
}

// ---- #992: learned-sink pre-gate ------------------------------------------

TEST(AttnDispatchTable, SinksRouteToFMHAEvenWhenFA2Accepts) {
    // gpt-oss learned sinks: only the FP16 WMMA FMHA folds them into its
    // online softmax — FA2/MXFP4/FP8 must not serve sink configs.
    EXPECT_EQ(select_attn_prefill_path(default_cfg(), all_accept(), /*has_sinks=*/true),
              AttnPrefillPath::FMHA_SM120);
}

TEST(AttnDispatchTable, SinksWithFMHADeclineIsNoneNotBlackwell) {
    // A sink-blind fallback would produce silently wrong output — the
    // dispatcher throws (NONE), it must NOT fall through to Blackwell.
    auto sup = all_accept();
    sup.fmha_sm120_accepts = false;
    EXPECT_EQ(select_attn_prefill_path(default_cfg(), sup, /*has_sinks=*/true), AttnPrefillPath::NONE);
}

TEST(AttnDispatchTable, SinksWithFMHANeverIsNone) {
    auto cfg = default_cfg();
    cfg.attention.fmha_sm120 = "never";
    EXPECT_EQ(select_attn_prefill_path(cfg, all_accept(), /*has_sinks=*/true), AttnPrefillPath::NONE);
}

// --------------------------------------------------------------------------
// The coupling that makes this file mean something (audit finding F-3)
// --------------------------------------------------------------------------
//
// Until #1210 this model was TEST-ONLY: attention_dispatch.cu mentioned
// select_attn_prefill_path() in a comment and never called it, so a reorder in
// the dispatch left every test above green while the header described a routing
// order that no longer existed.
//
// The dispatch now replays the model at the moment a tier wins, against the
// booleans it observed on the way down, and logs a divergence. These tests pin
// the invariant that replay depends on: for each tier, the observation pattern
// the dispatch can actually produce when THAT tier wins must make the model
// name the same tier. If it did not, the production check would cry wolf on a
// correct dispatch and get muted.
namespace {

// What attention_dispatch.cu has filled in by the time `winner` commits: every
// tier it walked past declined (false), the winner accepted, and tiers below
// the winner were never evaluated so they stay false.
AttnKernelSupport observed_when(AttnPrefillPath winner) {
    AttnKernelSupport s;  // all false — the dispatch's starting state
    switch (winner) {
        case AttnPrefillPath::MXFP4:
            s.mxfp4_available = true;
            s.mxfp4_accepts = true;
            break;
        case AttnPrefillPath::FA2:
            s.fa2_accepts = true;
            break;
        case AttnPrefillPath::FP8:
            s.fp8_accepts = true;
            break;
        case AttnPrefillPath::FMHA_SM120:
            s.fmha_sm120_accepts = true;
            break;
        case AttnPrefillPath::BLACKWELL:
            s.blackwell_accepts = true;
            break;
        case AttnPrefillPath::NONE:
            break;
    }
    return s;
}

}  // namespace

TEST(AttnRoutingModelCoupling, EveryTierReplaysToItself) {
    auto cfg = default_cfg();
    cfg.attention.fp8_fmha = "on";  // so the FP8 tier is reachable at all

    const AttnPrefillPath tiers[] = {AttnPrefillPath::FA2, AttnPrefillPath::FP8, AttnPrefillPath::FMHA_SM120,
                                     AttnPrefillPath::BLACKWELL};
    for (AttnPrefillPath t : tiers) {
        EXPECT_EQ(select_attn_prefill_path(cfg, observed_when(t)), t)
            << "the dispatch would run " << attn_prefill_path_name(t) << " but the model replays to "
            << attn_prefill_path_name(select_attn_prefill_path(cfg, observed_when(t)))
            << " — verify_against_routing_model() would fire on a correct dispatch";
    }
}

TEST(AttnRoutingModelCoupling, Mxfp4ReplaysToItselfWhenAvailable) {
    EXPECT_EQ(select_attn_prefill_path(default_cfg(), observed_when(AttnPrefillPath::MXFP4)),
              AttnPrefillPath::MXFP4);
}

// The sinks pre-gate (#992) is a separate entry point in the dispatch: it either
// runs the FP16 WMMA tier or throws. Both arms must replay.
TEST(AttnRoutingModelCoupling, SinksReplayToFmhaOrNone) {
    auto sup = observed_when(AttnPrefillPath::FMHA_SM120);
    EXPECT_EQ(select_attn_prefill_path(default_cfg(), sup, /*has_sinks=*/true), AttnPrefillPath::FMHA_SM120);

    AttnKernelSupport declined;  // sink-capable tier said no -> dispatch throws
    EXPECT_EQ(select_attn_prefill_path(default_cfg(), declined, /*has_sinks=*/true), AttnPrefillPath::NONE);
}

// The chain-exhausted case: nothing accepted, so the dispatch throws (#654) and
// the model must agree rather than naming a tier that never ran.
TEST(AttnRoutingModelCoupling, NothingAcceptsReplaysToNone) {
    EXPECT_EQ(select_attn_prefill_path(default_cfg(), AttnKernelSupport{}), AttnPrefillPath::NONE);
}

// --------------------------------------------------------------------------
// Tier PRECEDENCE — the tests that actually catch a reorder
// --------------------------------------------------------------------------
//
// The replay tests above cannot: each of their support patterns has exactly ONE
// accepting tier, so swapping two tiers in the chain leaves every answer
// unchanged. Verified by mutation — reordering FP8 ahead of FA2 in
// select_attn_prefill_path() left all of them green.
//
// Neither could the sixteen AttnDispatchTable cases, for a different reason:
// every one that involves FP8 sets fmha_fa2="never" first, so FA2 and FP8 are
// never both live. The relative order of the chain — the thing a reorder breaks
// and the thing attention_dispatch.cu is now checked against at runtime — was
// not asserted anywhere.
//
// These pin it: both gates on, both kernels accepting, earlier tier must win.
TEST(AttnTierPrecedence, Mxfp4BeatsFA2) {
    auto cfg = default_cfg();
    auto sup = all_accept();
    sup.mxfp4_available = true;
    ASSERT_TRUE(sup.mxfp4_accepts && sup.fa2_accepts);
    EXPECT_EQ(select_attn_prefill_path(cfg, sup), AttnPrefillPath::MXFP4);
}

TEST(AttnTierPrecedence, FA2BeatsFP8WhenBothOnAndBothAccept) {
    auto cfg = default_cfg();
    cfg.attention.fmha_fa2 = "on";
    cfg.attention.fp8_fmha = "on";
    auto sup = all_accept();
    ASSERT_TRUE(sup.fa2_accepts && sup.fp8_accepts);
    EXPECT_EQ(select_attn_prefill_path(cfg, sup), AttnPrefillPath::FA2)
        << "FA2 must precede FP8: the raw-e4m3 FP8 tier compounds ~10% score error "
           "per layer (#511) and must never win a tie";
}

TEST(AttnTierPrecedence, FP8BeatsFmhaSm120WhenBothAccept) {
    auto cfg = default_cfg();
    cfg.attention.fmha_fa2 = "never";
    cfg.attention.fp8_fmha = "on";
    auto sup = all_accept();
    ASSERT_TRUE(sup.fp8_accepts && sup.fmha_sm120_accepts);
    EXPECT_EQ(select_attn_prefill_path(cfg, sup), AttnPrefillPath::FP8);
}

TEST(AttnTierPrecedence, FmhaSm120BeatsBlackwellWhenBothAccept) {
    auto cfg = default_cfg();
    cfg.attention.fmha_fa2 = "never";
    cfg.attention.fp8_fmha = "never";
    auto sup = all_accept();
    ASSERT_TRUE(sup.fmha_sm120_accepts && sup.blackwell_accepts);
    EXPECT_EQ(select_attn_prefill_path(cfg, sup), AttnPrefillPath::FMHA_SM120);
}

// --------------------------------------------------------------------------
// MoE model/dispatch COUPLING — the same treatment the attention half got
// --------------------------------------------------------------------------
//
// executor_forward_moe_cutlass.cu now replays select_moe_prefill_path() against
// what the chain observed and logs loudly on divergence
// (verify_against_moe_routing_model). These pin the replay so the check cannot
// fire on a *correct* dispatch.
//
// The observations mirror what the .cu records, which is what each tier DID —
// not what its preconditions promised. Device-args and smallM can both pass
// their gate and then fail inside, falling through to a later tier, so the
// dispatch sets the flag only after the tier has actually completed.

namespace {

MoePrefillWorkspace moe_observed_when(MoePrefillPath winner) {
    MoePrefillWorkspace ws;  // all false — the dispatch's starting state
    switch (winner) {
        case MoePrefillPath::DEVICE_ARGS:
            ws.grouped_available = true;
            ws.device_args_ready = true;
            break;
        case MoePrefillPath::SMALL_M:
            ws.grouped_available = true;
            ws.smallM_available = true;
            ws.smallM_under_threshold = true;
            break;
        case MoePrefillPath::GROUPED:
            ws.grouped_available = true;
            ws.grouped_ready = true;
            break;
        case MoePrefillPath::LEGACY:
            break;  // the six entry gates refused → grouped_available stays false
    }
    return ws;
}

}  // namespace

TEST(MoeRoutingModelCoupling, EveryTierReplaysToItself) {
    auto cfg = default_cfg();
    cfg.moe.nvfp4_smallM = true;  // so the smallM tier is reachable at all

    const MoePrefillPath tiers[] = {MoePrefillPath::DEVICE_ARGS, MoePrefillPath::SMALL_M,
                                    MoePrefillPath::GROUPED, MoePrefillPath::LEGACY};
    for (MoePrefillPath t : tiers) {
        EXPECT_EQ(select_moe_prefill_path(ModelArch::QWEN3_MOE, cfg, moe_observed_when(t)), t)
            << "the dispatch would run " << moe_prefill_path_name(t) << " but the model replays to "
            << moe_prefill_path_name(select_moe_prefill_path(ModelArch::QWEN3_MOE, cfg, moe_observed_when(t)))
            << " — verify_against_moe_routing_model() would fire on a correct dispatch";
    }
}

// gpt-oss enters the same chain but is arch-gated off the two fast tiers, so its
// only non-legacy answer is GROUPED. The dispatch records exactly that.
TEST(MoeRoutingModelCoupling, GptOssReplaysToGrouped) {
    auto cfg = default_cfg();
    cfg.moe.nvfp4_smallM = true;
    EXPECT_EQ(select_moe_prefill_path(ModelArch::GPT_OSS, cfg, moe_observed_when(MoePrefillPath::GROUPED)),
              MoePrefillPath::GROUPED);
}

// --------------------------------------------------------------------------
// MoE tier PRECEDENCE — same reasoning as AttnTierPrecedence above
// --------------------------------------------------------------------------
//
// The replay tests cannot catch a reorder: each observation pattern has exactly
// one eligible tier, so swapping two tiers in select_moe_prefill_path() leaves
// every answer unchanged. These make two tiers eligible at once and assert the
// earlier one wins.

TEST(MoeTierPrecedence, DeviceArgsBeatsSmallM) {
    auto cfg = default_cfg();
    cfg.moe.nvfp4_smallM = true;
    auto ws = all_ready();
    ASSERT_TRUE(cfg.moe.nvfp4_device_args && ws.device_args_ready);
    ASSERT_TRUE(ws.smallM_available && ws.smallM_under_threshold);
    EXPECT_EQ(select_moe_prefill_path(ModelArch::QWEN3_MOE, cfg, ws), MoePrefillPath::DEVICE_ARGS);
}

TEST(MoeTierPrecedence, SmallMBeatsGrouped) {
    auto cfg = default_cfg();
    cfg.moe.nvfp4_device_args = false;  // device-args out of the way
    cfg.moe.nvfp4_smallM = true;
    auto ws = all_ready();
    ASSERT_TRUE(ws.smallM_available && ws.smallM_under_threshold && ws.grouped_ready);
    EXPECT_EQ(select_moe_prefill_path(ModelArch::QWEN3_MOE, cfg, ws), MoePrefillPath::SMALL_M);
}

TEST(MoeTierPrecedence, GroupedBeatsLegacyWhenReady) {
    auto cfg = default_cfg();
    cfg.moe.nvfp4_device_args = false;
    auto ws = all_ready();
    ws.smallM_available = false;  // smallM out of the way
    ASSERT_TRUE(ws.grouped_available && ws.grouped_ready);
    EXPECT_EQ(select_moe_prefill_path(ModelArch::QWEN3_MOE, cfg, ws), MoePrefillPath::GROUPED);
}

// The entry gate short-circuits every CUTLASS tier, however ready they look.
TEST(MoeTierPrecedence, EntryGateBeatsEveryReadyTier) {
    auto cfg = default_cfg();
    cfg.moe.nvfp4_smallM = true;
    auto ws = all_ready();
    ws.grouped_available = false;  // the six early returns in the .cu
    EXPECT_EQ(select_moe_prefill_path(ModelArch::QWEN3_MOE, cfg, ws), MoePrefillPath::LEGACY);
}
