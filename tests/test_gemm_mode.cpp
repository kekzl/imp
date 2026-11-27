// GemmMode (audit D-7): the 4 per-forward GEMM flags travel in every GemmContext instead of
// executor members. CPU lane: the structure, the decode-row rule and the copies, no kernel launch.

#include <gtest/gtest.h>

#include "exec/gemm_context.h"
#include "exec/gemm_mode.h"
#include "exec/quant_scratch.h"
#include "exec/weight_caches.h"

namespace {

using imp::GemmContext;
using imp::GemmMode;

TEST(GemmMode, HoldsTheFourForwardFlagsAllOffByDefault) {
    const GemmMode m{};
    const auto& [force_fp16, spec_verify, verify_smallm, decode_rows] = m;  // a 5th field fails to compile
    EXPECT_FALSE(force_fp16);
    EXPECT_FALSE(spec_verify);
    EXPECT_FALSE(verify_smallm);
    EXPECT_FALSE(decode_rows);
    EXPECT_FALSE(GemmContext{}.mode.decode_rows);
}

TEST(GemmMode, DecodeRowsAreGeneratedRowsOnly) {
    using imp::gemm_decode_rows;
    // One row is the M=1 decode, never a decode-row GEMM.
    EXPECT_FALSE(gemm_decode_rows(1, false, false));
    EXPECT_FALSE(gemm_decode_rows(1, true, true));
    // Batched decode.
    EXPECT_TRUE(gemm_decode_rows(8, false, false));
    // Prompt rows.
    EXPECT_FALSE(gemm_decode_rows(8, true, false));
    // A verify chunk forwards with is_prefill=true; its rows are draft tokens.
    EXPECT_TRUE(gemm_decode_rows(5, true, true));
}

TEST(GemmMode, MakeCarriesTheModeAndGatesVerifySmallMPerSite) {
    imp::WeightCaches wc;
    imp::QuantScratch qs;
    const GemmMode m{.force_fp16 = true, .spec_verify = true, .verify_smallm = true, .decode_rows = true};

    const GemmContext site = GemmContext::make(nullptr, wc, qs, m, /*force_mmvq=*/true);
    EXPECT_TRUE(site.mode.force_fp16);
    EXPECT_TRUE(site.mode.spec_verify);
    EXPECT_TRUE(site.mode.verify_smallm);
    EXPECT_TRUE(site.mode.decode_rows);
    EXPECT_TRUE(site.force_mmvq);
    EXPECT_TRUE(site.spec_verify_small_m);

    // The LM head and the MoE shared expert keep the mode but stay off the verify small-M routes.
    const GemmContext head = GemmContext::make(nullptr, wc, qs, m, false, /*verify_small_m_routes=*/false);
    EXPECT_TRUE(head.mode.spec_verify);
    EXPECT_FALSE(head.spec_verify_small_m);

    // No verify chunk: no site takes the verify routes.
    EXPECT_FALSE(GemmContext::make(nullptr, wc, qs, GemmMode{.decode_rows = true}).spec_verify_small_m);
}

// The decision helpers the executor's non-GEMM sites call with their caller's ctx.mode (audit D-7
// slice 2): each answer follows the mode argument, nothing else.
TEST(GemmMode, FusionRowRulesFollowTheModeArgument) {
    using imp::fp32_resid_mode_rows;
    using imp::fp8_smallm_mode_rows;
    using imp::m1_fusion_mode_ok;
    const GemmMode prompt{};
    const GemmMode decode{.decode_rows = true};
    const GemmMode verify{.spec_verify = true, .decode_rows = true};

    // FP8 x FP8 small-M: decode / verify rows, prompt rows only under the M=1 floor.
    EXPECT_FALSE(fp8_smallm_mode_rows(prompt, 2));
    EXPECT_TRUE(fp8_smallm_mode_rows(decode, 2));
    EXPECT_TRUE(fp8_smallm_mode_rows(verify, 2));
    EXPECT_TRUE(fp8_smallm_mode_rows(prompt, 1));

    // FP32-residual fusion: one row always, more rows only as decode / verify rows.
    EXPECT_TRUE(fp32_resid_mode_rows(prompt, 1));
    EXPECT_FALSE(fp32_resid_mode_rows(prompt, 8));
    EXPECT_TRUE(fp32_resid_mode_rows(decode, 8));
    EXPECT_FALSE(fp32_resid_mode_rows(decode, 0));

    // M=1 launch fusions: off in a verify chunk only.
    EXPECT_TRUE(m1_fusion_mode_ok(prompt));
    EXPECT_TRUE(m1_fusion_mode_ok(decode));
    EXPECT_FALSE(m1_fusion_mode_ok(verify));
}

TEST(GemmMode, DerivedContextsKeepTheMode) {
    imp::WeightCaches wc;
    imp::QuantScratch qs;
    const GemmContext ctx = GemmContext::make(nullptr, wc, qs,
                                              GemmMode{.spec_verify = true, .decode_rows = true});
    const int tag = 0;
    for (const GemmContext& c : {ctx.with_beta(1.0f), ctx.with_act_quant_hint(&tag, 4, 64)}) {
        EXPECT_TRUE(c.mode.spec_verify);
        EXPECT_TRUE(c.mode.decode_rows);
        EXPECT_FALSE(c.mode.force_fp16);
        EXPECT_TRUE(c.spec_verify_small_m);
    }
}

}  // namespace
