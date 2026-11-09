// speculative.mtp_k tri-state resolution (tools/common/mtp_auto.*): a checkpoint without an
// MTP head must not end up with n-gram off and nothing drafting; an operator-set ngram value
// must be kept. CPU-only, plain config in/out.

#include <gtest/gtest.h>
#include "common/mtp_auto.h"
#include "runtime/config.h"
#include "../tools/imp-cli/args.h"

namespace imp {
namespace {

constexpr int kSingleStream = 1;
constexpr int kServingAutoBatch = 0;

TEST(MtpAuto, DefaultIsTheAutoSentinel) {
    RuntimeConfig cfg;
    EXPECT_EQ(cfg.speculative.mtp_k, -1) << "auto is the shipped default";
    EXPECT_TRUE(cfg.speculative.ngram);
}

TEST(MtpAuto, SingleStreamWithHeadTakesThePair) {
    RuntimeConfig cfg;
    const int k = tools::mtp_auto_request_k(cfg, kSingleStream);
    EXPECT_EQ(k, tools::kMtpAutoK);
    tools::mtp_auto_finalize(cfg, k, /*head_loaded=*/true);
    EXPECT_EQ(cfg.speculative.mtp_k, tools::kMtpAutoK);
    EXPECT_FALSE(cfg.speculative.ngram) << "the measured recommendation is the pair";
}

TEST(MtpAuto, NoHeadFallsBackWithTheMatcherIntact) {
    RuntimeConfig cfg;
    const int k = tools::mtp_auto_request_k(cfg, kSingleStream);
    tools::mtp_auto_finalize(cfg, k, /*head_loaded=*/false);
    EXPECT_EQ(cfg.speculative.mtp_k, 0);
    EXPECT_TRUE(cfg.speculative.ngram) << "ngram off with nothing drafting is worse than the default";
}

// speculative.mtp_concurrent: a concurrent GDN-hybrid server takes the pair (batch > 1 never verifies there).
TEST(MtpAuto, ConcurrentHybridTakesThePair) {
    RuntimeConfig cfg;
    const int k = tools::mtp_auto_request_k(cfg, kServingAutoBatch);
    EXPECT_EQ(k, tools::kMtpAutoK) << "phase 1 asks for the head; phase 2 knows the model";
    const auto serving = tools::mtp_auto_serving(cfg, kServingAutoBatch, /*gdn_hybrid=*/true);
    EXPECT_EQ(serving, tools::MtpAutoServing::ConcurrentHybrid);
    tools::mtp_auto_finalize(cfg, k, /*head_loaded=*/true, /*head_auto_k_cap=*/0, serving);
    EXPECT_EQ(cfg.speculative.mtp_k, tools::kMtpAutoK);
    EXPECT_FALSE(cfg.speculative.ngram);
}

// A concurrent hybrid server keeps its KV pool: no NVFP4 draft head copy, one draft per bound row.
TEST(MtpAuto, ConcurrentHybridKeepsThePool) {
    RuntimeConfig cfg;
    const int k = tools::mtp_auto_request_k(cfg, kServingAutoBatch);
    tools::mtp_auto_finalize(cfg, k, /*head_loaded=*/true, 0, tools::MtpAutoServing::ConcurrentHybrid);
    EXPECT_FALSE(cfg.speculative.mtp_nvfp4_head);
    EXPECT_EQ(cfg.speculative.mtp_batch_verify_k, 1);
    EXPECT_EQ(cfg.speculative.mtp_draft_vocab, 98304) << "FP8 draft head scores the first 98304 rows";
}

TEST(MtpAuto, SingleStreamKeepsTheFullDraft) {
    RuntimeConfig cfg;
    tools::mtp_auto_finalize(cfg, tools::mtp_auto_request_k(cfg, kSingleStream), /*head_loaded=*/true);
    EXPECT_TRUE(cfg.speculative.mtp_nvfp4_head) << "1 chat at 50k: 138 vs 123 tok/s";
    EXPECT_EQ(cfg.speculative.mtp_batch_verify_k, 2);
    EXPECT_EQ(cfg.speculative.mtp_draft_vocab, 0);
}

TEST(MtpAuto, ConcurrentPoolDefaultsYieldToTheOperator) {
    RuntimeConfig cfg;
    ASSERT_TRUE(
        cfg.apply_overrides({"speculative.mtp_nvfp4_head=true", "speculative.mtp_batch_verify_k=2"}).empty());
    tools::mtp_auto_finalize(cfg, tools::mtp_auto_request_k(cfg, kServingAutoBatch), /*head_loaded=*/true, 0,
                             tools::MtpAutoServing::ConcurrentHybrid);
    EXPECT_TRUE(cfg.speculative.mtp_nvfp4_head);
    EXPECT_EQ(cfg.speculative.mtp_batch_verify_k, 2);
    EXPECT_EQ(cfg.speculative.mtp_draft_vocab, 0) << "the NVFP4 draft head scores every row";
}

TEST(MtpAuto, ExplicitDraftVocabSurvivesTheConcurrentDefaults) {
    RuntimeConfig cfg;
    ASSERT_TRUE(cfg.apply_overrides({"speculative.mtp_draft_vocab=0"}).empty());
    tools::mtp_auto_finalize(cfg, tools::mtp_auto_request_k(cfg, kServingAutoBatch), /*head_loaded=*/true, 0,
                             tools::MtpAutoServing::ConcurrentHybrid);
    EXPECT_FALSE(cfg.speculative.mtp_nvfp4_head);
    EXPECT_EQ(cfg.speculative.mtp_draft_vocab, 0);
}

TEST(MtpAuto, ExplicitDepthOnAConcurrentHybridKeepsThePoolToo) {
    RuntimeConfig cfg;
    ASSERT_TRUE(cfg.apply_overrides({"speculative.mtp_k=2"}).empty());
    tools::mtp_auto_finalize(cfg, 2, /*head_loaded=*/true, 0, tools::MtpAutoServing::ConcurrentHybrid);
    EXPECT_FALSE(cfg.speculative.mtp_nvfp4_head);
    EXPECT_EQ(cfg.speculative.mtp_batch_verify_k, 1);
}

TEST(MtpAuto, MtpOffLeavesThePoolKeysAlone) {
    RuntimeConfig cfg;
    ASSERT_TRUE(cfg.apply_overrides({"speculative.mtp_k=0"}).empty());
    tools::mtp_auto_finalize(cfg, 0, /*head_loaded=*/true, 0, tools::MtpAutoServing::ConcurrentHybrid);
    EXPECT_TRUE(cfg.speculative.mtp_nvfp4_head);
    EXPECT_EQ(cfg.speculative.mtp_batch_verify_k, 2);
    EXPECT_EQ(cfg.speculative.mtp_draft_vocab, 0);
}

// Off a GDN hybrid, batch > 1 round-robin verifies with the matcher: the pair would turn it off.
TEST(MtpAuto, ConcurrentNonHybridDeclines) {
    RuntimeConfig cfg;
    const int k = tools::mtp_auto_request_k(cfg, kServingAutoBatch);
    const auto serving = tools::mtp_auto_serving(cfg, kServingAutoBatch, /*gdn_hybrid=*/false);
    EXPECT_EQ(serving, tools::MtpAutoServing::ConcurrentOther);
    tools::mtp_auto_finalize(cfg, k, /*head_loaded=*/true, /*head_auto_k_cap=*/0, serving);
    EXPECT_EQ(cfg.speculative.mtp_k, 0);
    EXPECT_TRUE(cfg.speculative.ngram);
}

TEST(MtpAuto, ConcurrentOptOutRestoresTheDecline) {
    RuntimeConfig cfg;
    ASSERT_TRUE(cfg.apply_overrides({"speculative.mtp_concurrent=false"}).empty());
    EXPECT_EQ(tools::mtp_auto_request_k(cfg, kServingAutoBatch), 0) << "the head is not even loaded";
    EXPECT_EQ(tools::mtp_auto_serving(cfg, kServingAutoBatch, /*gdn_hybrid=*/true),
              tools::MtpAutoServing::ConcurrentOther);
    EXPECT_EQ(tools::mtp_auto_request_k(cfg, kSingleStream), tools::kMtpAutoK) << "single stream unchanged";
}

// A Qwen4Exp head declines on a concurrent hybrid exactly as on a single stream.
TEST(MtpAuto, ConcurrentHybridKeepsTheHeadDecline) {
    RuntimeConfig cfg;
    const int k = tools::mtp_auto_request_k(cfg, kServingAutoBatch);
    tools::mtp_auto_finalize(cfg, k, /*head_loaded=*/true, /*head_auto_k_cap=*/-1,
                             tools::MtpAutoServing::ConcurrentHybrid);
    EXPECT_EQ(cfg.speculative.mtp_k, 0);
}

TEST(MtpAuto, DeterminismOutranksTheSpeedup) {
    RuntimeConfig cfg;
    cfg.runtime.deterministic = true;
    EXPECT_EQ(tools::mtp_auto_request_k(cfg, kSingleStream), 0);
}

TEST(MtpAuto, ExplicitDepthIsUntouched) {
    RuntimeConfig cfg;
    ASSERT_TRUE(cfg.apply_overrides({"speculative.mtp_k=3"}).empty());
    EXPECT_EQ(tools::mtp_auto_request_k(cfg, kSingleStream), 3);
    tools::mtp_auto_finalize(cfg, 3, /*head_loaded=*/true);
    EXPECT_EQ(cfg.speculative.mtp_k, 3);
    EXPECT_TRUE(cfg.speculative.ngram) << "an explicit depth does not drag ngram with it";
}

TEST(MtpAuto, ExplicitOffStaysOff) {
    RuntimeConfig cfg;
    ASSERT_TRUE(cfg.apply_overrides({"speculative.mtp_k=0"}).empty());
    EXPECT_EQ(tools::mtp_auto_request_k(cfg, kSingleStream), 0);
    tools::mtp_auto_finalize(cfg, 0, /*head_loaded=*/true);
    EXPECT_EQ(cfg.speculative.mtp_k, 0);
    EXPECT_TRUE(cfg.speculative.ngram);
}

TEST(MtpAuto, ExplicitNgramSurvivesTheAutoPair) {
    RuntimeConfig cfg;
    ASSERT_TRUE(cfg.apply_overrides({"speculative.ngram=true"}).empty());
    const int k = tools::mtp_auto_request_k(cfg, kSingleStream);
    tools::mtp_auto_finalize(cfg, k, /*head_loaded=*/true);
    EXPECT_EQ(cfg.speculative.mtp_k, tools::kMtpAutoK);
    EXPECT_TRUE(cfg.speculative.ngram) << "auto must not overrule a key the operator set";
}

// A head with its own auto depth (qwen4_exp: 1) caps auto only; explicit settings and the tool flag pass.
TEST(MtpAuto, HeadAutoKCapClampsAutoOnly) {
    RuntimeConfig cfg;
    const int k = tools::mtp_auto_request_k(cfg, kSingleStream);
    ASSERT_GT(k, 1);
    tools::mtp_auto_finalize(cfg, k, /*head_loaded=*/true, /*head_auto_k_cap=*/1);
    EXPECT_EQ(cfg.speculative.mtp_k, 1);
    EXPECT_FALSE(cfg.speculative.ngram) << "the auto pair still turns the matcher off";

    RuntimeConfig pinned;
    ASSERT_TRUE(pinned.apply_overrides({"speculative.mtp_k=2"}).empty());
    tools::mtp_auto_finalize(pinned, 2, /*head_loaded=*/true, /*head_auto_k_cap=*/1);
    EXPECT_EQ(pinned.speculative.mtp_k, 2);

    RuntimeConfig flag;
    EXPECT_EQ(tools::mtp_auto_after_load(flag, 2, /*head_loaded=*/true, /*explicit_flag=*/2,
                                         /*head_auto_k_cap=*/1),
              2);
}

// A head whose verify costs more than it emits (qwen4_exp) turns auto off; explicit settings pass.
TEST(MtpAuto, HeadAutoDeclineTurnsAutoOff) {
    RuntimeConfig cfg;
    const int k = tools::mtp_auto_request_k(cfg, kSingleStream);
    ASSERT_GT(k, 0);
    tools::mtp_auto_finalize(cfg, k, /*head_loaded=*/true, /*head_auto_k_cap=*/-1);
    EXPECT_EQ(cfg.speculative.mtp_k, 0);

    RuntimeConfig pinned;
    ASSERT_TRUE(pinned.apply_overrides({"speculative.mtp_k=1"}).empty());
    tools::mtp_auto_finalize(pinned, 1, /*head_loaded=*/true, /*head_auto_k_cap=*/-1);
    EXPECT_EQ(pinned.speculative.mtp_k, 1);
}

// The gated bench measures RAW decode: auto drafting with an MTP head would
// redefine what tests/perf_baseline.json pins, silently.
TEST(MtpAuto, BenchModePinsAutoOff) {
    RuntimeConfig cfg;
    CliArgs args;
    args.bench = true;
    apply_config_pins(cfg, args);
    EXPECT_EQ(cfg.speculative.mtp_k, 0);
    EXPECT_EQ(tools::mtp_auto_request_k(cfg, kSingleStream), 0);
}

TEST(MtpAuto, BenchModeKeepsAnExplicitDepth) {
    RuntimeConfig cfg;
    CliArgs args;
    args.bench = true;
    args.config_overrides = {"speculative.mtp_k=2"};
    ASSERT_TRUE(cfg.apply_overrides(args.config_overrides).empty());
    apply_config_pins(cfg, args);
    EXPECT_EQ(cfg.speculative.mtp_k, 2) << "an explicit --set outranks the bench pin";
}

TEST(MtpAuto, NonBenchRunKeepsAuto) {
    RuntimeConfig cfg;
    CliArgs args;
    apply_config_pins(cfg, args);
    EXPECT_EQ(cfg.speculative.mtp_k, -1) << "only --bench pins the sentinel away";
}

TEST(ConfigExplicitKeys, RecordsWhatTheOperatorSet) {
    RuntimeConfig cfg;
    EXPECT_FALSE(cfg.was_set("speculative.ngram"));
    ASSERT_TRUE(cfg.apply_overrides({"speculative.ngram=false"}).empty());
    EXPECT_TRUE(cfg.was_set("speculative.ngram"));
    EXPECT_FALSE(cfg.was_set("speculative.mtp_k")) << "an untouched key must not report as set";
}

}  // namespace
}  // namespace imp
