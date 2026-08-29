// speculative.mtp_k tri-state resolution (tools/common/mtp_auto.*).
//
// The rule decides a DEFAULT behaviour change on every checkpoint that ships
// an MTP head, and two of its branches are failure-shaped: a checkpoint
// without a head must not end up with the n-gram matcher switched off and
// nothing drafting, and an operator who set `ngram` themselves must keep it.
// CPU-only: plain config in, plain config out.

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

TEST(MtpAuto, ConcurrentServingDeclines) {
    RuntimeConfig cfg;
    const int k = tools::mtp_auto_request_k(cfg, kServingAutoBatch);
    EXPECT_EQ(k, 0) << "the head binds one request and costs every slot's budget";
    tools::mtp_auto_finalize(cfg, k, /*head_loaded=*/false);
    EXPECT_EQ(cfg.speculative.mtp_k, 0);
    EXPECT_TRUE(cfg.speculative.ngram);
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
