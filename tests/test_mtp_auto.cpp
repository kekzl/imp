// speculative.mtp_k tri-state resolution (tools/common/mtp_auto.*): a checkpoint without an
// MTP head must not end up with n-gram off and nothing drafting; an operator-set ngram value
// must be kept. CPU-only, plain config in/out.

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

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
    const int k = tools::mtp_auto_request_k(cfg);
    EXPECT_EQ(k, tools::kMtpAutoK);
    tools::mtp_auto_finalize(cfg, k, /*head_loaded=*/true);
    EXPECT_EQ(cfg.speculative.mtp_k, tools::kMtpAutoK);
    EXPECT_FALSE(cfg.speculative.ngram) << "the measured recommendation is the pair";
}

TEST(MtpAuto, NoHeadFallsBackWithTheMatcherIntact) {
    RuntimeConfig cfg;
    const int k = tools::mtp_auto_request_k(cfg);
    tools::mtp_auto_finalize(cfg, k, /*head_loaded=*/false);
    EXPECT_EQ(cfg.speculative.mtp_k, 0);
    EXPECT_TRUE(cfg.speculative.ngram) << "ngram off with nothing drafting is worse than the default";
}

// A concurrent GDN-hybrid server takes the pair (batch > 1 never verifies there).
TEST(MtpAuto, ConcurrentHybridTakesThePair) {
    RuntimeConfig cfg;
    const int k = tools::mtp_auto_request_k(cfg);
    EXPECT_EQ(k, tools::kMtpAutoK) << "phase 1 asks for the head; phase 2 knows the model";
    const auto serving = tools::mtp_auto_serving(cfg, kServingAutoBatch, /*gdn_hybrid=*/true);
    EXPECT_EQ(serving, tools::MtpAutoServing::ConcurrentHybrid);
    tools::mtp_auto_finalize(cfg, k, /*head_loaded=*/true, /*head_auto_k_cap=*/0, serving);
    EXPECT_EQ(cfg.speculative.mtp_k, 7) << "the bound verify stages up to 7 drafts per request (streak)";
    EXPECT_FALSE(cfg.speculative.ngram);
}

// A concurrent hybrid server keeps its KV pool: the NVFP4 draft head copy holds rows [0, 98304) only
// (270 MiB of 682 MiB on Qwen3.8-27B); drafts 1 and 2 leave as factor rows.
TEST(MtpAuto, ConcurrentHybridKeepsThePool) {
    RuntimeConfig cfg;
    const int k = tools::mtp_auto_request_k(cfg);
    tools::mtp_auto_finalize(cfg, k, /*head_loaded=*/true, 0, tools::MtpAutoServing::ConcurrentHybrid);
    EXPECT_TRUE(cfg.speculative.mtp_nvfp4_head) << "1 chat: 204.26 vs 198.04 tok/s on the FP8 head";
    EXPECT_EQ(cfg.speculative.mtp_batch_verify_k, 4);
    EXPECT_EQ(cfg.speculative.mtp_draft_vocab, 98304) << "drafts score the first 98304 rows";
}

TEST(MtpAuto, SingleStreamKeepsTheFullDraft) {
    RuntimeConfig cfg;
    tools::mtp_auto_finalize(cfg, tools::mtp_auto_request_k(cfg), /*head_loaded=*/true);
    EXPECT_TRUE(cfg.speculative.mtp_nvfp4_head) << "1 chat at 50k: 138 vs 123 tok/s";
    EXPECT_EQ(cfg.speculative.mtp_batch_verify_k, 2);
    EXPECT_EQ(cfg.speculative.mtp_draft_vocab, 0);
}

TEST(MtpAuto, ConcurrentPoolDefaultsYieldToTheOperator) {
    RuntimeConfig cfg;
    ASSERT_TRUE(
        cfg.apply_overrides({"speculative.mtp_nvfp4_head=true", "speculative.mtp_batch_verify_k=2"}).empty());
    tools::mtp_auto_finalize(cfg, tools::mtp_auto_request_k(cfg), /*head_loaded=*/true, 0,
                             tools::MtpAutoServing::ConcurrentHybrid);
    EXPECT_TRUE(cfg.speculative.mtp_nvfp4_head);
    EXPECT_EQ(cfg.speculative.mtp_batch_verify_k, 2);
    EXPECT_EQ(cfg.speculative.mtp_draft_vocab, 98304) << "an unset prefix follows the server default";
}

TEST(MtpAuto, FullDraftVocabTurnsTheHeadCopyOff) {
    RuntimeConfig cfg;
    ASSERT_TRUE(cfg.apply_overrides({"speculative.mtp_draft_vocab=0"}).empty());
    tools::mtp_auto_finalize(cfg, tools::mtp_auto_request_k(cfg), /*head_loaded=*/true, 0,
                             tools::MtpAutoServing::ConcurrentHybrid);
    EXPECT_FALSE(cfg.speculative.mtp_nvfp4_head) << "the 682 MiB full copy is KV pool: 1990 blocks";
    EXPECT_EQ(cfg.speculative.mtp_draft_vocab, 0);
}

TEST(MtpAuto, HeadOffKeepsTheFp8PrefixDraft) {
    RuntimeConfig cfg;
    ASSERT_TRUE(cfg.apply_overrides({"speculative.mtp_nvfp4_head=false"}).empty());
    tools::mtp_auto_finalize(cfg, tools::mtp_auto_request_k(cfg), /*head_loaded=*/true, 0,
                             tools::MtpAutoServing::ConcurrentHybrid);
    EXPECT_FALSE(cfg.speculative.mtp_nvfp4_head);
    EXPECT_EQ(cfg.speculative.mtp_draft_vocab, 98304) << "the FP8 head scores the first 98304 rows";
}

TEST(MtpAuto, HostSpillKeepsBothDraftsAndThePrefixCopy) {
    RuntimeConfig cfg;
    ASSERT_TRUE(cfg.apply_overrides({"kv_cache.host_spill_mb=16384"}).empty());
    tools::mtp_auto_finalize(cfg, tools::mtp_auto_request_k(cfg), /*head_loaded=*/true, 0,
                             tools::MtpAutoServing::ConcurrentHybrid);
    EXPECT_GT(cfg.speculative.mtp_k, 0);
    EXPECT_TRUE(cfg.speculative.mtp_nvfp4_head);
    EXPECT_EQ(cfg.speculative.mtp_batch_verify_k, 4) << "spill restores evicted prefixes";
    EXPECT_EQ(cfg.speculative.mtp_draft_vocab, 98304);
}

TEST(MtpAuto, ExplicitDepthOnAConcurrentHybridKeepsThePoolToo) {
    RuntimeConfig cfg;
    ASSERT_TRUE(cfg.apply_overrides({"speculative.mtp_k=2"}).empty());
    tools::mtp_auto_finalize(cfg, 2, /*head_loaded=*/true, 0, tools::MtpAutoServing::ConcurrentHybrid);
    EXPECT_TRUE(cfg.speculative.mtp_nvfp4_head);
    EXPECT_EQ(cfg.speculative.mtp_draft_vocab, 98304);
    EXPECT_EQ(cfg.speculative.mtp_batch_verify_k, 2);
}

// Without the factor row a second draft is a second spare slot per row (79.5 MiB each on Qwen3.8-27B).
TEST(MtpAuto, ConcurrentHybridWithoutTheFactorRowVerifiesOneDraft) {
    RuntimeConfig cfg;
    ASSERT_TRUE(cfg.apply_overrides({"speculative.mtp_batch_verify_factor_row=false"}).empty());
    tools::mtp_auto_finalize(cfg, tools::mtp_auto_request_k(cfg), /*head_loaded=*/true, 0,
                             tools::MtpAutoServing::ConcurrentHybrid);
    EXPECT_EQ(cfg.speculative.mtp_batch_verify_k, 1);
    RuntimeConfig set;
    ASSERT_TRUE(set.apply_overrides(
                       {"speculative.mtp_batch_verify_factor_row=false", "speculative.mtp_batch_verify_k=2"})
                    .empty());
    tools::mtp_auto_finalize(set, tools::mtp_auto_request_k(set), /*head_loaded=*/true, 0,
                             tools::MtpAutoServing::ConcurrentHybrid);
    EXPECT_EQ(set.speculative.mtp_batch_verify_k, 2) << "the operator set it";
    EXPECT_EQ(cfg.speculative.mtp_k, 2) << "no factor rows: the step stages 2 drafts at most";
}

// mtp_k=auto on a concurrent hybrid drafts what the bound verify stages; unset, the verify depth follows a
// smaller mtp_k (a row past the drafted depth is a pad).
TEST(MtpAuto, AutoDepthAndVerifyDepthFollowEachOther) {
    const auto resolve = [](std::vector<std::string> sets) {
        RuntimeConfig cfg;
        EXPECT_TRUE(cfg.apply_overrides(sets).empty());
        tools::mtp_auto_finalize(cfg, tools::mtp_auto_request_k(cfg), /*head_loaded=*/true, 0,
                                 tools::MtpAutoServing::ConcurrentHybrid);
        return std::pair{cfg.speculative.mtp_k, cfg.speculative.mtp_batch_verify_k};
    };
    EXPECT_EQ(resolve({}), (std::pair{7, 4})) << "the streak key reaches 7 drafts";
    EXPECT_EQ(resolve({"speculative.mtp_batch_verify_k_streak=0"}), (std::pair{4, 4}));
    EXPECT_EQ(resolve({"speculative.mtp_batch_verify_k=2"}), (std::pair{7, 2}))
        << "the streak starts at the key";
    EXPECT_EQ(resolve({"speculative.mtp_batch_verify_k=2", "speculative.mtp_batch_verify_k_streak=0"}),
              (std::pair{2, 2}))
        << "auto follows the key";
    EXPECT_EQ(resolve({"speculative.mtp_batch_verify_k=1"}), (std::pair{2, 1})) << "auto keeps kMtpAutoK";
    EXPECT_EQ(resolve({"speculative.mtp_batch_verify=0"}), (std::pair{2, 2})) << "no bound verify";
    EXPECT_EQ(resolve({"speculative.mtp_history_slots=1"}), (std::pair{2, 2})) << "one binding";
    EXPECT_EQ(resolve({"speculative.hybrid=false"}), (std::pair{2, 2})) << "no grouped verify scan";
    EXPECT_EQ(resolve({"speculative.mtp_k=1"}), (std::pair{1, 1})) << "2 rows per request, no pad";
    EXPECT_EQ(resolve({"speculative.mtp_k=2"}), (std::pair{2, 2}));
    EXPECT_EQ(resolve({"speculative.mtp_k=3"}), (std::pair{3, 3}));
    EXPECT_EQ(resolve({"speculative.mtp_k=6"}), (std::pair{6, 4}));
    EXPECT_EQ(resolve({"speculative.mtp_k=1", "speculative.mtp_batch_verify_k=3"}), (std::pair{1, 3}))
        << "the operator set it";
    EXPECT_EQ(resolve({"speculative.mtp_batch_verify_factor_row=false", "kv_cache.host_spill_mb=16384"}),
              (std::pair{2, 2}))
        << "a second spare per row holds 2 drafts";
    RuntimeConfig cap;
    tools::mtp_auto_finalize(cap, tools::mtp_auto_request_k(cap), /*head_loaded=*/true,
                             /*head_auto_k_cap=*/1, tools::MtpAutoServing::ConcurrentHybrid);
    EXPECT_EQ(cap.speculative.mtp_k, 1) << "the head's own cap outranks the verify depth";
    EXPECT_EQ(cap.speculative.mtp_batch_verify_k, 1);
}

TEST(MtpAuto, MtpOffLeavesThePoolKeysAlone) {
    RuntimeConfig cfg;
    ASSERT_TRUE(cfg.apply_overrides({"speculative.mtp_k=0"}).empty());
    tools::mtp_auto_finalize(cfg, 0, /*head_loaded=*/true, 0, tools::MtpAutoServing::ConcurrentHybrid);
    EXPECT_TRUE(cfg.speculative.mtp_nvfp4_head);
    EXPECT_EQ(cfg.speculative.mtp_batch_verify_k, 4);
    EXPECT_EQ(cfg.speculative.mtp_draft_vocab, 0);
}

// Off a GDN hybrid, batch > 1 round-robin verifies with the matcher: the pair would turn it off.
TEST(MtpAuto, ConcurrentNonHybridDeclines) {
    RuntimeConfig cfg;
    const int k = tools::mtp_auto_request_k(cfg);
    const auto serving = tools::mtp_auto_serving(cfg, kServingAutoBatch, /*gdn_hybrid=*/false);
    EXPECT_EQ(serving, tools::MtpAutoServing::ConcurrentOther);
    tools::mtp_auto_finalize(cfg, k, /*head_loaded=*/true, /*head_auto_k_cap=*/0, serving);
    EXPECT_EQ(cfg.speculative.mtp_k, 0);
    EXPECT_TRUE(cfg.speculative.ngram);
}


// A one-chat GDN hybrid verifies on the bound path: engine batch 2, admission and decode stay 1.
TEST(MtpAuto, OneChatHybridTakesTheBoundVerifyBatch) {
    RuntimeConfig cfg;
    const int k = tools::mtp_auto_request_k(cfg);
    const int batch = tools::mtp_auto_engine_batch(cfg, kSingleStream, /*gdn_hybrid=*/true, k,
                                                   /*head_loaded=*/true);
    EXPECT_EQ(batch, 2) << "the bound verify stages rows from engine batch 2 up";
    EXPECT_EQ(cfg.runtime.max_batch_size, 1) << "admission and decode stay at one chat";
    const auto serving = tools::mtp_auto_serving(cfg, batch, /*gdn_hybrid=*/true);
    EXPECT_EQ(serving, tools::MtpAutoServing::ConcurrentHybrid);
    tools::mtp_auto_finalize(cfg, k, /*head_loaded=*/true, /*head_auto_k_cap=*/0, serving);
    EXPECT_EQ(cfg.speculative.mtp_k, 7) << "agent replay 1 chat: 205.58 vs 69.28 tok/s on the chain";
    EXPECT_EQ(cfg.speculative.mtp_draft_vocab, 98304);
}

// imp-cli resolves through mtp_auto_after_load: a chat run takes the batch, depth and pool of the lane.
TEST(MtpAuto, AfterLoadTakesTheServingOfTheEngineBatch) {
    RuntimeConfig cfg;
    const int k = tools::mtp_auto_request_k(cfg);
    const int batch = tools::mtp_auto_engine_batch(cfg, kSingleStream, /*gdn_hybrid=*/true, k,
                                                   /*head_loaded=*/true);
    EXPECT_EQ(tools::mtp_auto_after_load(cfg, k, /*head_loaded=*/true, /*explicit_flag=*/0,
                                         /*head_auto_k_cap=*/0,
                                         tools::mtp_auto_serving(cfg, batch, /*gdn_hybrid=*/true)),
              7);
    const RuntimeConfig pending = take_pending_runtime_config();
    EXPECT_EQ(pending.speculative.mtp_k, 7) << "the engine takes the resolved config";
    EXPECT_EQ(pending.speculative.mtp_draft_vocab, 98304);
    EXPECT_EQ(pending.runtime.max_batch_size, 1);

    RuntimeConfig chain;
    EXPECT_EQ(tools::mtp_auto_after_load(chain, k, /*head_loaded=*/true, /*explicit_flag=*/0),
              tools::kMtpAutoK)
        << "the default serving stays the single-stream chain";
    EXPECT_EQ(take_pending_runtime_config().speculative.mtp_draft_vocab, 0);
}

// A reload (resume, idle unload, swap) resolves auto again: plain probe 179.66 -> 133.63 tok/s without.
TEST(MtpAuto, ReloadResolvesAgainstTheStartupConfig) {
    RuntimeConfig cfg;
    std::optional<RuntimeConfig> startup;
    auto load = [&](bool gdn_hybrid, bool head_loaded) {
        tools::mtp_auto_restore_startup(cfg, startup);
        EXPECT_EQ(cfg.speculative.mtp_k, -1) << "auto again, not the previous load's depth";
        const int k = tools::mtp_auto_request_k(cfg);
        const int batch = tools::mtp_auto_engine_batch(cfg, kSingleStream, gdn_hybrid, k, head_loaded);
        tools::mtp_auto_finalize(cfg, k, head_loaded, /*head_auto_k_cap=*/0,
                                 tools::mtp_auto_serving(cfg, batch, gdn_hybrid));
        return batch;
    };
    EXPECT_EQ(load(/*gdn_hybrid=*/true, /*head_loaded=*/true), 2);
    EXPECT_EQ(cfg.speculative.mtp_k, 7);
    EXPECT_EQ(load(true, true), 2) << "the resumed one-chat server keeps the bound verify";
    EXPECT_EQ(cfg.speculative.mtp_k, 7);
    EXPECT_EQ(cfg.runtime.max_batch_size, 1);
    EXPECT_EQ(load(false, false), 1) << "a head-less model";
    EXPECT_EQ(cfg.speculative.mtp_k, 0);
    EXPECT_EQ(load(true, true), 2) << "swapping back drafts again";
    EXPECT_EQ(cfg.speculative.mtp_k, 7);
    EXPECT_FALSE(cfg.speculative.ngram);
}

// Everything else keeps its batch and its runtime.max_batch_size.
TEST(MtpAuto, BoundVerifyBatchIsForAOneChatAutoHybridOnly) {
    auto batch = [](const std::vector<std::string>& ov, int configured, bool gdn_hybrid, bool head_loaded,
                    int head_auto_k_cap) {
        RuntimeConfig cfg;
        EXPECT_TRUE(cfg.apply_overrides(ov).empty());
        const int before = cfg.runtime.max_batch_size;
        const int k = tools::mtp_auto_request_k(cfg);
        const int b = tools::mtp_auto_engine_batch(cfg, configured, gdn_hybrid, k, head_loaded,
                                                   head_auto_k_cap);
        if (b == configured)
            EXPECT_EQ(cfg.runtime.max_batch_size, before) << "a decline writes nothing";
        return b;
    };
    EXPECT_EQ(batch({}, kSingleStream, false, true, 0), 1) << "not a GDN hybrid";
    EXPECT_EQ(batch({}, kSingleStream, true, false, 0), 1) << "no head";
    EXPECT_EQ(batch({}, kSingleStream, true, true, -1), 1) << "the head declines auto";
    EXPECT_EQ(batch({"speculative.mtp_k=2"}, kSingleStream, true, true, 0), 1)
        << "explicit depth keeps the chain";
    EXPECT_EQ(batch({"speculative.mtp_k=0"}, kSingleStream, true, true, 0), 1);
    EXPECT_EQ(batch({"runtime.deterministic=true"}, kSingleStream, true, true, 0), 1);
    EXPECT_EQ(batch({"speculative.mtp_batch_verify=0"}, kSingleStream, true, true, 0), 1)
        << "no bound verify";
    EXPECT_EQ(batch({"speculative.batch_verify=true"}, kSingleStream, true, true, 0), 1);
    EXPECT_EQ(batch({}, kServingAutoBatch, true, true, 0), 0) << "an auto batch stays auto";
    EXPECT_EQ(batch({}, 4, true, true, 0), 4);
    EXPECT_EQ(batch({"runtime.max_batch_size=1"}, kSingleStream, true, true, 0), 2)
        << "the imp.conf key counts";
}

// A Qwen4Exp head declines on a concurrent hybrid exactly as on a single stream.
TEST(MtpAuto, ConcurrentHybridKeepsTheHeadDecline) {
    RuntimeConfig cfg;
    const int k = tools::mtp_auto_request_k(cfg);
    tools::mtp_auto_finalize(cfg, k, /*head_loaded=*/true, /*head_auto_k_cap=*/-1,
                             tools::MtpAutoServing::ConcurrentHybrid);
    EXPECT_EQ(cfg.speculative.mtp_k, 0);
}

TEST(MtpAuto, DeterminismOutranksTheSpeedup) {
    RuntimeConfig cfg;
    cfg.runtime.deterministic = true;
    EXPECT_EQ(tools::mtp_auto_request_k(cfg), 0);
}

TEST(MtpAuto, ExplicitDepthIsUntouched) {
    RuntimeConfig cfg;
    ASSERT_TRUE(cfg.apply_overrides({"speculative.mtp_k=3"}).empty());
    EXPECT_EQ(tools::mtp_auto_request_k(cfg), 3);
    tools::mtp_auto_finalize(cfg, 3, /*head_loaded=*/true);
    EXPECT_EQ(cfg.speculative.mtp_k, 3);
    EXPECT_TRUE(cfg.speculative.ngram) << "an explicit depth does not drag ngram with it";
}

TEST(MtpAuto, ExplicitOffStaysOff) {
    RuntimeConfig cfg;
    ASSERT_TRUE(cfg.apply_overrides({"speculative.mtp_k=0"}).empty());
    EXPECT_EQ(tools::mtp_auto_request_k(cfg), 0);
    tools::mtp_auto_finalize(cfg, 0, /*head_loaded=*/true);
    EXPECT_EQ(cfg.speculative.mtp_k, 0);
    EXPECT_TRUE(cfg.speculative.ngram);
}

TEST(MtpAuto, ExplicitNgramSurvivesTheAutoPair) {
    RuntimeConfig cfg;
    ASSERT_TRUE(cfg.apply_overrides({"speculative.ngram=true"}).empty());
    const int k = tools::mtp_auto_request_k(cfg);
    tools::mtp_auto_finalize(cfg, k, /*head_loaded=*/true);
    EXPECT_EQ(cfg.speculative.mtp_k, tools::kMtpAutoK);
    EXPECT_TRUE(cfg.speculative.ngram) << "auto must not overrule a key the operator set";
}

// A head with its own auto depth (qwen4_exp: 1) caps auto only; explicit settings and the tool flag pass.
TEST(MtpAuto, HeadAutoKCapClampsAutoOnly) {
    RuntimeConfig cfg;
    const int k = tools::mtp_auto_request_k(cfg);
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
    const int k = tools::mtp_auto_request_k(cfg);
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
    EXPECT_EQ(tools::mtp_auto_request_k(cfg), 0);
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
