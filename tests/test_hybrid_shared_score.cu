// /v1/decide shared mode on a hybrid (#2198): score rows restoring ONE shared-prefix recurrent
// snapshot in one ragged prefill forward. Rows must be independent of their neighbours and of
// their row position (bit-equal logits). Ragged vs solo is a rounding class: printed, not asserted.
// Needs IMP_TEST_MODEL_GDN; runs from make test-e2e.

#include <gtest/gtest.h>

#include "api/imp_internal.h"
#include "imp/imp.h"
#include "model/model.h"
#include "runtime/config.h"
#include "runtime/engine.h"
#include "runtime/request.h"
#include "runtime/snapshot_boundary.h"
#include "test_models.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace imp {
namespace {

constexpr int kItems = 8;  // wave 2 rows; item 0 runs alone first and publishes the prefix

constexpr const char* kEvidenceLine =
    "Record %d: the parcel left the depot on day %d, weighed %d kilograms and was signed for by clerk %d.\n";

constexpr const char* kSuffixes[kItems + 1] = {
    "Question: on which day did record 3 leave? Answer:",
    "Question: how heavy was record 7? Answer:",
    "Question: which clerk signed record 11, and was the parcel late? Answer:",
    "Question: did record 2 weigh more than record 5? Answer:",
    "Question: list the records that left on the same day as record 9, if any. Answer:",
    "Question: which record was the lightest? Answer:",
    "Question: who signed record 0? Answer:",
    "Question: is there a record signed by clerk 4 that weighed under ten kilograms? Answer:",
    "Question: which day saw the most departures in the whole list of records above? Answer:",
};

class HybridSharedScoreTest : public ::testing::Test {
protected:
    void SetUp() override {
        const std::string path = imp_test::env_path(imp_test::kEnvModelGdn);
        if (path.empty())
            GTEST_SKIP() << "Set " << imp_test::kEnvModelGdn << " to run the hybrid shared-score test";
        ASSERT_NO_FATAL_FAILURE(imp_test::require_readable(path.c_str(), imp_test::kEnvModelGdn));
        path_ = path;
        ASSERT_EQ(load(), IMP_SUCCESS);
    }
    ImpError load() {
        const bool gguf = path_.size() >= 5 && path_.substr(path_.size() - 5) == ".gguf";
        return imp_model_load(path_.c_str(), gguf ? IMP_FORMAT_GGUF : IMP_FORMAT_SAFETENSORS, &model_);
    }
    void TearDown() override {
        if (ctx_)
            imp_context_free(ctx_);
        if (model_)
            imp_model_free(model_);
    }

    std::vector<int32_t> tokenize(const std::string& s) {
        std::vector<int32_t> toks(8192);
        int n = 0;
        EXPECT_EQ(imp_tokenize(model_, s.c_str(), toks.data(), &n, static_cast<int>(toks.size())),
                  IMP_SUCCESS);
        toks.resize(n > 0 ? static_cast<size_t>(n) : 0);
        return toks;
    }

    // Fresh engine per arm: a finished row publishes its own prompt, which would change what a
    // later arm restores. An engine consumes its model's source tensors: reload per engine.
    Engine* fresh_engine() {
        if (ctx_) {
            imp_context_free(ctx_);
            ctx_ = nullptr;
            imp_model_free(model_);
            model_ = nullptr;
            EXPECT_EQ(load(), IMP_SUCCESS);
        }
        RuntimeConfig rc;
        rc.runtime.deterministic = true;
        set_pending_runtime_config(rc);
        ImpConfig cfg = imp_config_default();
        cfg.max_seq_len = 4096;
        cfg.max_batch_size = kItems + 1;
        cfg.enable_cuda_graphs = 1;
        cfg.use_prefix_caching = 1;
        EXPECT_EQ(imp_context_create(model_, &cfg, &ctx_), IMP_SUCCESS);
        return ctx_ ? ctx_->engine.get() : nullptr;
    }

    std::string path_;
    ImpModel model_ = nullptr;
    ImpContext ctx_ = nullptr;
};

struct Row {
    std::vector<float> logits;
    int cached = -1;
};

std::shared_ptr<Request> score_req(const std::vector<int32_t>& toks, int hint,
                                   const std::vector<int32_t>& ids) {
    auto r = std::make_shared<Request>();
    r->input_tokens = toks;
    r->max_tokens = 1;
    r->temperature = 0.0f;
    r->score_token_ids = ids;
    r->snapshot_hint_tokens = hint;
    r->status = RequestStatus::PENDING;
    return r;
}

// Adds every prompt before the first step (one wave), steps until all finished. Priority = index:
// the scheduler sorts by priority before length, so the batch row order is the vector order.
std::vector<Row> run_wave(Engine* e, const std::vector<std::vector<int32_t>>& prompts, int hint,
                          const std::vector<int32_t>& ids) {
    std::vector<std::shared_ptr<Request>> reqs;
    for (const auto& p : prompts) {
        reqs.push_back(score_req(p, hint, ids));
        reqs.back()->priority = static_cast<int>(reqs.size());
    }
    for (auto& r : reqs)
        e->add_request(r);
    for (int i = 0; i < 4096; ++i) {
        const bool busy = std::any_of(reqs.begin(), reqs.end(), [](const auto& r) {
            return r->status != RequestStatus::FINISHED && r->status != RequestStatus::CANCELLED;
        });
        if (!busy)
            break;
        (void)e->step();
    }
    std::vector<Row> out;
    for (const auto& r : reqs) {
        EXPECT_EQ(r->status, RequestStatus::FINISHED);
        EXPECT_EQ(r->score_out.size(), ids.size());
        out.push_back({r->score_out, r->cached_tokens});
    }
    return out;
}

float max_abs_delta(const std::vector<float>& a, const std::vector<float>& b) {
    float d = 0.0f;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i)
        d = std::max(d, std::fabs(a[i] - b[i]));
    return d;
}

}  // namespace

TEST_F(HybridSharedScoreTest, RowsRestoringOneSnapshotAreIndependent) {
    if (!(model_ && model_->model && model_->model->config().ssm_inner_size > 0))
        GTEST_SKIP() << "SKIPPED ON A DENSE CHECKPOINT: " << imp_test::kEnvModelGdn
                     << " points at a model with ssm_inner_size == 0.";
    std::string evidence;
    char line[256];
    for (int k = 0; k < 24; ++k) {
        std::snprintf(line, sizeof(line), kEvidenceLine, k, 1 + (k * 7) % 30, 3 + (k * 11) % 40, k % 6);
        evidence += line;
    }
    std::vector<std::vector<int32_t>> prompts;
    for (const char* s : kSuffixes)
        prompts.push_back(tokenize(evidence + s));
    const int hint = common_prefix_tokens(prompts);
    std::vector<int32_t> ids;
    for (int32_t id = 1000; id < 1016; ++id)
        ids.push_back(id);

    Engine* e = fresh_engine();
    ASSERT_NE(e, nullptr);
    ASSERT_NE(e->ssm_state(), nullptr) << "hybrid checkpoint without a recurrent state pool";
    const int bs = kKVBlockSize;
    const int restore_at = (hint / bs) * bs;
    ASSERT_GE(restore_at, e->runtime_config().server.snapshot_min_prompt_tokens)
        << "shared evidence too short to be snapshotted";
    const std::vector<std::vector<int32_t>> wave(prompts.begin() + 1, prompts.end());
    std::vector<std::vector<int32_t>> wave_rev(wave.rbegin(), wave.rend());

    // Arm 1: item 0 alone publishes the prefix, then the wave in order.
    (void)run_wave(e, {prompts[0]}, hint, ids);
    e->reset_ragged_prefill_max_seqs();
    const std::vector<Row> fwd = run_wave(e, wave, hint, ids);
    EXPECT_EQ(e->ragged_prefill_max_seqs(), kItems) << "the wave did not run as one ragged forward";
    for (int i = 0; i < kItems; ++i)
        EXPECT_EQ(fwd[static_cast<size_t>(i)].cached, restore_at)
            << "row " << i << " did not restore the snapshot";

    // Arm 2: same wave, reversed row order. Every item must get bit-equal logits.
    e = fresh_engine();
    ASSERT_NE(e, nullptr);
    (void)run_wave(e, {prompts[0]}, hint, ids);
    const std::vector<Row> rev = run_wave(e, wave_rev, hint, ids);
    int order_mismatch = 0;
    for (int i = 0; i < kItems; ++i) {
        const auto& a = fwd[static_cast<size_t>(i)].logits;
        const auto& b = rev[static_cast<size_t>(kItems - 1 - i)].logits;
        if (a != b) {
            ++order_mismatch;
            std::printf("[shared] item %d: reversed row order moves logits by %g\n", i + 1,
                        max_abs_delta(a, b));
        }
    }
    EXPECT_EQ(order_mismatch, 0) << "rows depend on their position in the ragged batch";

    // Arm 3: kItems copies of item 1 restoring the same snapshot in one forward: identical rows.
    e = fresh_engine();
    ASSERT_NE(e, nullptr);
    (void)run_wave(e, {prompts[0]}, hint, ids);
    const std::vector<Row> same = run_wave(e, std::vector<std::vector<int32_t>>(kItems, prompts[1]), hint,
                                           ids);
    for (int i = 1; i < kItems; ++i)
        EXPECT_EQ(same[static_cast<size_t>(i)].logits, same[0].logits)
            << "copy " << i << " differs from copy 0";

    // Diagnostic: ragged vs solo (each item alone after the same item 0). Rounding class.
    e = fresh_engine();
    ASSERT_NE(e, nullptr);
    (void)run_wave(e, {prompts[0]}, hint, ids);
    float worst = 0.0f;
    for (int i = 0; i < kItems; ++i) {
        const std::vector<Row> solo = run_wave(e, {wave[static_cast<size_t>(i)]}, hint, ids);
        EXPECT_EQ(solo[0].cached, restore_at);
        worst = std::max(worst, max_abs_delta(solo[0].logits, fwd[static_cast<size_t>(i)].logits));
    }
    std::printf("[shared] ragged vs solo: max |dlogit| = %g over %d rows (rounding class, not asserted)\n",
                worst, kItems);
}

}  // namespace imp
