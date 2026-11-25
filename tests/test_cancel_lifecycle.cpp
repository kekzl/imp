// Engine::cancel_request on a real checkpoint (audit C-1, C-4, C-5; C-14 found them untested).
// C-1: cancelling a queued request (no recurrent slot yet) leaves the running sequence's GDN state;
//      the reverted fallback `id % cap` zeroed it. Needs IMP_TEST_MODEL_GDN (make test-hybrid-restore).
// C-4/C-5: a cancel during a graph burst waits the burst out and empties every table, and a burst
//      abandoned under a cancel appends no token. Needs IMP_TEST_MODEL (make test-e2e).

#include <gtest/gtest.h>

#include "api/imp_internal.h"
#include "imp/imp.h"
#include "memory/kv_cache_manager.h"
#include "memory/ssm_state.h"
#include "model/model.h"
#include "runtime/config.h"
#include "runtime/engine.h"
#include "runtime/request.h"
#include "test_models.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace imp {
namespace {

constexpr const char* kPrompt = "Write a long story about a lighthouse keeper who counts the waves at night.";

class CancelLifecycleTest : public ::testing::Test {
protected:
    virtual const char* model_env() const { return imp_test::kEnvModel; }
    void SetUp() override {
        const std::string path = imp_test::env_path(model_env());
        if (path.empty())
            GTEST_SKIP() << "Set " << model_env() << " to run the cancel lifecycle tests";
        ASSERT_NO_FATAL_FAILURE(imp_test::require_readable(path.c_str(), model_env()));
        ASSERT_EQ(imp_model_load(path.c_str(), imp_test::model_format(path), &model_), IMP_SUCCESS);
    }
    void TearDown() override {
        if (ctx_) {
            imp_context_free(ctx_);
            EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "context teardown faulted in-flight work";
        }
        if (model_)
            imp_model_free(model_);
    }
    // Speculation off: a lone decoding request then runs the graph burst (kDecodeBurst steps).
    Engine* create(int max_batch, bool graphs) {
        RuntimeConfig rc;
        rc.server.prefix_cache = false;
        const auto bad = rc.apply_overrides(
            {"speculative.ngram=false", "speculative.suffix=false", "speculative.mtp_k=0"});
        EXPECT_TRUE(bad.empty()) << bad[0];
        set_pending_runtime_config(rc);
        ImpConfig cfg = imp_config_default();
        cfg.max_seq_len = 2048;
        cfg.max_batch_size = max_batch;
        cfg.enable_cuda_graphs = graphs ? 1 : 0;
        cfg.use_prefix_caching = 0;
        EXPECT_EQ(imp_context_create(model_, &cfg, &ctx_), IMP_SUCCESS);
        return ctx_ ? ctx_->engine.get() : nullptr;
    }
    std::shared_ptr<Request> make_req(int max_tokens) {
        auto req = std::make_shared<Request>();
        std::vector<int32_t> toks(4096);
        int n = 0;
        EXPECT_EQ(imp_tokenize(model_, kPrompt, toks.data(), &n, static_cast<int>(toks.size())), IMP_SUCCESS);
        toks.resize(n > 0 ? n : 0);
        req->input_tokens = toks;
        req->max_tokens = max_tokens;
        req->temperature = 0.0f;
        req->top_p = 1.0f;
        req->top_k = 0;
        req->ignore_eos = true;
        req->status = RequestStatus::PENDING;
        return req;
    }
    // Steps until a graph burst holds req; false when none launched within 64 steps.
    static bool step_into_burst(Engine* engine, const std::shared_ptr<Request>& req) {
        for (int i = 0; i < 64 && !engine->graph_burst_holds(req->id); ++i)
            (void)engine->step();
        return engine->graph_burst_holds(req->id) && req->status == RequestStatus::DECODING;
    }
    ImpModel model_ = nullptr;
    ImpContext ctx_ = nullptr;
};

class CancelLifecycleHybridTest : public CancelLifecycleTest {
protected:
    const char* model_env() const override { return imp_test::kEnvModelGdn; }
};

}  // namespace

// C-5 + C-4: Wait collects the burst before KV and tables go. Mutants: drop the
// async_graph_abandon_() call in cancel_request (burst still holds the id), or the
// release_request_tables_ call (block table still allocated).
TEST_F(CancelLifecycleTest, CancelDuringAGraphBurstWaitsItOutAndReleasesEveryTable) {
    Engine* engine = create(/*max_batch=*/2, /*graphs=*/true);
    ASSERT_NE(engine, nullptr);
    auto req = make_req(/*max_tokens=*/512);
    engine->add_request(req);
    ASSERT_TRUE(step_into_burst(engine, req)) << "no graph burst launched: nothing in flight to cancel";
    ASSERT_FALSE(engine->kv_manager()->block_table(req->id).empty());

    ASSERT_TRUE(engine->cancel_request(req, CancelLanes::Wait));
    const size_t emitted = req->output_tokens.size();

    EXPECT_EQ(req->status, RequestStatus::CANCELLED);
    EXPECT_FALSE(engine->graph_burst_holds(req->id)) << "the burst still references a cancelled request";
    EXPECT_TRUE(engine->kv_manager()->block_table(req->id).empty()) << "KV blocks outlived the cancel";
    EXPECT_FALSE(engine->cancel_request(req)) << "a second cancel found the request still held";
    for (int i = 0; i < 8; ++i)
        (void)engine->step();
    EXPECT_EQ(req->output_tokens.size(), emitted) << "tokens appended after the cancel";
    EXPECT_EQ(req->status, RequestStatus::CANCELLED) << "finished after it was cancelled";

    // The engine serves the next request to its end.
    auto next = make_req(/*max_tokens=*/16);
    engine->add_request(next);
    for (int i = 0; i < 256 && next->status != RequestStatus::FINISHED; ++i)
        (void)engine->step();
    EXPECT_EQ(next->status, RequestStatus::FINISHED);
    EXPECT_EQ(next->output_tokens.size(), 16u);
}

// C-5: a cancel that could not wait (CancelLanes::Abandon) leaves the burst running; the next
// harvest must append nothing and finish nothing. Mutant: drop the status check at the top of
// step_async_graph_resume (the burst's tokens land on the cancelled request).
TEST_F(CancelLifecycleTest, AbandonedBurstAppendsNoTokenAfterTheCancel) {
    Engine* engine = create(/*max_batch=*/2, /*graphs=*/true);
    ASSERT_NE(engine, nullptr);
    auto req = make_req(/*max_tokens=*/512);
    engine->add_request(req);
    ASSERT_TRUE(step_into_burst(engine, req)) << "no graph burst launched: nothing in flight to abandon";

    ASSERT_TRUE(engine->cancel_request(req, CancelLanes::Abandon));
    const size_t emitted = req->output_tokens.size();
    for (int i = 0; i < 8; ++i)
        (void)engine->step();

    EXPECT_EQ(req->output_tokens.size(), emitted) << "the abandoned burst appended tokens after the cancel";
    EXPECT_EQ(req->status, RequestStatus::CANCELLED);
    EXPECT_FALSE(engine->graph_burst_holds(req->id)) << "the next step did not drop the abandoned burst";
}

// C-1: request B waits for admission (no slot) with B.id % cap == the slot request A decodes in.
// Cancelling B must not touch A's slab. Mutant: in cancel_request, reset slot `req->id % cap` when
// recurrent_slot_for_reset() has no entry: A's slab reads all zero.
TEST_F(CancelLifecycleHybridTest, CancelOfAQueuedRequestKeepsTheRunningSequencesState) {
    if (!(model_ && model_->model && model_->model->config().ssm_inner_size > 0))
        GTEST_SKIP() << "SKIPPED ON A DENSE CHECKPOINT: " << model_env() << " has ssm_inner_size == 0";
    // No graphs: no burst runs on past the cancel's pipeline drain, so A's slab is still at the read.
    Engine* engine = create(/*max_batch=*/4, /*graphs=*/false);
    ASSERT_NE(engine, nullptr);
    ASSERT_NE(engine->ssm_state(), nullptr);
    const int cap = engine->ssm_state()->max_sequences();
    ASSERT_GT(cap, 0);

    auto a = make_req(/*max_tokens=*/24);
    engine->add_request(a);
    for (int i = 0; i < 64 && a->output_tokens.size() < 4; ++i)
        (void)engine->step();
    ASSERT_EQ(a->status, RequestStatus::DECODING);
    const int slot = engine->recurrent_slot(a->id);
    ASSERT_GE(slot, 0);

    // Ids are sequential: within cap requests one aliases A's slot. The others are cancelled too.
    std::shared_ptr<Request> queued;
    for (int i = 0; i < cap && !queued; ++i) {
        auto b = make_req(/*max_tokens=*/8);
        engine->add_request(b);
        if (b->id % cap == slot)
            queued = b;
        else
            ASSERT_TRUE(engine->cancel_request(b));
    }
    ASSERT_NE(queued, nullptr);
    ASSERT_EQ(engine->recurrent_slot(queued->id), -1) << "the queued request already holds a slot";

    ASSERT_TRUE(engine->cancel_request(queued));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<uint8_t> slab(engine->ssm_state()->per_seq_bytes());
    ASSERT_EQ(cudaMemcpy(slab.data(), engine->ssm_state()->seq_base(slot), slab.size(),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    const auto nonzero = std::count_if(slab.begin(), slab.end(), [](uint8_t v) { return v != 0; });
    EXPECT_GT(nonzero, 0) << "request " << a->id << "'s recurrent slot " << slot
                          << " was zeroed by the cancel of request " << queued->id;
    EXPECT_EQ(engine->recurrent_slot(a->id), slot);

    for (int i = 0; i < 256 && a->status != RequestStatus::FINISHED; ++i)
        (void)engine->step();
    EXPECT_EQ(a->status, RequestStatus::FINISHED);
    EXPECT_EQ(a->output_tokens.size(), 24u);
}

}  // namespace imp
