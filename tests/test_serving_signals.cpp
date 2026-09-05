// AUDIT_arch_2026 dispatch #7 (C-1, C-5, C-6): the serving signals a client
// or a scrape can act on, driven through the server's BatchingEngine on a
// real model:
//   - a decode that runs the KV pool dry finishes "capacity", never
//     "cancelled" (the value that means "your client went away");
//   - queue_ms is the wait behind max_batch_size / KV admission, and the
//     waiting/running split is visible while it happens;
//   - the per-request speculation counters reach the request object.
// Requires IMP_TEST_MODEL (default /models/Qwen3-8B-Q8_0.gguf).
#include <gtest/gtest.h>
#include "imp/imp.h"
#include "api/imp_internal.h"
#include "batching_engine.h"
#include "model/model.h"
#include "model/tokenizer.h"
#include "runtime/engine.h"
#include "test_models.h"
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>

namespace {

const char* model_path() {
    return imp_test::env_cstr_or(imp_test::kEnvModel, "/models/Qwen3-8B-Q8_0.gguf");
}

bool model_exists() {
    FILE* f = std::fopen(model_path(), "r");
    if (!f)
        return false;
    std::fclose(f);
    return true;
}

struct Served {
    std::shared_ptr<ServerRequest> sr;
    std::string finish;
    int tokens = 0;
};

std::shared_ptr<ServerRequest> make_request(ImpContext ctx, const std::string& prompt, int max_tokens,
                                            float temperature) {
    auto sr = std::make_shared<ServerRequest>();
    sr->request = std::make_shared<imp::Request>();
    sr->request->input_tokens = ctx->engine->model()->tokenizer()->encode(prompt);
    sr->request->max_tokens = max_tokens;
    sr->request->temperature = temperature;
    sr->request->ignore_eos = true;
    return sr;
}

// Pops every event of one request; false when the deadline passes first.
bool drain(Served& r, int timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    TokenEvent ev{};
    for (;;) {
        if (r.sr->pop_token(ev, 500)) {
            if (ev.token_id >= 0)
                r.tokens++;
            if (ev.is_last) {
                r.finish = ev.finish_reason ? ev.finish_reason : "";
                return true;
            }
        }
        if (std::chrono::steady_clock::now() > deadline)
            return false;
    }
}

struct Loaded {
    ImpModel model = nullptr;
    ImpContext ctx = nullptr;
    ~Loaded() {
        if (ctx)
            imp_context_free(ctx);
        if (model)
            imp_model_free(model);
    }
    bool open(int max_batch_size, size_t kv_blocks) {
        if (imp_model_load(model_path(), IMP_FORMAT_GGUF, &model) != IMP_SUCCESS)
            return false;
        ImpConfig config = imp_config_default();
        config.max_seq_len = 1024;
        config.max_batch_size = max_batch_size;
        config.kv_cache_max_blocks = kv_blocks;
        // The F16-KV eviction valve would answer pressure by dropping context
        // instead of cancelling; the typed cancel is what this file tests.
        config.streaming_kv_auto = 0;
        config.streaming_kv_enabled = 0;
        return imp_context_create(model, &config, &ctx) == IMP_SUCCESS;
    }
};

std::string long_prompt() {
    std::string p;
    for (int i = 0; i < 4; i++)
        p += "The three primary colors are red, blue, and yellow. Mixing two primary colors "
             "produces a secondary color: red and blue make purple, blue and yellow make green, "
             "and red and yellow make orange. ";
    return p;
}

TEST(ServingSignalsTest, MidDecodeKvExhaustionFinishesAsCapacity) {
    if (!model_exists())
        GTEST_SKIP() << "Model not found: " << model_path();
    Loaded m;
    ASSERT_TRUE(m.open(/*max_batch_size=*/1, /*kv_blocks=*/24));  // 24 x 16 = 384 tokens
    BatchingEngine be;
    be.start(m.ctx);
    // ~120 prompt tokens fit; 600 generated tokens cannot, so the pool runs
    // dry mid-decode and the append fails on the decode path.
    Served r{make_request(m.ctx, long_prompt(), 600, 1.0f)};
    be.submit(r.sr);
    ASSERT_TRUE(drain(r, 180000)) << "the request never finished";
    EXPECT_GT(r.tokens, 0) << "the pool did not even hold the prompt; the cancel must come mid-decode";
    EXPECT_EQ(r.finish, "capacity");
    EXPECT_EQ(r.sr->request->cancel_reason, imp::CancelReason::KvCapacity);
    be.stop();
}

TEST(ServingSignalsTest, QueueTimeIsTheSchedulerWaitAndTheSplitIsVisible) {
    if (!model_exists())
        GTEST_SKIP() << "Model not found: " << model_path();
    Loaded m;
    ASSERT_TRUE(m.open(/*max_batch_size=*/1, /*kv_blocks=*/0));
    BatchingEngine be;
    be.start(m.ctx);
    Served r1{make_request(m.ctx, long_prompt(), 64, 1.0f)};
    Served r2{make_request(m.ctx, long_prompt(), 4, 1.0f)};
    be.submit(r1.sr);
    be.submit(r2.sr);
    // While r1 generates, r2 sits behind max_batch_size: one running, one
    // waiting. The worker refreshes the split every loop.
    bool saw_split = false;
    for (int i = 0; i < 300 && !saw_split; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        saw_split = be.queue_running.load() == 1 && be.queue_waiting.load() == 1;
    }
    EXPECT_TRUE(saw_split) << "running/waiting never read 1/1 while r1 was in flight";
    ASSERT_TRUE(drain(r1, 180000));
    ASSERT_TRUE(drain(r2, 60000));
    const double q1 = r1.sr->queue_ms.load();
    const double q2 = r2.sr->queue_ms.load();
    EXPECT_GE(q1, 0.0);
    EXPECT_GT(q2, 100.0) << "r2 waited for r1's 64 tokens; a stamp at worker pickup reads ~0 here";
    EXPECT_GT(q2, q1);
    be.stop();
}

TEST(ServingSignalsTest, SpeculationCountersReachTheRequest) {
    if (!model_exists())
        GTEST_SKIP() << "Model not found: " << model_path();
    Loaded m;
    ASSERT_TRUE(m.open(/*max_batch_size=*/1, /*kv_blocks=*/0));
    BatchingEngine be;
    be.start(m.ctx);
    // Greedy over a counting cycle that stops mid-cycle: the continuation is
    // the cycle itself, so the n-gram drafter finds its proposals in the
    // prompt from the first generated token on, verify steps run and the
    // counters move. (A prompt whose continuation is NOT in the context
    // misses a handful of times and the drafter gives up: 0 verify steps.)
    std::string prompt;
    for (int i = 0; i < 8; i++)
        prompt += "1 2 3 4 5 6 7 8 9 10 11 12 ";
    prompt += "1 2 3 4 5 6";
    Served r{make_request(m.ctx, prompt, 64, 0.0f)};
    be.submit(r.sr);
    ASSERT_TRUE(drain(r, 180000));
    const auto& req = *r.sr->request;
    EXPECT_GT(req.spec_verifies, 0) << "no verify step ran: is the n-gram drafter on for this model?";
    EXPECT_GT(req.spec_drafted, 0);
    EXPECT_GE(req.spec_accepted, 0);
    EXPECT_LE(req.spec_accepted, req.spec_drafted);
    be.stop();
}

}  // namespace
