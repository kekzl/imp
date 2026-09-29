// #2257: the prompt_logprobs chunk logits (up to 1024 x vocab FP32) are released when the pass
// returns. Rule: VRAMAllocator bytes after a prompt_logprobs request <= plain-request bytes + 8 MiB.
// Needs a GPU and IMP_TEST_MODEL (skipped otherwise).

#include <gtest/gtest.h>
#include "imp/imp.h"
#include "api/imp_internal.h"
#include "runtime/engine.h"
#include "runtime/request.h"
#include "test_models.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace {

class PromptLogprobsE2ETest : public ::testing::Test {
protected:
    void SetUp() override {
        const char* path = std::getenv(imp_test::kEnvModel);
        if (!path)
            GTEST_SKIP() << "Set IMP_TEST_MODEL to run the prompt_logprobs scratch-release test";
        ASSERT_NO_FATAL_FAILURE(imp_test::require_readable(path, imp_test::kEnvModel));
        const std::string p(path);
        const bool gguf = p.size() >= 5 && p.substr(p.size() - 5) == ".gguf";
        ASSERT_EQ(imp_model_load(path, gguf ? IMP_FORMAT_GGUF : IMP_FORMAT_SAFETENSORS, &model_),
                  IMP_SUCCESS);
        ImpConfig cfg = imp_config_default();
        cfg.max_seq_len = 2048;
        cfg.max_batch_size = 1;
        cfg.use_prefix_caching = 0;
        ASSERT_EQ(imp_context_create(model_, &cfg, &ctx_), IMP_SUCCESS);
    }

    void TearDown() override {
        if (ctx_)
            imp_context_free(ctx_);
        if (model_)
            imp_model_free(model_);
    }

    // One engine-loop request over `tokens`; returns it finished.
    std::shared_ptr<imp::Request> run(const std::vector<int32_t>& tokens, int prompt_logprobs) {
        imp::Engine* engine = ctx_->engine.get();
        auto req = std::make_shared<imp::Request>();
        req->input_tokens = tokens;
        req->max_tokens = 1;
        req->temperature = 0.0f;
        req->top_p = 1.0f;
        req->top_k = 0;
        req->ignore_eos = true;
        req->prompt_logprobs = prompt_logprobs;
        req->status = imp::RequestStatus::PENDING;
        engine->add_request(req);
        for (int i = 0; i < 256; ++i) {
            if (req->status == imp::RequestStatus::FINISHED || req->status == imp::RequestStatus::CANCELLED)
                break;
            (void)engine->step();
        }
        EXPECT_EQ(req->status, imp::RequestStatus::FINISHED);
        return req;
    }

    ImpModel model_ = nullptr;
    ImpContext ctx_ = nullptr;
};

TEST_F(PromptLogprobsE2ETest, ChunkLogitsReleasedAfterPass) {
    imp::Engine* engine = ctx_->engine.get();
    ASSERT_NE(engine, nullptr);
    // 1024 ordinary ids (100..20099): valid in every tested vocab, no special tokens.
    std::vector<int32_t> tokens(1024);
    for (int i = 0; i < 1024; ++i)
        tokens[i] = 100 + (i * 7919) % 20000;

    run(tokens, -1);  // warms every lazily grown workspace of a plain prefill
    const size_t plain = engine->vram_allocator().allocated();

    tokens[0] = 101;  // distinct prompt
    const auto req = run(tokens, 0);
    EXPECT_EQ(req->prompt_lp.rows, 1023) << "prompt_logprobs pass did not score every prompt row";
    const size_t after = engine->vram_allocator().allocated();
    constexpr size_t kSlack = size_t{8} << 20;  // targets/rank/lp scratch: 1024 x 12 B
    EXPECT_LE(after, plain + kSlack) << "prompt_logprobs left " << (after - plain) << " bytes allocated";
}

}  // namespace
