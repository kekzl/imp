// gemm.fp32_residual_fuse (exec/executor_fp32_residual.cpp) changes launches, not bits: 16 greedy
// requests stepped together through one engine (deterministic, prefix cache and CUDA graphs off, so
// every step runs the host path of the policy in force), the policy flipped between rounds. Output
// tokens and chosen-token logprobs must be equal; a fused-fused repeat is the control.
#include <gtest/gtest.h>
#include "imp/imp.h"
#include "api/imp_internal.h"
#include "exec/executor.h"
#include "runtime/engine.h"
#include "runtime/request.h"
#include "test_models.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr int kRows = 16;
constexpr int kTokens = 48;

struct Round {
    std::vector<std::vector<int32_t>> tokens;
    std::vector<std::vector<float>> logprobs;
    bool operator==(const Round&) const = default;
};

Round run_round(ImpModel model, imp::Engine* engine) {
    Round out;
    std::vector<std::shared_ptr<imp::Request>> reqs;
    for (int i = 0; i < kRows; ++i) {
        const std::string prompt = "Request " + std::to_string(i) + ": explain in plain words how " +
                                   (i % 2 ? "a glacier moves" : "a bridge carries load") + ".";
        int32_t toks[256];
        int n = 0;
        EXPECT_EQ(imp_tokenize(model, prompt.c_str(), toks, &n, 256), IMP_SUCCESS);
        auto req = std::make_shared<imp::Request>();
        req->input_tokens.assign(toks, toks + n);
        req->max_tokens = kTokens;
        req->temperature = 0.0f;
        req->top_p = 1.0f;
        req->top_k = 0;
        req->ignore_eos = true;
        req->logprobs = true;
        req->status = imp::RequestStatus::PENDING;
        engine->add_request(req);
        reqs.push_back(req);
    }
    for (int step = 0; step < 4096; ++step) {
        bool done = true;
        for (const auto& r : reqs)
            done = done &&
                   (r->status == imp::RequestStatus::FINISHED || r->status == imp::RequestStatus::CANCELLED);
        if (done)
            break;
        (void)engine->step();
    }
    for (const auto& r : reqs) {
        EXPECT_EQ(r->status, imp::RequestStatus::FINISHED);
        out.tokens.push_back(r->output_tokens);
        std::vector<float> lp;
        for (const auto& t : r->output_logprobs)
            lp.push_back(t.logprob);
        out.logprobs.push_back(lp);
    }
    return out;
}

TEST(Fp32ResidualFuseE2ETest, BatchedDecodeBitEqualGemma4) {
    const char* path = std::getenv(imp_test::kEnvModelGemma4Nvfp4);
    if (!path)
        GTEST_SKIP() << "Set " << imp_test::kEnvModelGemma4Nvfp4 << " to run the fused-norm equivalence";
    ASSERT_NO_FATAL_FAILURE(imp_test::require_readable(path, imp_test::kEnvModelGemma4Nvfp4));
    setenv("IMP_DETERMINISTIC", "1", 1);
    ImpModel model = nullptr;
    ASSERT_EQ(imp_model_load(path, IMP_FORMAT_SAFETENSORS, &model), IMP_SUCCESS);
    ImpConfig cfg = imp_config_default();
    cfg.enable_cuda_graphs = 0;  // a replayed graph would keep the policy it was captured under
    cfg.use_prefix_caching = 0;
    cfg.max_batch_size = kRows;
    ImpContext ctx = nullptr;
    ASSERT_EQ(imp_context_create(model, &cfg, &ctx), IMP_SUCCESS);
    imp::Engine* engine = ctx->engine.get();
    imp::GraphExecutor* ex = engine->executor();
    ASSERT_NE(ex, nullptr);
    const imp::DispatchPolicy& own = ex->dispatch_policy();
    const imp::DispatchPolicy fused = own;
    imp::DispatchPolicy unfused = own;
    unfused.gemm.fp32_residual_fuse = false;
    EXPECT_TRUE(fused.gemm.fp32_residual_fuse) << "default changed: the rounds below would not differ";

    ex->set_dispatch_policy(fused);
    (void)run_round(model, engine);  // warm: every measured round follows one identical round
    const Round a = run_round(model, engine);
    ex->set_dispatch_policy(unfused);
    const Round off = run_round(model, engine);
    ex->set_dispatch_policy(fused);
    const Round again = run_round(model, engine);
    ex->set_dispatch_policy(own);

    EXPECT_TRUE(a == again) << "control: fused vs fused differ, the comparison means nothing";
    EXPECT_TRUE(a == off) << "fused vs unfused: tokens or chosen-token logprobs differ";
    imp_context_free(ctx);
    imp_model_free(model);
}

}  // namespace
