// Batched sampling keeps each row's OWN banned-token list through the penalty stash (#2019).
// Two rows enqueue with different lists in one step; the flush must ban 5 for row 0 and 7
// for row 1. With one shared device copy (the pre-fix cache) row 0 read row 1's list at the
// flush and banned 7 instead: on Qwen3.8-27B at 8 thinking streams that was <|im_end|>
// banned on every closed row while a later row still thought, so the closed rows
// re-emitted </think> + their answer every ~100 tokens until the last thinker closed.
// Needs IMP_TEST_MODEL (any checkpoint); runs from make test-e2e.

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include "api/imp_internal.h"
#include "core/tensor.h"
#include "exec/executor.h"
#include "exec/inference_state.h"
#include "imp/imp.h"
#include "runtime/config.h"
#include "runtime/engine.h"
#include "test_models.h"

#include <string>
#include <vector>

namespace imp {
namespace {

class SamplerRowBansTest : public ::testing::Test {
protected:
    void SetUp() override {
        const std::string path = imp_test::env_path(imp_test::kEnvModel);
        if (path.empty())
            GTEST_SKIP() << "Set " << imp_test::kEnvModel << " to run the sampler row-ban test";
        ASSERT_NO_FATAL_FAILURE(imp_test::require_readable(path.c_str(), imp_test::kEnvModel));
        const bool gguf = path.size() >= 5 && path.substr(path.size() - 5) == ".gguf";
        ASSERT_EQ(imp_model_load(path.c_str(), gguf ? IMP_FORMAT_GGUF : IMP_FORMAT_SAFETENSORS, &model_),
                  IMP_SUCCESS);
        RuntimeConfig rc;
        rc.server.prefix_cache = false;
        set_pending_runtime_config(rc);
        ImpConfig cfg = imp_config_default();
        cfg.max_seq_len = 512;
        cfg.max_batch_size = 4;
        cfg.enable_cuda_graphs = 1;
        cfg.use_prefix_caching = 0;
        ASSERT_EQ(imp_context_create(model_, &cfg, &ctx_), IMP_SUCCESS);
    }
    void TearDown() override {
        if (ctx_)
            imp_context_free(ctx_);
        if (model_)
            imp_model_free(model_);
    }
    ImpModel model_ = nullptr;
    ImpContext ctx_ = nullptr;
};

}  // namespace

TEST_F(SamplerRowBansTest, StashedRowsKeepTheirOwnBanList) {
    Engine* engine = ctx_->engine.get();
    ASSERT_NE(engine, nullptr);
    GraphExecutor* ex = engine->executor();
    ASSERT_NE(ex, nullptr);
    ASSERT_TRUE(ex->sample_pipeline_ready()) << "row-batched sampler scratch not armed";

    constexpr int kRows = 2;
    const int vocab = 64;
    // Row 0: 7 leads, 5 second. Row 1: 5 leads, 7 second. Bans decide the argmax.
    std::vector<float> h(static_cast<size_t>(kRows) * vocab, 0.0f);
    h[7] = 10.0f;
    h[5] = 9.0f;
    h[static_cast<size_t>(vocab) + 5] = 10.0f;
    h[static_cast<size_t>(vocab) + 7] = 9.0f;
    float* d = nullptr;
    ASSERT_EQ(cudaMalloc(&d, h.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    const int64_t shape[2] = {kRows, vocab};
    Tensor logits(d, QType::F32, 2, shape, true);

    const std::vector<int32_t> ban_row0 = {5};
    const std::vector<int32_t> ban_row1 = {7};
    InferenceState st[kRows];
    for (int r = 0; r < kRows; ++r) {
        st[r].temperature = 0.0f;
        st[r].top_k = 0;
        st[r].n_sequences = 1;
        st[r].banned_tokens = r == 0 ? ban_row0.data() : ban_row1.data();
        st[r].n_banned_tokens = 1;
    }
    cudaStream_t stream = nullptr;
    // Both rows enqueue before either samples: the stash flush is where a shared
    // device copy of the ban list would serve row 1's list to row 0.
    for (int r = 0; r < kRows; ++r)
        ASSERT_TRUE(ex->sample_single_from_logits_async(logits.slice(r, r + 1), st[r], r, stream))
            << "row " << r << " declined the async path";
    const int32_t* toks = ex->collect_sampled_tokens(kRows, stream);
    ASSERT_NE(toks, nullptr);
    EXPECT_EQ(toks[0], 7) << "row 0 bans 5 only";
    EXPECT_EQ(toks[1], 5) << "row 1 bans 7 only";
    cudaFree(d);
}

}  // namespace imp
