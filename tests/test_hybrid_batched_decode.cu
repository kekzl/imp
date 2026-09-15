// Batched GDN decode (runtime.gdn_batched_decode) at the ENGINE level, greedy under
// runtime.deterministic. Asserted: eight rows of ONE prompt in a batch decode identically
// (per-row plumbing: slot table, conv state, residual and alpha/beta paths, sampling state
// copies). Printed only: how many rows of eight different prompts differ from their solo run,
// which the M=1 (GEMV) vs small-M (GEMM) rounding makes a near-tie count, not a defect.
// Context: Qwen3.8-27B-NVFP4-vllm at 8 thinking sessions re-closes its think block and repeats
// the answer on 2-5 rows, never solo, never with gdn_batched_decode=false; rows there ARE
// row-invariant too, so that defect needs a quality oracle (#2019). Needs IMP_TEST_MODEL_GDN;
// runs from make test-e2e.

#include <gtest/gtest.h>

#include "api/imp_internal.h"
#include "imp/imp.h"
#include "model/model.h"
#include "runtime/config.h"
#include "runtime/engine.h"
#include "runtime/request.h"
#include "test_models.h"

#include <memory>
#include <string>
#include <vector>

namespace imp {
namespace {

constexpr int kRows = 8;
constexpr int kGreedy = 48;

constexpr const char* kPrompts[kRows] = {
    "Explain how a paged KV cache shares blocks between sequences.",
    "Why does a recurrent state make prefix caching harder than plain attention?",
    "Give a short argument for measuring bandwidth instead of trusting cudaMalloc.",
    "What is the cost of one extra chunk boundary in a chunked prefill?",
    "Describe what a block table is and who reads it during decode.",
    "When is a snapshot of a recurrent state safe to restore, and when is it not?",
    "Name two reasons a greedy continuation can differ between two runs of one engine.",
    "How does a delta rule scan keep its state across a chunk boundary?",
};

class HybridBatchedDecodeTest : public ::testing::Test {
protected:
    void SetUp() override {
        const std::string path = imp_test::env_path(imp_test::kEnvModelGdn);
        if (path.empty())
            GTEST_SKIP() << "Set " << imp_test::kEnvModelGdn << " to run the batched hybrid decode test";
        ASSERT_NO_FATAL_FAILURE(imp_test::require_readable(path.c_str(), imp_test::kEnvModelGdn));
        const bool gguf = path.size() >= 5 && path.substr(path.size() - 5) == ".gguf";
        ASSERT_EQ(imp_model_load(path.c_str(), gguf ? IMP_FORMAT_GGUF : IMP_FORMAT_SAFETENSORS, &model_),
                  IMP_SUCCESS);
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

TEST_F(HybridBatchedDecodeTest, RowsOfOnePromptDecodeIdentically) {
    if (!(model_ && model_->model && model_->model->config().ssm_inner_size > 0))
        GTEST_SKIP() << "SKIPPED ON A DENSE CHECKPOINT: " << imp_test::kEnvModelGdn
                     << " points at a model with ssm_inner_size == 0.";

    RuntimeConfig rc;
    rc.runtime.deterministic = true;
    rc.server.prefix_cache = false;
    set_pending_runtime_config(rc);

    ImpConfig cfg = imp_config_default();
    cfg.max_seq_len = 2048;
    cfg.max_batch_size = kRows;
    cfg.enable_cuda_graphs = 1;
    cfg.use_prefix_caching = 0;
    ASSERT_EQ(imp_context_create(model_, &cfg, &ctx_), IMP_SUCCESS);
    Engine* engine = ctx_->engine.get();
    ASSERT_NE(engine, nullptr);
    ASSERT_NE(engine->ssm_state(), nullptr) << "hybrid checkpoint without a recurrent state pool";

    auto make_req = [&](int i) {
        auto req = std::make_shared<Request>();
        std::vector<int32_t> toks(4096);
        int n = 0;
        EXPECT_EQ(imp_tokenize(model_, kPrompts[i], toks.data(), &n, static_cast<int>(toks.size())),
                  IMP_SUCCESS);
        toks.resize(n > 0 ? n : 0);
        req->input_tokens = toks;
        req->max_tokens = kGreedy;
        req->temperature = 0.0f;
        req->top_p = 1.0f;
        req->top_k = 0;
        req->ignore_eos = true;
        req->status = RequestStatus::PENDING;
        return req;
    };
    auto drain = [&](std::vector<std::shared_ptr<Request>>& reqs) {
        for (int i = 0; i < 65536; ++i) {
            bool busy = false;
            for (auto& r : reqs)
                busy = busy || (r->status != RequestStatus::FINISHED && r->status != RequestStatus::CANCELLED);
            if (!busy)
                break;
            (void)engine->step();
        }
        for (auto& r : reqs)
            EXPECT_EQ(r->status, RequestStatus::FINISHED) << "request did not finish";
    };

    // Solo arms: one request at a time.
    std::vector<std::vector<int32_t>> solo(kRows);
    for (int i = 0; i < kRows; ++i) {
        std::vector<std::shared_ptr<Request>> one{make_req(i)};
        engine->add_request(one[0]);
        drain(one);
        solo[static_cast<size_t>(i)] = one[0]->output_tokens;
        ASSERT_EQ(static_cast<int>(solo[static_cast<size_t>(i)].size()), kGreedy);
    }
    // Control: solo again must be bit-identical, or the comparison below decides nothing.
    {
        std::vector<std::shared_ptr<Request>> one{make_req(0)};
        engine->add_request(one[0]);
        drain(one);
        ASSERT_EQ(one[0]->output_tokens, solo[0]) << "solo run of prompt 0 is not reproducible";
    }

    // Batched arm: all eight admitted before the first step.
    std::vector<std::shared_ptr<Request>> batch;
    for (int i = 0; i < kRows; ++i)
        batch.push_back(make_req(i));
    for (auto& r : batch)
        engine->add_request(r);
    drain(batch);

    int mismatched = 0;
    for (int i = 0; i < kRows; ++i) {
        const auto& got = batch[static_cast<size_t>(i)]->output_tokens;
        const auto& want = solo[static_cast<size_t>(i)];
        size_t first = 0;
        while (first < got.size() && first < want.size() && got[first] == want[first])
            ++first;
        if (got != want) {
            ++mismatched;
            std::printf("[batched] row %d diverges from solo at token %zu of %d\n", i, first, kGreedy);
        }
    }
    // Diagnostic, not asserted: the batched step runs the small-M GEMMs and the batched scan,
    // solo runs the GEMV path (gdn.m1_fused), so their rounding differs and greedy flips at a
    // near-tie are expected (2 of 8 rows at tokens 9 and 18 on Qwen3.5-4B-mxfp4, 2026-09-15).
    std::printf("[batched] %d of %d rows differ from their solo run (rounding class, not asserted)\n",
                mismatched, kRows);

    // Row invariance, the rounding-proof half: eight copies of ONE prompt in one batch must
    // decode identically (same tokens, same kernels, same M), whatever they differ from solo by.
    std::vector<std::shared_ptr<Request>> same;
    for (int i = 0; i < kRows; ++i)
        same.push_back(make_req(0));
    for (auto& r : same)
        engine->add_request(r);
    drain(same);
    int row_mismatch = 0;
    for (int i = 1; i < kRows; ++i) {
        const auto& got = same[static_cast<size_t>(i)]->output_tokens;
        const auto& ref = same[0]->output_tokens;
        size_t first = 0;
        while (first < got.size() && first < ref.size() && got[first] == ref[first])
            ++first;
        if (got != ref) {
            ++row_mismatch;
            std::printf("[batched] identical prompt: row %d diverges from row 0 at token %zu\n", i, first);
        }
    }
    EXPECT_EQ(row_mismatch, 0) << row_mismatch << " of " << kRows - 1
                               << " rows with the SAME prompt decoded differently from row 0";
    std::printf("[batched] identical-prompt rows vs solo: %s\n",
                same[0]->output_tokens == solo[0] ? "row 0 equals solo" : "row 0 differs from solo");
}

}  // namespace imp
