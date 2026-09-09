// =============================================================================
// Decode-graph buckets on a GDN/SSM hybrid: does the token stream depend on how
// many sequences happen to share the step?
//
// WHY THIS EXISTS
//   The engine keeps ONE captured decode graph per batch size
//   (`decode_graph_pool_[n_sequences - 1]`, engine.h). For a hybrid, each of
//   those graphs also encodes how the GDN scan reaches its per-sequence
//   recurrent state: at n == 1 the slot travels as a captured scalar and a slot
//   change forces a recapture (engine_scheduler.cpp), at n > 1 it travels
//   through the device slot table `d_ssm_seq_slots_`, which the graph reads by
//   pointer. Two different mechanisms, one per bucket, and nothing in the tree
//   exercised a capture-and-replay of bucket i against bucket 1 on a recurrent
//   model: `test_gdn_batched.cu` covers the scan kernel, `test_graph_slots.cpp`
//   covers a different pool (GraphSlotPool), `test_continuous_batching.cpp`
//   runs the scheduler with no model at all.
//
//   The failure this catches is a bucket whose graph reads the wrong slab: the
//   answer stays fluent, so no degeneration battery sees it, and it only
//   differs from the single-stream answer that every benchmark measures.
//
// WHAT IT IS NOT
//   Not a batch-invariance guarantee (imp deliberately has none,
//   docs/determinism.md): bucket n is never compared with bucket 1. Each arm is
//   compared with a PERMUTATION of itself (same prompts, same batch shape,
//   reversed slot assignment), so the GEMM tiles and the per-row reduction
//   order match and only the slot addressing can differ.
//
// GPU lane: needs a real hybrid checkpoint (IMP_TEST_MODEL with GDN layers).
// =============================================================================

#include <gtest/gtest.h>

#include "imp/imp.h"
#include "api/imp_internal.h"
#include "model/model.h"
#include "runtime/engine.h"
#include "runtime/request.h"
#include "test_models.h"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace imp {
namespace {

constexpr int kGen = 24;  // tokens per arm
constexpr int kMaxSteps = 8192;

// Ordinary prose, long enough to leave the low-entropy regime a two-word prompt
// would sit in (there every arm agrees for the wrong reason).
constexpr const char* kPrompt =
    "A decode step reads every weight of the model exactly once, so its floor is set by how many "
    "bytes the weights occupy divided by how fast the device can stream them. Recurrent layers "
    "carry a state that is cumulative over the whole prefix, which is why a cache of keys and "
    "values is not by itself enough to skip work. Explain the consequence for a server that "
    "batches requests:";

bool is_safetensors_dir(const std::string& p) { return p.size() < 5 || p.substr(p.size() - 5) != ".gguf"; }

class GdnGraphBucketTest : public ::testing::Test {
protected:
    void SetUp() override {
        // IMP_TEST_MODEL_GDN is the hybrid slot in the suite's model map; fall
        // back to the generic one so a run that points IMP_TEST_MODEL at a
        // hybrid still covers this, and skip below when it is dense.
        const char* var = std::getenv(imp_test::kEnvModelGdn) ? imp_test::kEnvModelGdn : imp_test::kEnvModel;
        const std::string path = imp_test::env_path(var);
        if (path.empty())
            GTEST_SKIP() << "Set " << imp_test::kEnvModelGdn << " to a hybrid (GDN/SSM) checkpoint";
        ASSERT_NO_FATAL_FAILURE(imp_test::require_readable(path.c_str(), var));
        const ImpModelFormat fmt = is_safetensors_dir(path) ? IMP_FORMAT_SAFETENSORS : IMP_FORMAT_GGUF;
        ASSERT_EQ(imp_model_load(path.c_str(), fmt, &model_), IMP_SUCCESS);
        if (!(model_ && model_->model && model_->model->config().ssm_inner_size > 0))
            GTEST_SKIP() << "SKIPPED ON A DENSE CHECKPOINT: " << path
                         << " has ssm_inner_size == 0, so there is no per-sequence recurrent slot "
                            "and no bucket to compare. Point IMP_TEST_MODEL_GDN at a GDN/SSM "
                            "checkpoint (the `test-e2e` target does).";

        ImpConfig cfg = imp_config_default();
        cfg.max_seq_len = 1024;
        cfg.max_batch_size = kMaxBatch;
        cfg.enable_cuda_graphs = 1;
        // Prefix caching OFF: the arms share a prompt on purpose, and a cache
        // hit would make the second arm skip the prefill the first one ran.
        // What is under test is the decode graph, not the cache.
        cfg.use_prefix_caching = 0;
        ASSERT_EQ(imp_context_create(model_, &cfg, &ctx_), IMP_SUCCESS);
        engine_ = ctx_->engine.get();
        ASSERT_NE(engine_, nullptr);

        int n = 0;
        tokens_.resize(2048);
        ASSERT_EQ(imp_tokenize(model_, kPrompt, tokens_.data(), &n, static_cast<int>(tokens_.size())),
                  IMP_SUCCESS);
        ASSERT_GT(n, 0);
        tokens_.resize(n);
    }

    void TearDown() override {
        if (ctx_)
            imp_context_free(ctx_);
        if (model_)
            imp_model_free(model_);
    }

    // N concurrent greedy requests carrying the SAME prompt. Returns one output
    // token vector per request, in submission order.
    std::vector<std::vector<int32_t>> run_concurrent(int n_seqs) {
        return run_concurrent(std::vector<std::vector<int32_t>>(n_seqs, tokens_));
    }

    // Prompt i for an n-way arm: the fixture prompt minus its last i tokens, so
    // every row carries different content and a different length.
    std::vector<std::vector<int32_t>> distinct_prompts(int n_seqs) const {
        std::vector<std::vector<int32_t>> out;
        for (int i = 0; i < n_seqs; ++i)
            out.emplace_back(tokens_.begin(), tokens_.end() - i);
        return out;
    }

    // One greedy request per prompt, all in flight together. Returns one output
    // token vector per request, in submission order.
    std::vector<std::vector<int32_t>> run_concurrent(const std::vector<std::vector<int32_t>>& prompts) {
        const int n_seqs = static_cast<int>(prompts.size());
        std::vector<std::shared_ptr<Request>> reqs;
        for (int i = 0; i < n_seqs; ++i) {
            auto req = std::make_shared<Request>();
            req->input_tokens = prompts[i];
            req->max_tokens = kGen;
            req->temperature = 0.0f;
            req->top_p = 1.0f;
            req->top_k = 0;
            req->ignore_eos = true;  // fixed length, so the arms are comparable
            req->status = RequestStatus::PENDING;
            reqs.push_back(req);
            engine_->add_request(req);
        }
        for (int step = 0; step < kMaxSteps; ++step) {
            bool all_done = true;
            for (const auto& r : reqs)
                if (r->status != RequestStatus::FINISHED && r->status != RequestStatus::CANCELLED)
                    all_done = false;
            if (all_done)
                break;
            (void)engine_->step();
        }
        std::vector<std::vector<int32_t>> out;
        for (const auto& r : reqs) {
            EXPECT_EQ(r->status, RequestStatus::FINISHED)
                << "a request in the " << n_seqs << "-way arm did not finish (status "
                << static_cast<int>(r->status) << ")";
            out.push_back(r->output_tokens);
        }
        return out;
    }

    static constexpr int kMaxBatch = 8;

    ImpModel model_ = nullptr;
    ImpContext ctx_ = nullptr;
    Engine* engine_ = nullptr;
    std::vector<int32_t> tokens_;
};

// Every bucket must address each row's own recurrent slab. The oracle is a
// permutation of the SAME n-way batch: n distinct prompts submitted in order
// and then in reverse order share the batch shape (so the GEMM tiles and the
// reduction order per row are the same) and differ only in which slot each
// request lands in. A bucket whose captured graph reads the slab of a
// neighbouring slot answers fluently and differently for one of the two
// orders. Comparing bucket n against the n=1 run is NOT a valid oracle: the
// batched GEMM path (M >= 2) is a different kernel from the M=1 GEMV and imp has
// no batch-invariance guarantee (docs/determinism.md); measured on
// Qwen3.5-4B-mxfp4 the n=4 and n=8 buckets leave the n=1 stream at token 16
// while every row inside a bucket agrees.
TEST_F(GdnGraphBucketTest, EveryBatchBucketAddressesItsOwnRecurrentSlot) {
    for (int n : {2, 4, 8}) {
        const auto prompts = distinct_prompts(n);
        auto reversed = prompts;
        std::reverse(reversed.begin(), reversed.end());

        const auto forward = run_concurrent(prompts);
        const auto backward = run_concurrent(reversed);
        ASSERT_EQ(static_cast<int>(forward.size()), n);
        ASSERT_EQ(static_cast<int>(backward.size()), n);
        for (int i = 0; i < n; ++i) {
            ASSERT_FALSE(forward[i].empty()) << "bucket " << n << " row " << i << " produced no tokens";
            EXPECT_EQ(forward[i], backward[n - 1 - i])
                << "batch bucket " << n << ", prompt " << i << ": the same prompt in the same " << n
                << "-way batch shape produced a different stream when it was submitted in slot "
                << (n - 1 - i) << " instead of slot " << i
                << ". Same GEMM shapes, same per-row reduction order: that is state addressing "
                   "(d_ssm_seq_slots_ / the captured slot scalar), not FP associativity.";
        }
    }
}

// The n == 1 graph captures the recurrent SLOT, not a pointer to a table, so a
// slot change has to force a recapture. After a batch has cycled slots through
// the pool, a fresh single request must still read its own state.
TEST_F(GdnGraphBucketTest, SingleSequenceIsUnchangedAfterASlotIsRecycled) {
    const auto ref = run_concurrent(1);
    ASSERT_FALSE(ref[0].empty());

    // Fill and drain the pool so the next single request is handed a slot some
    // other sequence used and left dirty.
    (void)run_concurrent(kMaxBatch);

    const auto after = run_concurrent(1);
    ASSERT_EQ(after.size(), 1u);
    EXPECT_EQ(after[0], ref[0])
        << "a single sequence taking a recycled recurrent slot diverged: either the slot was not "
           "zeroed on re-acquire, or the n=1 decode graph replayed a captured slot id that now "
           "belongs to nobody.";
}

}  // namespace
}  // namespace imp
