// Batched GDN decode (runtime.gdn_batched_decode) at the ENGINE level, greedy under
// runtime.deterministic. Asserted: eight different prompts admitted in reverse row order give
// every request the tokens it got in forward order (per-row plumbing: slot table, conv state,
// residual and alpha/beta paths, sampling state copies), and eight rows of ONE prompt decode
// identically. Mutation-validated 2026-09-15: row 1 reading row 0's state slot passes the
// one-prompt arm (same prompt, same state) and fails the reversed-order arm on 4 of 8 prompts.
// Printed only: how many rows differ from their solo run, which the M=1 (GEMV) vs small-M
// (GEMM) rounding makes a near-tie count, not a defect.
// Context: Qwen3.8-27B-NVFP4-vllm at 8 thinking sessions re-closed its think block and repeated
// the answer on 2-5 rows (#2019): the sampler's shared banned-list buffer, fixed and gated by
// SamplerRowBansTest; the instruments below (logit delta vs solo, server-shaped think repro)
// stay for the next quality question. Needs IMP_TEST_MODEL_GDN; runs from make test-e2e.

#include <gtest/gtest.h>

#include "api/imp_internal.h"
#include "imp/imp.h"
#include "model/model.h"
#include "model/tokenizer.h"
#include "runtime/config.h"
#include "runtime/engine.h"
#include "runtime/request.h"
#include "test_models.h"

#include <cmath>
#include <cstdlib>
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

TEST_F(HybridBatchedDecodeTest, RowsDecodeIndependentlyOfTheirOrder) {
    if (!(model_ && model_->model && model_->model->config().ssm_inner_size > 0))
        GTEST_SKIP() << "SKIPPED ON A DENSE CHECKPOINT: " << imp_test::kEnvModelGdn
                     << " points at a model with ssm_inner_size == 0.";

    RuntimeConfig rc;
    rc.runtime.deterministic = true;
    rc.server.prefix_cache = false;
    // IMP_TEST_SET="key=value,..." applies knob arms (same syntax as the instruments below).
    if (const char* kv = std::getenv("IMP_TEST_SET"); kv && *kv) {
        std::vector<std::string> sets;
        std::string s(kv);
        for (size_t p = 0; p <= s.size();) {
            const size_t q = std::min(s.find(',', p), s.size());
            if (q > p)
                sets.push_back(s.substr(p, q - p));
            p = q + 1;
        }
        const auto bad = rc.apply_overrides(sets);
        ASSERT_TRUE(bad.empty()) << "unknown IMP_TEST_SET key: " << bad[0];
    }
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

    // Row-order invariance, the half a slot mix-up cannot hide from: the same eight prompts
    // admitted in reverse order must give every request the tokens it got in the first batch.
    // Identical prompts share a state, so a row reading a neighbour's slot passes the check
    // above (mutant survived 2026-09-15); different prompts in a different row order do not.
    std::vector<std::shared_ptr<Request>> rev;
    for (int i = kRows - 1; i >= 0; --i)
        rev.push_back(make_req(i));
    for (auto& r : rev)
        engine->add_request(r);
    drain(rev);
    int order_mismatch = 0;
    for (int i = 0; i < kRows; ++i) {
        const auto& got = rev[static_cast<size_t>(kRows - 1 - i)]->output_tokens;  // prompt i
        const auto& ref = batch[static_cast<size_t>(i)]->output_tokens;
        size_t first = 0;
        while (first < got.size() && first < ref.size() && got[first] == ref[first])
            ++first;
        if (got != ref) {
            ++order_mismatch;
            std::printf("[batched] reversed order: prompt %d (row %d) diverges from row %d at token %zu\n", i,
                        kRows - 1 - i, i, first);
        }
    }
    EXPECT_EQ(order_mismatch, 0) << order_mismatch << " of " << kRows
                                 << " prompts decoded differently in reversed row order";
    std::printf("[batched] identical-prompt rows vs solo: %s\n",
                same[0]->output_tokens == solo[0] ? "row 0 equals solo" : "row 0 differs from solo");
}

// Magnitude instrument (diagnostic, asserts nothing): per row, the logprob delta batched vs solo
// at the prefill step (k=0, same kernels expected) and at the first decode step (k=1, same
// context, batched kernels vs the GEMV path), plus the solo top-1 margin where greedy flips.
// Rounding class is |delta| ~ 1e-2; a delta of the margin's size is a defect signal.
TEST_F(HybridBatchedDecodeTest, BatchedLogitDeltaVsSolo) {
    if (!(model_ && model_->model && model_->model->config().ssm_inner_size > 0))
        GTEST_SKIP() << "SKIPPED ON A DENSE CHECKPOINT";
    RuntimeConfig rc;
    rc.runtime.deterministic = true;
    rc.server.prefix_cache = false;
    // IMP_TEST_SET="gdn.ref_kernel=true,gemm.nvfp4_smallm=false": knob ablation with this oracle.
    if (const char* kv = std::getenv("IMP_TEST_SET"); kv && *kv) {
        std::vector<std::string> sets;
        std::string s(kv);
        for (size_t p = 0; p <= s.size();) {
            const size_t q = std::min(s.find(',', p), s.size());
            if (q > p)
                sets.push_back(s.substr(p, q - p));
            p = q + 1;
        }
        const auto bad = rc.apply_overrides(sets);
        ASSERT_TRUE(bad.empty()) << "unknown IMP_TEST_SET key: " << bad[0];
    }
    // IMP_TEST_DUMP=1: eager, prompt 0 only (solo + eight copies), IMP_TEST_STEPS decode tokens
    // (default 24), so diagnostics.dump_hidden_dir gets solo step01 vs batched step00 per layer.
    const bool dump_mode = std::getenv("IMP_TEST_DUMP") != nullptr;
    const int kSteps = std::getenv("IMP_TEST_STEPS") ? std::atoi(std::getenv("IMP_TEST_STEPS")) : 24;
    if (dump_mode)
        rc.runtime.cuda_graphs = "never";
    set_pending_runtime_config(rc);
    ImpConfig cfg = imp_config_default();
    cfg.max_seq_len = 2048;
    cfg.max_batch_size = kRows;
    cfg.enable_cuda_graphs = dump_mode ? 0 : 1;
    cfg.use_prefix_caching = 0;
    ASSERT_EQ(imp_context_create(model_, &cfg, &ctx_), IMP_SUCCESS);
    Engine* engine = ctx_->engine.get();
    ASSERT_NE(engine, nullptr);
    auto make_req = [&](int i) {
        auto req = std::make_shared<Request>();
        std::vector<int32_t> toks(4096);
        int n = 0;
        EXPECT_EQ(imp_tokenize(model_, kPrompts[i], toks.data(), &n, static_cast<int>(toks.size())),
                  IMP_SUCCESS);
        toks.resize(n > 0 ? n : 0);
        req->input_tokens = toks;
        req->max_tokens = kSteps;
        req->temperature = 0.0f;
        req->top_p = 1.0f;
        req->top_k = 0;
        req->ignore_eos = true;
        req->logprobs = true;
        req->top_logprobs = 20;
        req->status = RequestStatus::PENDING;
        return req;
    };
    auto drain = [&](std::vector<std::shared_ptr<Request>>& reqs) {
        for (int i = 0; i < 65536; ++i) {
            bool busy = false;
            for (auto& r : reqs)
                busy = busy ||
                       (r->status != RequestStatus::FINISHED && r->status != RequestStatus::CANCELLED);
            if (!busy)
                break;
            (void)engine->step();
        }
    };
    // Max |delta| over solo's top-20 that batched also lists, plus the delta of solo's top-1.
    auto delta = [](const TokenLogprobInfo& a, const TokenLogprobInfo& b, float* top1_delta) {
        float mx = 0.0f;
        *top1_delta = 0.0f;
        for (size_t i = 0; i < a.top.size(); ++i)
            for (const auto& t : b.top)
                if (t.token == a.top[i].token) {
                    const float d = std::fabs(t.logprob - a.top[i].logprob);
                    mx = std::max(mx, d);
                    if (i == 0)
                        *top1_delta = d;
                }
        return mx;
    };
    auto margin = [](const TokenLogprobInfo& a) {
        return a.top.size() >= 2 ? a.top[0].logprob - a.top[1].logprob : 99.0f;
    };
    std::vector<std::shared_ptr<Request>> solo;
    for (int i = 0; i < kRows; ++i) {
        std::vector<std::shared_ptr<Request>> one{make_req(dump_mode ? 0 : i)};
        engine->add_request(one[0]);
        drain(one);
        solo.push_back(one[0]);
        ASSERT_EQ(static_cast<int>(one[0]->output_logprobs.size()), kSteps);
    }
    auto report = [&](const char* label, std::vector<std::shared_ptr<Request>>& rows, int prompt_of_row) {
        for (int r = 0; r < kRows; ++r) {
            const int p = prompt_of_row < 0 ? r : prompt_of_row;
            const auto& s = *solo[static_cast<size_t>(p)];
            const auto& b = *rows[static_cast<size_t>(r)];
            ASSERT_EQ(static_cast<int>(b.output_logprobs.size()), kSteps);
            size_t first = 0;
            while (first < b.output_tokens.size() && first < s.output_tokens.size() &&
                   b.output_tokens[first] == s.output_tokens[first])
                ++first;
            float t0 = 0, t1 = 0, td = 0;
            const float d0 = delta(s.output_logprobs[0], b.output_logprobs[0], &t0);
            const float d1 = first >= 1 ? delta(s.output_logprobs[1], b.output_logprobs[1], &t1) : -1.0f;
            const bool flipped = first < static_cast<size_t>(kSteps);
            const float dd = flipped ? delta(s.output_logprobs[first], b.output_logprobs[first], &td) : -1.0f;
            std::printf(
                "[%s] row %d prompt %d: k0 max|d| %.4f top1 %.4f | k1 max|d| %.4f top1 %.4f margin %.3f | "
                "flip at %zu: solo margin %.3f max|d| %.4f top1 %.4f\n",
                label, r, p, d0, t0, d1, t1, margin(s.output_logprobs[1]), first,
                flipped ? margin(s.output_logprobs[first]) : -1.0f, dd, td);
        }
    };
    if (!dump_mode) {
        std::vector<std::shared_ptr<Request>> batch;
        for (int i = 0; i < kRows; ++i)
            batch.push_back(make_req(i));
        for (auto& r : batch)
            engine->add_request(r);
        drain(batch);
        report("mixed", batch, -1);
    }
    std::vector<std::shared_ptr<Request>> same;
    for (int i = 0; i < kRows; ++i)
        same.push_back(make_req(0));
    for (auto& r : same)
        engine->add_request(r);
    drain(same);
    report("same", same, 0);
    // Horizon trace, row 0 vs solo: every 8th step while the tokens still agree, then the flip.
    {
        const auto& s = *solo[0];
        const auto& b = *same[0];
        for (int k = 0; k < kSteps; ++k) {
            const bool agree = b.output_tokens[static_cast<size_t>(k)] ==
                               s.output_tokens[static_cast<size_t>(k)];
            float t1 = 0;
            const float d = delta(s.output_logprobs[static_cast<size_t>(k)],
                                  b.output_logprobs[static_cast<size_t>(k)], &t1);
            if (k % 8 == 0 || !agree)
                std::printf("[trace] k %d max|d| %.4f top1 %.4f solo margin %.3f %s\n", k, d, t1,
                            margin(s.output_logprobs[static_cast<size_t>(k)]), agree ? "" : "FLIP");
            if (!agree)
                break;
        }
    }
}

// Server-shaped repro of the #2019 re-close loop at the engine level: the probe's chat
// transcripts (system + user + assistant "<think>\n"), greedy, repetition_penalty 1.05,
// logprobs, think_budget 0.5 with the request primed in-think (the server's shape), eight
// rows in one batch vs their solo runs. Printed, not asserted: per row the </think> count,
// the token count, the first divergence from solo and the logprob delta there. The pre-fix
// sampler (one shared device copy of the banned list) shows here as closed rows that cannot
// stop while row 7 thinks: batched 574-1306 tokens against 319-644 fixed on Qwen3.8-27B,
// no re-close (the server probe re-closed 5 of 8; which symptom a row picks is a trajectory
// matter). SamplerRowBansTest is the gate. IMP_TEST_STEPS (default 1500) is max_tokens;
// IMP_TEST_STAGGER=1 admits one row every second step.
TEST_F(HybridBatchedDecodeTest, ThinkCloseRepeatsBatchedVsSolo) {
    if (!(model_ && model_->model && model_->model->config().ssm_inner_size > 0))
        GTEST_SKIP() << "SKIPPED ON A DENSE CHECKPOINT";
    RuntimeConfig rc;
    rc.runtime.deterministic = true;
    rc.server.prefix_cache = false;
    if (const char* kv = std::getenv("IMP_TEST_SET"); kv && *kv) {
        std::vector<std::string> sets;
        std::string s(kv);
        for (size_t p = 0; p <= s.size();) {
            const size_t q = std::min(s.find(',', p), s.size());
            if (q > p)
                sets.push_back(s.substr(p, q - p));
            p = q + 1;
        }
        const auto bad = rc.apply_overrides(sets);
        ASSERT_TRUE(bad.empty()) << "unknown IMP_TEST_SET key: " << bad[0];
    }
    set_pending_runtime_config(rc);
    ImpConfig cfg = imp_config_default();
    cfg.max_seq_len = 4096;
    cfg.max_batch_size = kRows;
    cfg.enable_cuda_graphs = 1;
    cfg.use_prefix_caching = 0;
    ASSERT_EQ(imp_context_create(model_, &cfg, &ctx_), IMP_SUCCESS);
    Engine* engine = ctx_->engine.get();
    ASSERT_NE(engine, nullptr);
    const int kSteps = std::getenv("IMP_TEST_STEPS") ? std::atoi(std::getenv("IMP_TEST_STEPS")) : 1500;
    const bool stagger = std::getenv("IMP_TEST_STAGGER") != nullptr;
    Tokenizer* tok = model_->model->tokenizer();
    ASSERT_NE(tok, nullptr);
    const auto think_end = tok->encode("</think>", true);
    ASSERT_EQ(think_end.size(), 1u);
    auto make_req = [&](int i) {
        auto req = std::make_shared<Request>();
        // Prompt 1 thinks longest; on rows 3 and 7 it keeps the LAST enqueued row in-think while the
        // others answer, the order the aliasing needed (the last upload won for every stashed row).
        // IMP_TEST_PROMPT_PAD=N prepends N filler sentences (~14 tokens each) to the user turn: long
        // prompts split at the snapshot boundary into a big chunk plus a 1-2 row tail chunk, the
        // shape the mixed hybrid step met in the 1k-prompt burst.
        std::string filler;
        if (const char* pad = std::getenv("IMP_TEST_PROMPT_PAD"))
            for (int k = 0; k < std::atoi(pad); ++k)
                filler +=
                    "A translation lookaside buffer caches recent page mappings so most loads skip the "
                    "walk. ";
        const std::string text = "<|im_start|>system\nSession " + std::to_string(i) +
                                 ". Answer briefly.<|im_end|>\n<|im_start|>user\n" + filler +
                                 kPrompts[(i + 2) % 4] + "<|im_end|>\n<|im_start|>assistant\n<think>\n";
        req->input_tokens = tok->encode(text, true);
        req->max_tokens = kSteps;
        req->temperature = 0.0f;
        req->top_p = 1.0f;
        req->top_k = 0;
        req->repetition_penalty = 1.05f;
        // IMP_TEST_NOLP=1: no logprobs, so the rows qualify as mixed-step riders
        // (prefill_ragged_req_ok_ refuses logprobs rows); the delta columns then read -1.
        req->logprobs = std::getenv("IMP_TEST_NOLP") == nullptr;
        req->top_logprobs = req->logprobs ? 5 : 0;
        // The server primes a thinking request this way (handlers_chat_core.cpp): the
        // budget's recount starts in-think and the sampler masks stop ids while thinking.
        req->think_budget = 0.5f;
        req->started_in_think = true;
        req->in_think_block = true;
        req->status = RequestStatus::PENDING;
        return req;
    };
    auto drain = [&](std::vector<std::shared_ptr<Request>>& reqs) {
        for (int i = 0; i < 65536; ++i) {
            bool busy = false;
            for (auto& r : reqs)
                busy = busy ||
                       (r->status != RequestStatus::FINISHED && r->status != RequestStatus::CANCELLED);
            if (!busy)
                break;
            (void)engine->step();
        }
    };
    auto closes = [&](const Request& r) {
        int n = 0;
        for (int32_t t : r.output_tokens)
            n += (t == think_end[0]);
        return n;
    };
    std::vector<std::shared_ptr<Request>> solo;
    for (int i = 0; i < kRows; ++i) {
        std::vector<std::shared_ptr<Request>> one{make_req(i)};
        engine->add_request(one[0]);
        drain(one);
        solo.push_back(one[0]);
    }
    std::vector<std::shared_ptr<Request>> batch;
    for (int i = 0; i < kRows; ++i)
        batch.push_back(make_req(i));
    if (stagger) {
        for (int i = 0; i < kRows; ++i) {
            engine->add_request(batch[static_cast<size_t>(i)]);
            (void)engine->step();
            (void)engine->step();
        }
    } else {
        for (auto& r : batch)
            engine->add_request(r);
    }
    drain(batch);
    std::printf("[think] prompt tokens %zu, first id %d, im_start single=%d\n", solo[0]->input_tokens.size(),
                solo[0]->input_tokens.empty() ? -1 : solo[0]->input_tokens[0],
                tok->encode("<|im_start|>", true).size() == 1 ? 1 : 0);
    int looped_batched = 0, looped_solo = 0;
    for (int r = 0; r < kRows; ++r) {
        const auto& s = *solo[static_cast<size_t>(r)];
        const auto& b = *batch[static_cast<size_t>(r)];
        size_t first = 0;
        while (first < b.output_tokens.size() && first < s.output_tokens.size() &&
               b.output_tokens[first] == s.output_tokens[first])
            ++first;
        const bool flipped = first < b.output_tokens.size() && first < s.output_tokens.size();
        float dmax = -1.0f, margin_s = -1.0f;
        if (flipped && first < s.output_logprobs.size() && first < b.output_logprobs.size()) {
            const auto& a = s.output_logprobs[first];
            const auto& c = b.output_logprobs[first];
            margin_s = a.top.size() >= 2 ? a.top[0].logprob - a.top[1].logprob : 99.0f;
            dmax = 0.0f;
            for (const auto& x : a.top)
                for (const auto& y : c.top)
                    if (x.token == y.token)
                        dmax = std::max(dmax, std::fabs(x.logprob - y.logprob));
        }
        const int cs = closes(s), cb = closes(b);
        looped_solo += cs > 1;
        looped_batched += cb > 1;
        // Single stream (async graph loop): the budget must close the block exactly once. Pre-fix
        // the loop relaunched between the forced "\n" and "</think>" and 4 of 8 rows never closed.
        EXPECT_EQ(cs, 1) << "solo row " << r << " must close its think block once under the budget";
        if (cb > 1 || cs > 1) {
            std::printf("[think]   </think> at solo:");
            for (size_t k = 0; k < s.output_tokens.size(); ++k)
                if (s.output_tokens[k] == think_end[0])
                    std::printf(" %zu", k);
            std::printf(" | batched:");
            for (size_t k = 0; k < b.output_tokens.size(); ++k)
                if (b.output_tokens[k] == think_end[0])
                    std::printf(" %zu", k);
            std::printf("\n");
        }
        std::printf(
            "[think] row %d: solo %zu tokens </think> x%d | batched %zu tokens </think> x%d | "
            "first diff %zu solo margin %.3f max|d| %.4f\n",
            r, s.output_tokens.size(), cs, b.output_tokens.size(), cb, first, margin_s, dmax);
    }
    std::printf("[think] rows closing more than once: solo %d, batched %d of %d\n", looped_solo,
                looped_batched, kRows);
    // Asserted for the pipeline path (IMP_TEST_NOLP=1): the think-budget force must land once
    // per row. Pre-fix the chained step repeated "\n" and "</think>" (8 of 8 rows closed at
    // 103 and 104 with a 100-token budget, solo once at 101); the eager path never did.
    if (looped_solo == 0)
        EXPECT_EQ(looped_batched, 0) << "batched rows closed their think block more than once";
}

}  // namespace imp
