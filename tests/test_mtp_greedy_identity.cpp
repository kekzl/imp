// Speculative decoding may only move greedy output at a near-tie (batch-shape effect, SETTLED
// D-2/#1924: M=32 vs M=1 mean|dlogp| 0.242, max 1.636 nats, 7/64 tokens flip); wider divergence
// is a verify/accept defect. Oracle: mtp_k=0 vs mtp_k=2 (ngram=false), margin at the first
// diverging token must lie inside the batch-shape envelope. GPU: needs checkpoint + MTP head.
#include "imp/imp.h"
#include "api/imp_internal.h"
#include "runtime/config.h"
#include "runtime/engine.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <utility>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

const char* model_or(const char* env, const char* fallback) {
    const char* v = getenv(env);
    return (v && *v) ? v : fallback;
}

bool present(const std::string& dir) { return fs::exists(dir + "/config.json"); }

size_t device_free_mib() {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess)
        return 0;
    return free_b >> 20;
}

// Both arms deterministic: without it the forward itself is not reproducible
// (93.6 % of floats differ between two runs of the same binary) and a token
// comparison measures nothing.
imp::RuntimeConfig arm_config(int mtp_k) {
    imp::RuntimeConfig rc;
    rc.runtime.deterministic = true;
    // Prefix cache disabled: a request reaching max_tokens registers block hashes, so a second
    // run of the same prompt would prefill at a different chunk shape, moving a near-tie
    // (AUDIT_qwen38_nvfp4.md P7). Every run here must be cold.
    rc.server.prefix_cache = false;
    rc.speculative.ngram = false;
    rc.speculative.suffix = false;
    rc.speculative.token_recycling = false;
    rc.speculative.mtp_k = mtp_k;
    rc.speculative.mtp_econ_min_emit = 0.0f;
    rc.speculative.give_up_after = 0;
    return rc;
}

constexpr int kGenTokens = 128;

// SETTLED D-2: batch shape moves a logp by up to 1.636 nats (M=32 vs M=1). A flip needs
// top-1 and top-2 to move toward each other, so the bridgeable margin is twice that;
// anything wider is not the batch shape.
constexpr float kBatchShapeMarginNats = 2.0f * 1.636f;

struct GreedyRun {
    std::vector<int32_t> tokens;
    // top-1 minus top-2 logp at each step, as this arm saw it. +inf when the
    // engine returned fewer than two alternatives for a step.
    std::vector<float> margin;
};

// Speculation gates refuse a request carrying logprobs (engine_spec_ngram.cpp,
// constrained_decode), so margins come from a logprobs run of the plain arm, valid only
// as far as its tokens agree with the logprobs-free run.
GreedyRun greedy_run(ImpModel model, ImpContext ctx, const char* prompt, int n_gen, bool want_margins) {
    ImpGenerateParams params = imp_generate_params_default();
    params.temperature = 0.0f;
    params.top_k = 1;
    params.top_p = 1.0f;
    params.seed = 42;
    params.max_tokens = n_gen;
    params.logprobs = want_margins ? 1 : 0;
    params.top_logprobs = want_margins ? 2 : 0;

    GreedyRun run;
    std::vector<int32_t> prompt_tokens(2048);
    int n_prompt = 0;
    if (imp_tokenize(model, prompt, prompt_tokens.data(), &n_prompt, 2048) != IMP_SUCCESS)
        return run;
    prompt_tokens.resize(n_prompt);

    if (imp_context_reset(ctx) != IMP_SUCCESS)
        return run;
    if (imp_prefill_with_params(ctx, prompt_tokens.data(), n_prompt, &params) != IMP_SUCCESS)
        return run;
    // Hold the request: the C API drops active_request the moment the request
    // finishes (max_tokens reached inside the last decode_step), and
    // output_logprobs lives on the request.
    std::shared_ptr<imp::Request> req = ctx->active_request;

    run.tokens.reserve(n_gen);
    for (int i = 0; i < n_gen; i++) {
        int32_t tok = -1;
        if (imp_decode_step(ctx, &params, &tok) != IMP_SUCCESS || tok < 0)
            break;
        run.tokens.push_back(tok);
    }
    // output_logprobs is parallel to output_tokens on the request that produced
    // them.
    if (req) {
        for (const auto& info : req->output_logprobs) {
            float m = std::numeric_limits<float>::infinity();
            if (info.top.size() >= 2)
                m = info.top[0].logprob - info.top[1].logprob;
            run.margin.push_back(m);
        }
    }
    return run;
}

// Prompts long enough for the chain to engage repeatedly (the head drafts off
// the previous step's hidden state, so a three-token answer proves nothing) and
// varied enough that one of them is draft-poor.
constexpr const char* kPrompts[] = {
    "Explain how a paged KV cache works in an LLM inference engine, and why block size matters.",
    "Q: What is 17 + 25? Show the addition digit by digit.\nA:",
    "List the first ten prime numbers, separated by commas.",
};

TEST(MtpGreedyIdentityTest, MtpDoesNotChangeGreedyTokens) {
    const std::string dir = model_or("IMP_TEST_MODEL_MTP", "/models/Qwen3.8-27B-NVFP4-vllm");
    if (!present(dir))
        GTEST_SKIP() << "checkpoint not present at " << dir;
    // Two full loads back to back plus the head. An OOM here poisons every
    // later test in this binary (see the note in test_mtp_forward.cpp).
    constexpr size_t kNeededMiB = 24000;
    const size_t free_mib = device_free_mib();
    if (free_mib < kNeededMiB)
        GTEST_SKIP() << "needs ~" << kNeededMiB << " MiB free, card has " << free_mib << " MiB";

    // ---- arm A: no speculation at all.
    std::vector<GreedyRun> baseline;
    {
        imp::set_pending_runtime_config(arm_config(0));
        ImpModel model = nullptr;
        ASSERT_EQ(imp_model_load_ex(dir.c_str(), IMP_FORMAT_SAFETENSORS, /*load_mtp_head=*/0, &model),
                  IMP_SUCCESS);
        ImpConfig cfg = imp_config_default();
        cfg.max_seq_len = 2048;
        cfg.max_batch_size = 1;
        ImpContext ctx = nullptr;
        ASSERT_EQ(imp_context_create(model, &cfg, &ctx), IMP_SUCCESS);
        for (const char* p : kPrompts) {
            const GreedyRun first = greedy_run(model, ctx, p, kGenTokens, /*want_margins=*/false);
            GreedyRun plain = greedy_run(model, ctx, p, kGenTokens, /*want_margins=*/false);
            const GreedyRun with_lp = greedy_run(model, ctx, p, kGenTokens, /*want_margins=*/true);
            // Same engine/params, deterministic, cold: neither a second run nor the logprobs flag may
            // move a token (AUDIT_qwen38_nvfp4.md P7: trajectory depends on prior engine state).
            // Margins cannot be read off a different sequence.
            const auto report = [&](const GreedyRun& x, const GreedyRun& y) {
                size_t i = 0;
                while (i < std::min(x.tokens.size(), y.tokens.size()) && x.tokens[i] == y.tokens[i])
                    ++i;
                std::printf("[mtp-identity] arm A '%s': parts at token %zu of %zu/%zu", p, i,
                            x.tokens.size(), y.tokens.size());
                if (i < x.tokens.size() && i < y.tokens.size())
                    std::printf(" (%d vs %d)", x.tokens[i], y.tokens[i]);
                if (i < y.margin.size())
                    std::printf(", logprobs-run margin there %.3f nats", y.margin[i]);
                std::printf("\n");
            };
            if (first.tokens != plain.tokens)
                report(first, plain);
            ASSERT_EQ(first.tokens, plain.tokens)
                << "prompt " << p << ": two cold deterministic runs of the plain arm differ";
            // Logprobs run is a different LM-head/sampling path and parts from the plain run at a
            // near-tie (measured on Qwen3.8-27B-NVFP4: token 81 of 127, 0.086 nats). Its margins equal
            // the plain run's exactly as far as the two agree, unreadable past that point.
            size_t lp_split = 0;
            while (lp_split < std::min(plain.tokens.size(), with_lp.tokens.size()) &&
                   plain.tokens[lp_split] == with_lp.tokens[lp_split])
                ++lp_split;
            if (lp_split < plain.tokens.size())
                report(plain, with_lp);
            plain.margin.assign(with_lp.margin.begin(),
                                with_lp.margin.begin() + static_cast<std::ptrdiff_t>(
                                                            std::min(lp_split, with_lp.margin.size())));
            baseline.push_back(std::move(plain));
        }
        imp_context_free(ctx);
        imp_model_free(model);
    }
    for (size_t i = 0; i < baseline.size(); ++i) {
        ASSERT_GE(baseline[i].tokens.size(), 32u)
            << "prompt " << i << " produced " << baseline[i].tokens.size()
            << " tokens; too few for the comparison to mean anything";
        ASSERT_GT(baseline[i].margin.size(), 0u)
            << "prompt " << i << ": the logprobs run returned no margins, the oracle has nothing to read";
    }

    // ---- arm B: the documented MTP pair, depth 2.
    imp::set_pending_runtime_config(arm_config(2));
    ImpModel model = nullptr;
    // imp_model_load hard-codes load_mtp_head=0; the head is the point here.
    ASSERT_EQ(imp_model_load_ex(dir.c_str(), IMP_FORMAT_SAFETENSORS, /*load_mtp_head=*/1, &model),
              IMP_SUCCESS);
    ImpConfig cfg = imp_config_default();
    cfg.max_seq_len = 2048;
    cfg.max_batch_size = 1;
    ImpContext ctx = nullptr;
    ASSERT_EQ(imp_context_create(model, &cfg, &ctx), IMP_SUCCESS);
    if (imp_enable_mtp_spec_decode(ctx, 2) != IMP_SUCCESS) {
        imp_context_free(ctx);
        imp_model_free(model);
        GTEST_SKIP() << "checkpoint at " << dir << " carries no loadable MTP head";
    }

    std::vector<GreedyRun> speculated;
    for (const char* p : kPrompts)
        speculated.push_back(greedy_run(model, ctx, p, kGenTokens, /*want_margins=*/false));

    const auto stats = ctx->engine->spec_stats();
    imp_context_free(ctx);
    imp_model_free(model);

    // A green test proves nothing if the head never drafted (arms would be the non-speculative
    // path vs itself, #1321, why SpecStats exists) or if verify accepts nothing (plain path with
    // extra work, invisible to any token comparison).
    ASSERT_GT(stats.mtp.drafted, 0)
        << "the MTP head drafted nothing over " << (kGenTokens * 3)
        << " greedy tokens, so this test compared the plain decode path against itself";
    ASSERT_GT(stats.mtp.accepted, 0)
        << "the MTP head drafted " << stats.mtp.drafted << " tokens and the verify accepted none";

    for (size_t i = 0; i < baseline.size(); ++i) {
        const auto& a = baseline[i].tokens;
        const auto& b = speculated[i].tokens;
        const size_t n = std::min(a.size(), b.size());
        size_t first = 0;
        while (first < n && a[first] == b[first])
            ++first;
        if (first == n && a.size() == b.size()) {
            std::printf("[mtp-identity] prompt %zu: %zu tokens identical\n", i, n);
            continue;
        }
        ASSERT_LT(first, n) << "prompt " << i << ": one arm stopped early (" << a.size() << " vs "
                            << b.size() << " tokens) with an identical prefix";
        if (first >= baseline[i].margin.size()) {
            // The logprobs run parted from the plain run before this point, so
            // the margin here is unreadable: reported, not judged.
            std::printf("[mtp-identity] prompt %zu: parts at token %zu (mtp_k=0 %d, mtp_k=2 %d), margin "
                        "unreadable (logprobs run parted at %zu)\n",
                        i, first, a[first], b[first], baseline[i].margin.size());
            continue;
        }
        const float margin = baseline[i].margin[first];
        std::printf("[mtp-identity] prompt %zu: parts at token %zu (mtp_k=0 %d, mtp_k=2 %d), margin %.3f nats\n",
                    i, first, a[first], b[first], margin);
        EXPECT_LE(margin, kBatchShapeMarginNats)
            << "prompt " << i << " (" << kPrompts[i] << "): greedy output parted at token " << first
            << " (mtp_k=0 chose " << a[first] << ", mtp_k=2 chose " << b[first] << ") where the "
            << "mtp_k=0 arm's top-1/top-2 margin was " << margin << " nats.\n"
            << "The batch shape of the verify chunk can only flip a near-tie (SETTLED D-2, #1924: "
            << "max 1.636 nats per token). A flip at this margin is a verify/accept defect: an "
            << "accepted draft the model would not have sampled.";
    }
}

}  // namespace
