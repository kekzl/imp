// Speculative decoding may only move greedy output at a near-tie (batch-shape effect, SETTLED
// D-2/#1924: M=32 vs M=1 mean|dlogp| 0.242, max 1.636 nats, 7/64 tokens flip); wider divergence
// is a verify/accept defect. Oracle: plain vs mtp_k=2 or the DSpark drafter (ngram=false), margin at
// the first diverging token must lie inside the batch-shape envelope. GPU: needs the checkpoints.
#include "imp/imp.h"
#include "api/imp_internal.h"
#include "compute/json_constrain.h"
#include "model/chat_template.h"
#include "model/tokenizer.h"
#include "runtime/config.h"
#include "runtime/engine.h"
#include "runtime/request.h"
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
    // (AUDIT_QWEN38_NVFP4.md P7). Every run here must be cold.
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

bool chat_mode() {
    const char* v = getenv("IMP_TEST_MTP_CHAT");
    return v && *v == '1';
}

// Speculation gates refuse a request carrying logprobs (engine_spec_ngram.cpp,
// constrained_decode), so margins come from a logprobs run of the plain arm, valid only
// as far as its tokens agree with the logprobs-free run.
GreedyRun greedy_run(ImpModel model, ImpContext ctx, const char* prompt, int n_gen, bool want_margins,
                     bool json = false) {
    ImpGenerateParams params = imp_generate_params_default();
    params.temperature = 0.0f;
    params.top_k = 1;
    params.top_p = 1.0f;
    params.seed = 42;
    params.max_tokens = n_gen;
    params.logprobs = want_margins ? 1 : 0;
    params.top_logprobs = want_margins ? 2 : 0;
    params.json_mode = json ? 1 : 0;

    GreedyRun run;
    std::vector<int32_t> prompt_tokens(2048);
    int n_prompt = 0;
    if (chat_mode()) {
        const std::vector<imp::ChatMessage> msgs = {{"user", prompt}};
        prompt_tokens = ctx->engine->chat_template().apply(*model->model->tokenizer(), msgs);
    } else if (imp_tokenize(model, prompt, prompt_tokens.data(), &n_prompt, 2048) != IMP_SUCCESS) {
        return run;
    } else {
        prompt_tokens.resize(n_prompt);
    }
    n_prompt = static_cast<int>(prompt_tokens.size());

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
// IMP_TEST_MTP_CHAT=1: the prompts of scripts/mtp_accuracy_bench.sh, chat-templated (imp-cli path).
constexpr const char* kChatPrompts[] = {
    "What is the chemical formula for water, and what are its boiling and freezing points at standard "
    "atmospheric pressure?",
    "Explain why the sky appears blue during the day but red during sunset. Walk me through the physics step "
    "by step.",
    "Write a Python function that computes the nth Fibonacci number using dynamic programming. Include a "
    "docstring.",
    "Compose a polite email to a colleague asking them to review a draft document by end of week. Keep it "
    "under 80 words.",
};
std::vector<const char*> prompts() {
    if (chat_mode())
        return {std::begin(kChatPrompts), std::end(kChatPrompts)};
    return {std::begin(kPrompts), std::end(kPrompts)};
}

// Arm A: no speculation at all.
void plain_baseline(const std::string& dir, std::vector<GreedyRun>* baseline, bool json = false) {
    imp::set_pending_runtime_config(arm_config(0));
    ImpModel model = nullptr;
    ASSERT_EQ(imp_model_load_ex(dir.c_str(), IMP_FORMAT_SAFETENSORS, /*load_mtp_head=*/0, &model),
              IMP_SUCCESS);
    ImpConfig cfg = imp_config_default();
    cfg.max_seq_len = 2048;
    cfg.max_batch_size = 1;
    ImpContext ctx = nullptr;
    ASSERT_EQ(imp_context_create(model, &cfg, &ctx), IMP_SUCCESS);
    for (const char* p : prompts()) {
        const GreedyRun first = greedy_run(model, ctx, p, kGenTokens, /*want_margins=*/false, json);
        GreedyRun plain = greedy_run(model, ctx, p, kGenTokens, /*want_margins=*/false, json);
        const GreedyRun with_lp = greedy_run(model, ctx, p, kGenTokens, /*want_margins=*/true, json);
        // Same engine/params, deterministic, cold: neither a second run nor the logprobs flag may
        // move a token (AUDIT_QWEN38_NVFP4.md P7: trajectory depends on prior engine state).
        // Margins cannot be read off a different sequence.
        const auto report = [&](const GreedyRun& x, const GreedyRun& y) {
            size_t i = 0;
            while (i < std::min(x.tokens.size(), y.tokens.size()) && x.tokens[i] == y.tokens[i])
                ++i;
            std::printf("[spec-identity] arm A '%s': parts at token %zu of %zu/%zu", p, i, x.tokens.size(),
                        y.tokens.size());
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
                            with_lp.margin.begin() +
                                static_cast<std::ptrdiff_t>(std::min(lp_split, with_lp.margin.size())));
        baseline->push_back(std::move(plain));
    }
    imp_context_free(ctx);
    imp_model_free(model);
    for (size_t i = 0; i < baseline->size(); ++i) {
        ASSERT_GE((*baseline)[i].tokens.size(), 32u)
            << "prompt " << i << " produced " << (*baseline)[i].tokens.size()
            << " tokens; too few for the comparison to mean anything";
        ASSERT_GT((*baseline)[i].margin.size(), 0u)
            << "prompt " << i << ": the logprobs run returned no margins, the oracle has nothing to read";
    }
}

// First diverging token per prompt; its arm-A margin must lie inside the batch-shape envelope.
void expect_near_tie_only(const std::vector<GreedyRun>& baseline, const std::vector<GreedyRun>& speculated,
                          const char* arm) {
    for (size_t i = 0; i < baseline.size(); ++i) {
        const auto& a = baseline[i].tokens;
        const auto& b = speculated[i].tokens;
        const size_t n = std::min(a.size(), b.size());
        size_t first = 0;
        while (first < n && a[first] == b[first])
            ++first;
        if (first == n && a.size() == b.size()) {
            std::printf("[spec-identity] %s prompt %zu: %zu tokens identical\n", arm, i, n);
            continue;
        }
        ASSERT_LT(first, n) << "prompt " << i << ": one arm stopped early (" << a.size() << " vs " << b.size()
                            << " tokens) with an identical prefix";
        if (first >= baseline[i].margin.size()) {
            // The logprobs run parted from the plain run before this point, so
            // the margin here is unreadable: reported, not judged.
            std::printf(
                "[spec-identity] %s prompt %zu: parts at token %zu (plain %d, spec %d), margin "
                "unreadable (logprobs run parted at %zu)\n",
                arm, i, first, a[first], b[first], baseline[i].margin.size());
            continue;
        }
        const float margin = baseline[i].margin[first];
        std::printf(
            "[spec-identity] %s prompt %zu: parts at token %zu (plain %d, spec %d), margin %.3f nats\n", arm,
            i, first, a[first], b[first], margin);
        EXPECT_LE(margin, kBatchShapeMarginNats)
            << "prompt " << i << " (" << prompts()[i] << "): greedy output parted at token " << first
            << " (plain chose " << a[first] << ", " << arm << " chose " << b[first] << ") where the "
            << "plain arm's top-1/top-2 margin was " << margin << " nats.\n"
            << "The batch shape of the verify chunk can only flip a near-tie (SETTLED D-2, #1924: "
            << "max 1.636 nats per token). A flip at this margin is a verify/accept defect: an "
            << "accepted draft the model would not have sampled.";
    }
}

// json: both arms json_mode, so the MTP arm's verify keeps a row only if the mask allows it
// (speculative.constrained, engine_spec_constrained.cpp).
void mtp_identity(bool json) {
    const std::string dir = model_or("IMP_TEST_MODEL_MTP", "/models/Qwen3.8-27B-NVFP4-vllm");
    if (!present(dir))
        GTEST_SKIP() << "checkpoint not present at " << dir;
    // Two full loads back to back plus the head. An OOM here poisons every
    // later test in this binary (see the note in test_mtp_forward.cpp).
    constexpr size_t kNeededMiB = 24000;
    const size_t free_mib = device_free_mib();
    if (free_mib < kNeededMiB)
        GTEST_SKIP() << "needs ~" << kNeededMiB << " MiB free, card has " << free_mib << " MiB";

    std::vector<GreedyRun> baseline;
    plain_baseline(dir, &baseline, json);
    if (::testing::Test::HasFatalFailure())
        return;

    // ---- arm B: the documented MTP pair, depth 2 (IMP_TEST_MTP_K overrides: Qwen3.8-Flash-Next runs 1).
    const char* k_env = getenv("IMP_TEST_MTP_K");
    const int k = (k_env && *k_env) ? std::atoi(k_env) : 2;
    imp::set_pending_runtime_config(arm_config(k));
    ImpModel model = nullptr;
    // imp_model_load hard-codes load_mtp_head=0; the head is the point here.
    ASSERT_EQ(imp_model_load_ex(dir.c_str(), IMP_FORMAT_SAFETENSORS, /*load_mtp_head=*/1, &model),
              IMP_SUCCESS);
    ImpConfig cfg = imp_config_default();
    cfg.max_seq_len = 2048;
    cfg.max_batch_size = 1;
    ImpContext ctx = nullptr;
    ASSERT_EQ(imp_context_create(model, &cfg, &ctx), IMP_SUCCESS);
    if (imp_enable_mtp_spec_decode(ctx, k) != IMP_SUCCESS) {
        imp_context_free(ctx);
        imp_model_free(model);
        GTEST_SKIP() << "checkpoint at " << dir << " carries no loadable MTP head";
    }

    std::vector<GreedyRun> speculated;
    for (const char* p : prompts())
        speculated.push_back(greedy_run(model, ctx, p, kGenTokens, /*want_margins=*/false, json));

    // The margin oracle cannot see a mask violation (an unmasked first token passed it at 3.151
    // nats): every character the json arm emitted must advance the JSON grammar.
    for (size_t i = 0; json && i < speculated.size(); ++i) {
        std::vector<char> buf(16384, '\0');
        ASSERT_EQ(imp_detokenize(model, speculated[i].tokens.data(),
                                 static_cast<int>(speculated[i].tokens.size()), buf.data(), buf.size()),
                  IMP_SUCCESS);
        const std::string text(buf.data());
        imp::JsonGrammar g;
        g.reset();
        size_t bad = 0;
        while (bad < text.size() && g.advance_char(text[bad]))
            ++bad;
        const bool eos_after_done = g.done() && text.compare(bad, 2, "<|") == 0;  // detokenized EOS
        EXPECT_TRUE(bad == text.size() || eos_after_done)
            << "prompt " << i << ": the json arm left the grammar at char " << bad << " of \""
            << text.substr(0, 120) << "\"";
    }

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

    expect_near_tie_only(baseline, speculated, json ? "mtp-json" : "mtp");
}

TEST(MtpGreedyIdentityTest, MtpDoesNotChangeGreedyTokens) { mtp_identity(false); }

TEST(MtpGreedyIdentityTest, MtpDoesNotChangeConstrainedGreedyTokens) { mtp_identity(true); }

// Block drafter: DSpark 7 drafts per step (8-row verify), DFlash 15 (16-row verify).
void block_drafter_identity(const std::string& drafter, const char* tag) {
    const std::string dir = model_or("IMP_TEST_MODEL_DSPARK_TARGET", "/models/Qwen3-8B-NVFP4-cortecs");
    if (!present(dir) || !present(drafter))
        GTEST_SKIP() << "checkpoint not present at " << dir << " or " << drafter;
    constexpr size_t kNeededMiB = 12000;  // one 8B NVFP4 load at a time + 2 GB drafter
    const size_t free_mib = device_free_mib();
    if (free_mib < kNeededMiB)
        GTEST_SKIP() << "needs ~" << kNeededMiB << " MiB free, card has " << free_mib << " MiB";

    std::vector<GreedyRun> baseline;
    plain_baseline(dir, &baseline);
    if (::testing::Test::HasFatalFailure())
        return;

    imp::RuntimeConfig rc = arm_config(0);
    rc.speculative.dspark_model = drafter;
    imp::set_pending_runtime_config(rc);
    ImpModel model = nullptr;
    ASSERT_EQ(imp_model_load_ex(dir.c_str(), IMP_FORMAT_SAFETENSORS, /*load_mtp_head=*/0, &model),
              IMP_SUCCESS);
    ImpConfig cfg = imp_config_default();
    cfg.max_seq_len = 2048;
    cfg.max_batch_size = 1;
    ImpContext ctx = nullptr;
    ASSERT_EQ(imp_context_create(model, &cfg, &ctx), IMP_SUCCESS);
    std::vector<GreedyRun> speculated;
    for (const char* p : prompts())
        speculated.push_back(greedy_run(model, ctx, p, kGenTokens, /*want_margins=*/false));
    const auto stats = ctx->engine->spec_stats();
    imp_context_free(ctx);
    imp_model_free(model);

    ASSERT_GT(stats.drafted, 0) << tag << " drafted nothing: the plain path was compared against itself";
    ASSERT_GT(stats.accepted, 0) << tag << " drafted " << stats.drafted
                                 << " tokens and the verify accepted none";
    std::printf("[spec-identity] %s drafted=%lld accepted=%lld verify_steps=%lld\n", tag, stats.drafted,
                stats.accepted, stats.verify_steps);
    expect_near_tie_only(baseline, speculated, tag);
}

TEST(MtpGreedyIdentityTest, DSparkDoesNotChangeGreedyTokens) {
    block_drafter_identity(model_or("IMP_TEST_MODEL_DSPARK", "/models/dflash_qwen3_8b_block7"), "dspark");
}

TEST(MtpGreedyIdentityTest, DFlashDoesNotChangeGreedyTokens) {
    block_drafter_identity(model_or("IMP_TEST_MODEL_DFLASH", "/models/Qwen3-8B-DFlash-b16"), "dflash");
}

}  // namespace
