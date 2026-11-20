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
    // Logprob of the sampled token and the top-20 alternatives per step (logprobs run only).
    std::vector<float> lp;
    std::vector<std::vector<std::pair<int32_t, float>>> top;
    int32_t prefill_token = -1;
    std::string text;  // prefill token + tokens, detokenized
};

std::string detok(ImpModel model, const GreedyRun& run) {
    std::vector<int32_t> ids;
    if (run.prefill_token >= 0)
        ids.push_back(run.prefill_token);
    ids.insert(ids.end(), run.tokens.begin(), run.tokens.end());
    std::vector<char> buf(16384, '\0');
    if (imp_detokenize(model, ids.data(), static_cast<int>(ids.size()), buf.data(), buf.size()) !=
        IMP_SUCCESS)
        return "<detokenize failed>";
    return buf.data();
}

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
    params.top_logprobs = want_margins ? 20 : 0;
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

    // Output token 0 is sampled by the prefill: not in tokens (imp_decode_step starts at token 1,
    // output_logprobs has no row for it either), but the text needs it.
    run.prefill_token = ctx->prefill_token;
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
            run.lp.push_back(info.logprob);
            std::vector<std::pair<int32_t, float>> top;
            for (const auto& t : info.top)
                top.emplace_back(t.token, t.logprob);
            run.top.push_back(std::move(top));
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
// json arm: a prompt that does not ask for JSON closes after "[1]" (16 tokens on
// Qwen3.8-27B-NVFP4-vllm), below the 32-token floor.
constexpr const char* kJsonPrompts[] = {
    "A JSON array of twelve programming languages, each an object with the keys name, year, paradigm and "
    "a one-sentence description:\n",
    "A JSON object describing the eight planets of the solar system: for each planet its diameter in km, "
    "its number of moons and one sentence about its atmosphere:\n",
    "A JSON array with the first twenty prime numbers, each as an object with the keys index, value and "
    "is_twin_prime:\n",
};
// n-gram arms: the drafter matches against the prompt, so the continuation has to copy from it (the
// prose prompts drafted 13-32 tokens in 1-2 verify steps over 381 tokens).
constexpr const char* kCopyPrompts[] = {
    "Text: The paged KV cache stores keys and values in fixed-size blocks, so a request grows by "
    "appending a block instead of reallocating, and a finished request returns its blocks to the pool.\n"
    "The same text again, word for word, three times:\n",
    "def clamp(value, low, high):\n    if value < low:\n        return low\n    if value > high:\n"
    "        return high\n    return value\n\n# The same function again, renamed to clamp2, clamp3 and "
    "clamp4:\n",
    "| city | country | river |\n|---|---|---|\n| Paris | France | Seine |\n| Vienna | Austria | Danube |\n"
    "| Budapest | Hungary | Danube |\n| Cologne | Germany | Rhine |\n\nThe table repeated twice:\n",
};
const std::vector<const char*>* g_prompt_override = nullptr;
std::vector<const char*> prompts(bool json = false) {
    if (g_prompt_override != nullptr)
        return *g_prompt_override;
    if (json)
        return {std::begin(kJsonPrompts), std::end(kJsonPrompts)};
    if (chat_mode())
        return {std::begin(kChatPrompts), std::end(kChatPrompts)};
    return {std::begin(kPrompts), std::end(kPrompts)};
}

// Regret of a speculated sequence on the plain arm: the sequence is forced through the per-step decode
// route and each token scored against that step's argmax (top-1 logp minus the token's logp, 0 = the
// plain arm's own choice). Unlike the first-divergence check it covers every emitted token.
std::vector<float> forced_regret(ImpModel model, ImpContext ctx, const char* prompt, const GreedyRun& spec,
                                 bool json) {
    std::vector<float> gaps;
    std::vector<int32_t> seq;
    if (spec.prefill_token < 0 || spec.tokens.empty())
        return gaps;
    seq.push_back(spec.prefill_token);
    seq.insert(seq.end(), spec.tokens.begin(), spec.tokens.end());
    const int steps = static_cast<int>(seq.size()) - 1;

    ImpGenerateParams params = imp_generate_params_default();
    params.temperature = 0.0f;
    params.top_k = 1;
    params.top_p = 1.0f;
    params.seed = 42;
    params.ignore_eos = 1;
    params.max_tokens = steps + 2;  // prefill's token plus one spare keeps the request open
    params.logprobs = 1;
    params.top_logprobs = 2;
    params.json_mode = json ? 1 : 0;

    std::vector<int32_t> prompt_tokens(2048);
    int n_prompt = 0;
    if (chat_mode()) {
        const std::vector<imp::ChatMessage> msgs = {{"user", prompt}};
        prompt_tokens = ctx->engine->chat_template().apply(*model->model->tokenizer(), msgs);
    } else if (imp_tokenize(model, prompt, prompt_tokens.data(), &n_prompt, 2048) != IMP_SUCCESS) {
        return gaps;
    } else {
        prompt_tokens.resize(n_prompt);
    }
    if (imp_context_reset(ctx) != IMP_SUCCESS ||
        imp_prefill_with_params(ctx, prompt_tokens.data(), static_cast<int>(prompt_tokens.size()), &params) !=
            IMP_SUCCESS ||
        imp_set_forced_decode(ctx, seq.data(), static_cast<int>(seq.size())) != IMP_SUCCESS)
        return gaps;
    std::shared_ptr<imp::Request> req = ctx->active_request;
    for (int s = 0; s < steps; s++) {
        int32_t tok = -1;
        if (imp_decode_step(ctx, &params, &tok) != IMP_SUCCESS)
            break;
    }
    if (!req)
        return gaps;
    // Row s scores seq[s + 1]; the prefill ran without logprobs and left no row.
    for (const auto& info : req->output_logprobs)
        gaps.push_back(info.top.empty() ? 0.0f : std::max(0.0f, info.top[0].logprob - info.logprob));
    return gaps;
}

// Loads the plain arm again and prices every speculated token. The arm fails with more tokens above the
// batch-shape envelope than the plain arm's own output has: 0 unconstrained, and under json the mask
// moves the plain arm off its raw argmax too (budget-forced closer: 15.526 nats at token 123 of 127).
void expect_low_regret(const std::string& dir, const std::vector<GreedyRun>& baseline,
                       const std::vector<GreedyRun>& speculated, const char* arm, bool json = false) {
    imp::set_pending_runtime_config(arm_config(0));
    ImpModel model = nullptr;
    ASSERT_EQ(imp_model_load_ex(dir.c_str(), IMP_FORMAT_SAFETENSORS, /*load_mtp_head=*/0, &model),
              IMP_SUCCESS);
    ImpConfig cfg = imp_config_default();
    cfg.max_seq_len = 2048;
    cfg.max_batch_size = 1;
    ImpContext ctx = nullptr;
    ASSERT_EQ(imp_context_create(model, &cfg, &ctx), IMP_SUCCESS);
    const auto ps = prompts(json);
    for (size_t i = 0; i < speculated.size(); ++i) {
        const std::vector<float> gaps = forced_regret(model, ctx, ps[i], speculated[i], json);
        EXPECT_EQ(gaps.size(), speculated[i].tokens.size())
            << arm << " prompt " << i << ": the plain arm scored " << gaps.size() << " of "
            << speculated[i].tokens.size() << " speculated tokens";
        size_t worst = 0, above = 0;
        double sum = 0.0;
        for (size_t t = 0; t < gaps.size(); ++t) {
            sum += gaps[t];
            if (gaps[t] > gaps[worst])
                worst = t;
            if (gaps[t] > kBatchShapeMarginNats)
                ++above;
        }
        std::printf(
            "[spec-regret] %s prompt %zu: %zu tokens, mean %.4f nats, max %.3f at token %zu, %zu above "
            "%.3f\n",
            arm, i, gaps.size(), gaps.empty() ? 0.0 : sum / static_cast<double>(gaps.size()),
            gaps.empty() ? 0.0f : gaps[worst], worst, above, kBatchShapeMarginNats);
        size_t plain_above = 0;
        if (json) {
            for (const float g : forced_regret(model, ctx, ps[i], baseline[i], json))
                plain_above += g > kBatchShapeMarginNats ? 1 : 0;
            std::printf("[spec-regret] %s prompt %zu: plain arm's own output has %zu above\n", arm, i,
                        plain_above);
        }
        EXPECT_LE(above, plain_above) << arm << " prompt " << i << " (" << ps[i] << "): " << above
                                      << " speculated tokens sit more than " << kBatchShapeMarginNats
                                      << " nats below the plain arm's choice at their step (plain arm's own "
                                      << "output: " << plain_above << "), worst "
                                      << (gaps.empty() ? 0.0f : gaps[worst]) << " nats at token " << worst;
    }
    imp_context_free(ctx);
    imp_model_free(model);
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
    for (const char* p : prompts(json)) {
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
        const size_t keep = std::min(lp_split, with_lp.margin.size());
        plain.lp.assign(with_lp.lp.begin(), with_lp.lp.begin() + static_cast<std::ptrdiff_t>(keep));
        plain.top.assign(with_lp.top.begin(), with_lp.top.begin() + static_cast<std::ptrdiff_t>(keep));
        plain.text = detok(model, plain);
        baseline->push_back(std::move(plain));
    }
    imp_context_free(ctx);
    imp_model_free(model);
    for (size_t i = 0; i < baseline->size(); ++i) {
        ASSERT_GE((*baseline)[i].tokens.size(), 32u)
            << "prompt " << i << " produced " << (*baseline)[i].tokens.size()
            << " tokens; too few for the comparison to mean anything: \"" << (*baseline)[i].text << "\"";
        ASSERT_GT((*baseline)[i].margin.size(), 0u)
            << "prompt " << i << ": the logprobs run returned no margins, the oracle has nothing to read";
    }
}

// First diverging token per prompt; its arm-A margin must lie inside the batch-shape envelope.
void expect_near_tie_only(const std::vector<GreedyRun>& baseline, const std::vector<GreedyRun>& speculated,
                          const char* arm, bool json = false) {
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
        // Gap between the two choices as the plain arm scored them: lp(plain) - lp(spec). The spec
        // token outside the plain top-20 gives a lower bound (lp(plain) - lp(rank 20)).
        const auto& top = baseline[i].top[first];
        float lp_spec = top.empty() ? baseline[i].lp[first] : top.back().second;
        bool ranked = false;
        for (const auto& t : top) {
            if (t.first == b[first]) {
                lp_spec = t.second;
                ranked = true;
            }
        }
        const float gap = baseline[i].lp[first] - lp_spec;
        std::printf(
            "[spec-identity] %s prompt %zu: parts at token %zu (plain %d, spec %d), gap %s%.3f nats, "
            "plain top-1/top-2 margin %.3f\n",
            arm, i, first, a[first], b[first], ranked ? "" : ">= ", gap, baseline[i].margin[first]);
        EXPECT_LE(gap, kBatchShapeMarginNats)
            << "prompt " << i << " (" << prompts(json)[i] << "): greedy output parted at token " << first
            << " (plain chose " << a[first] << ", " << arm << " chose " << b[first] << "), which the plain "
            << "arm scored " << gap << " nats apart" << (ranked ? "" : " or more") << ".\n"
            << "The batch shape of the verify chunk can only flip a near-tie (SETTLED D-2, #1924: "
            << "max 1.636 nats per token). A flip at this gap is a verify/accept defect: an "
            << "accepted draft the model would not have sampled.";
    }
}

// json: both arms json_mode, so the MTP arm's verify keeps a row only if the mask allows it
// (engine_spec_constrained.cpp).
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
    for (const char* p : prompts(json))
        speculated.push_back(greedy_run(model, ctx, p, kGenTokens, /*want_margins=*/false, json));

    // The margin oracle cannot see a mask violation (an unmasked first token passed it at 3.151
    // nats): every character the json arm emitted must advance the JSON grammar.
    for (size_t i = 0; json && i < speculated.size(); ++i) {
        const std::string text = detok(model, speculated[i]);
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
    std::printf("[spec-identity] %s drafted=%lld accepted=%lld\n", json ? "mtp-json" : "mtp",
                static_cast<long long>(stats.mtp.drafted), static_cast<long long>(stats.mtp.accepted));
    ASSERT_GT(stats.mtp.drafted, 0)
        << "the MTP head drafted nothing over " << (kGenTokens * 3)
        << " greedy tokens, so this test compared the plain decode path against itself";
    ASSERT_GT(stats.mtp.accepted, 0)
        << "the MTP head drafted " << stats.mtp.drafted << " tokens and the verify accepted none";

    expect_near_tie_only(baseline, speculated, json ? "mtp-json" : "mtp", json);
    expect_low_regret(dir, baseline, speculated, json ? "mtp-json" : "mtp", json);
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
    expect_low_regret(dir, baseline, speculated, tag);
}

// n-gram drafter alone (no head, no block drafter): its chunks run past kSpecVerifyA16MaxRows rows.
void ngram_identity(const std::string& dir, const char* tag, size_t needed_mib) {
    if (!present(dir))
        GTEST_SKIP() << "checkpoint not present at " << dir;
    const size_t free_mib = device_free_mib();
    if (free_mib < needed_mib)
        GTEST_SKIP() << "needs ~" << needed_mib << " MiB free, card has " << free_mib << " MiB";

    const std::vector<const char*> copy_prompts(std::begin(kCopyPrompts), std::end(kCopyPrompts));
    struct Override {
        explicit Override(const std::vector<const char*>* p) { g_prompt_override = p; }
        ~Override() { g_prompt_override = nullptr; }
    } override_guard(&copy_prompts);

    std::vector<GreedyRun> baseline;
    plain_baseline(dir, &baseline);
    if (::testing::Test::HasFatalFailure())
        return;

    imp::RuntimeConfig rc = arm_config(0);
    rc.speculative.ngram = true;
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

    std::printf("[spec-identity] %s drafted=%lld accepted=%lld verify_steps=%lld\n", tag, stats.drafted,
                stats.accepted, stats.verify_steps);
    ASSERT_GE(stats.verify_steps, 10)
        << tag << " verified " << stats.verify_steps << " chunks over " << (kGenTokens * 3)
        << " tokens: the plain path was compared against itself";
    ASSERT_GT(stats.accepted, 0) << tag << " drafted " << stats.drafted
                                 << " tokens and the verify accepted none";
    expect_near_tie_only(baseline, speculated, tag);
    expect_low_regret(dir, baseline, speculated, tag);
}

TEST(MtpGreedyIdentityTest, NgramDoesNotChangeGreedyTokensDense) {
    ngram_identity(model_or("IMP_TEST_MODEL_DSPARK_TARGET", "/models/Qwen3-8B-NVFP4-cortecs"), "ngram-dense",
                   12000);
}

TEST(MtpGreedyIdentityTest, NgramDoesNotChangeGreedyTokensHybrid) {
    ngram_identity(model_or("IMP_TEST_MODEL_MTP", "/models/Qwen3.8-27B-NVFP4-vllm"), "ngram-hybrid", 24000);
}

TEST(MtpGreedyIdentityTest, DSparkDoesNotChangeGreedyTokens) {
    block_drafter_identity(model_or("IMP_TEST_MODEL_DSPARK", "/models/dflash_qwen3_8b_block7"), "dspark");
}

TEST(MtpGreedyIdentityTest, DFlashDoesNotChangeGreedyTokens) {
    block_drafter_identity(model_or("IMP_TEST_MODEL_DFLASH", "/models/Qwen3-8B-DFlash-b16"), "dflash");
}

}  // namespace
