// =============================================================================
// test_mtp_greedy_identity.cpp - MTP must not change greedy output
// =============================================================================
//
// Speculative decoding is a SPEED optimisation: the verify step accepts a draft
// token only when it equals what the model would have sampled anyway, so at
// temperature 0 the token sequence with the head on must equal the sequence
// with it off. Nothing in this tree asserted that.
//
// What existed instead: test_spec_capture_fidelity.cpp compares a CAPTURED
// verify chunk against an EAGER forward of the same state - it gates the graph,
// not the speculation - and test_e2e_greedy_lock.cpp freezes token sequences
// without pinning `speculative.*` at all, so a lock recorded on one drafter
// configuration is silently re-run on another (docs/LIMITATIONS.md: "Golden
// tests must pin speculative.ngram and speculative.mtp_k"). Between them, an
// MTP defect that changes output rather than crashing - a mis-consumed row, an
// off-by-one in the accepted prefix, the banned-mask bug of #1796 - had no gate.
//
// This is the missing one: same prompt, same greedy params, mtp_k=0 against
// mtp_k=2 with ngram=false, 128 tokens, token-identical.
//
// GPU lane: needs the checkpoint and its MTP head (~0.79 GiB on top of the
// model). Skips when either is absent.
// =============================================================================

#include "imp/imp.h"
#include "api/imp_internal.h"
#include "runtime/config.h"
#include "runtime/engine.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
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

// The two arms differ in ONE pair of keys. Everything else that could move a
// token is pinned identically: no n-gram matcher, no suffix index, no token
// recycling, no economics give-up (a mid-run give-up would make the arms differ
// in WHICH steps speculated, which is exactly what must not matter).
imp::RuntimeConfig arm_config(int mtp_k) {
    imp::RuntimeConfig rc;
    rc.speculative.ngram = false;
    rc.speculative.suffix = false;
    rc.speculative.token_recycling = false;
    rc.speculative.mtp_k = mtp_k;
    rc.speculative.mtp_econ_min_emit = 0.0f;
    rc.speculative.give_up_after = 0;
    return rc;
}

constexpr int kGenTokens = 128;

// Token-level generation, not text: a detokenized comparison hides a token
// split that renders to the same string.
std::vector<int32_t> greedy_tokens(ImpModel model, ImpContext ctx, const char* prompt, int n_gen) {
    ImpGenerateParams params = imp_generate_params_default();
    params.temperature = 0.0f;
    params.top_k = 1;
    params.top_p = 1.0f;
    params.seed = 42;
    params.max_tokens = n_gen;

    std::vector<int32_t> prompt_tokens(2048);
    int n_prompt = 0;
    if (imp_tokenize(model, prompt, prompt_tokens.data(), &n_prompt, 2048) != IMP_SUCCESS)
        return {};
    prompt_tokens.resize(n_prompt);

    if (imp_context_reset(ctx) != IMP_SUCCESS)
        return {};
    if (imp_prefill_with_params(ctx, prompt_tokens.data(), n_prompt, &params) != IMP_SUCCESS)
        return {};

    std::vector<int32_t> out;
    out.reserve(n_gen);
    for (int i = 0; i < n_gen; i++) {
        int32_t tok = -1;
        if (imp_decode_step(ctx, &params, &tok) != IMP_SUCCESS || tok < 0)
            break;
        out.push_back(tok);
    }
    return out;
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
    std::vector<std::vector<int32_t>> baseline;
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
        for (const char* p : kPrompts)
            baseline.push_back(greedy_tokens(model, ctx, p, kGenTokens));
        imp_context_free(ctx);
        imp_model_free(model);
    }
    for (size_t i = 0; i < baseline.size(); ++i)
        ASSERT_GE(baseline[i].size(), 32u)
            << "prompt " << i << " produced " << baseline[i].size()
            << " tokens; too few for the comparison to mean anything";

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

    std::vector<std::vector<int32_t>> speculated;
    for (const char* p : kPrompts)
        speculated.push_back(greedy_tokens(model, ctx, p, kGenTokens));

    const auto stats = ctx->engine->spec_stats();
    imp_context_free(ctx);
    imp_model_free(model);

    // A green identity test proves nothing if the head never drafted: without
    // this the arms are the non-speculative path compared against itself
    // (#1321, the reason SpecStats exists at all).
    ASSERT_GT(stats.mtp.drafted, 0)
        << "the MTP head drafted nothing over " << (kGenTokens * 3)
        << " greedy tokens, so this test compared the plain decode path against itself";

    for (size_t i = 0; i < baseline.size(); ++i) {
        EXPECT_EQ(baseline[i], speculated[i])
            << "prompt " << i << " (" << kPrompts[i] << "): greedy output changed with MTP on.\n"
            << "  mtp_k=0: " << baseline[i].size() << " tokens\n"
            << "  mtp_k=2: " << speculated[i].size() << " tokens\n"
            << "Speculation is a speed optimisation: the verify step accepts a draft only when it "
               "equals the token the model would have sampled, so a divergence here is a defect in "
               "the verify/accept path, never an acceptable difference.";
    }
}

}  // namespace
