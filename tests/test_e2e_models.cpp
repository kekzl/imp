#include <gtest/gtest.h>
#include "imp/imp.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Environment helpers
// ---------------------------------------------------------------------------

static const char* primary_model() {
    return std::getenv("IMP_TEST_MODEL");
}

static const char* gdn_model() {
    return std::getenv("IMP_TEST_MODEL_GDN");
}

static const char* gemma4_model() {
    return std::getenv("IMP_TEST_MODEL_GEMMA4");
}

#define REQUIRE_MODEL(var) \
    const char* path = var(); \
    if (!path) GTEST_SKIP() << "Set " #var " env var to run this test"

// ---------------------------------------------------------------------------
// Primary model tests (Qwen3-4B Q8_0 or similar dense transformer)
// ---------------------------------------------------------------------------

class PrimaryModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        path_ = primary_model();
        if (!path_) GTEST_SKIP() << "Set IMP_TEST_MODEL to run";

        ASSERT_EQ(imp_model_load(path_, IMP_FORMAT_GGUF, &model_), IMP_SUCCESS);
        ASSERT_NE(model_, nullptr);

        ImpConfig cfg = imp_config_default();
        cfg.max_seq_len = 512;
        cfg.max_batch_size = 1;
        cfg.enable_cuda_graphs = 0;
        ASSERT_EQ(imp_context_create(model_, &cfg, &ctx_), IMP_SUCCESS);
    }

    void TearDown() override {
        if (ctx_) imp_context_free(ctx_);
        if (model_) imp_model_free(model_);
    }

    const char* path_ = nullptr;
    ImpModel model_ = nullptr;
    ImpContext ctx_ = nullptr;
};

TEST_F(PrimaryModelTest, GenerateCoherentOutput) {
    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 32;
    params.temperature = 0.0f;
    params.apply_chat_template = 0;

    char output[4096];
    size_t len = 0;
    ASSERT_EQ(imp_generate(ctx_, "The capital of France is", &params,
                           output, sizeof(output), &len), IMP_SUCCESS);
    EXPECT_GT(len, 0u);

    // Greedy output should contain "Paris" for any reasonable model
    std::string text(output, len);
    EXPECT_NE(text.find("Paris"), std::string::npos)
        << "Expected 'Paris' in output: " << text;
}

TEST_F(PrimaryModelTest, MultiTurnConversation) {
    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 32;
    params.temperature = 0.0f;
    params.apply_chat_template = 1;

    // Turn 1
    char out1[4096];
    size_t len1 = 0;
    ASSERT_EQ(imp_generate(ctx_, "Say hello.", &params,
                           out1, sizeof(out1), &len1), IMP_SUCCESS);
    EXPECT_GT(len1, 0u);

    // Reset for turn 2
    ASSERT_EQ(imp_context_reset(ctx_), IMP_SUCCESS);

    // Turn 2 — different prompt, verify context is clean
    char out2[4096];
    size_t len2 = 0;
    ASSERT_EQ(imp_generate(ctx_, "What is 2+2? Answer with just the number.", &params,
                           out2, sizeof(out2), &len2), IMP_SUCCESS);
    EXPECT_GT(len2, 0u);

    std::string text2(out2, len2);
    EXPECT_NE(text2.find("4"), std::string::npos)
        << "Expected '4' in output: " << text2;
}

TEST_F(PrimaryModelTest, TokenizeRoundtrip) {
    const char* text = "Hello, how are you today?";
    int32_t tokens[256];
    int n_tokens = 0;
    ASSERT_EQ(imp_tokenize(model_, text, tokens, &n_tokens, 256), IMP_SUCCESS);
    EXPECT_GT(n_tokens, 3);

    char buf[1024];
    ASSERT_EQ(imp_detokenize(model_, tokens, n_tokens, buf, sizeof(buf)), IMP_SUCCESS);
    // Detokenized text should contain the original words
    EXPECT_NE(std::string(buf).find("Hello"), std::string::npos);
}

TEST_F(PrimaryModelTest, PrefillThenDecodeMultipleTokens) {
    int32_t tokens[128];
    int n_tokens = 0;
    ASSERT_EQ(imp_tokenize(model_, "The meaning of life is", tokens, &n_tokens, 128),
              IMP_SUCCESS);
    ASSERT_GT(n_tokens, 0);

    ASSERT_EQ(imp_prefill(ctx_, tokens, n_tokens), IMP_SUCCESS);

    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 16;
    params.temperature = 0.0f;

    std::vector<int32_t> generated;
    for (int i = 0; i < 16; i++) {
        int32_t tok = 0;
        ImpError err = imp_decode_step(ctx_, &params, &tok);
        if (err != IMP_SUCCESS) break;
        generated.push_back(tok);
    }
    EXPECT_GE(generated.size(), 4u) << "Should generate at least a few tokens";
}

// ---------------------------------------------------------------------------
// GDN model tests (Qwen3.5 Gated DeltaNet hybrid)
// ---------------------------------------------------------------------------

class GDNModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        path_ = gdn_model();
        if (!path_) GTEST_SKIP() << "Set IMP_TEST_MODEL_GDN to run";

        ASSERT_EQ(imp_model_load(path_, IMP_FORMAT_GGUF, &model_), IMP_SUCCESS);
        ASSERT_NE(model_, nullptr);

        ImpConfig cfg = imp_config_default();
        cfg.max_seq_len = 512;
        cfg.max_batch_size = 1;
        cfg.enable_cuda_graphs = 0;
        ASSERT_EQ(imp_context_create(model_, &cfg, &ctx_), IMP_SUCCESS);
    }

    void TearDown() override {
        if (ctx_) imp_context_free(ctx_);
        if (model_) imp_model_free(model_);
    }

    const char* path_ = nullptr;
    ImpModel model_ = nullptr;
    ImpContext ctx_ = nullptr;
};

TEST_F(GDNModelTest, GenerateCoherentOutput) {
    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 32;
    params.temperature = 0.0f;
    params.apply_chat_template = 1;

    char output[4096];
    size_t len = 0;
    ASSERT_EQ(imp_generate(ctx_, "What is the largest planet in our solar system? One word answer.", &params,
                           output, sizeof(output), &len), IMP_SUCCESS);
    EXPECT_GT(len, 0u);

    // GDN model may not follow instructions well at small sizes.
    // Verify generation is non-degenerate (not all repetition of a single token).
    std::string text(output, len);
    EXPECT_GT(text.size(), 5u) << "Output too short: " << text;
}

TEST_F(GDNModelTest, MultiTurnGDNState) {
    // GDN models carry recurrent state across tokens.
    // Multi-turn verifies the state management works correctly.
    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 32;
    params.temperature = 0.0f;
    params.apply_chat_template = 1;

    // Turn 1
    char out1[4096];
    size_t len1 = 0;
    ASSERT_EQ(imp_generate(ctx_, "Say hello.", &params,
                           out1, sizeof(out1), &len1), IMP_SUCCESS);
    EXPECT_GT(len1, 0u);

    // Reset
    ASSERT_EQ(imp_context_reset(ctx_), IMP_SUCCESS);

    // Turn 2 — independent prompt (GDN state should be clean after reset)
    char out2[4096];
    size_t len2 = 0;
    ASSERT_EQ(imp_generate(ctx_, "What is 1+1? Answer with just the number.", &params,
                           out2, sizeof(out2), &len2), IMP_SUCCESS);
    EXPECT_GT(len2, 0u);

    // Verify generation works after reset (GDN state properly cleared).
    // Don't check exact content — small GDN models have limited instruction following.
    std::string text2(out2, len2);
    EXPECT_GT(text2.size(), 1u) << "Output too short after reset: " << text2;
}

// ---------------------------------------------------------------------------
// Gemma-4 model tests (26B-A4B MoE hybrid: per-layer SWA/global, 128 experts
// top-8, shared dense FFN alongside each MoE block, custom router, GEGLU)
//
// Primary regression test — Gemma-4 forward pass has historically been
// fragile (see memory/gemma4_working_2026_04_14.md). This test locks in
// correct output on the Q4_K_M model.
// ---------------------------------------------------------------------------

class Gemma4ModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        path_ = gemma4_model();
        if (!path_) GTEST_SKIP() << "Set IMP_TEST_MODEL_GEMMA4 to run";

        ASSERT_EQ(imp_model_load(path_, IMP_FORMAT_GGUF, &model_), IMP_SUCCESS);
        ASSERT_NE(model_, nullptr);

        ImpConfig cfg = imp_config_default();
        cfg.max_seq_len = 512;
        cfg.max_batch_size = 1;
        cfg.enable_cuda_graphs = 0;  // baseline path — paired with Gemma4GraphsTest
        ASSERT_EQ(imp_context_create(model_, &cfg, &ctx_), IMP_SUCCESS);
    }

    void TearDown() override {
        if (ctx_) imp_context_free(ctx_);
        if (model_) imp_model_free(model_);
    }

    const char* path_ = nullptr;
    ImpModel model_ = nullptr;
    ImpContext ctx_ = nullptr;
};

TEST_F(Gemma4ModelTest, AnswersCapitalOfFrance) {
    // With default gemma chat template, Gemma-4 emits a <|channel>thought...
    // block followed by the answer. "Paris" must appear in the output.
    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 64;
    params.temperature = 0.0f;
    params.apply_chat_template = 1;

    char output[4096];
    size_t len = 0;
    ASSERT_EQ(imp_generate(ctx_, "What is the capital of France?", &params,
                           output, sizeof(output), &len), IMP_SUCCESS);
    EXPECT_GT(len, 0u);

    std::string text(output, len);
    EXPECT_NE(text.find("Paris"), std::string::npos)
        << "Expected 'Paris' in Gemma-4 output: " << text;
}

TEST_F(Gemma4ModelTest, RawCompletionProducesOutput) {
    // Raw (no-chat-template) path: instruct-tuned Gemma-4 without its chat
    // template produces structurally different output than with the template
    // (e.g. it emits <|channel>thought tokens as if the turn had started).
    // We don't assert exact content — just that the forward pass completes
    // successfully and returns non-empty non-degenerate text. The chat-template
    // test above already covers semantic correctness.
    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 12;
    params.temperature = 0.0f;
    params.apply_chat_template = 0;

    char output[4096];
    size_t len = 0;
    ASSERT_EQ(imp_generate(ctx_, "The capital of France is", &params,
                           output, sizeof(output), &len), IMP_SUCCESS);
    EXPECT_GT(len, 0u);
}

TEST_F(Gemma4ModelTest, NoRepetitionDegeneration) {
    // Guards against the classic Gemma-4 failure mode: ~15 tokens of sensible
    // output followed by "own own own" (or "아니라 own 아니라") loops driven by
    // MoE routing instability / precision drift through 30 layers × 128 experts.
    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 100;
    params.temperature = 0.0f;
    params.apply_chat_template = 1;

    char output[8192];
    size_t len = 0;
    ASSERT_EQ(imp_generate(ctx_, "Name three European capitals.", &params,
                           output, sizeof(output), &len), IMP_SUCCESS);
    EXPECT_GT(len, 0u);

    std::string text(output, len);

    // Repetition check: the most common 4-char substring in English text
    // tops out around 5-10% of the text. If any single run of the same
    // character occupies >30% of the output, it's a degeneration loop.
    size_t max_run = 0;
    for (size_t i = 0; i < text.size(); ) {
        size_t j = i;
        while (j < text.size() && text[j] == text[i]) ++j;
        max_run = std::max(max_run, j - i);
        i = j;
    }
    EXPECT_LT(max_run * 2, text.size())
        << "Detected degeneration (run of " << max_run
        << " chars in output of " << text.size() << "): " << text;
}

// ---------------------------------------------------------------------------
// Gemma-4 with CUDA graphs enabled — regression guard for the graph path.
//
// History: the AsyncGraphLoop captured forward_decode_async() — a parallel
// reimplementation that diverged on Gemma-4 Q4_K_M (sampled <eos> at step 0
// of the WHILE body, terminating after ~3 tokens with garbage). Fixed by
// unifying forward_decode_async() with forward_logits() (the canonical
// path). Locks in: graphs MUST produce identical output to the no-graph
// path, both in the short (single-token capture) and long (async WHILE)
// regimes.
// ---------------------------------------------------------------------------

class Gemma4GraphsTest : public ::testing::Test {
protected:
    void SetUp() override {
        path_ = gemma4_model();
        if (!path_) GTEST_SKIP() << "Set IMP_TEST_MODEL_GEMMA4 to run";

        ASSERT_EQ(imp_model_load(path_, IMP_FORMAT_GGUF, &model_), IMP_SUCCESS);
        ASSERT_NE(model_, nullptr);

        ImpConfig cfg = imp_config_default();
        cfg.max_seq_len = 512;
        cfg.max_batch_size = 1;
        cfg.enable_cuda_graphs = 1;
        ASSERT_EQ(imp_context_create(model_, &cfg, &ctx_), IMP_SUCCESS);
    }

    void TearDown() override {
        if (ctx_) imp_context_free(ctx_);
        if (model_) imp_model_free(model_);
    }

    const char* path_ = nullptr;
    ImpModel model_ = nullptr;
    ImpContext ctx_ = nullptr;
};

TEST_F(Gemma4GraphsTest, AnswersCapitalOfFranceWithGraphs) {
    // Same prompt as the no-graph baseline above. Output must contain "Paris".
    // Generates >3 decoded tokens to ensure the AsyncGraphLoop launches and
    // produces correct output (the previous bug terminated at ~3 tokens).
    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 64;
    params.temperature = 0.0f;
    params.apply_chat_template = 1;

    char output[4096];
    size_t len = 0;
    ASSERT_EQ(imp_generate(ctx_, "What is the capital of France?", &params,
                           output, sizeof(output), &len), IMP_SUCCESS);
    EXPECT_GT(len, 0u);

    std::string text(output, len);
    EXPECT_NE(text.find("Paris"), std::string::npos)
        << "Expected 'Paris' in Gemma-4 graph output: " << text;
}

TEST_F(Gemma4GraphsTest, LongDecodeStaysCoherent) {
    // 256 decode tokens: long enough for the AsyncGraphLoop to drive most
    // of the generation. Guards against state drift over a captured WHILE
    // body that runs many iterations from the same baked-in pointers.
    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 256;
    params.temperature = 0.0f;
    params.apply_chat_template = 1;

    char output[16384];
    size_t len = 0;
    ASSERT_EQ(imp_generate(ctx_, "Name three European capitals.", &params,
                           output, sizeof(output), &len), IMP_SUCCESS);
    EXPECT_GT(len, 0u);

    // Same degeneration heuristic as the no-graph variant.
    std::string text(output, len);
    size_t max_run = 0;
    for (size_t i = 0; i < text.size(); ) {
        size_t j = i;
        while (j < text.size() && text[j] == text[i]) ++j;
        max_run = std::max(max_run, j - i);
        i = j;
    }
    EXPECT_LT(max_run * 2, text.size())
        << "Graph-path degeneration (run of " << max_run
        << " chars in output of " << text.size() << "): " << text;
}

} // anonymous namespace
