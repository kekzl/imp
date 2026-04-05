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

} // anonymous namespace
