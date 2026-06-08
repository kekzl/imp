// Test for output degeneration detection.
//
// Detects repetition loops, empty outputs, and gibberish that indicate
// broken inference pipelines (wrong state management, quantization errors,
// numerical instability).
//
// These tests require a real model and GPU. Skip via IMP_TEST_MODEL env var
// or run with: imp-tests --gtest_filter="DegenerationTest.*"

#include <gtest/gtest.h>
#include "imp/imp.h"
#include "test_models.h"

#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

namespace {

static const char* get_model_path() {
    return imp_test::env_cstr_or(imp_test::kEnvModel, "/models/Qwen3-8B-Q8_0.gguf");
}

static bool model_exists() {
    FILE* f = fopen(get_model_path(), "r");
    if (f) {
        fclose(f);
        return true;
    }
    return false;
}

#define SKIP_IF_NO_MODEL()                                           \
    do {                                                             \
        if (!model_exists())                                         \
            GTEST_SKIP() << "Model not found: " << get_model_path(); \
    } while (0)

// Helper: generate text via imp API and return as string
static std::string generate(ImpModel model, ImpContext ctx, const std::string& prompt, int max_tokens,
                            float temperature = 0.7f) {
    (void)model;
    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = max_tokens;
    params.temperature = temperature;
    params.seed = 42;
    params.apply_chat_template = 1;

    char output[4096];
    size_t output_len = 0;
    ImpError err = imp_generate(ctx, prompt.c_str(), &params, output, sizeof(output), &output_len);
    if (err != IMP_SUCCESS)
        return "";
    return std::string(output, output_len);
}

// Helper: detect n-gram repetition in a token sequence.
// Returns the fraction of tokens that are part of a repeating n-gram.
static float repetition_ratio(const std::string& text, int ngram_size = 3) {
    if (text.size() < static_cast<size_t>(ngram_size * 3))
        return 0.0f;

    // Split into words (crude but effective)
    std::vector<std::string> words;
    std::string word;
    for (char c : text) {
        if (c == ' ' || c == '\n' || c == '\t') {
            if (!word.empty()) {
                words.push_back(word);
                word.clear();
            }
        } else {
            word += c;
        }
    }
    if (!word.empty())
        words.push_back(word);

    if (words.size() < static_cast<size_t>(ngram_size * 3))
        return 0.0f;

    // Count n-gram occurrences
    std::unordered_map<std::string, int> ngram_counts;
    for (size_t i = 0; i + ngram_size <= words.size(); i++) {
        std::string ngram;
        for (int j = 0; j < ngram_size; j++) {
            if (j > 0)
                ngram += " ";
            ngram += words[i + j];
        }
        ngram_counts[ngram]++;
    }

    // Find max repeating n-gram
    int max_count = 0;
    for (const auto& [ng, count] : ngram_counts) {
        if (count > max_count)
            max_count = count;
    }

    // Ratio: repeated tokens / total tokens
    int repeated_tokens = (max_count > 1) ? max_count * ngram_size : 0;
    return static_cast<float>(repeated_tokens) / static_cast<float>(words.size());
}

class DegenerationTest : public ::testing::Test {
protected:
    ImpModel model_ = nullptr;
    ImpContext ctx_ = nullptr;

    // Set CUBLAS_WORKSPACE_CONFIG=:4096:8 once per test suite. Required for
    // greedy-deterministic behavior on Blackwell sm_120; harmless on others.
    // Set before any test creates a cuBLAS handle. setenv is idempotent.
    static void SetUpTestSuite() {
        // The 0 (overwrite) means: only set if not already set in env.
        // Production environments that need a different value can override.
        setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", /*overwrite=*/0);
    }

    void SetUp() override {
        SKIP_IF_NO_MODEL();

        ASSERT_EQ(imp_model_load(get_model_path(), IMP_FORMAT_GGUF, &model_), IMP_SUCCESS);

        ImpConfig config = imp_config_default();
        config.max_seq_len = 2048;
        config.max_batch_size = 1;

        ImpError err = imp_context_create(model_, &config, &ctx_);
        if (err != IMP_SUCCESS) {
            imp_model_free(model_);
            model_ = nullptr;
            GTEST_SKIP() << "Context creation failed: " << imp_error_string(err);
        }
    }

    void TearDown() override {
        if (ctx_)
            imp_context_free(ctx_);
        if (model_)
            imp_model_free(model_);
    }
};

// Test 1: Short prompt should produce non-empty, non-repeating output
TEST_F(DegenerationTest, ShortPromptNoRepetition) {
    std::string out = generate(model_, ctx_, "What is the capital of France?", 50);
    EXPECT_GT(out.size(), 0u) << "Empty output for simple prompt";

    float rep = repetition_ratio(out, 3);
    EXPECT_LT(rep, 0.5f) << "High repetition ratio (" << (rep * 100)
                         << "%) in output: " << out.substr(0, 200);
}

// Test 2: Second request after first should still produce coherent output
TEST_F(DegenerationTest, SecondRequestNotCorrupt) {
    // First request
    std::string out1 = generate(model_, ctx_, "Say hello", 20);

    // Reset context
    imp_context_reset(ctx_);

    // Second request — should NOT be corrupted by first request's state
    std::string out2 = generate(model_, ctx_, "What is 2+2?", 30);
    EXPECT_GT(out2.size(), 0u) << "Empty output on second request (state leak?)";

    // Should contain "4" somewhere (basic sanity)
    bool has_four = out2.find("4") != std::string::npos;
    // Relaxed: if not "4", at least no heavy repetition
    if (!has_four) {
        float rep = repetition_ratio(out2, 2);
        EXPECT_LT(rep, 0.6f) << "Second request degenerated: " << out2.substr(0, 200);
    }
}

// Test 3: Long generation should not degenerate into repetition loop
TEST_F(DegenerationTest, LongGenerationStability) {
    std::string out = generate(model_, ctx_, "Write a short paragraph about the history of computing.", 200);
    EXPECT_GT(out.size(), 50u) << "Output too short for 200-token generation";

    // Check 3-gram repetition
    float rep3 = repetition_ratio(out, 3);
    EXPECT_LT(rep3, 0.4f) << "3-gram repetition at " << (rep3 * 100) << "%: " << out.substr(0, 300);

    // Check 5-gram repetition (stricter — longer patterns)
    float rep5 = repetition_ratio(out, 5);
    EXPECT_LT(rep5, 0.3f) << "5-gram repetition at " << (rep5 * 100) << "%: " << out.substr(0, 300);
}

// Test 4: Greedy (temp=0) should be deterministic across calls
// Greedy (temp=0) should be deterministic across context resets.
//
// Fixed 2026-05-14: set CUBLAS_WORKSPACE_CONFIG=:4096:8 in the test process
// before the cuBLAS handle is created (via setenv in the test body). On
// Blackwell sm_120 without this, cuBLAS picks different algorithms on
// successive calls within the same process, producing FP16 rounding drift
// that cascades into divergent greedy output. The env var pins the
// workspace size and forces deterministic algo selection.
//
// Note: the env var must be set BEFORE any cuBLAS call in this process.
// Test class SetUp() builds the engine which creates the cuBLAS handle, so
// setenv must run before that. We use SetUpTestSuite (once per fixture).
TEST_F(DegenerationTest, GreedyDeterminism) {
    auto gen_greedy = [&](const std::string& prompt) {
        imp_context_reset(ctx_);
        ImpGenerateParams p = imp_generate_params_default();
        p.max_tokens = 30;
        p.temperature = 0.0f;
        p.seed = 42;
        p.apply_chat_template = 0;
        char buf[2048];
        size_t len = 0;
        imp_generate(ctx_, prompt.c_str(), &p, buf, sizeof(buf), &len);
        return std::string(buf, len);
    };

    std::string out1 = gen_greedy("The answer is");
    std::string out2 = gen_greedy("The answer is");
    EXPECT_EQ(out1, out2) << "Greedy sampling not deterministic!\n  Run 1: " << out1 << "\n  Run 2: " << out2;
}

// Test 5: Output should not contain raw special tokens
TEST_F(DegenerationTest, NoLeakedSpecialTokens) {
    std::string out = generate(model_, ctx_, "Tell me a fun fact about dolphins.", 100);

    // These should never appear in user-visible output
    std::vector<std::string> leaked_tokens = {"<|im_start|>", "<|im_end|>", "<|endoftext|>",
                                              "<s>",          "</s>",       "<pad>"};

    for (const auto& tok : leaked_tokens) {
        EXPECT_EQ(out.find(tok), std::string::npos)
            << "Leaked special token '" << tok << "' in output: " << out.substr(0, 200);
    }
}

}  // namespace
