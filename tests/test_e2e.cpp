#include <gtest/gtest.h>
#include "imp/imp.h"

#include <cstdlib>
#include <cstring>
#include <string>

namespace {

// Helper: get model path from environment variable IMP_TEST_MODEL.
// Tests that require a model are skipped if not set.
static const char* test_model_path() {
    return std::getenv("IMP_TEST_MODEL");
}

// --- API sanity tests (no model required) ---

TEST(EndToEndTest, VersionString) {
    const char* version = imp_version();
    EXPECT_NE(version, nullptr);
    EXPECT_GT(strlen(version), 0u);
}

TEST(EndToEndTest, ConfigDefault) {
    ImpConfig config = imp_config_default();
    EXPECT_GE(config.max_batch_size, 1);
    EXPECT_GE(config.max_seq_len, 1);
    EXPECT_EQ(config.compute_dtype, IMP_DTYPE_FP16);
    EXPECT_EQ(config.enable_pdl, 1);
    EXPECT_EQ(config.enable_cuda_graphs, 1);
    EXPECT_EQ(config.gpu_layers, -1);
}

TEST(EndToEndTest, GenerateParamsDefault) {
    ImpGenerateParams params = imp_generate_params_default();
    EXPECT_GT(params.temperature, 0.0f);
    EXPECT_GT(params.top_p, 0.0f);
    EXPECT_GE(params.top_k, 0);
    EXPECT_GT(params.max_tokens, 0);
    EXPECT_EQ(params.seed, -1);
    EXPECT_EQ(params.apply_chat_template, 1);
}

TEST(EndToEndTest, ErrorStrings) {
    EXPECT_STREQ(imp_error_string(IMP_SUCCESS), "success");
    EXPECT_STREQ(imp_error_string(IMP_ERROR_INVALID_ARG), "invalid argument");
    EXPECT_STREQ(imp_error_string(IMP_ERROR_OUT_OF_MEMORY), "out of memory");
    EXPECT_STREQ(imp_error_string(IMP_ERROR_CUDA), "CUDA error");
    EXPECT_STREQ(imp_error_string(IMP_ERROR_FILE_NOT_FOUND), "file not found");
    EXPECT_STREQ(imp_error_string(IMP_ERROR_INVALID_MODEL), "invalid model");
}

TEST(EndToEndTest, LoadNonexistentModel) {
    ImpModel model = nullptr;
    ImpError err = imp_model_load("/nonexistent/path/model.gguf", IMP_FORMAT_GGUF, &model);
    EXPECT_NE(err, IMP_SUCCESS);
    EXPECT_EQ(model, nullptr);
}

TEST(EndToEndTest, NullArguments) {
    // model_load with null path
    ImpModel model = nullptr;
    EXPECT_EQ(imp_model_load(nullptr, IMP_FORMAT_GGUF, &model), IMP_ERROR_INVALID_ARG);

    // model_load with null output
    EXPECT_EQ(imp_model_load("test.gguf", IMP_FORMAT_GGUF, nullptr), IMP_ERROR_INVALID_ARG);

    // tokenize with null model
    int32_t tokens[64];
    int n_tokens = 0;
    EXPECT_EQ(imp_tokenize(nullptr, "hello", tokens, &n_tokens, 64), IMP_ERROR_INVALID_ARG);

    // context_create with null model
    ImpConfig cfg = imp_config_default();
    ImpContext ctx = nullptr;
    EXPECT_EQ(imp_context_create(nullptr, &cfg, &ctx), IMP_ERROR_INVALID_ARG);

    // context_reset with null
    EXPECT_EQ(imp_context_reset(nullptr), IMP_ERROR_INVALID_ARG);

    // generate with null context
    ImpGenerateParams params = imp_generate_params_default();
    char buf[256];
    size_t len;
    EXPECT_EQ(imp_generate(nullptr, "test", &params, buf, sizeof(buf), &len), IMP_ERROR_INVALID_ARG);

    // decode_step with null context
    int32_t tok;
    EXPECT_EQ(imp_decode_step(nullptr, &params, &tok), IMP_ERROR_INVALID_ARG);
}

// --- Model-dependent tests (require IMP_TEST_MODEL env var) ---

TEST(EndToEndModelTest, LoadModel) {
    const char* path = test_model_path();
    if (!path) GTEST_SKIP() << "Set IMP_TEST_MODEL to run model tests";

    ImpModel model = nullptr;
    ImpError err = imp_model_load(path, IMP_FORMAT_GGUF, &model);
    ASSERT_EQ(err, IMP_SUCCESS);
    ASSERT_NE(model, nullptr);

    EXPECT_GT(imp_model_n_layers(model), 0);
    EXPECT_GT(imp_model_d_model(model), 0);
    EXPECT_GT(imp_model_vocab_size(model), 0);

    imp_model_free(model);
}

TEST(EndToEndModelTest, Tokenize) {
    const char* path = test_model_path();
    if (!path) GTEST_SKIP() << "Set IMP_TEST_MODEL to run model tests";

    ImpModel model = nullptr;
    ASSERT_EQ(imp_model_load(path, IMP_FORMAT_GGUF, &model), IMP_SUCCESS);

    int32_t tokens[256];
    int n_tokens = 0;
    ASSERT_EQ(imp_tokenize(model, "Hello world", tokens, &n_tokens, 256), IMP_SUCCESS);
    EXPECT_GT(n_tokens, 0);
    EXPECT_LE(n_tokens, 256);

    // Roundtrip: detokenize should produce something non-empty
    char buf[1024];
    ASSERT_EQ(imp_detokenize(model, tokens, n_tokens, buf, sizeof(buf)), IMP_SUCCESS);
    EXPECT_GT(strlen(buf), 0u);

    imp_model_free(model);
}

TEST(EndToEndModelTest, CreateContextAndGenerate) {
    const char* path = test_model_path();
    if (!path) GTEST_SKIP() << "Set IMP_TEST_MODEL to run model tests";

    ImpModel model = nullptr;
    ASSERT_EQ(imp_model_load(path, IMP_FORMAT_GGUF, &model), IMP_SUCCESS);

    ImpConfig config = imp_config_default();
    config.max_seq_len = 512;
    config.max_batch_size = 1;
    config.enable_cuda_graphs = 0;  // Simpler for testing

    ImpContext ctx = nullptr;
    ASSERT_EQ(imp_context_create(model, &config, &ctx), IMP_SUCCESS);
    ASSERT_NE(ctx, nullptr);

    // Generate a short completion
    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 16;
    params.temperature = 0.0f;  // Greedy for determinism
    params.apply_chat_template = 0;

    char output[4096];
    size_t output_len = 0;
    ImpError err = imp_generate(ctx, "The capital of France is", &params,
                                 output, sizeof(output), &output_len);
    ASSERT_EQ(err, IMP_SUCCESS);
    EXPECT_GT(output_len, 0u);

    imp_context_free(ctx);
    imp_model_free(model);
}

TEST(EndToEndModelTest, PrefillDecodeStep) {
    const char* path = test_model_path();
    if (!path) GTEST_SKIP() << "Set IMP_TEST_MODEL to run model tests";

    ImpModel model = nullptr;
    ASSERT_EQ(imp_model_load(path, IMP_FORMAT_GGUF, &model), IMP_SUCCESS);

    ImpConfig config = imp_config_default();
    config.max_seq_len = 256;
    config.max_batch_size = 1;
    config.enable_cuda_graphs = 0;

    ImpContext ctx = nullptr;
    ASSERT_EQ(imp_context_create(model, &config, &ctx), IMP_SUCCESS);

    // Tokenize a prompt
    int32_t tokens[128];
    int n_tokens = 0;
    ASSERT_EQ(imp_tokenize(model, "Hello", tokens, &n_tokens, 128), IMP_SUCCESS);
    ASSERT_GT(n_tokens, 0);

    // Prefill
    ASSERT_EQ(imp_prefill(ctx, tokens, n_tokens), IMP_SUCCESS);

    // Decode a few tokens
    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 8;
    params.temperature = 0.0f;

    for (int i = 0; i < 4; i++) {
        int32_t token = 0;
        ImpError err = imp_decode_step(ctx, &params, &token);
        if (err != IMP_SUCCESS) break;  // Request may finish early (EOS)
        EXPECT_GT(token, 0);
    }

    // Reset and verify we can reuse the context
    ASSERT_EQ(imp_context_reset(ctx), IMP_SUCCESS);

    imp_context_free(ctx);
    imp_model_free(model);
}

} // namespace
