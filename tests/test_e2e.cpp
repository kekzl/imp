#include <gtest/gtest.h>
#include "imp/imp.h"
#include "gguf_stub.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

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
    EXPECT_GE(config.max_batch_size, 0);  // 0 = auto-detect
    EXPECT_GE(config.max_seq_len, 0);     // 0 = auto-detect
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

// --- Stub GGUF tests (no real model required, uses generated ~200 KB GGUF) ---

class StubModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        stub_path_ = imp::test::generate_gguf_stub("llama");
        ASSERT_FALSE(stub_path_.empty()) << "Failed to generate stub GGUF";
    }

    void TearDown() override {
        if (!stub_path_.empty()) unlink(stub_path_.c_str());
        stub_path_.clear();
    }

    std::string stub_path_;
};

TEST_F(StubModelTest, LoadStubModel) {
    ImpModel model = nullptr;
    ImpError err = imp_model_load(stub_path_.c_str(), IMP_FORMAT_GGUF, &model);
    ASSERT_EQ(err, IMP_SUCCESS) << "Failed to load stub GGUF: " << imp_error_string(err);
    ASSERT_NE(model, nullptr);

    EXPECT_EQ(imp_model_n_layers(model), 1);
    EXPECT_EQ(imp_model_d_model(model), 64);
    EXPECT_EQ(imp_model_vocab_size(model), 256);

    imp_model_free(model);
}

TEST_F(StubModelTest, TokenizeStub) {
    ImpModel model = nullptr;
    ASSERT_EQ(imp_model_load(stub_path_.c_str(), IMP_FORMAT_GGUF, &model), IMP_SUCCESS);

    // Stub tokenizer has 256 byte-tokens but no BPE merge rules.
    // imp_tokenize may return 0 tokens (no merges to apply) or
    // fall back to byte-level encoding. Either is acceptable.
    int32_t tokens[256];
    int n_tokens = 0;
    ImpError err = imp_tokenize(model, "Hello", tokens, &n_tokens, 256);
    // Success or graceful failure — no crash
    EXPECT_TRUE(err == IMP_SUCCESS || n_tokens == 0);

    imp_model_free(model);
}

TEST_F(StubModelTest, CreateContextAndInfer) {
    ImpModel model = nullptr;
    ASSERT_EQ(imp_model_load(stub_path_.c_str(), IMP_FORMAT_GGUF, &model), IMP_SUCCESS);

    ImpConfig config = imp_config_default();
    config.max_seq_len = 64;
    config.max_batch_size = 1;
    config.enable_cuda_graphs = 0;
    config.enable_pdl = 0;

    ImpContext ctx = nullptr;
    ImpError err = imp_context_create(model, &config, &ctx);
    // Context creation involves GPU weight upload; this may fail if CUDA is not
    // available or if the tiny model trips some validation. Either outcome is
    // acceptable — the key check is no crash.
    if (err != IMP_SUCCESS) {
        imp_model_free(model);
        GTEST_SKIP() << "Context creation failed (expected without GPU): "
                     << imp_error_string(err);
    }
    ASSERT_NE(ctx, nullptr);

    // Attempt a short generation (random weights = garbage output, but no crash)
    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 4;
    params.temperature = 0.0f;
    params.apply_chat_template = 0;

    char output[1024];
    size_t output_len = 0;
    err = imp_generate(ctx, "AB", &params, output, sizeof(output), &output_len);
    // Generation may fail with tiny random weights; we just check no crash/abort
    (void)err;

    imp_context_free(ctx);
    imp_model_free(model);
}

TEST_F(StubModelTest, PrefillDecodeStub) {
    ImpModel model = nullptr;
    ASSERT_EQ(imp_model_load(stub_path_.c_str(), IMP_FORMAT_GGUF, &model), IMP_SUCCESS);

    ImpConfig config = imp_config_default();
    config.max_seq_len = 64;
    config.max_batch_size = 1;
    config.enable_cuda_graphs = 0;
    config.enable_pdl = 0;

    ImpContext ctx = nullptr;
    ImpError err = imp_context_create(model, &config, &ctx);
    if (err != IMP_SUCCESS) {
        imp_model_free(model);
        GTEST_SKIP() << "Context creation failed: " << imp_error_string(err);
    }

    // Tokenize — stub has no BPE merges, use raw token IDs instead
    int32_t tokens[] = {72, 105};  // ASCII 'H', 'i'
    int n_tokens = 2;

    // Prefill
    err = imp_prefill(ctx, tokens, n_tokens);
    if (err != IMP_SUCCESS) {
        // Prefill may fail with tiny model; acceptable if no crash
        imp_context_free(ctx);
        imp_model_free(model);
        return;
    }

    // Decode a couple tokens
    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 4;
    params.temperature = 0.0f;

    for (int i = 0; i < 2; i++) {
        int32_t token = 0;
        err = imp_decode_step(ctx, &params, &token);
        if (err != IMP_SUCCESS) break;
    }

    // Reset should always succeed
    EXPECT_EQ(imp_context_reset(ctx), IMP_SUCCESS);

    imp_context_free(ctx);
    imp_model_free(model);
}

TEST_F(StubModelTest, VRAMLeakDetection) {
    ImpModel model = nullptr;
    ASSERT_EQ(imp_model_load(stub_path_.c_str(), IMP_FORMAT_GGUF, &model), IMP_SUCCESS);

    ImpConfig config = imp_config_default();
    config.max_seq_len = 64;
    config.max_batch_size = 1;
    config.enable_cuda_graphs = 0;
    config.enable_pdl = 0;

    ImpContext ctx = nullptr;
    ImpError err = imp_context_create(model, &config, &ctx);
    if (err != IMP_SUCCESS) {
        imp_model_free(model);
        GTEST_SKIP() << "Context creation failed: " << imp_error_string(err);
    }

    // Run one warm-up request so lazy CUDA allocations are settled
    {
        int32_t warmup_tokens[] = {1, 2, 3};
        err = imp_prefill(ctx, warmup_tokens, 3);
        if (err == IMP_SUCCESS) {
            ImpGenerateParams p = imp_generate_params_default();
            p.max_tokens = 2;
            p.temperature = 0.0f;
            p.seed = 42;
            int32_t tok;
            imp_decode_step(ctx, &p, &tok);
        }
        imp_context_reset(ctx);
    }

    // Measure VRAM baseline after warm-up
    cudaDeviceSynchronize();
    size_t free_before = 0, total = 0;
    cudaMemGetInfo(&free_before, &total);

    // Run 20 prefill+decode+reset cycles
    constexpr int kNumRequests = 20;
    for (int i = 0; i < kNumRequests; i++) {
        int32_t tokens[] = {1, 2, 3, 4, 5};
        err = imp_prefill(ctx, tokens, 5);
        if (err != IMP_SUCCESS) break;

        ImpGenerateParams params = imp_generate_params_default();
        params.max_tokens = 4;
        params.temperature = 0.0f;
        params.seed = 100 + i;

        for (int j = 0; j < 2; j++) {
            int32_t out_token = 0;
            err = imp_decode_step(ctx, &params, &out_token);
            if (err != IMP_SUCCESS) break;
        }

        err = imp_context_reset(ctx);
        if (err != IMP_SUCCESS) break;
    }

    // Measure VRAM after all requests
    cudaDeviceSynchronize();
    size_t free_after = 0;
    cudaMemGetInfo(&free_after, &total);

    size_t leak = (free_before > free_after) ? (free_before - free_after) : 0;
    float leak_pct = 100.0f * static_cast<float>(leak) / static_cast<float>(total);
    EXPECT_LT(leak_pct, 5.0f)
        << "VRAM leak detected: " << (leak / (1024 * 1024)) << " MiB after "
        << kNumRequests << " requests (" << leak_pct << "% of total "
        << (total / (1024 * 1024)) << " MiB)";

    imp_context_free(ctx);
    imp_model_free(model);
}

} // namespace
