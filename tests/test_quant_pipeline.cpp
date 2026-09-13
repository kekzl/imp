// QuantPipeline build() post-condition (D2 extraction): drives the pipeline through the real
// engine/C-API path and asserts the observable result (coherent, non-degenerate decode),
// since a standalone unit test would need reproducing ~half of engine init (bare Model only
// reachable via the opaque C-API handle, VRAMAllocator owned by Engine::init()).
// FOLLOW-UP: replace with a true bare-QuantPipeline unit test once a lightweight
// Model+VRAMAllocator fixture exists. Requires a real model and GPU.

#include <gtest/gtest.h>
#include "imp/imp.h"
#include "test_models.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

static std::string model_path() {
    return imp_test::env_path_or(imp_test::kEnvModel, "/models/Qwen3-8B-Q8_0.gguf");
}

static bool model_exists(const std::string& p) {
    FILE* f = fopen(p.c_str(), "r");
    if (f) {
        fclose(f);
        return true;
    }
    return false;
}

// Count "word-like" whitespace-separated tokens that contain at least one
// alphanumeric character. A populated decode cache produces real words; a
// pipeline that failed to fill its caches yields empty / garbage output.
static int wordlike_token_count(const std::string& text) {
    int count = 0;
    bool in_word = false;
    bool has_alnum = false;
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (in_word && has_alnum)
                ++count;
            in_word = false;
            has_alnum = false;
        } else {
            in_word = true;
            if (std::isalnum(static_cast<unsigned char>(c)))
                has_alnum = true;
        }
    }
    if (in_word && has_alnum)
        ++count;
    return count;
}

class QuantPipelineTest : public ::testing::Test {
protected:
    ImpModel model_ = nullptr;
    ImpContext ctx_ = nullptr;

    // Pin the cuBLAS workspace for greedy-deterministic behavior on sm_120.
    static void SetUpTestSuite() {
        setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", /*overwrite=*/0);
    }

    void SetUp() override {
        const std::string path = model_path();
        if (!model_exists(path))
            GTEST_SKIP() << "Model not found: " << path;

        // imp_model_load + imp_context_create run engine init, which calls
        // QuantPipeline::build() (via GraphExecutor::pre_dequant_weights).
        ASSERT_EQ(imp_model_load(path.c_str(), IMP_FORMAT_GGUF, &model_), IMP_SUCCESS);

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

// build() post-condition: with the decode caches populated, greedy decode
// produces coherent, non-degenerate output (>= 10 word-like tokens).
TEST_F(QuantPipelineTest, BuildPopulatesDecodeCachesForCoherentDecode) {
    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 64;
    params.temperature = 0.0f;  // greedy: exercises the decode caches build() filled
    params.seed = 42;
    params.apply_chat_template = 1;

    char output[4096];
    size_t output_len = 0;
    ImpError err = imp_generate(ctx_, "List three primary colors.", &params, output,
                                sizeof(output), &output_len);
    ASSERT_EQ(err, IMP_SUCCESS) << "Decode failed: " << imp_error_string(err);

    std::string out(output, output_len);
    ASSERT_GT(out.size(), 0u) << "Empty decode output — decode caches likely unpopulated";

    int words = wordlike_token_count(out);
    EXPECT_GE(words, 10) << "Decode produced only " << words
                         << " word-like tokens (expected >= 10); a coherent decode "
                            "indicates QuantPipeline::build() populated the caches. "
                            "Output: " << out.substr(0, 200);
}

}  // namespace
