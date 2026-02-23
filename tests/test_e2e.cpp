#include <gtest/gtest.h>
#include "imp/imp.h"

namespace {

TEST(EndToEndTest, Placeholder) {
    // TODO: Full end-to-end test loading a small model and generating text
    EXPECT_TRUE(true);
}

TEST(EndToEndTest, VersionString) {
    const char* version = imp_version();
    EXPECT_NE(version, nullptr);
    EXPECT_GT(strlen(version), 0u);
}

TEST(EndToEndTest, ConfigDefault) {
    ImpConfig config = imp_config_default();
    EXPECT_GE(config.max_batch_size, 1);
    EXPECT_GE(config.max_seq_len, 1);
}

TEST(EndToEndTest, GenerateParamsDefault) {
    ImpGenerateParams params = imp_generate_params_default();
    EXPECT_GT(params.temperature, 0.0f);
    EXPECT_GT(params.top_p, 0.0f);
    EXPECT_GE(params.top_k, 0); // 0 means disabled (use all)
    EXPECT_GT(params.max_tokens, 0);
}

TEST(EndToEndTest, ErrorStrings) {
    EXPECT_NE(imp_error_string(IMP_SUCCESS), nullptr);
    EXPECT_NE(imp_error_string(IMP_ERROR_INVALID_ARG), nullptr);
    EXPECT_NE(imp_error_string(IMP_ERROR_FILE_NOT_FOUND), nullptr);
    EXPECT_NE(imp_error_string(IMP_ERROR_CUDA), nullptr);
}

TEST(EndToEndTest, LoadNonexistentModel) {
    ImpModel model = nullptr;
    ImpError err = imp_model_load("/nonexistent/path/model.gguf", IMP_FORMAT_GGUF, &model);
    EXPECT_NE(err, IMP_SUCCESS);
    EXPECT_EQ(model, nullptr);
}

TEST(EndToEndTest, TokenizeNullModel) {
    int32_t tokens[64];
    int n_tokens = 0;
    ImpError err = imp_tokenize(nullptr, "hello", tokens, &n_tokens, 64);
    EXPECT_NE(err, IMP_SUCCESS);
}

} // namespace
