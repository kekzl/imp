#include <gtest/gtest.h>
#include "model/gguf_loader.h"

namespace imp {
namespace {

TEST(GgufLoaderTest, LoadNonexistentFile) {
    auto model = load_gguf("/nonexistent/path/model.gguf");
    EXPECT_EQ(model, nullptr);
}

TEST(GgufLoaderTest, GGMLTypeHelpers) {
    // Block sizes
    EXPECT_EQ(ggml_blck_size(GGMLType::F32), 1);
    EXPECT_EQ(ggml_blck_size(GGMLType::F16), 1);
    EXPECT_EQ(ggml_blck_size(GGMLType::Q4_0), 32);
    EXPECT_EQ(ggml_blck_size(GGMLType::Q8_0), 32);
    EXPECT_EQ(ggml_blck_size(GGMLType::Q4_K), 256);

    // Type sizes
    EXPECT_EQ(ggml_type_size(GGMLType::F32), 4u);
    EXPECT_EQ(ggml_type_size(GGMLType::F16), 2u);
    EXPECT_EQ(ggml_type_size(GGMLType::BF16), 2u);
    EXPECT_EQ(ggml_type_size(GGMLType::Q4_0), 18u);
    EXPECT_EQ(ggml_type_size(GGMLType::Q8_0), 34u);

    // Row size
    EXPECT_EQ(ggml_row_size(GGMLType::F32, 4096), 4096u * 4);
    EXPECT_EQ(ggml_row_size(GGMLType::F16, 4096), 4096u * 2);
    // Q4_0: 4096 elements / 32 elements_per_block * 18 bytes_per_block
    EXPECT_EQ(ggml_row_size(GGMLType::Q4_0, 4096), (4096u / 32) * 18);

    // Type names
    EXPECT_STREQ(ggml_type_name(GGMLType::F32), "F32");
    EXPECT_STREQ(ggml_type_name(GGMLType::Q4_K), "Q4_K");

    // DType conversion
    EXPECT_EQ(ggml_type_to_dtype(GGMLType::F32), DType::FP32);
    EXPECT_EQ(ggml_type_to_dtype(GGMLType::F16), DType::FP16);
    EXPECT_EQ(ggml_type_to_dtype(GGMLType::BF16), DType::BF16);
    EXPECT_EQ(ggml_type_to_dtype(GGMLType::Q4_0), DType::INT4);
}

TEST(GgufLoaderTest, InvalidMagic) {
    // Create a small file with wrong magic to test error handling
    auto model = load_gguf("/dev/null");
    EXPECT_EQ(model, nullptr);
}

} // namespace
} // namespace imp
