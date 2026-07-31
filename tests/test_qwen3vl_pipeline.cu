// Image bytes through the real Qwen3-VL tower.
//
// The synthetic encoder test (`test_qwen3vl_encoder.cu`) proves the algorithm
// against a CPU reference; this proves the plumbing against the actual
// checkpoint — real shapes (hidden 1024, head_dim 64, 24 blocks, 3 DeepStack
// taps), real patch counts, real smart_resize output. Those are exactly the
// things a synthetic tower with round numbers cannot catch.
//
// Skipped unless IMP_TEST_MODEL_QWEN3VL points at a Qwen3-VL checkpoint
// directory, so it costs nothing where the model is not staged.

#include "memory/vram_allocator.h"
#include "model/model.h"
#include "model/safetensors_loader.h"
#include "vision/image_processor.h"
#include "vision/qwen3vl_pipeline.h"
#include "vision/vision_model.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

namespace imp {
namespace {

const char* model_dir() { return std::getenv("IMP_TEST_MODEL_QWEN3VL"); }

std::vector<float> read_back(const half* d, size_t n) {
    std::vector<half> h(n);
    EXPECT_EQ(cudaMemcpy(h.data(), d, n * sizeof(half), cudaMemcpyDeviceToHost), cudaSuccess);
    std::vector<float> f(n);
    for (size_t i = 0; i < n; ++i)
        f[i] = __half2float(h[i]);
    return f;
}

class Qwen3VLPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!model_dir())
            GTEST_SKIP() << "IMP_TEST_MODEL_QWEN3VL not set";
        model_ = load_safetensors(model_dir(), /*load_mtp_head=*/false);
        ASSERT_NE(model_, nullptr);
        ASSERT_NE(model_->vision_tower, nullptr) << "checkpoint carries no vision tower";
        ASSERT_TRUE(alloc_.init(0.10f));
        // 4096 patches = a 1024x1024 image at patch 16, and enough to cross the
        // encoder's attention chunk boundary.
        ASSERT_TRUE(pipeline_.init(*model_->vision_tower, alloc_, 4096));
    }

    std::unique_ptr<Model> model_;
    VRAMAllocator alloc_;
    Qwen3VLPipeline pipeline_;
};

TEST_F(Qwen3VLPipelineTest, EncodesTheFixtureImage) {
    Qwen3VLImage img;
    ASSERT_TRUE(pipeline_.encode_file("tests/fixtures/vision_test_64.png", img, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const VisionConfig& c = model_->vision_tower->config;
    EXPECT_GT(img.tokens, 0);
    EXPECT_EQ(img.tokens, img.grid_rows * img.grid_cols);
    ASSERT_EQ(img.d_deepstack.size(), c.deepstack_indexes.size());

    const size_t n = static_cast<size_t>(img.tokens) * c.out_hidden_size;
    const auto emb = read_back(img.d_embeddings, n);

    // A NaN or an inf here means an overflow somewhere in 24 blocks, and it
    // would reach the LM as a token that poisons the whole sequence.
    double max_abs = 0.0;
    for (float v : emb) {
        ASSERT_TRUE(std::isfinite(v)) << "non-finite embedding";
        max_abs = std::max<double>(max_abs, std::fabs(v));
    }
    EXPECT_GT(max_abs, 1e-3) << "the encoder returned an all-zero embedding";
    EXPECT_LT(max_abs, 1e4) << "embedding magnitude suggests an overflow";

    for (size_t d = 0; d < img.d_deepstack.size(); ++d) {
        const auto ds = read_back(img.d_deepstack[d], n);
        for (float v : ds)
            ASSERT_TRUE(std::isfinite(v)) << "non-finite DeepStack embedding, tap " << d;
        double diff = 0.0;
        for (size_t i = 0; i < n; ++i)
            diff = std::max<double>(diff, std::fabs(ds[i] - emb[i]));
        EXPECT_GT(diff, 1e-3) << "DeepStack tap " << d << " is indistinguishable from the main output";
    }
}

// The encoder feeds a KV cache and a prefix cache downstream; a run-to-run
// difference would surface there as a cache that never hits.
TEST_F(Qwen3VLPipelineTest, TheSameImageEncodesIdentically) {
    Qwen3VLImage a;
    ASSERT_TRUE(pipeline_.encode_file("tests/fixtures/vision_test_64.png", a, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const size_t n = static_cast<size_t>(a.tokens) * model_->vision_tower->config.out_hidden_size;
    const auto first = read_back(a.d_embeddings, n);

    Qwen3VLImage b;
    ASSERT_TRUE(pipeline_.encode_file("tests/fixtures/vision_test_64.png", b, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(b.tokens, a.tokens);
    const auto second = read_back(b.d_embeddings, n);

    for (size_t i = 0; i < n; ++i)
        ASSERT_FLOAT_EQ(first[i], second[i]) << "element " << i;
}

// The counterweight to the test above: an encoder that ignored its input would
// also be perfectly reproducible.
TEST_F(Qwen3VLPipelineTest, DifferentImagesEncodeDifferently) {
    const char* other = std::getenv("IMP_TEST_IMAGE_ALT");
    if (!other)
        GTEST_SKIP() << "IMP_TEST_IMAGE_ALT not set";

    Qwen3VLImage a, b;
    ASSERT_TRUE(pipeline_.encode_file("tests/fixtures/vision_test_64.png", a, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const size_t na = static_cast<size_t>(a.tokens) * model_->vision_tower->config.out_hidden_size;
    const auto first = read_back(a.d_embeddings, na);

    ASSERT_TRUE(pipeline_.encode_file(other, b, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const size_t nb = static_cast<size_t>(b.tokens) * model_->vision_tower->config.out_hidden_size;
    const auto second = read_back(b.d_embeddings, nb);

    if (a.tokens != b.tokens)
        SUCCEED() << "different images produced different token counts (" << a.tokens << " vs " << b.tokens
                  << ")";
    else {
        double diff = 0.0;
        for (size_t i = 0; i < na; ++i)
            diff = std::max<double>(diff, std::fabs(first[i] - second[i]));
        EXPECT_GT(diff, 1e-2) << "two different images produced the same embedding";
    }
}

// The token count is what the prompt has to reserve placeholders for. If the
// pipeline and `smart_resize` disagreed, every position after the image would
// shift.
TEST_F(Qwen3VLPipelineTest, TokenCountMatchesTheResizedGrid) {
    Qwen3VLImage img;
    ASSERT_TRUE(pipeline_.encode_file("tests/fixtures/vision_test_64.png", img, nullptr));
    const VisionConfig& c = model_->vision_tower->config;
    const int factor = c.patch_size * c.merge_size;
    const SmartResize sr = qwen_smart_resize(64, 64, factor, 65536, pipeline_.max_pixels());
    ASSERT_TRUE(sr.ok);
    EXPECT_EQ(img.grid_rows, sr.height / factor);
    EXPECT_EQ(img.grid_cols, sr.width / factor);
    EXPECT_EQ(img.tokens, (sr.height / factor) * (sr.width / factor));
}

}  // namespace
}  // namespace imp
