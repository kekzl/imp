// Plumbing test against the actual checkpoint (real shapes: hidden 1024, head_dim 64,
// 24 blocks, 3 DeepStack taps, real patch counts, real smart_resize) - what a synthetic
// tower with round numbers can't catch. Skipped unless IMP_TEST_MODEL_QWEN3VL is set.

#include "memory/vram_allocator.h"
#include "model/model.h"
#include "model/safetensors_loader.h"
#include "model/tokenizer.h"
#include "vision/image_processor.h"
#include "vision/qwen3vl_pipeline.h"
#include "vision/qwen3vl_vision_load.h"
#include "scoped_engine_arena.h"
#include "vision/vision_model.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <span>
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
        // Tower is a T2 arena tenant: fixture opens/sizes the arena exactly like Engine::init does,
        // from the tower's own tensor list rather than a guess that rots when the checkpoint changes.
        const size_t vision_bytes = qwen3vl_vision_tower_device_bytes(*model_->vision_tower) +
                                    Qwen3VLPipeline::demand_bytes(*model_->vision_tower, 4096);
        arena_ = std::make_unique<ScopedEngineArena>(vision_bytes + vision_bytes / 8);
        ASSERT_TRUE(arena_->opened());
        ASSERT_TRUE(alloc_.init(0.10f));
        // 4096 patches = a 1024x1024 image at patch 16, and enough to cross the
        // encoder's attention chunk boundary.
        ASSERT_TRUE(pipeline_.init(*model_->vision_tower, 4096));
    }

    std::unique_ptr<Model> model_;
    std::unique_ptr<ScopedEngineArena> arena_;
    VRAMAllocator alloc_;
    Qwen3VLPipeline pipeline_;
};

// Reservation and allocation are two separate expressions of the same buffer list; nothing
// but this test stops them drifting - a buffer added to init() without updating demand_bytes()
// under-reserves the arena, surfacing as exhaustion on whichever model is tight.
TEST_F(Qwen3VLPipelineTest, ReservedBytesMatchTakenBytes) {
    EXPECT_EQ(pipeline_.taken_bytes(), Qwen3VLPipeline::demand_bytes(*model_->vision_tower, 4096));
}

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

// --- Video ---------------------------------------------------------------------------------

const std::string kRedPanda = "tests/fixtures/qwen3vl_video/red_panda";

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

// Raw 16-bit LE file -> floats; bf16 = high half of an fp32, f16 = IEEE half.
std::vector<float> read_16bit(const std::string& path, bool bf16) {
    const auto raw = read_file(path);
    std::vector<float> out(raw.size() / 2);
    for (size_t i = 0; i < out.size(); ++i) {
        const uint16_t b = static_cast<uint16_t>(raw[2 * i] | (raw[2 * i + 1] << 8));
        if (bf16) {
            const uint32_t u = static_cast<uint32_t>(b) << 16;
            std::memcpy(&out[i], &u, sizeof(float));
        } else {
            __half h;
            std::memcpy(&h, &b, sizeof(b));
            out[i] = __half2float(h);
        }
    }
    return out;
}

double rel_l2(const std::vector<float>& a, const std::vector<float>& ref) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        num += (static_cast<double>(a[i]) - ref[i]) * (static_cast<double>(a[i]) - ref[i]);
        den += static_cast<double>(ref[i]) * ref[i];
    }
    return std::sqrt(num / den);
}

// 8 frames of red-panda.mp4 at 192x320 (tools/qwen3vl_video_fixture/run_encoder_ref.sh): 4 frame
// pairs, each encoded like an image, concatenated == HF's video merger output. Bound 1e-2 relL2 vs
// the HF FP32 tower; HF's own BF16 run sits at 9.84e-2 from it, FP16 at 6.53e-3 (encoder_ref.txt).
TEST_F(Qwen3VLPipelineTest, VideoFramePairsMatchHfReference) {
    std::vector<std::vector<uint8_t>> frames;
    for (int k = 1; k <= 8; ++k)
        frames.push_back(read_file(kRedPanda + "/frame_" + std::to_string(k) + ".png"));
    ASSERT_FALSE(frames.back().empty()) << "missing fixture " << kRedPanda;
    std::vector<std::span<const uint8_t>> spans(frames.begin(), frames.end());
    std::vector<QwenPatches> groups;
    ASSERT_TRUE(pipeline_.preprocess_video(spans, groups));
    ASSERT_EQ(groups.size(), 4u);
    EXPECT_EQ(groups[0].grid_h, 20);
    EXPECT_EQ(groups[0].grid_w, 12);

    const int dim = pipeline_.embedding_dim();
    const size_t per_group = static_cast<size_t>(pipeline_.merged_tokens_of(groups[0])) * dim;
    half* d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_out, per_group * groups.size() * sizeof(half)), cudaSuccess);
    for (size_t g = 0; g < groups.size(); ++g) {
        Qwen3VLImage shape;
        ASSERT_TRUE(pipeline_.encode_patches_to(groups[g], d_out + g * per_group, {}, shape, nullptr));
        EXPECT_EQ(shape.grid_rows, 10);
        EXPECT_EQ(shape.grid_cols, 6);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto got = read_back(d_out, per_group * groups.size());
    cudaFree(d_out);

    const auto fp32 = read_16bit(kRedPanda + "/encoder_ref_fp32.f16", false);
    const auto bf16 = read_16bit(kRedPanda + "/encoder_ref.bf16", true);
    ASSERT_EQ(fp32.size(), got.size());
    ASSERT_EQ(bf16.size(), got.size());
    const double r32 = rel_l2(got, fp32), r16 = rel_l2(got, bf16);
    std::printf("[video-encoder] tokens=%zu dim=%d relL2_vs_hf_fp32=%.4e relL2_vs_hf_bf16=%.4e\n",
                got.size() / dim, dim, r32, r16);
    RecordProperty("relL2_vs_hf_fp32", std::to_string(r32));
    RecordProperty("relL2_vs_hf_bf16", std::to_string(r16));
    EXPECT_LE(r32, 1e-2);
}

// "<x.x seconds>" goes through imp's tokenizer on the server and CLI paths: ids must equal HF's.
TEST(Qwen3VLPipelineVideoStamps, TimestampIdsMatchHf) {
    if (!model_dir())
        GTEST_SKIP() << "IMP_TEST_MODEL_QWEN3VL not set";
    Tokenizer tok;
    ASSERT_TRUE(tok.load(std::string(model_dir()) + "/tokenizer.json"));
    EXPECT_EQ(tok.encode("<0.1 seconds>"), (std::vector<int32_t>{27, 15, 13, 16, 6486, 29}));
    EXPECT_EQ(tok.encode("<1.2 seconds>"), (std::vector<int32_t>{27, 16, 13, 17, 6486, 29}));
}

}  // namespace
}  // namespace imp
