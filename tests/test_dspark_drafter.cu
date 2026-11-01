// DSpark block drafter (src/runtime/dspark_drafter.cu) against the FP32 reference forward
// tools/analysis/dspark_reference.py on deepseek-ai/dflash_qwen3_8b_block7, same deterministic inputs.
#include "memory/vram_allocator.h"
#include "runtime/dspark_drafter.h"

#include <gtest/gtest.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <vector>

namespace {

std::string drafter_dir() {
    const char* v = std::getenv("IMP_TEST_MODEL_DSPARK");
    return (v && *v) ? v : "/models/dflash_qwen3_8b_block7";
}

// dspark_reference.py: feature [i][j] = 2 sin(0.37 i + 0.011 j), rounded to BF16.
std::vector<half> features(int rows, int cols) {
    std::vector<half> f(static_cast<size_t>(rows) * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            f[(static_cast<size_t>(i) * cols) + j] = __float2half(
                __bfloat162float(__float2bfloat16(2.0f * std::sin((0.37f * i) + (0.011f * j)))));
    return f;
}

// `python3 -I tools/analysis/dspark_reference.py <drafter> <target> out.json 24` (FP32, CPU).
const std::vector<int32_t> kRefIds = {305, 6174, 432, 374,
                                      374, 264,  374};  // block row r -> position C + r + 1

constexpr imp::DSparkOptions kEager16{.fp8 = false, .graph = false};

class DSparkDrafterTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(drafter_dir() + "/model.safetensors"))
            GTEST_SKIP() << "no drafter checkpoint at " << drafter_dir() << " (IMP_TEST_MODEL_DSPARK)";
        ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
        ASSERT_TRUE(va_.init());
        const std::string err = dr_.load(va_, drafter_dir(), 4096, 36, 256, stream_, kEager16);
        ASSERT_TRUE(err.empty()) << err;
        const int cols = dr_.feature_dim();
        ASSERT_EQ(cols, 5 * 4096);
        host_ = features(kCtx, cols);
        ASSERT_EQ(cudaMalloc(&feat_, host_.size() * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(feat_, host_.data(), host_.size() * sizeof(half), cudaMemcpyHostToDevice),
                  cudaSuccess);
    }
    void TearDown() override {
        cudaFree(feat_);
        if (stream_)
            cudaStreamDestroy(stream_);
    }
    const half* rows(int r0) const { return feat_ + (static_cast<size_t>(r0) * dr_.feature_dim()); }

    static constexpr int kCtx = 24;
    imp::VRAMAllocator va_;  // outlives dr_
    imp::DSparkDrafter dr_;
    cudaStream_t stream_ = nullptr;
    std::vector<half> host_;
    half* feat_ = nullptr;
};

TEST_F(DSparkDrafterTest, BlockMatchesReference) {
    dr_.append_ctx(rows(0), dr_.feature_dim(), kCtx, 0, stream_);
    EXPECT_EQ(dr_.ctx_rows(), kCtx);
    EXPECT_EQ(dr_.propose(785, kCtx, stream_), kRefIds);
}

TEST_F(DSparkDrafterTest, ContextInTwoPartsAndRollback) {
    dr_.append_ctx(rows(0), dr_.feature_dim(), kCtx / 2, 0, stream_);
    dr_.append_ctx(rows(kCtx / 2), dr_.feature_dim(), kCtx - (kCtx / 2), kCtx / 2, stream_);
    EXPECT_EQ(dr_.propose(785, kCtx, stream_), kRefIds) << "context K/V appended in two parts";
    // A verify rolls back: rows from position 20 on are replaced by the same rows again.
    dr_.append_ctx(rows(20), dr_.feature_dim(), kCtx - 20, 20, stream_);
    EXPECT_EQ(dr_.ctx_rows(), kCtx);
    EXPECT_EQ(dr_.propose(785, kCtx, stream_), kRefIds) << "rollback to position 20 + re-append";
    dr_.reset();
    EXPECT_EQ(dr_.ctx_rows(), 0);
}

TEST_F(DSparkDrafterTest, Fp8WeightsTrackReference) {
    imp::DSparkDrafter f8;
    ASSERT_TRUE(f8.load(va_, drafter_dir(), 4096, 36, 256, stream_, {.fp8 = true, .graph = false}).empty());
    EXPECT_LT(f8.device_bytes(), dr_.device_bytes() * 6 / 10);
    f8.append_ctx(rows(0), f8.feature_dim(), kCtx, 0, stream_);
    const std::vector<int32_t> ids = f8.propose(785, kCtx, stream_);
    ASSERT_EQ(ids.size(), kRefIds.size());
    int same = 0;
    for (size_t i = 0; i < ids.size(); ++i)
        same += ids[i] == kRefIds[i] ? 1 : 0;
    EXPECT_GE(same, 6) << "FP8 drafts agreeing with the FP32 reference (FP16 weights: 7 of 7)";
    f8.append_ctx(rows(20), f8.feature_dim(), kCtx - 20, 20, stream_);
    EXPECT_EQ(f8.propose(785, kCtx, stream_), ids) << "rollback + re-append";
}

// Propose 1 eager, 2 captured + replayed, 3 replayed after a rollback: the context length the graph
// reads moves 12 -> 24 -> 12.
TEST_F(DSparkDrafterTest, GraphReplayFollowsContextLength) {
    imp::DSparkDrafter g;
    ASSERT_TRUE(g.load(va_, drafter_dir(), 4096, 36, 256, stream_, {.fp8 = false, .graph = true}).empty());
    g.append_ctx(rows(0), g.feature_dim(), kCtx / 2, 0, stream_);
    const std::vector<int32_t> eager12 = g.propose(785, kCtx / 2, stream_);
    ASSERT_EQ(eager12.size(), kRefIds.size());
    EXPECT_EQ(g.graph_replays(), 0);
    g.append_ctx(rows(kCtx / 2), g.feature_dim(), kCtx - (kCtx / 2), kCtx / 2, stream_);
    EXPECT_EQ(g.propose(785, kCtx, stream_), kRefIds);
    EXPECT_EQ(g.graph_replays(), 1);
    g.append_ctx(nullptr, g.feature_dim(), 0, kCtx / 2, stream_);
    EXPECT_EQ(g.propose(785, kCtx / 2, stream_), eager12);
    EXPECT_EQ(g.graph_replays(), 2);
}

// z-lab DFlash layout (z-lab/Qwen3-8B-DFlash-b16): 16-row block, rows 1..15 draft, embed / lm_head borrowed
// from the target checkpoint. Same reference script and features; FP16 eager, then FP8 + graph.
std::string env_or(const char* name, const char* dflt) {
    const char* v = std::getenv(name);
    return (v && *v) ? v : dflt;
}

const std::vector<int32_t> kDFlashRefIds = {277, 11,  11,  594, 374, 374, 304, 304,
                                            382, 382, 382, 279, 311, 311, 311};  // row r -> C + r

TEST(DFlashDrafterTest, BlockMatchesReference) {
    const std::string dir = env_or("IMP_TEST_MODEL_DFLASH", "/models/Qwen3-8B-DFlash-b16");
    const std::string target = env_or("IMP_TEST_MODEL_DSPARK_TARGET", "/models/Qwen3-8B-NVFP4-cortecs");
    if (!std::filesystem::exists(dir + "/model.safetensors") || !std::filesystem::exists(target))
        GTEST_SKIP() << "no DFlash drafter at " << dir << " or target at " << target;
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    imp::VRAMAllocator va;
    ASSERT_TRUE(va.init());
    constexpr int kCtx = 24;
    {
        imp::DSparkDrafter none;
        EXPECT_FALSE(none.load(va, dir, 4096, 36, 256, stream, kEager16).empty())
            << "no target path: embed / lm_head must be refused, not read from nowhere";
    }
    imp::DSparkDrafter dr;
    imp::DSparkOptions opt = kEager16;
    opt.target_path = target;
    const std::string err = dr.load(va, dir, 4096, 36, 256, stream, opt);
    ASSERT_TRUE(err.empty()) << err;
    ASSERT_EQ(dr.block_size(), 16);
    ASSERT_EQ(dr.draft_len(), 15);
    const std::vector<half> host = features(kCtx, dr.feature_dim());
    half* feat = nullptr;
    ASSERT_EQ(cudaMalloc(&feat, host.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(feat, host.data(), host.size() * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);
    dr.append_ctx(feat, dr.feature_dim(), kCtx, 0, stream);
    EXPECT_EQ(dr.propose(785, kCtx, stream), kDFlashRefIds);
    {
        // Target residual outliers (Qwen3-8B-NVFP4 features reach 7096): an FP16 fc output overflowed to inf
        // and every draft came back NaN (argmax INT_MAX, cut by propose).
        std::vector<half> big = host;
        for (size_t i = 0; i < big.size(); i += 61)
            big[i] = __float2half(7096.0f);
        half* bf = nullptr;
        ASSERT_EQ(cudaMalloc(&bf, big.size() * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(bf, big.data(), big.size() * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);
        imp::DSparkDrafter o;
        ASSERT_TRUE(o.load(va, dir, 4096, 36, 256, stream, opt).empty());
        o.append_ctx(bf, o.feature_dim(), kCtx, 0, stream);
        EXPECT_EQ(o.propose(785, kCtx, stream).size(), kDFlashRefIds.size())
            << "non-finite drafts on outliers";
        cudaFree(bf);
    }

    imp::DSparkDrafter g;
    opt.fp8 = true;
    opt.graph = true;
    ASSERT_TRUE(g.load(va, dir, 4096, 36, 256, stream, opt).empty());
    g.append_ctx(feat, g.feature_dim(), kCtx, 0, stream);
    const std::vector<int32_t> eager = g.propose(785, kCtx, stream);
    ASSERT_EQ(eager.size(), kDFlashRefIds.size());
    int same = 0;
    for (size_t i = 0; i < eager.size(); ++i)
        same += eager[i] == kDFlashRefIds[i] ? 1 : 0;
    EXPECT_GE(same, 12) << "FP8 drafts agreeing with the FP32 reference (measured 12 of 15; FP16 15 of 15)";
    EXPECT_EQ(g.propose(785, kCtx, stream), eager) << "graph replay";
    EXPECT_EQ(g.graph_replays(), 1);
    cudaFree(feat);
    cudaStreamDestroy(stream);
}

}  // namespace
