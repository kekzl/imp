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

class DSparkDrafterTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(drafter_dir() + "/model.safetensors"))
            GTEST_SKIP() << "no drafter checkpoint at " << drafter_dir() << " (IMP_TEST_MODEL_DSPARK)";
        ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
        ASSERT_TRUE(va_.init());
        const std::string err = dr_.load(va_, drafter_dir(), 4096, 36, 256, stream_);
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

}  // namespace
