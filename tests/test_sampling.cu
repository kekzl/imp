#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "compute/sampling.h"
#include "core/tensor.h"

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <map>

namespace imp {
namespace {

// Helper: create a 1D FP32 GPU tensor from host data
Tensor make_logits(const float* data, int64_t vocab_size) {
    Tensor t;
    t.qtype = QType::F32;
    t.ndim = 1;
    t.shape[0] = vocab_size;
    t.compute_strides();
    t.on_device = true;
    cudaMalloc(&t.data, t.nbytes());
    cudaMemcpy(t.data, data, t.nbytes(), cudaMemcpyHostToDevice);
    return t;
}

void free_gpu_tensor(Tensor& t) {
    if (t.data) {
        cudaFree(t.data);
        t.data = nullptr;
    }
}

// =========================================================================
// Greedy sampling tests
// =========================================================================

TEST(SamplingTest, GreedyBasic) {
    // Token 2 has the highest logit
    std::vector<float> logits = {1.0f, 3.0f, 5.0f, 2.0f, 0.0f};
    Tensor d_logits = make_logits(logits.data(), logits.size());

    int32_t token = sample_greedy(d_logits);
    EXPECT_EQ(token, 2);

    free_gpu_tensor(d_logits);
}

TEST(SamplingTest, GreedyTieBreak) {
    // Two equal max values — should prefer lower index
    std::vector<float> logits = {1.0f, 5.0f, 3.0f, 5.0f, 2.0f};
    Tensor d_logits = make_logits(logits.data(), logits.size());

    int32_t token = sample_greedy(d_logits);
    EXPECT_EQ(token, 1);

    free_gpu_tensor(d_logits);
}

TEST(SamplingTest, GreedyNegativeLogits) {
    // All negative — token 3 is least negative
    std::vector<float> logits = {-5.0f, -3.0f, -10.0f, -1.0f, -4.0f};
    Tensor d_logits = make_logits(logits.data(), logits.size());

    int32_t token = sample_greedy(d_logits);
    EXPECT_EQ(token, 3);

    free_gpu_tensor(d_logits);
}

TEST(SamplingTest, GreedySingleToken) {
    std::vector<float> logits = {42.0f};
    Tensor d_logits = make_logits(logits.data(), logits.size());

    int32_t token = sample_greedy(d_logits);
    EXPECT_EQ(token, 0);

    free_gpu_tensor(d_logits);
}

TEST(SamplingTest, GreedyLargeVocab) {
    // 32K vocab, peak at position 12345
    constexpr int V = 32768;
    std::vector<float> logits(V, 0.0f);
    logits[12345] = 100.0f;
    Tensor d_logits = make_logits(logits.data(), V);

    int32_t token = sample_greedy(d_logits);
    EXPECT_EQ(token, 12345);

    free_gpu_tensor(d_logits);
}

TEST(SamplingTest, GreedyPreallocated) {
    std::vector<float> logits = {0.1f, 0.9f, 0.5f};
    Tensor d_logits = make_logits(logits.data(), logits.size());

    int32_t* d_result = nullptr;
    cudaMalloc(&d_result, sizeof(int32_t));

    int32_t token = sample_greedy(d_logits, d_result);
    EXPECT_EQ(token, 1);

    cudaFree(d_result);
    free_gpu_tensor(d_logits);
}

// =========================================================================
// Top-k + top-p sampling tests
// =========================================================================

TEST(SamplingTest, TopKDeterministic) {
    // With top_k=1, should always return argmax regardless of seed
    std::vector<float> logits = {1.0f, 5.0f, 3.0f, 2.0f};
    Tensor d_logits = make_logits(logits.data(), logits.size());

    for (unsigned int seed = 0; seed < 10; seed++) {
        int32_t token = sample_topk_topp(d_logits, /*top_k=*/1, /*top_p=*/1.0f,
                                         /*temperature=*/1.0f, seed);
        EXPECT_EQ(token, 1) << "top_k=1 should always pick argmax, seed=" << seed;
    }

    free_gpu_tensor(d_logits);
}

TEST(SamplingTest, TopKRespectsK) {
    // One dominant logit, rest are very small. top_k=3 should still pick token 0
    constexpr int V = 100;
    std::vector<float> logits(V, -100.0f);
    logits[0] = 10.0f;
    logits[50] = -90.0f;
    logits[99] = -95.0f;
    Tensor d_logits = make_logits(logits.data(), V);

    for (unsigned int seed = 0; seed < 20; seed++) {
        int32_t token = sample_topk_topp(d_logits, /*top_k=*/3, /*top_p=*/1.0f,
                                         /*temperature=*/1.0f, seed);
        EXPECT_EQ(token, 0) << "Dominant logit should always be picked, seed=" << seed;
    }

    free_gpu_tensor(d_logits);
}

TEST(SamplingTest, TemperatureZeroIsGreedy) {
    // Temperature near zero should behave like greedy
    std::vector<float> logits = {1.0f, 3.0f, 2.0f, 5.0f, 4.0f};
    Tensor d_logits = make_logits(logits.data(), logits.size());

    for (unsigned int seed = 0; seed < 10; seed++) {
        int32_t token = sample_topk_topp(d_logits, /*top_k=*/128, /*top_p=*/1.0f,
                                         /*temperature=*/0.01f, seed);
        EXPECT_EQ(token, 3) << "Very low temperature should pick argmax, seed=" << seed;
    }

    free_gpu_tensor(d_logits);
}

TEST(SamplingTest, TopPFiltering) {
    // Two tokens with equal high logits, rest negligible.
    // top_p=0.5 should restrict to just one of the top tokens
    constexpr int V = 10;
    std::vector<float> logits(V, -100.0f);
    logits[2] = 5.0f;
    logits[7] = 5.0f;  // Equal probability with token 2
    Tensor d_logits = make_logits(logits.data(), V);

    // With many seeds, we should only ever see tokens 2 or 7
    for (unsigned int seed = 0; seed < 50; seed++) {
        int32_t token = sample_topk_topp(d_logits, /*top_k=*/128, /*top_p=*/0.99f,
                                         /*temperature=*/1.0f, seed);
        EXPECT_TRUE(token == 2 || token == 7)
            << "Should only sample from top-2 tokens, got " << token << " seed=" << seed;
    }

    free_gpu_tensor(d_logits);
}

TEST(SamplingTest, SamplingDistribution) {
    // Verify sampling roughly follows the probability distribution
    // Token 0: logit 2.0, Token 1: logit 1.0, Token 2: logit 0.0
    // After softmax: ~0.665, ~0.245, ~0.090
    std::vector<float> logits = {2.0f, 1.0f, 0.0f};
    Tensor d_logits = make_logits(logits.data(), logits.size());

    std::map<int32_t, int> counts;
    constexpr int N = 1000;
    for (unsigned int seed = 0; seed < N; seed++) {
        int32_t token = sample_topk_topp(d_logits, /*top_k=*/128, /*top_p=*/1.0f,
                                         /*temperature=*/1.0f, seed);
        counts[token]++;
    }

    // Token 0 should be most frequent (>40% of samples)
    EXPECT_GT(counts[0], N * 4 / 10) << "Token 0 (highest logit) should appear >40% of the time";
    // Token 2 should be least frequent (<30% of samples)
    EXPECT_LT(counts[2], N * 3 / 10) << "Token 2 (lowest logit) should appear <30% of the time";

    free_gpu_tensor(d_logits);
}

// =========================================================================
// Edge case tests
// =========================================================================

TEST(SamplingTest, NaNLogits) {
    // If any logit is NaN, verify no crash (graceful handling)
    float nan_val = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> logits = {1.0f, nan_val, 3.0f, 2.0f};
    Tensor d_logits = make_logits(logits.data(), logits.size());

    // Should not crash — result may be any token
    int32_t token = sample_greedy(d_logits);
    EXPECT_GE(token, 0);
    EXPECT_LT(token, static_cast<int32_t>(logits.size()));

    free_gpu_tensor(d_logits);
}

TEST(SamplingTest, AllIdenticalLogits) {
    // All logits equal — any token is valid (uniform distribution)
    constexpr int V = 16;
    std::vector<float> logits(V, 1.0f);
    Tensor d_logits = make_logits(logits.data(), V);

    // Greedy: should return a valid token (ties → lowest index)
    int32_t token = sample_greedy(d_logits);
    EXPECT_EQ(token, 0);

    // Stochastic: should produce valid tokens across seeds
    for (unsigned int seed = 0; seed < 20; seed++) {
        int32_t t = sample_topk_topp(d_logits, /*top_k=*/128, /*top_p=*/1.0f,
                                     /*temperature=*/1.0f, seed);
        EXPECT_GE(t, 0);
        EXPECT_LT(t, V);
    }

    free_gpu_tensor(d_logits);
}

TEST(SamplingTest, SingleNonNegInf) {
    // One logit=0, rest=-inf → must always select that token
    constexpr int V = 32;
    float neg_inf = -std::numeric_limits<float>::infinity();
    std::vector<float> logits(V, neg_inf);
    logits[17] = 0.0f;
    Tensor d_logits = make_logits(logits.data(), V);

    // Greedy
    EXPECT_EQ(sample_greedy(d_logits), 17);

    // Stochastic with various seeds
    for (unsigned int seed = 0; seed < 20; seed++) {
        int32_t t = sample_topk_topp(d_logits, /*top_k=*/128, /*top_p=*/1.0f,
                                     /*temperature=*/1.0f, seed);
        EXPECT_EQ(t, 17) << "Only non-(-inf) token should be sampled, seed=" << seed;
    }

    free_gpu_tensor(d_logits);
}

}  // namespace
}  // namespace imp
