// topk_gating_kernel with a group limit (moe_topk_gating, grp) against Ling-3.0's
// BailingMoeV3Gate (tests/refs/moe_group_topk_golden.h). The kernel reads the expert bias as FP16;
// the golden bias (|b| <= 0.2) keeps every selection margin above that rounding.

#include "compute/moe_routing.h"
#include "kda_reference.h"
#include "refs/moe_group_topk_golden.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace imp {
namespace {

namespace gold = moe_group_topk_golden;

TEST(MoeGroupTopk, KernelMatchesLingGate) {
    kda_ref::Lcg rng(gold::kSeed);
    const auto logits = rng.fill(static_cast<size_t>(gold::kTokens) * gold::kExperts, 4.0);
    const auto bias = rng.fill(gold::kExperts, 0.2);
    std::vector<half> bias_h(bias.size());
    for (size_t i = 0; i < bias.size(); ++i)
        bias_h[i] = __float2half(bias[i]);

    float* d_logits = nullptr;
    half* d_bias = nullptr;
    ASSERT_EQ(cudaMalloc(&d_logits, logits.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_bias, bias_h.size() * sizeof(half)), cudaSuccess);
    cudaMemcpy(d_logits, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias_h.data(), bias_h.size() * sizeof(half), cudaMemcpyHostToDevice);
    int64_t shape[2] = {gold::kTokens, gold::kExperts};
    Tensor t(d_logits, QType::F32, 2, shape, true);

    MoeRoutingResult r;
    moe_topk_gating(t, gold::kTopK, r, nullptr, /*use_sigmoid=*/true, /*normalize_weights=*/true, d_bias,
                    MoeGroupLimit{gold::kNGroup, gold::kTopkGroup});
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const size_t n = static_cast<size_t>(gold::kTokens) * gold::kTopK;
    std::vector<int32_t> idx(n);
    std::vector<float> w(n);
    cudaMemcpy(idx.data(), r.expert_indices.data, n * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(w.data(), r.expert_weights.data, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int tk = 0; tk < gold::kTokens; ++tk) {
        std::vector<std::pair<int, float>> row;
        for (int k = 0; k < gold::kTopK; ++k)
            row.emplace_back(idx[(tk * gold::kTopK) + k], w[(tk * gold::kTopK) + k]);
        std::sort(row.begin(), row.end());
        for (int k = 0; k < gold::kTopK; ++k) {
            EXPECT_EQ(row[k].first, gold::kIdx[(tk * gold::kTopK) + k]) << "token " << tk;
            // The kernel normalises; routed_scaling_factor is applied by the executor.
            EXPECT_NEAR(row[k].second * gold::kScale, gold::kWeight[(tk * gold::kTopK) + k], 1e-5);
        }
    }
    cudaFree(r.expert_indices.data), cudaFree(r.expert_weights.data);
    cudaFree(r.sorted_token_ids.data), cudaFree(r.expert_offsets.data);
    cudaFree(d_logits), cudaFree(d_bias);
}

}  // namespace
}  // namespace imp
