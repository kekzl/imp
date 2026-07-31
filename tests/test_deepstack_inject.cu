// DeepStack injection: add visual features at image-token positions.
//
// Two mistakes this guards, both of which leave a running model:
//   - adding at the wrong positions (text tokens, or the wrong occurrence);
//   - replacing instead of adding, which throws away everything the first
//     layers computed for those positions.

#include "vision/deepstack_inject.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace imp {
namespace {

constexpr int kD = 8;
constexpr int kImageToken = 77;

bool gpu_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

struct InjectResult {
    std::vector<float> hidden;
};

InjectResult inject(const std::vector<int32_t>& tokens, const std::vector<float>& hidden_in,
                    const std::vector<float>& emb, int n_vision_tokens) {
    const int n = static_cast<int>(tokens.size());
    std::vector<half> h(hidden_in.size()), e(emb.size());
    for (size_t i = 0; i < hidden_in.size(); ++i)
        h[i] = __float2half(hidden_in[i]);
    for (size_t i = 0; i < emb.size(); ++i)
        e[i] = __float2half(emb[i]);

    half *d_h = nullptr, *d_e = nullptr;
    int32_t* d_t = nullptr;
    EXPECT_EQ(cudaMalloc(&d_h, h.size() * sizeof(half)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&d_e, std::max<size_t>(e.size(), 1) * sizeof(half)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&d_t, tokens.size() * sizeof(int32_t)), cudaSuccess);
    cudaMemcpy(d_h, h.data(), h.size() * sizeof(half), cudaMemcpyHostToDevice);
    if (!e.empty())
        cudaMemcpy(d_e, e.data(), e.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t, tokens.data(), tokens.size() * sizeof(int32_t), cudaMemcpyHostToDevice);

    launch_add_vision_embeddings(d_h, d_t, d_e, kImageToken, n, kD, n_vision_tokens, nullptr);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    cudaMemcpy(h.data(), d_h, h.size() * sizeof(half), cudaMemcpyDeviceToHost);
    cudaFree(d_h);
    cudaFree(d_e);
    cudaFree(d_t);

    InjectResult r;
    r.hidden.resize(h.size());
    for (size_t i = 0; i < h.size(); ++i)
        r.hidden[i] = __half2float(h[i]);
    return r;
}

std::vector<float> ramp(size_t n, float base) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = base + 0.25f * static_cast<float>(i % 13);
    return v;
}

TEST(DeepStackInject, AddsAtImageTokensAndLeavesTextAlone) {
    if (!gpu_available())
        GTEST_SKIP() << "no CUDA device";

    //          0    1     2     3     4     5
    const std::vector<int32_t> tokens = {5, kImageToken, kImageToken, 9, kImageToken, 3};
    const auto hidden = ramp(tokens.size() * kD, 1.0f);
    const auto emb = ramp(3 * kD, -0.5f);

    const InjectResult r = inject(tokens, hidden, emb, 3);

    int vision_idx = 0;
    for (size_t t = 0; t < tokens.size(); ++t) {
        for (int d = 0; d < kD; ++d) {
            const size_t at = t * kD + d;
            if (tokens[t] == kImageToken) {
                const float want = hidden[at] + emb[static_cast<size_t>(vision_idx) * kD + d];
                EXPECT_NEAR(r.hidden[at], want, 2e-2) << "token " << t << " dim " << d;
            } else {
                EXPECT_FLOAT_EQ(r.hidden[at], hidden[at]) << "text token " << t << " was touched";
            }
        }
        if (tokens[t] == kImageToken)
            ++vision_idx;
    }
}

// Replacing would discard whatever the first layers produced at those
// positions, which is a different model, not a subtly worse one.
TEST(DeepStackInject, AddsRatherThanReplaces) {
    if (!gpu_available())
        GTEST_SKIP() << "no CUDA device";

    const std::vector<int32_t> tokens = {kImageToken, kImageToken};
    const std::vector<float> hidden(tokens.size() * kD, 3.0f);
    const std::vector<float> emb(tokens.size() * kD, 0.5f);
    const InjectResult r = inject(tokens, hidden, emb, 2);
    for (float v : r.hidden)
        EXPECT_NEAR(v, 3.5f, 1e-2) << "a replace would read 0.5 here";
}

// Two injections in a row must stack — that is what "DeepStack" means, and it
// is how layers 0, 1 and 2 each contribute.
TEST(DeepStackInject, RepeatedInjectionsAccumulate) {
    if (!gpu_available())
        GTEST_SKIP() << "no CUDA device";

    const std::vector<int32_t> tokens = {kImageToken};
    std::vector<float> hidden(kD, 1.0f);
    const std::vector<float> emb(kD, 0.25f);
    for (int round = 0; round < 3; ++round)
        hidden = inject(tokens, hidden, emb, 1).hidden;
    for (float v : hidden)
        EXPECT_NEAR(v, 1.75f, 1e-2);
}

// A prompt with more placeholders than the encoder produced must leave the
// surplus untouched rather than read past the embedding buffer.
TEST(DeepStackInject, SurplusPlaceholdersAreLeftAlone) {
    if (!gpu_available())
        GTEST_SKIP() << "no CUDA device";

    const std::vector<int32_t> tokens = {kImageToken, kImageToken, kImageToken};
    const auto hidden = ramp(tokens.size() * kD, 2.0f);
    const auto emb = ramp(1 * kD, 0.5f);
    const InjectResult r = inject(tokens, hidden, emb, 1);  // only ONE embedding

    for (int d = 0; d < kD; ++d)
        EXPECT_NEAR(r.hidden[d], hidden[d] + emb[d], 2e-2);
    for (size_t at = kD; at < hidden.size(); ++at)
        EXPECT_FLOAT_EQ(r.hidden[at], hidden[at]) << "position " << at / kD << " was written";
}

TEST(DeepStackInject, NoImageTokensIsANoOp) {
    if (!gpu_available())
        GTEST_SKIP() << "no CUDA device";

    const std::vector<int32_t> tokens = {1, 2, 3};
    const auto hidden = ramp(tokens.size() * kD, 4.0f);
    const auto emb = ramp(2 * kD, 1.0f);
    const InjectResult r = inject(tokens, hidden, emb, 2);
    for (size_t i = 0; i < hidden.size(); ++i)
        EXPECT_FLOAT_EQ(r.hidden[i], hidden[i]);
}

}  // namespace
}  // namespace imp
