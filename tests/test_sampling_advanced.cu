// Three shipped samplers with no test in any lane (AUDIT_arch_2026 I-4): DRY, mirostat v2,
// logit_bias. Each driven at the kernel entry point (sampling.h) against a host copy of the
// same arithmetic on small vocabularies. GPU lane (test-compute).
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "compute/sampling.h"
#include "core/cuda_static_reset.h"
#include "core/tensor.h"

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>
#include "scoped_engine_arena.h"

namespace imp {
IMP_TEST_ENGINE_ARENA(64ull << 20);  // the DRY and logit_bias slots live in the T2 arena
namespace {

Tensor make_logits(const std::vector<float>& data) {
    Tensor t;
    t.qtype = QType::F32;
    t.ndim = 1;
    t.shape[0] = static_cast<int64_t>(data.size());
    t.compute_strides();
    t.on_device = true;
    cudaMalloc(&t.data, t.nbytes());
    cudaMemcpy(t.data, data.data(), t.nbytes(), cudaMemcpyHostToDevice);
    return t;
}

std::vector<float> read_back(const Tensor& t) {
    std::vector<float> h(static_cast<size_t>(t.shape[0]));
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), t.data, t.nbytes(), cudaMemcpyDeviceToHost);
    return h;
}

void free_gpu(Tensor& t) {
    if (t.data) {
        cudaFree(t.data);
        t.data = nullptr;
    }
}

// ---------------------------------------------------------------------------
// logit_bias: the listed ids move by their bias, nothing else moves.
// ---------------------------------------------------------------------------
TEST(SamplingAdvancedTest, LogitBiasMovesOnlyTheListedTokens) {
    sampling_preallocate_logit_bias(16);
    std::vector<float> base = {0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f};
    Tensor d = make_logits(base);
    const std::vector<std::pair<int32_t, float>> bias = {{1, -100.0f}, {4, 7.25f}};
    apply_logit_bias(static_cast<float*>(d.data), static_cast<int>(base.size()), bias.data(),
                     static_cast<int>(bias.size()));
    const auto out = read_back(d);
    for (size_t i = 0; i < base.size(); i++) {
        float expected = base[i];
        if (i == 1)
            expected += -100.0f;
        if (i == 4)
            expected += 7.25f;
        EXPECT_FLOAT_EQ(out[i], expected) << "token " << i;
    }
    // With the bias the greedy choice moves from 5 to 4 (2.5 + 7.25 = 9.75).
    EXPECT_EQ(sample_greedy(d), 4);
    free_gpu(d);
}

TEST(SamplingAdvancedTest, LogitBiasIgnoresOutOfRangeIds) {
    sampling_preallocate_logit_bias(16);
    std::vector<float> base = {1.0f, 2.0f, 3.0f};
    Tensor d = make_logits(base);
    const std::vector<std::pair<int32_t, float>> bias = {{-1, 50.0f}, {3, 50.0f}, {99999, 50.0f}};
    apply_logit_bias(static_cast<float*>(d.data), 3, bias.data(), static_cast<int>(bias.size()));
    const auto out = read_back(d);
    EXPECT_EQ(out, base) << "an id outside the vocabulary must neither write nor crash";
    free_gpu(d);
}

// DRY: a token extending a repeated n-gram is penalised by multiplier*base^(match_len -
// allowed_length); tokens not extending any repetition stay put.
TEST(SamplingAdvancedTest, DryPenalisesTheTokenThatWouldExtendARepeat) {
    sampling_preallocate_dry(64);
    // History "1 2 3 4 1 2 3": the suffix "1 2 3" already occurred, followed
    // by 4. DRY's scan finds match_len 3 for token 4 (the continuation) and
    // nothing else above allowed_length 2.
    const std::vector<int32_t> history = {1, 2, 3, 4, 1, 2, 3};
    std::vector<float> base(8, 0.0f);
    Tensor d = make_logits(base);
    const float multiplier = 1.5f, dry_base = 2.0f;
    const int allowed_length = 2;
    apply_dry_penalty(static_cast<float*>(d.data), 8, history.data(), static_cast<int>(history.size()),
                      multiplier, dry_base, allowed_length, /*penalty_last_n=*/0);
    const auto out = read_back(d);
    const float expected_penalty = multiplier * std::pow(dry_base, 3.0f - allowed_length);  // 1.5 * 2 = 3
    EXPECT_NEAR(out[4], -expected_penalty, 1e-5f) << "the continuation of the repeated trigram";
    for (int t = 0; t < 8; t++) {
        if (t == 4)
            continue;
        EXPECT_FLOAT_EQ(out[t], 0.0f) << "token " << t << " extends no repetition";
    }
    free_gpu(d);
}

TEST(SamplingAdvancedTest, DryLeavesShortMatchesAndDisabledMultiplierAlone) {
    sampling_preallocate_dry(64);
    const std::vector<int32_t> history = {1, 2, 3, 4, 1, 2, 3};
    {
        // allowed_length 3: the match of exactly 3 is not above it.
        std::vector<float> base(8, 0.0f);
        Tensor d = make_logits(base);
        apply_dry_penalty(static_cast<float*>(d.data), 8, history.data(), 7, 1.5f, 2.0f, /*allowed_length=*/3, 0);
        EXPECT_EQ(read_back(d), base);
        free_gpu(d);
    }
    {
        // multiplier 0 = off, whatever the history.
        std::vector<float> base(8, 0.0f);
        Tensor d = make_logits(base);
        apply_dry_penalty(static_cast<float*>(d.data), 8, history.data(), 7, 0.0f, 2.0f, 1, 0);
        EXPECT_EQ(read_back(d), base);
        free_gpu(d);
    }
    {
        // penalty_last_n 2: the window holds only "2 3", no repeat to find.
        std::vector<float> base(8, 0.0f);
        Tensor d = make_logits(base);
        apply_dry_penalty(static_cast<float*>(d.data), 8, history.data(), 7, 1.5f, 2.0f, 1, /*penalty_last_n=*/2);
        EXPECT_EQ(read_back(d), base);
        free_gpu(d);
    }
}

// Mirostat v2: returns a token of the vocabulary, follows the argmax on a spike distribution,
// and moves mu toward the target surprise tau.
TEST(SamplingAdvancedTest, MirostatPicksTheSpikeAndMovesMuTowardTau) {
    std::vector<float> spike(16, 0.0f);
    spike[9] = 40.0f;  // p(9) ~ 1, surprise ~ 0
    Tensor d = make_logits(spike);
    const float tau = 5.0f, eta = 0.1f;
    float mu = 2.0f * tau;
    const int32_t tok = sample_mirostat_v2(d, /*temperature=*/1.0f, tau, eta, &mu, /*seed=*/7);
    EXPECT_EQ(tok, 9);
    // surprise of a certainty is ~0, so mu -= eta * (0 - tau) = mu + 0.5
    EXPECT_NEAR(mu, 2.0f * tau + eta * tau, 1e-3f);
    free_gpu(d);
}

TEST(SamplingAdvancedTest, MirostatStaysInsideTheVocabularyAndIsSeedStable) {
    std::vector<float> flat(32, 0.0f);
    for (int i = 0; i < 32; i++)
        flat[static_cast<size_t>(i)] = 0.05f * static_cast<float>(i);
    Tensor d = make_logits(flat);
    float mu_a = 10.0f, mu_b = 10.0f;
    const int32_t a = sample_mirostat_v2(d, 1.0f, 5.0f, 0.1f, &mu_a, /*seed=*/1234);
    const int32_t b = sample_mirostat_v2(d, 1.0f, 5.0f, 0.1f, &mu_b, /*seed=*/1234);
    EXPECT_GE(a, 0);
    EXPECT_LT(a, 32);
    EXPECT_EQ(a, b) << "same logits, same seed, same mu: the draw must repeat";
    EXPECT_FLOAT_EQ(mu_a, mu_b);
    EXPECT_NE(mu_a, 10.0f) << "mu must move after a draw";
    free_gpu(d);
}

}  // namespace
}  // namespace imp
