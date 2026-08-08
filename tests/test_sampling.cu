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
#include <random>
#include "scoped_engine_arena.h"

namespace imp {

IMP_TEST_ENGINE_ARENA(64ull << 20);  // T2 arena for the migrated scratches (A7 step 8)
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

// =========================================================================
// top_p must actually truncate (#1300)
//
// TopPFiltering above documents top_p=0.5 but passes 0.99, against logits whose
// tail mass is ~e^-105 — nucleus truncation is a no-op on that fixture, so both
// sampler paths pass with the top_p cutoff removed entirely (mutants M20/M21).
// Across the whole suite top_p only ever took the values 1.0, 0.95 and 0.99.
//
// This builds a genuinely spread distribution and computes the nucleus IN the
// test from the same probabilities — an oracle, not a golden. The control
// assertion (the same seeds DO reach outside the nucleus at top_p=1.0) is what
// stops this test from degenerating the way the old one did: a fixture whose
// tail is unreachable would satisfy the main assertion while proving nothing.
// =========================================================================
void run_top_p_truncation_case(int top_k, const char* path_name, float logit_offset = 0.0f) {
    constexpr int V = 512;
    constexpr int kSeeds = 200;
    constexpr float kTopP = 0.75f;

    // p: 0.40 / 0.25 / 0.15 on the head (cum 0.80 crosses kTopP at rank 2),
    // 0.02 on each of ten tail tokens (0.20 of reachable mass), ~0 elsewhere.
    //
    // The head deliberately does NOT sit at index 0: a sampler that silently
    // returns its error sentinel (token 0) must be distinguishable from one
    // that always returns the argmax, and both from one that samples.
    constexpr int kHead[3] = {137, 42, 301};
    constexpr int kTail[10] = {5, 63, 99, 150, 188, 222, 260, 333, 400, 470};
    std::vector<float> probs(V, 1e-9f);
    probs[kHead[0]] = 0.40f;
    probs[kHead[1]] = 0.25f;
    probs[kHead[2]] = 0.15f;
    for (int t : kTail)
        probs[t] = 0.02f;

    // Softmax is shift-invariant, so adding a constant to every logit must not
    // change the sampled distribution at all. Both offsets are exercised below.
    std::vector<float> logits(V);
    for (int i = 0; i < V; i++)
        logits[i] = std::log(probs[i]) + logit_offset;

    // Oracle: the nucleus is the shortest descending prefix whose cumulative
    // probability reaches kTopP. The token that crosses the threshold is IN,
    // matching the `cumsum >= top_p` cutoff in both kernels.
    std::vector<int> order(V);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return probs[a] > probs[b]; });

    std::vector<char> in_nucleus(V, 0);
    int nucleus_size = 0;
    float cum = 0.0f;
    for (int i = 0; i < std::min(top_k, V); i++) {
        in_nucleus[order[i]] = 1;
        nucleus_size++;
        cum += probs[order[i]];
        if (cum >= kTopP)
            break;
    }
    ASSERT_LT(nucleus_size, top_k) << path_name
                                   << ": fixture is degenerate — the nucleus is the whole "
                                      "candidate list, so top_p cannot be observed";

    Tensor d_logits = make_logits(logits.data(), V);

    std::map<int32_t, int> drawn;
    for (unsigned int seed = 0; seed < kSeeds; seed++) {
        int32_t tok = sample_topk_topp(d_logits, top_k, kTopP, /*temperature=*/1.0f, seed);
        drawn[tok]++;
        EXPECT_TRUE(tok >= 0 && tok < V && in_nucleus[tok])
            << path_name << ": sampled token " << tok << " lies outside the top_p=" << kTopP
            << " nucleus (seed " << seed << ")";
    }
    if (drawn.size() == 1) {
        const int32_t only = drawn.begin()->first;
        ADD_FAILURE() << path_name << ": all " << kSeeds << " seeds drew token " << only
                      << (only == kHead[0]  ? " (the argmax — the sampler is ignoring the "
                                              "distribution and behaving greedily)"
                          : only == 0       ? " (token 0 — the sampler's error sentinel)"
                                            : " (a single fixed token)");
    }

    // Control: without truncation the same seeds must land outside the nucleus.
    int outside = 0;
    for (unsigned int seed = 0; seed < kSeeds; seed++) {
        int32_t tok = sample_topk_topp(d_logits, top_k, /*top_p=*/1.0f, /*temperature=*/1.0f, seed);
        if (tok >= 0 && tok < V && !in_nucleus[tok])
            outside++;
    }
    EXPECT_GT(outside, 0) << path_name
                          << ": the tail is unreachable even at top_p=1.0 — this fixture cannot "
                             "tell a working nucleus filter from a missing one";

    free_gpu_tensor(d_logits);
}

TEST(SamplingTest, TopPTruncatesMultiblockPath) {
    run_top_p_truncation_case(/*top_k=*/50, "multiblock, all-negative logits");
}

TEST(SamplingTest, TopPTruncatesMultiblockPathShifted) {
    run_top_p_truncation_case(/*top_k=*/50, "multiblock, shifted logits", /*logit_offset=*/5.0f);
}

TEST(SamplingTest, TopPTruncatesCubPath) {
    // top_k > SAMPLE_MAX_TOP_K (128) routes to sample_topk_topp_cub().
    run_top_p_truncation_case(/*top_k=*/200, "CUB, all-negative logits");
}

TEST(SamplingTest, TopPTruncatesCubPathShifted) {
    // Same distribution, every logit shifted by +5 so the maximum is positive.
    // Softmax is shift-invariant: this must behave identically to the case
    // above. If only one of the two passes, the sampler's max reduction is
    // sign-dependent.
    run_top_p_truncation_case(/*top_k=*/200, "CUB, shifted logits", /*logit_offset=*/5.0f);
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

// =========================================================================
// Async (enqueue-only) sampler variants — the batched-decode fast path
// (executor sample_from_logits) enqueues one sampler per sequence into its
// own SAMPLE_SCRATCH_BYTES slot and gathers all tokens with one strided
// pinned D2H + one sync. Tokens must be bit-identical to the synchronous
// per-sequence variants for the same logits and seeds.
// =========================================================================

TEST(SamplingTest, AsyncBatchedSlotsMatchSynchronousPerSequence) {
    const int n_seq = 8;
    const int vocab = 4096;
    const int top_k = 50;
    const float top_p = 0.9f, temperature = 0.7f;

    // Distinct pseudo-random logits per sequence (fixed pattern, no RNG).
    std::vector<std::vector<float>> h_logits(n_seq, std::vector<float>(vocab));
    for (int i = 0; i < n_seq; i++)
        for (int v = 0; v < vocab; v++)
            h_logits[i][v] = std::sin(0.37f * v + 1.13f * i) * 7.0f + ((v * 31 + i * 17) % 97) * 0.01f;

    std::vector<Tensor> logits(n_seq);
    for (int i = 0; i < n_seq; i++)
        logits[i] = make_logits(h_logits[i].data(), vocab);

    // Reference: synchronous per-sequence sampling (pre-allocated d_result).
    int32_t* d_ref = nullptr;
    ASSERT_EQ(cudaMalloc(&d_ref, SAMPLE_SCRATCH_BYTES), cudaSuccess);
    std::vector<int32_t> ref(n_seq), ref_greedy(n_seq);
    for (int i = 0; i < n_seq; i++) {
        ref[i] = sample_topk_topp(logits[i], top_k, top_p, temperature, 42u + i, d_ref, nullptr);
        ref_greedy[i] = sample_greedy(logits[i], d_ref, nullptr);
    }

    // Batched: enqueue all sequences into per-slot scratch, one strided D2H.
    char* d_slots = nullptr;
    ASSERT_EQ(cudaMalloc(&d_slots, static_cast<size_t>(n_seq) * SAMPLE_SCRATCH_BYTES), cudaSuccess);
    int32_t* h_pinned = nullptr;
    ASSERT_EQ(cudaHostAlloc(&h_pinned, n_seq * sizeof(int32_t), cudaHostAllocDefault), cudaSuccess);

    for (int i = 0; i < n_seq; i++) {
        auto* slot = reinterpret_cast<int32_t*>(d_slots + static_cast<size_t>(i) * SAMPLE_SCRATCH_BYTES);
        ASSERT_TRUE(sample_topk_topp_async(logits[i], top_k, top_p, temperature, 42u + i, slot, nullptr));
    }
    ASSERT_EQ(cudaMemcpy2DAsync(h_pinned, sizeof(int32_t), d_slots, SAMPLE_SCRATCH_BYTES, sizeof(int32_t),
                                n_seq, cudaMemcpyDeviceToHost, nullptr),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
    for (int i = 0; i < n_seq; i++)
        EXPECT_EQ(h_pinned[i], ref[i]) << "topk/topp token diverged for sequence " << i;

    // Greedy async path, same slot mechanics.
    for (int i = 0; i < n_seq; i++) {
        auto* slot = reinterpret_cast<int32_t*>(d_slots + static_cast<size_t>(i) * SAMPLE_SCRATCH_BYTES);
        sample_greedy_async(logits[i], slot, nullptr);
    }
    ASSERT_EQ(cudaMemcpy2DAsync(h_pinned, sizeof(int32_t), d_slots, SAMPLE_SCRATCH_BYTES, sizeof(int32_t),
                                n_seq, cudaMemcpyDeviceToHost, nullptr),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
    for (int i = 0; i < n_seq; i++)
        EXPECT_EQ(h_pinned[i], ref_greedy[i]) << "greedy token diverged for sequence " << i;

    // The CUB regime (top_k > SAMPLE_MAX_TOP_K) must decline the async path.
    EXPECT_FALSE(sample_topk_topp_async(logits[0], SAMPLE_MAX_TOP_K + 1, top_p, temperature, 42u,
                                        reinterpret_cast<int32_t*>(d_slots), nullptr));
    EXPECT_FALSE(sample_topk_topp_async(logits[0], /*top_k=*/0, top_p, temperature, 42u,
                                        reinterpret_cast<int32_t*>(d_slots), nullptr))
        << "top_k<=0 normalizes to vocab and must take the CUB path";

    cudaFreeHost(h_pinned);
    cudaFree(d_slots);
    cudaFree(d_ref);
    for (int i = 0; i < n_seq; i++)
        free_gpu_tensor(logits[i]);
}

}  // namespace
// Issue #1142: the sampler drew the SAME quantile on every token.
//
// The engine hands the samplers `base_seed + step` — consecutive integers, one
// per generated token — and the device side took a single LCG step from it. An
// LCG's first output is affine in its seed, so seed+1 moved the drawn float by
// 1664525 / 2^32 ~= 0.0004: over a whole generation the draw is effectively
// constant and the sampler keeps picking the same RANK. At small top_k that
// rank is the argmax, which reads as fluent greedy text and is why this hid;
// at top_k = 2000 it is a fixed token deep in the tail and the model emits it
// forever.
//
// The probe is the shape of the bug: 200 CONSECUTIVE seeds over a distribution
// whose top token holds well under all the mass. Before the scramble this
// returned exactly ONE distinct token on both sides of the k=128 path split.
TEST(SamplerSeeding, ConsecutiveSeedsDoNotAllDrawTheSameToken) {
    const int V = 4096;
    std::vector<float> h(V);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 2.0f);
    for (auto& v : h)
        v = nd(rng);
    // Three near-equal winners: a correct sampler spreads over them, a
    // fixed-quantile sampler cannot.
    h[1000] = 12.0f;
    h[2000] = 11.9f;
    h[3000] = 11.8f;

    Tensor d_logits = make_logits(h.data(), h.size());

    auto draw = [&](int top_k) {
        std::map<int32_t, int> hist;
        for (unsigned s = 1; s <= 200; ++s)
            hist[sample_topk_topp(d_logits, top_k, /*top_p=*/0.95f, /*temperature=*/1.0f, s)]++;
        int top3 = 0;
        for (int tok : {1000, 2000, 3000})
            if (auto it = hist.find(tok); it != hist.end())
                top3 += it->second;
        fprintf(stderr, "[#1142] k=%d: %zu distinct tokens, top-3 share %.2f\n", top_k, hist.size(),
                top3 / 200.0);
        return hist;
    };

    // Both sides of the SAMPLE_MAX_TOP_K = 128 split: the multiblock kernel and
    // the CUB kernel take the seed through the same path and both regressed.
    for (int top_k : {128, 256}) {
        const auto hist = draw(top_k);
        EXPECT_GT(hist.size(), 1u)
            << "top_k=" << top_k
            << ": 200 consecutive seeds produced one token — the seed is not being decorrelated";
        int most = 0;
        for (const auto& [tok, n] : hist)
            most = std::max(most, n);
        EXPECT_LT(most, 190) << "top_k=" << top_k << ": one token took " << most
                             << "/200 draws from three near-equal candidates";
    }

    free_gpu_tensor(d_logits);
}

// Issue #1142: the CUB path (top_k > SAMPLE_MAX_TOP_K) served STALE candidates.
//
// cub::DeviceTopK::MaxPairs filled its output on the first call, wrote nothing
// on the second while still returning cudaSuccess, and failed permanently with
// `invalid device ordinal` from the fourth — all on one thread, one stream,
// device 0. Nobody checked the return code, so the sampler kept drawing from
// whatever the previous call had left in the buffer. On a real model that is a
// token loop; every existing test missed it for one reason:
//
//   THE LOGITS HAVE TO CHANGE BETWEEN CALLS. Feed the same distribution twice
//   and the stale candidates ARE the correct candidates, so a broken run is
//   indistinguishable from a working one. This test moves the winner every
//   iteration, which is what a real decode step does.
TEST(SamplerCubPath, EachCallSamplesFromItsOwnLogitsNotThePreviousCalls) {
    const int V = 151936;   // the vocabulary the issue was reported on
    const int top_k = 200;  // > SAMPLE_MAX_TOP_K, so the CUB path runs
    std::vector<float> h(V, -20.0f);

    int wrong_step = -1, wrong_token = -1, expected = -1;
    for (int step = 0; step < 12 && wrong_step < 0; ++step) {
        // One unmistakable winner, moved to a fresh place every step.
        const int winner = 1000 + step * 7919;
        std::fill(h.begin(), h.end(), -20.0f);
        h[winner] = 30.0f;

        Tensor d_logits = make_logits(h.data(), h.size());
        const int32_t got = sample_topk_topp(d_logits, top_k, /*top_p=*/0.95f, /*temperature=*/0.7f,
                                             1234u + step);
        free_gpu_tensor(d_logits);

        if (got != winner) {
            wrong_step = step;
            wrong_token = got;
            expected = winner;
        }
    }

    EXPECT_EQ(wrong_step, -1) << "step " << wrong_step << " sampled token " << wrong_token
                              << " but its logits put all the mass on " << expected
                              << " — the candidate list came from an earlier call";
}

}  // namespace imp
