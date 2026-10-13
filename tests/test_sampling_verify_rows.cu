// sample_verify_rows (sampled speculative verify, compute/sampling.h): row r draws what the eager
// sampler draws for seed seed0 + row0 + r, and accepting a draft while the draw equals it reproduces
// the target distribution (accept rate p(d), rejected draws follow p without d).

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "compute/sampling.h"
#include "core/tensor.h"
#include "scoped_engine_arena.h"

#include <cmath>
#include <vector>

namespace imp {

IMP_TEST_ENGINE_ARENA(64ull << 20);

namespace {

struct RowsFixture {
    int n_rows = 0;
    int vocab = 0;
    float* d_logits = nullptr;
    TopkRowArgs* h_args = nullptr;
    TopkRowArgs* d_args = nullptr;
    void* d_slots = nullptr;
    int32_t* d_out = nullptr;

    RowsFixture(int rows, int v) : n_rows(rows), vocab(v) {
        EXPECT_EQ(cudaMalloc(&d_logits, sizeof(float) * rows * v), cudaSuccess);
        EXPECT_EQ(cudaMallocHost(&h_args, sizeof(TopkRowArgs) * rows), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_args, sizeof(TopkRowArgs) * rows), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_slots, SAMPLE_SCRATCH_BYTES * rows), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_out, sizeof(int32_t) * rows), cudaSuccess);
    }
    ~RowsFixture() {
        cudaFree(d_logits);
        cudaFreeHost(h_args);
        cudaFree(d_args);
        cudaFree(d_slots);
        cudaFree(d_out);
    }
    void upload(const std::vector<float>& rows_logits) {
        ASSERT_EQ(cudaMemcpy(d_logits, rows_logits.data(), sizeof(float) * rows_logits.size(),
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
    }
    std::vector<int32_t> run(const VerifySampling& s, int row0) {
        EXPECT_EQ(sample_verify_rows(d_logits, n_rows, vocab, s, row0, h_args, d_args, d_slots, d_out,
                                     nullptr),
                  cudaSuccess);
        std::vector<int32_t> out(n_rows, -1);
        EXPECT_EQ(cudaMemcpy(out.data(), d_out, sizeof(int32_t) * n_rows, cudaMemcpyDeviceToHost),
                  cudaSuccess);
        return out;
    }
};

// Same row logits repeated n_rows times: log of a fixed distribution over the first entries, rest -inf-ish.
std::vector<float> repeated_rows(const std::vector<float>& probs, int n_rows, int vocab) {
    std::vector<float> rows(static_cast<size_t>(n_rows) * vocab, -1e30f);
    for (int r = 0; r < n_rows; ++r)
        for (size_t i = 0; i < probs.size(); ++i)
            rows[static_cast<size_t>(r) * vocab + i] = std::log(probs[i]);
    return rows;
}

}  // namespace

TEST(SampleVerifyRows, RowDrawsWhatTheEagerSamplerDrawsForItsStepSeed) {
    const int n_rows = 9, vocab = 4096, row0 = 3;
    std::vector<float> h(static_cast<size_t>(n_rows) * vocab);
    for (int r = 0; r < n_rows; ++r)
        for (int v = 0; v < vocab; ++v)
            h[static_cast<size_t>(r) * vocab + v] = std::sin(0.37f * v + 1.13f * r) * 4.0f +
                                                    ((v * 31 + r) % 97) * 0.01f;
    RowsFixture f(n_rows, vocab);
    f.upload(h);
    VerifySampling s;
    s.temperature = 0.7f;
    s.top_p = 0.9f;
    s.top_k = sample_effective_top_k(0, vocab);
    s.seed0 = 0x7ffffffd;  // wraps past 2^31 inside the chunk, like compute_step_seed
    const std::vector<int32_t> got = f.run(s, row0);

    int32_t* d_ref = nullptr;
    ASSERT_EQ(cudaMalloc(&d_ref, SAMPLE_SCRATCH_BYTES), cudaSuccess);
    for (int r = 0; r < n_rows; ++r) {
        Tensor t;
        t.qtype = QType::F32;
        t.ndim = 1;
        t.shape[0] = vocab;
        t.compute_strides();
        t.on_device = true;
        t.data = f.d_logits + static_cast<size_t>(r) * vocab;
        const unsigned seed = (static_cast<unsigned>(s.seed0) + static_cast<unsigned>(row0 + r)) &
                              0x7fffffffu;
        EXPECT_EQ(got[r], sample_topk_topp(t, s.top_k, s.top_p, s.temperature, seed, d_ref, nullptr))
            << "row " << r;
    }
    cudaFree(d_ref);
}

TEST(SampleVerifyRows, DraftAcceptanceMatchesTargetDistribution) {
    // Target p over 4 tokens; draft d = token 1. Accept rate must be p(1); rejected draws follow p without 1.
    const std::vector<float> p = {0.5f, 0.3f, 0.15f, 0.05f};
    const int n_rows = 64, vocab = 256, rounds = 250;
    RowsFixture f(n_rows, vocab);
    f.upload(repeated_rows(p, n_rows, vocab));
    VerifySampling s;
    s.temperature = 1.0f;
    s.top_p = 1.0f;
    s.top_k = 50;
    std::vector<long> counts(4, 0);
    long total = 0;
    for (int round = 0; round < rounds; ++round) {
        s.seed0 = 1000 + round * n_rows;
        for (int32_t tok : f.run(s, 0)) {
            ASSERT_GE(tok, 0);
            ASSERT_LT(tok, 4);
            counts[tok]++;
            total++;
        }
    }
    const double accept = static_cast<double>(counts[1]) / total;
    const double sd = std::sqrt(0.3 * 0.7 / total);
    EXPECT_NEAR(accept, 0.3, 5 * sd) << "n=" << total;
    const long rejected = total - counts[1];
    for (int t : {0, 2, 3}) {
        const double q = p[t] / (1.0 - p[1]);
        const double got = static_cast<double>(counts[t]) / rejected;
        EXPECT_NEAR(got, q, 5 * std::sqrt(q * (1 - q) / rejected)) << "token " << t;
    }
}

TEST(SampleVerifyRows, MinPRemovesTokensBelowTheThreshold) {
    // min_p 0.2 against max 0.6: 0.1 and 0.05 (ratio < 0.2) must never be drawn.
    const std::vector<float> p = {0.6f, 0.25f, 0.1f, 0.05f};
    const int n_rows = 64, vocab = 256;
    RowsFixture f(n_rows, vocab);
    VerifySampling s;
    s.min_p = 0.2f;
    s.top_k = 50;
    for (int round = 0; round < 40; ++round) {
        f.upload(repeated_rows(p, n_rows, vocab));  // min_p filters in place
        s.seed0 = 7 + round * n_rows;
        for (int32_t tok : f.run(s, 0))
            EXPECT_TRUE(tok == 0 || tok == 1) << "drew filtered token " << tok;
    }
}

TEST(SampleVerifyRows, RefusesTopKPastTheRowSampler) {
    RowsFixture f(2, 512);
    VerifySampling s;
    s.top_k = SAMPLE_MAX_TOP_K + 1;
    EXPECT_EQ(sample_verify_rows(f.d_logits, 2, 512, s, 0, f.h_args, f.d_args, f.d_slots, f.d_out, nullptr),
              cudaErrorInvalidValue);
}

}  // namespace imp
