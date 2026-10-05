// Split-K reduce (paged_attention_reduce_kernel) merges per-split (m,l,O_unnorm) partials;
// 8.2% of the decode step at 8k ctx on Qwen3-Coder-30B-A3B, previously untested directly.
// Checks: (1) correctness vs an fp64 host ref over num_splits {1,4,40,85} (85=GQA @4 KV
// heads); (2) staged (<=256 splits) vs unstaged paths must be BIT-IDENTICAL (empty-split
// sentinel m=-FLT_MAX,l=0 is exactly neutral, so both reduce the same real splits).

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cfloat>
#include <cmath>
#include <cstring>
#include <vector>

#include "compute/attention_paged.h"
#include "compute/gemm.h"

#include "test_cuda_skip.h"

using namespace imp;

namespace {

// Multiply-only f32 LCG (reproducible, no %13 periodicity, no size_t underflow).
struct Lcg {
    uint32_t s;
    explicit Lcg(uint32_t seed) : s(seed) {}
    float next() {  // (-1, 1), heavy-tailed via the cube
        s = s * 1664525u + 1013904223u;
        float u = static_cast<float>((s >> 8) & 0xFFFFFF) / 8388608.0f - 1.0f;
        return u * u * u;
    }
};

// Host fp64 reference for the split-K softmax merge, in the kernel's order.
std::vector<double> reference_merge(const std::vector<float>& partial, int n_heads, int head_dim,
                                    int num_splits) {
    const int stride = 2 + head_dim;
    std::vector<double> out(static_cast<size_t>(n_heads) * head_dim, 0.0);
    for (int h = 0; h < n_heads; ++h) {
        const float* base = partial.data() + static_cast<size_t>(h) * num_splits * stride;
        double gmax = -DBL_MAX;
        for (int s = 0; s < num_splits; ++s)
            gmax = std::max(gmax, static_cast<double>(base[s * stride]));
        double gl = 0.0;
        for (int s = 0; s < num_splits; ++s) {
            double m = base[s * stride], l = base[s * stride + 1];
            gl += std::exp(m - gmax) * l;
        }
        const double inv = (gl > 0.0) ? 1.0 / gl : 0.0;
        for (int d = 0; d < head_dim; ++d) {
            double acc = 0.0;
            for (int s = 0; s < num_splits; ++s) {
                double m = base[s * stride];
                acc += std::exp(m - gmax) * static_cast<double>(base[s * stride + 2 + d]);
            }
            out[static_cast<size_t>(h) * head_dim + d] = acc * inv;
        }
    }
    return out;
}

// Fill `num_real` splits with data; leave the rest as the empty-split sentinel
// the split kernels write (m = -FLT_MAX, l = 0, O = 0), which is exactly neutral.
std::vector<float> make_partials(int n_heads, int head_dim, int num_splits, int num_real, uint32_t seed) {
    const int stride = 2 + head_dim;
    std::vector<float> p(static_cast<size_t>(n_heads) * num_splits * stride, 0.0f);
    Lcg rng(seed);
    for (int h = 0; h < n_heads; ++h) {
        for (int s = 0; s < num_splits; ++s) {
            float* e = p.data() + (static_cast<size_t>(h) * num_splits + s) * stride;
            if (s < num_real) {
                e[0] = rng.next() * 8.0f;              // running max, spread over ~16
                e[1] = 0.25f + std::fabs(rng.next());  // denominator, strictly > 0
                for (int d = 0; d < head_dim; ++d)
                    e[2 + d] = rng.next() * 2.0f;
            } else {
                e[0] = -FLT_MAX;  // empty-split sentinel
                e[1] = 0.0f;
                for (int d = 0; d < head_dim; ++d)
                    e[2 + d] = 0.0f;
            }
        }
    }
    return p;
}

std::vector<half> run_reduce(const std::vector<float>& partial, int n_heads, int head_dim, int num_splits) {
    float* d_partial = nullptr;
    half* d_out = nullptr;
    const size_t pbytes = partial.size() * sizeof(float);
    const size_t obytes = static_cast<size_t>(n_heads) * head_dim * sizeof(half);
    EXPECT_EQ(cudaMalloc(&d_partial, pbytes), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&d_out, obytes), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d_partial, partial.data(), pbytes, cudaMemcpyHostToDevice), cudaSuccess);
    EXPECT_EQ(cudaMemset(d_out, 0, obytes), cudaSuccess);

    paged_attention_launch_reduce(d_partial, d_out, /*batch_size=*/1, n_heads, head_dim, num_splits,
                                  /*stream=*/nullptr, /*attn_sinks=*/nullptr);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);

    std::vector<half> out(static_cast<size_t>(n_heads) * head_dim);
    EXPECT_EQ(cudaMemcpy(out.data(), d_out, obytes, cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(d_partial);
    cudaFree(d_out);
    return out;
}

}  // namespace

// Correctness against an independent fp64 merge.
TEST(PagedAttentionReduce, MatchesFp64Reference) {
    SKIP_IF_NO_CUDA();
    constexpr int kHeads = 8, kHeadDim = 128;
    for (int num_splits : {1, 4, 40, 85}) {  // 85 = the GQA path's count at 4 KV heads
        auto partial = make_partials(kHeads, kHeadDim, num_splits, num_splits, 12345u + num_splits);
        auto got = run_reduce(partial, kHeads, kHeadDim, num_splits);
        auto want = reference_merge(partial, kHeads, kHeadDim, num_splits);

        double worst = 0.0;
        for (size_t i = 0; i < got.size(); ++i) {
            const float g = __half2float(got[i]);
            ASSERT_TRUE(std::isfinite(g)) << "non-finite output at " << i << ", splits=" << num_splits;
            // Compare against the reference rounded to f16, the precision the
            // kernel actually stores at. The remaining gap is f32-vs-f64
            // accumulation over num_splits terms.
            const float w = __half2float(__float2half(static_cast<float>(want[i])));
            const double denom = std::max(1e-3, std::fabs(static_cast<double>(w)));
            worst = std::max(worst, std::fabs(g - w) / denom);
        }
        // 2e-2: f16 storage is ~5e-4 relative, but merging num_splits products in f32 against an
        // f64 ref hits real cancellation on heavy-tailed LCG data. Ceiling, not a tight fit.
        EXPECT_LT(worst, 2e-2) << "splits=" << num_splits;
    }
}

// Staged (<=256 splits) vs unstaged (300) paths must agree BIT-EXACTLY; the empty-split
// sentinel makes padding neutral so both merge the same 64 real splits.
// Mutation-validated: reversing staged accumulation order -> CAUGHT (seed 7, elem 4753);
// expf -> __expf in the staged path -> SURVIVES (below f16 mantissa) - not a test hole.
TEST(PagedAttentionReduce, StagedAndUnstagedPathsAreBitIdentical) {
    SKIP_IF_NO_CUDA();
    constexpr int kHeads = 64, kHeadDim = 128, kReal = 64;
    constexpr int kStaged = 64;     // <= kMaxStagedSplits -> shared-memory path
    constexpr int kUnstaged = 300;  // >  kMaxStagedSplits -> straight from global
    const int stride = 2 + kHeadDim;

    size_t compared = 0;
    for (uint32_t seed : {7u, 101u, 777u, 4242u, 90210u, 1000003u}) {
        auto p_staged = make_partials(kHeads, kHeadDim, kStaged, kReal, seed);
        auto p_unstaged = make_partials(kHeads, kHeadDim, kUnstaged, kReal, seed);
        // Same seed and fill order, so the first kReal splits must be identical
        // bit-for-bit; the rest is the neutral sentinel.
        for (int h = 0; h < kHeads; ++h) {
            for (int s = 0; s < kReal; ++s) {
                const float* a = p_staged.data() + (static_cast<size_t>(h) * kStaged + s) * stride;
                const float* b = p_unstaged.data() + (static_cast<size_t>(h) * kUnstaged + s) * stride;
                ASSERT_EQ(std::memcmp(a, b, stride * sizeof(float)), 0)
                    << "fixture mismatch at head " << h << " split " << s;
            }
        }

        auto got_staged = run_reduce(p_staged, kHeads, kHeadDim, kStaged);
        auto got_unstaged = run_reduce(p_unstaged, kHeads, kHeadDim, kUnstaged);

        ASSERT_EQ(got_staged.size(), got_unstaged.size());
        for (size_t i = 0; i < got_staged.size(); ++i) {
            uint16_t a, b;
            std::memcpy(&a, &got_staged[i], sizeof(a));
            std::memcpy(&b, &got_unstaged[i], sizeof(b));
            ASSERT_EQ(a, b) << "seed " << seed << ": staged vs unstaged differ at element " << i << " ("
                            << __half2float(got_staged[i]) << " vs " << __half2float(got_unstaged[i])
                            << ") — the shared-memory staging changed the arithmetic";
        }
        compared += got_staged.size();
    }
    EXPECT_GE(compared, 49000u) << "sweep shrank; the check loses its resolution";
}

// Q8_1 epilogue (#2439): the armed reduce writes qs, d and d8 bit-identical to
// quantize_fp16_to_q8_1 over its own FP16 output; one-shot, and off when not armed.
TEST(PagedAttentionReduce, Q8EpilogueMatchesSeparateQuantize) {
    SKIP_IF_NO_CUDA();
    for (int head_dim : {64, 128, 256}) {
        constexpr int kHeads = 32, kSplits = 40;
        const int K = kHeads * head_dim, nb = K / 32;
        auto partial = make_partials(kHeads, head_dim, kSplits, kSplits, 4242u + head_dim);
        float *d_partial = nullptr, *d8_epi = nullptr, *d8_ref = nullptr;
        half* d_out = nullptr;
        block_q8_1 *q8_epi = nullptr, *q8_ref = nullptr;
        ASSERT_EQ(cudaMalloc(&d_partial, partial.size() * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_out, K * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&q8_epi, nb * sizeof(block_q8_1)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&q8_ref, nb * sizeof(block_q8_1)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d8_epi, nb * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d8_ref, nb * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_partial, partial.data(), partial.size() * sizeof(float),
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_EQ(cudaMemset(q8_epi, 0, nb * sizeof(block_q8_1)), cudaSuccess);
        ASSERT_EQ(cudaMemset(q8_ref, 0, nb * sizeof(block_q8_1)), cudaSuccess);

        paged_attention_arm_q8_epilogue(true, q8_epi, d8_epi);
        paged_attention_launch_reduce(d_partial, d_out, 1, kHeads, head_dim, kSplits, nullptr, nullptr);
        EXPECT_TRUE(paged_attention_take_q8_epilogue()) << "hd " << head_dim;
        EXPECT_FALSE(paged_attention_take_q8_epilogue()) << "one-shot";
        quantize_fp16_to_q8_1(d_out, q8_ref, d8_ref, K, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        std::vector<block_q8_1> a(nb), b(nb);
        std::vector<float> da(nb), db(nb);
        ASSERT_EQ(cudaMemcpy(a.data(), q8_epi, nb * sizeof(block_q8_1), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(b.data(), q8_ref, nb * sizeof(block_q8_1), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(da.data(), d8_epi, nb * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(db.data(), d8_ref, nb * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
        int nonzero = 0;
        for (int i = 0; i < nb; ++i) {
            ASSERT_EQ(std::memcmp(a[i].qs, b[i].qs, 32), 0) << "hd " << head_dim << " block " << i;
            ASSERT_EQ(std::memcmp(&a[i].d, &b[i].d, sizeof(half)), 0) << "hd " << head_dim << " block " << i;
            ASSERT_EQ(std::memcmp(&da[i], &db[i], sizeof(float)), 0) << "hd " << head_dim << " block " << i;
            nonzero += (db[i] != 0.0f);
        }
        EXPECT_GT(nonzero, nb / 2) << "fixture degenerate";

        // Not armed: the reduce leaves the Q8_1 buffer alone.
        ASSERT_EQ(cudaMemset(q8_epi, 0x5A, nb * sizeof(block_q8_1)), cudaSuccess);
        paged_attention_arm_q8_epilogue(false, q8_epi, d8_epi);
        paged_attention_launch_reduce(d_partial, d_out, 1, kHeads, head_dim, kSplits, nullptr, nullptr);
        EXPECT_FALSE(paged_attention_take_q8_epilogue());
        ASSERT_EQ(cudaMemcpy(a.data(), q8_epi, nb * sizeof(block_q8_1), cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_EQ(static_cast<uint8_t>(a[0].qs[0]), 0x5Au);

        cudaFree(d_partial);
        cudaFree(d_out);
        cudaFree(q8_epi);
        cudaFree(q8_ref);
        cudaFree(d8_epi);
        cudaFree(d8_ref);
    }
}
