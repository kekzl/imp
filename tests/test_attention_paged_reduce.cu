// Split-K reduce (paged_attention_reduce_kernel) — numeric oracle + a
// path-equivalence test for the shared-memory staging.
//
// The kernel merges the per-split (m, l, O_unnormalised-at-m) partials that
// every split-K paged decode kernel writes. It is 8.2 % of the decode step at
// 8k context on Qwen3-Coder-30B-A3B (48 layers x 5.3 us), and it had no direct
// numeric test: the paged oracle covers the attention kernels, not the merge.
//
// Two things are checked, for two different reasons:
//
//  1. CORRECTNESS vs an independent fp64 reference computed on the host from
//     the same partials, over num_splits ∈ {1, 4, 40, 85} (85 is what the GQA
//     path actually launches at 4 KV heads). The reference is rounded to f16
//     before comparing, because that is the precision the kernel stores at; the
//     residual is f32-vs-f64 accumulation over num_splits terms. Tolerance is
//     stated and justified at the assert.
//
//  2. PATH EQUIVALENCE, which is the actual claim behind the staging change.
//     The kernel stages the (m, l) pairs into shared memory when
//     num_splits <= 256 and reduces them in the SAME serial order; above that
//     it reads them straight from global. Both paths must produce BIT-IDENTICAL
//     output, otherwise "only the memory latency is parallel" is false.
//     Getting both paths onto the same data is possible because an empty split
//     is exactly neutral: the kernels write m = -FLT_MAX, l = 0 for a split
//     with no work, expf(-FLT_MAX - gmax) is 0, and adding 0 changes neither
//     the denominator nor the numerator. So the same 64 real splits are run
//     once at num_splits=64 (staged) and once padded to 300 (unstaged), and the
//     outputs are compared as raw bits.
//
// Data is the repo's heavy-tailed LCG regime (tests/refs/README.md §3), not
// sin/cos: a benign fill hides ordering effects behind values that are all the
// same magnitude, which is exactly what this test is looking for.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cfloat>
#include <cmath>
#include <cstring>
#include <vector>

#include "compute/attention_paged.h"

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
        // 2e-2: f16 storage is ~5e-4 relative, but the merge accumulates
        // num_splits products in f32 against an f64 reference, and the LCG data
        // is heavy-tailed on purpose so cancellation is real. Measured worst
        // case over these shapes was well inside this; it is a ceiling, not a
        // fit.
        EXPECT_LT(worst, 2e-2) << "splits=" << num_splits;
    }
}

// The staging claim: staged and unstaged paths must agree BIT-EXACTLY.
// num_splits <= 256 stages into shared memory; 300 does not. Padding with the
// empty-split sentinel is exactly neutral, so both runs merge the same 64 real
// splits and any difference is the staging changing arithmetic.
// The output is f16, so a few f32 ulp between the two paths get swallowed by the
// 10-bit mantissa. Mutation-validated, both directions:
//   * reversing the accumulation order in the staged path only -> CAUGHT
//     (seed 7, element 4753). That is the property this test exists for.
//   * swapping expf for __expf in the staged path -> SURVIVES, across all 49152
//     elements below. Not a hole in the test: at these magnitudes the ~1 vs ~2
//     ulp difference never reaches the f16 mantissa, so on this output type the
//     two are equivalent. Do not "fix" the test for it; it would only be
//     measurable if the kernel started storing f32.
// The sweep is what gives the order check its resolution: each element is an
// independent chance to land near an f16 rounding boundary.
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
