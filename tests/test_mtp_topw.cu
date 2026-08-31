// test_mtp_topw.cu — the two-pass serving top-W kernel against the probe's
// single-CTA reference on the same logits.
//
// Why it exists: mtp_topw_fast replaces a 713 us measurement-grade scan in the
// multi-candidate draft path (speculative.mtp_tree_width > 1). A wrong top-W
// cannot corrupt output — the verify accept is lossless — but it silently
// turns the branch candidates into noise, which reads as "the tree does not
// pay" instead of "the kernel is broken". Values are drawn DISTINCT: both
// kernels break exact-value ties by scan order, and the fast kernel's slice
// structure orders equal values differently — set equality on distinct inputs
// is the meaningful contract.

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <random>
#include <vector>

#include "compute/mtp_forward.h"

namespace imp {
namespace {

class MtpTopWTest : public ::testing::Test {
  protected:
    void SetUp() override {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0)
            GTEST_SKIP() << "no CUDA device";
    }
};

// Distinct pseudo-random FP16-representable values over a given vocab size.
std::vector<float> distinct_values(int vocab, unsigned seed) {
    // Base grid of strictly increasing values, then shuffle: distinctness by
    // construction, and FP16-exact so the FP16 arm loses nothing to rounding.
    std::vector<float> v(vocab);
    for (int i = 0; i < vocab; ++i)
        v[i] = -100.0f + 0.001f * static_cast<float>(i);
    std::mt19937 rng(seed);
    std::shuffle(v.begin(), v.end(), rng);
    return v;
}

void run_pair(int vocab, int top_w, bool fp32, unsigned seed) {
    MtpDraftWorkspace ws{};
    // The kernels only touch the top-W buffers; allocate just those.
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&ws.d_topk), kMtpMaxTopW * sizeof(int)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&ws.d_topk_part_val),
                         kMtpTopWBlocks * kMtpMaxTopW * sizeof(float)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&ws.d_topk_part_idx),
                         kMtpTopWBlocks * kMtpMaxTopW * sizeof(int)),
              cudaSuccess);

    const std::vector<float> host = distinct_values(vocab, seed);
    void* d_logits = nullptr;
    if (fp32) {
        ASSERT_EQ(cudaMalloc(&d_logits, vocab * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_logits, host.data(), vocab * sizeof(float),
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
    } else {
        std::vector<__half> h16(vocab);
        for (int i = 0; i < vocab; ++i) h16[i] = __float2half(host[i]);
        ASSERT_EQ(cudaMalloc(&d_logits, vocab * sizeof(__half)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_logits, h16.data(), vocab * sizeof(__half),
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
    }

    ASSERT_TRUE(mtp_topw_fast(d_logits, fp32, vocab, top_w, ws, nullptr));
    std::vector<int> fast(top_w);
    ASSERT_EQ(cudaMemcpy(fast.data(), ws.d_topk, top_w * sizeof(int), cudaMemcpyDeviceToHost),
              cudaSuccess);

    ASSERT_TRUE(mtp_topw_reference(d_logits, fp32, vocab, top_w, ws, nullptr));
    std::vector<int> ref(top_w);
    ASSERT_EQ(cudaMemcpy(ref.data(), ws.d_topk, top_w * sizeof(int), cudaMemcpyDeviceToHost),
              cudaSuccess);

    // Distinct values => the ORDERED sequences must agree exactly.
    EXPECT_EQ(fast, ref) << "vocab=" << vocab << " w=" << top_w << " fp32=" << fp32
                         << " seed=" << seed;

    // And against the host truth, not just kernel-vs-kernel (two kernels can
    // share a defect; the host sort cannot).
    std::vector<int> idx(vocab);
    for (int i = 0; i < vocab; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + top_w, idx.end(),
                      [&](int a, int b) { return host[a] > host[b]; });
    for (int w = 0; w < top_w; ++w)
        EXPECT_EQ(fast[w], idx[w]) << "rank " << w;

    cudaFree(d_logits);
    cudaFree(ws.d_topk);
    cudaFree(ws.d_topk_part_val);
    cudaFree(ws.d_topk_part_idx);
}

TEST_F(MtpTopWTest, MatchesReferenceAndHostFp16) {
    // 248320 is Qwen3.6/3.8's vocab — the shipped shape; 4097 exercises the
    // ragged last slice; 63 is smaller than the block count, so most pass-1
    // blocks own an empty slice and must emit losing sentinels.
    for (int vocab : {248320, 4097, 63})
        for (int w : {1, 2, 4, 8})
            run_pair(vocab, w, /*fp32=*/false, 0xC0FFEE + vocab + w);
}

TEST_F(MtpTopWTest, MatchesReferenceAndHostFp32) {
    for (int vocab : {248320, 4097})
        for (int w : {2, 8})
            run_pair(vocab, w, /*fp32=*/true, 0xBEEF + vocab + w);
}

}  // namespace
}  // namespace imp
