// test_mtp_topw.cu — the two-pass serving top-W kernel against the probe's
// single-CTA reference on the same logits.
//
// Why it exists: mtp_topw_fast replaces a 713 us measurement-grade scan in the
// multi-candidate draft path (speculative.mtp_tree_width > 1). A wrong top-W
// cannot corrupt output — the verify accept is lossless — but it silently
// turns the branch candidates into noise, which reads as "the tree does not
// pay" instead of "the kernel is broken". The top-W values are drawn DISTINCT
// (see distinct_values): both kernels break exact-value ties by scan order,
// and the fast kernel's slice structure orders equal values differently, so
// only a unique top-W is a meaningful contract.

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

// Logits with a well-defined ordered top-W. FP32: a strictly increasing grid,
// shuffled - every value distinct. FP16 cannot do that (there are ~63k finite
// half values, a 248k vocab must tie), so the FP16 arm draws the bulk from a
// coarse half-exact grid below -50 (ties allowed, none of them can win) and
// plants top_w distinct half-exact values above it at random positions: the
// ordered top-W is unique, everything below it is noise the kernels may
// order however they like.
std::vector<float> distinct_values(int vocab, int top_w, bool fp32, unsigned seed) {
    std::vector<float> v(vocab);
    std::mt19937 rng(seed);
    if (fp32) {
        for (int i = 0; i < vocab; ++i)
            v[i] = -100.0f + 0.001f * static_cast<float>(i);
        std::shuffle(v.begin(), v.end(), rng);
        return v;
    }
    for (int i = 0; i < vocab; ++i)
        v[i] = -100.0f + 0.0625f * static_cast<float>(i % 512);  // half-exact, ties
    std::vector<int> pos(vocab);
    for (int i = 0; i < vocab; ++i) pos[i] = i;
    std::shuffle(pos.begin(), pos.end(), rng);
    for (int k = 0; k < top_w; ++k)
        v[pos[k]] = -40.0f + 0.5f * static_cast<float>(k);  // half-exact, distinct, above the bulk
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

    const std::vector<float> host = distinct_values(vocab, top_w, fp32, seed);
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

    // Distinct top values => the ORDERED sequences must agree exactly.
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
