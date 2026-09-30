// GPU test for the fused prompt-logprobs row kernel (#2257) against a naive CPU reference with the
// semantics of the #2207 kernel: double logsumexp, 1-based rank, top-N by (value desc, index asc).

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "compute/prompt_logprobs_rows.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace imp {
namespace {

struct RowRef {
    float lp = 0.0f;
    int32_t rank = 0;
    std::vector<int32_t> top_ids;
    std::vector<float> top_lp;
};

RowRef ref_row(const float* x, int V, int32_t t, int top_n) {
    float mx = -std::numeric_limits<float>::infinity();
    for (int j = 0; j < V; ++j)
        mx = std::max(mx, x[j]);
    double sum = 0.0;
    int above = 0;
    for (int j = 0; j < V; ++j) {
        sum += std::exp(static_cast<double>(x[j]) - static_cast<double>(mx));
        above += (x[j] > x[t] || (x[j] == x[t] && j < t)) ? 1 : 0;
    }
    const double lse = std::log(sum) + static_cast<double>(mx);
    RowRef r;
    r.lp = static_cast<float>(static_cast<double>(x[t]) - lse);
    r.rank = above + 1;
    std::vector<int32_t> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + top_n, idx.end(),
                      [&](int32_t a, int32_t b) { return x[a] > x[b] || (x[a] == x[b] && a < b); });
    for (int k = 0; k < top_n; ++k) {
        r.top_ids.push_back(idx[k]);
        r.top_lp.push_back(static_cast<float>(static_cast<double>(x[idx[k]]) - lse));
    }
    return r;
}

// Row-major [rows x V] -> slab-major (compute/prompt_logprobs_rows.h).
std::vector<float> to_slabs(const std::vector<float>& rm, int rows, int V, int slab_w) {
    std::vector<float> out(rm.size());
    for (int n0 = 0; n0 < V; n0 += slab_w) {
        const int w = std::min(slab_w, V - n0);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < w; ++c)
                out[static_cast<size_t>(n0) * rows + static_cast<size_t>(r) * w + c] =
                    rm[static_cast<size_t>(r) * V + n0 + c];
    }
    return out;
}

// Values on a 1/8 grid in [-24, 24): many exact ties; every 997th entry -inf.
std::vector<float> make_logits(int rows, int V, uint32_t seed) {
    std::vector<float> x(static_cast<size_t>(rows) * V);
    uint32_t s = seed;
    for (size_t i = 0; i < x.size(); ++i) {
        s = s * 1664525u + 1013904223u;
        x[i] = static_cast<float>(static_cast<int>(s >> 8) % 384 - 192) / 8.0f;
        if (i % 997 == 5)
            x[i] = -std::numeric_limits<float>::infinity();
    }
    return x;
}

void run_case(int rows, int V, int slab_w, int top_n, uint32_t seed) {
    SCOPED_TRACE(testing::Message() << "rows=" << rows << " V=" << V << " slab_w=" << slab_w
                                    << " top_n=" << top_n);
    const std::vector<float> rm = make_logits(rows, V, seed);
    std::vector<int32_t> targets(rows);
    for (int r = 0; r < rows; ++r) {
        targets[r] = static_cast<int32_t>((static_cast<int64_t>(r) * 7919 + 13) % V);
        if (std::isinf(rm[static_cast<size_t>(r) * V + targets[r]]))
            targets[r] = (targets[r] + 1) % V;
    }
    // Row 0 targets its first row maximum: rank 1.
    const auto row0_max = std::max_element(rm.begin(), rm.begin() + V) - rm.begin();
    targets[0] = static_cast<int32_t>(row0_max);
    const std::vector<float> dev_layout = (slab_w >= V) ? rm : to_slabs(rm, rows, V, slab_w);

    float *d_x = nullptr, *d_lp = nullptr, *d_tlp = nullptr;
    int32_t *d_t = nullptr, *d_rank = nullptr, *d_tid = nullptr;
    const size_t ntop = static_cast<size_t>(rows) * std::max(top_n, 1);
    ASSERT_EQ(cudaMalloc(&d_x, dev_layout.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_lp, rows * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_tlp, ntop * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_t, rows * sizeof(int32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_rank, rows * sizeof(int32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_tid, ntop * sizeof(int32_t)), cudaSuccess);
    cudaMemcpy(d_x, dev_layout.data(), dev_layout.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t, targets.data(), rows * sizeof(int32_t), cudaMemcpyHostToDevice);
    prompt_logprobs_rows(d_x, rows, V, slab_w, d_t, top_n, d_lp, d_rank, top_n > 0 ? d_tid : nullptr,
                         top_n > 0 ? d_tlp : nullptr, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> lp(rows), tlp(ntop);
    std::vector<int32_t> rank(rows), tid(ntop);
    cudaMemcpy(lp.data(), d_lp, rows * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(rank.data(), d_rank, rows * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(tlp.data(), d_tlp, ntop * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(tid.data(), d_tid, ntop * sizeof(int32_t), cudaMemcpyDeviceToHost);
    for (void* p : {static_cast<void*>(d_x), static_cast<void*>(d_lp), static_cast<void*>(d_tlp),
                    static_cast<void*>(d_t), static_cast<void*>(d_rank), static_cast<void*>(d_tid)})
        cudaFree(p);

    float max_diff = 0.0f;
    for (int r = 0; r < rows; ++r) {
        const RowRef ref = ref_row(rm.data() + static_cast<size_t>(r) * V, V, targets[r], top_n);
        max_diff = std::max(max_diff, std::fabs(lp[r] - ref.lp));
        EXPECT_EQ(rank[r], ref.rank) << "row " << r;
        for (int k = 0; k < top_n; ++k) {
            const size_t o = static_cast<size_t>(r) * top_n + k;
            EXPECT_EQ(tid[o], ref.top_ids[k]) << "row " << r << " k " << k;
            max_diff = std::max(max_diff, std::fabs(tlp[o] - ref.top_lp[k]));
        }
    }
    EXPECT_EQ(rank[0], 1);
    EXPECT_LE(max_diff, 1e-4f);
}

TEST(PromptLogprobsRows, RowMajorMatchesReference) {
    for (int top_n : {0, 1, 5, kPlpMaxTopN})
        run_case(/*rows=*/9, /*V=*/151936, /*slab_w=*/151936, top_n, 17u + top_n);
}

TEST(PromptLogprobsRows, SlabMajorMatchesReference) {
    // 12288 = Qwen3-8B dequant-scratch slab; 5000 leaves a short last slab; 7 is narrower than the block.
    for (int slab_w : {12288, 5000, 7})
        run_case(/*rows=*/6, /*V=*/50021, slab_w, kPlpMaxTopN, 101u + slab_w);
}

TEST(PromptLogprobsRows, SmallVocabAndSingleRow) {
    run_case(/*rows=*/1, /*V=*/300, /*slab_w=*/300, /*top_n=*/20, 5u);
    run_case(/*rows=*/3, /*V=*/33, /*slab_w=*/33, /*top_n=*/20, 6u);
}

}  // namespace
}  // namespace imp
