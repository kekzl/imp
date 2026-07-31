// GPU tests for the row-wise top-M logit extraction
// (src/compute/rowwise_topm.cu) feeding the Token-Recycling adjacency
// table from the spec-verify chunk (speculative.token_recycling).

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "compute/rowwise_topm.h"

#include <algorithm>
#include <numeric>
#include <vector>
#include "scoped_engine_arena.h"

namespace imp {

IMP_TEST_ENGINE_ARENA(64ull << 20);  // T2 arena for the migrated scratches (A7 step 8)
namespace {

// CPU reference: indices of the m largest values, ties -> lowest index first.
std::vector<int32_t> ref_topm(const std::vector<float>& row, int m) {
    std::vector<int32_t> idx(row.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) {
        if (row[a] != row[b]) return row[a] > row[b];
        return a < b;
    });
    idx.resize(m);
    return idx;
}

std::vector<int32_t> run_topm(const std::vector<float>& logits, int rows, int V, int m) {
    float* d_logits = nullptr;
    int32_t* d_out = nullptr;
    cudaMalloc(&d_logits, logits.size() * sizeof(float));
    cudaMalloc(&d_out, static_cast<size_t>(rows) * m * sizeof(int32_t));
    cudaMemcpy(d_logits, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice);
    rowwise_topm(d_logits, rows, V, m, d_out, /*stream=*/nullptr);
    std::vector<int32_t> out(static_cast<size_t>(rows) * m);
    cudaMemcpy(out.data(), d_out, out.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_logits);
    cudaFree(d_out);
    return out;
}

TEST(RowwiseTopM, MatchesCpuReference) {
    const int rows = 3, V = 4099, m = 4;
    std::vector<float> logits(static_cast<size_t>(rows) * V);
    // Deterministic pseudo-random values, distinct per position.
    uint32_t s = 123456789u;
    for (auto& v : logits) {
        s = s * 1664525u + 1013904223u;
        v = static_cast<float>(s % 100000u) / 1000.0f - 50.0f;
    }
    auto out = run_topm(logits, rows, V, m);
    for (int r = 0; r < rows; ++r) {
        std::vector<float> row(logits.begin() + static_cast<size_t>(r) * V,
                               logits.begin() + static_cast<size_t>(r + 1) * V);
        auto ref = ref_topm(row, m);
        for (int j = 0; j < m; ++j)
            EXPECT_EQ(out[r * m + j], ref[j]) << "row " << r << " rank " << j;
    }
}

TEST(RowwiseTopM, TieBreaksLowestIndex) {
    const int V = 1024;
    std::vector<float> logits(V, 1.0f);  // all tied
    auto out = run_topm(logits, 1, V, 3);
    EXPECT_EQ(out[0], 0);
    EXPECT_EQ(out[1], 1);
    EXPECT_EQ(out[2], 2);
}

TEST(RowwiseTopM, M1MatchesArgmax) {
    const int V = 2048;
    std::vector<float> logits(V, 0.0f);
    logits[777] = 42.0f;
    auto out = run_topm(logits, 1, V, 1);
    EXPECT_EQ(out[0], 777);
}

}  // namespace
}  // namespace imp

// An exhausted arena must DEGRADE, not corrupt. This is the contract the
// attention_cublas pointer array turned out not to have (A7 step 8): migrating
// a buffer whose caller dereferences it unconditionally turns "no arena" into a
// null-pointer kernel launch, a sticky CUDA error, and every later test in the
// binary failing on a device query that returns zero. rowwise_topm has the
// contract — it returns without writing — and this pins it, because the whole
// argument for leaving this tenant UNCHARGED is that running out is survivable.
TEST(RowwiseTopM, AnExhaustedArenaLeavesTheOutputAloneInsteadOfCorruptingTheContext) {
    // Take the standing arena down and put up one far too small for the ask.
    const bool had_arena = imp::engine_arena().is_open();
    if (had_arena)
        imp::engine_arena_close();
    ASSERT_EQ(imp::engine_arena_open(imp::cuda_malloc_backend(), 4096), imp::MemError::Ok);

    const int rows = 512, V = 4096, m = 16;  // needs 512*32*16*8 B >> 4 KiB
    std::vector<float> h_logits(static_cast<size_t>(rows) * V, 0.0f);
    for (int r = 0; r < rows; ++r)
        h_logits[static_cast<size_t>(r) * V + (r % V)] = 10.0f;

    float* d_logits = nullptr;
    int32_t* d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_logits, h_logits.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out, static_cast<size_t>(rows) * m * sizeof(int32_t)), cudaSuccess);
    cudaMemcpy(d_logits, h_logits.data(), h_logits.size() * sizeof(float), cudaMemcpyHostToDevice);
    const int32_t sentinel = -12345;
    std::vector<int32_t> h_out(static_cast<size_t>(rows) * m, sentinel);
    cudaMemcpy(d_out, h_out.data(), h_out.size() * sizeof(int32_t), cudaMemcpyHostToDevice);

    imp::rowwise_topm(d_logits, rows, V, m, d_out, /*stream=*/nullptr);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "the exhausted path launched something anyway";
    EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "the exhausted path left a sticky CUDA error";

    std::vector<int32_t> got(h_out.size());
    cudaMemcpy(got.data(), d_out, got.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(got, h_out) << "output written from a scratch the arena never handed out";

    cudaFree(d_logits);
    cudaFree(d_out);
    imp::engine_arena_close();
    if (had_arena)
        ASSERT_EQ(imp::engine_arena_open(imp::cuda_malloc_backend(), 64ull << 20), imp::MemError::Ok);
}
