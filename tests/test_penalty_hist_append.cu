// =============================================================================
// test_penalty_hist_append.cu — the batched penalty-history append kernel
// =============================================================================
//
// One launch appends row i's sampled token (strided sample slots) into
// hist[slots[i] * cap + offs[i]]; offs[i] < 0 skips the row, offs >= cap is
// refused. This is the device half of the n>1 decode loop's per-request
// penalty histories (engine_scheduler.cpp sample_per_request), which replaced
// the per-row pageable re-upload of the whole output history.
//
// GPU required — skips cleanly without one.
// =============================================================================

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>

#include "compute/sampling.h"

namespace {

bool gpu_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

TEST(PenaltyHistAppendTest, AppendsSkipsAndBounds) {
    if (!gpu_available()) GTEST_SKIP() << "no CUDA device";

    constexpr int kStride = 64;  // bytes between sample slots
    constexpr int kRows = 5;
    constexpr int kSlots = 4;
    constexpr int kCap = 8;

    // Sample slots: token of row i = 100 + i at the head of each slot.
    std::vector<char> h_samples(kRows * kStride, 0);
    for (int i = 0; i < kRows; i++)
        *reinterpret_cast<int32_t*>(h_samples.data() + i * kStride) = 100 + i;
    void* d_samples = nullptr;
    ASSERT_EQ(cudaMalloc(&d_samples, h_samples.size()), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_samples, h_samples.data(), h_samples.size(), cudaMemcpyHostToDevice),
              cudaSuccess);

    std::vector<int32_t> h_hist(kSlots * kCap, -7);
    int32_t* d_hist = nullptr;
    ASSERT_EQ(cudaMalloc(&d_hist, h_hist.size() * sizeof(int32_t)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_hist, h_hist.data(), h_hist.size() * sizeof(int32_t),
                         cudaMemcpyHostToDevice), cudaSuccess);

    imp::PenaltyAppendArgs args;
    args.n = kRows;
    args.cap = kCap;
    // row 0 -> slot 2 off 0; row 1 -> slot 0 off 5; row 2 skipped (off -1);
    // row 3 -> slot 3 off 7 (last valid); row 4 -> slot 1 off 8 (== cap, refused)
    int slots[kRows] = {2, 0, 0, 3, 1};
    int offs[kRows] = {0, 5, -1, 7, 8};
    for (int i = 0; i < kRows; i++) {
        args.slots[i] = slots[i];
        args.offs[i] = offs[i];
    }
    imp::penalty_hist_append(d_samples, kStride, args, d_hist, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_hist.data(), d_hist, h_hist.size() * sizeof(int32_t),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(h_hist[2 * kCap + 0], 100);  // row 0
    EXPECT_EQ(h_hist[0 * kCap + 5], 101);  // row 1
    EXPECT_EQ(h_hist[3 * kCap + 7], 103);  // row 3
    // Skipped and out-of-cap rows leave the buffer untouched.
    int untouched = 0;
    for (size_t i = 0; i < h_hist.size(); i++)
        if (h_hist[i] == -7) untouched++;
    EXPECT_EQ(untouched, static_cast<int>(h_hist.size()) - 3);

    cudaFree(d_samples);
    cudaFree(d_hist);
}

}  // namespace
