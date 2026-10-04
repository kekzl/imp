// ActivationCalibrator::accumulate_experts: one statistic per expert segment of an expert-sorted
// buffer, kind "<KIND>.<e>", unrouted experts absent (imp-quantize group Y reads these).

#include "exec/activation_calibrator.h"

#include <gtest/gtest.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

using namespace imp;

TEST(ActivationCalibratorExperts, OneEntryPerRoutedExpertSegment) {
    VRAMAllocator alloc;
    ASSERT_TRUE(alloc.init());
    ActivationCalibrator calib(&alloc);

    constexpr int64_t K = 32;
    constexpr int ne = 3;
    // Expert 0: rows 0-1, expert 1: none, expert 2: rows 2-4.
    const std::vector<int32_t> offsets = {0, 2, 2, 5};
    const int64_t rows = offsets.back();
    std::vector<__half> h(static_cast<size_t>(rows * K));
    for (int64_t r = 0; r < rows; r++)
        for (int64_t j = 0; j < K; j++)
            h[static_cast<size_t>(r * K + j)] = __float2half((r < 2 ? -1.0f : 2.0f) *
                                                             static_cast<float>(r + 1));

    void* d_x = nullptr;
    int32_t* d_off = nullptr;
    ASSERT_EQ(cudaMalloc(&d_x, h.size() * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_off, offsets.size() * sizeof(int32_t)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_x, h.data(), h.size() * sizeof(__half), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_off, offsets.data(), offsets.size() * sizeof(int32_t), cudaMemcpyHostToDevice),
              cudaSuccess);

    int64_t shape[2] = {rows, K};
    Tensor x(d_x, QType::F16, 2, shape, true);
    // Twice: the second call accumulates onto the first.
    calib.accumulate_experts(4, TensorKind::EXPERT_DOWN, x, d_off, ne, nullptr);
    calib.accumulate_experts(4, TensorKind::EXPERT_DOWN, x, d_off, ne, nullptr);
    const CalibrationStats s = calib.snapshot("t");
    cudaFree(d_x);
    cudaFree(d_off);

    ASSERT_EQ(s.entries.size(), 2u);
    EXPECT_EQ(s.find(4, "EXPERT_DOWN.1"), nullptr);
    const CalibrationEntry* e0 = s.find(4, "EXPERT_DOWN.0");
    const CalibrationEntry* e2 = s.find(4, "EXPERT_DOWN.2");
    ASSERT_NE(e0, nullptr);
    ASSERT_NE(e2, nullptr);
    EXPECT_EQ(e0->rows, 4u);
    EXPECT_EQ(e2->rows, 6u);
    ASSERT_EQ(e0->mean_abs.size(), static_cast<size_t>(K));
    // Expert 0: |x| = 1, 2 -> mean 1.5, mean_sq 2.5. Expert 2: 6, 8, 10 -> 8, 200/3.
    for (int64_t j = 0; j < K; j++) {
        EXPECT_FLOAT_EQ(e0->mean_abs[static_cast<size_t>(j)], 1.5f);
        EXPECT_FLOAT_EQ(e0->mean_sq[static_cast<size_t>(j)], 2.5f);
        EXPECT_FLOAT_EQ(e2->mean_abs[static_cast<size_t>(j)], 8.0f);
        EXPECT_NEAR(e2->mean_sq[static_cast<size_t>(j)], 200.0f / 3.0f, 1e-4f);
    }
}
