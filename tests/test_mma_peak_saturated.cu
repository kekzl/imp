#include <gtest/gtest.h>
#include "bench/mma_peak_saturated.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace imp {
namespace {

// Saturated TC peak calibration (#595/#596): pins that f32-accumulate runs at ~1/4 the
// f16-accumulate rate on sm_120, and the FP4 block-scale path lands nearer half the
// 3354-TOPS datasheet number than the full one. Loose thresholds: tripwire, not a perf gate.
TEST(MmaPeakSaturatedTest, CalibrationInvariants) {
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    MmaPeakResult r = bench_mma_peak_saturated(stream);
    cudaStreamDestroy(stream);

    std::printf("\n=== Saturated mma.sync peaks (170 blocks x 8 warps x 4 chains) ===\n");
    std::printf("  FP4  mxf4nvf4 block_scale : %8.1f TOPS   (datasheet 3354)\n", r.fp4_blockscale_tops);
    std::printf("  FP16 f16acc               : %8.1f TFLOPS (datasheet 838 @2.407GHz)\n",
                r.fp16_f16acc_tflops);
    std::printf("  FP16 f32acc               : %8.1f TFLOPS\n", r.fp16_f32acc_tflops);
    std::printf("  FP8  e4m3 f32acc          : %8.1f TOPS\n\n", r.fp8_f32acc_tops);

    ASSERT_GT(r.fp4_blockscale_tops, 0.0);
    ASSERT_GT(r.fp16_f16acc_tflops, 0.0);

    // (1) f32acc is a small fraction of f16acc (measured ~7.7x apart).
    EXPECT_GT(r.fp16_f16acc_tflops, r.fp16_f32acc_tflops * 3.0)
        << "f32-accumulate penalty vanished — recalibrate tools/roofline/config.json "
           "and re-evaluate gemm.cublas_fp16_acc";

    // (2) FP4 peak is ~half datasheet (measured ~2019 of 3354 at boost).
    EXPECT_LT(r.fp4_blockscale_tops, 2700.0)
        << "FP4 mma.sync exceeds half-datasheet band — recalibrate tc_fp4";
    EXPECT_GT(r.fp4_blockscale_tops, 1500.0) << "FP4 peak collapsed vs 2026-06-07 calibration (2019 TOPS)";
}

}  // namespace
}  // namespace imp
