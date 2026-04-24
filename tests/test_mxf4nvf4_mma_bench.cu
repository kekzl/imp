#include <gtest/gtest.h>
#include "compute/mxf4nvf4_mma_bench.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace imp {
namespace {

// Raw-instruction perf comparison. Logs the numbers, doesn't assert a
// specific ratio — that would be fragile. But asserts BLOCKSCALE is at
// least faster than LEGACY, which is the go/no-go gate for Stage 4.
TEST(Mxf4nvf4MmaBenchTest, BlockScaleBeatsLegacy) {
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    // 170 warps = 1 per SM on RTX 5090 (full occupancy for the
    // instruction pipeline). Iterations tuned so total wall time
    // ≈ 100ms per variant.
    constexpr int WARPS = 170;
    constexpr int ITERATIONS = 1 << 20;  // 1M per warp

    MmaBenchResult r = bench_mma_comparison(WARPS, ITERATIONS, stream);
    cudaStreamDestroy(stream);

    std::printf("\n=== MMA instruction throughput (170 warps × 1M iters) ===\n");
    std::printf("  kind::f8f6f4.m16n8k32               %7.2f ms  %7.2f TOPS\n",
                r.legacy_ms, r.legacy_tops);
    std::printf("  kind::mxf4nvf4.block_scale.m16n8k64 %7.2f ms  %7.2f TOPS\n",
                r.blockscale_ms, r.blockscale_tops);
    std::printf("  Speedup (blockscale / legacy):      %5.2fx\n\n", r.speedup);

    // Numerical guards
    EXPECT_GT(r.legacy_tops, 0.0);
    EXPECT_GT(r.blockscale_tops, 0.0);

    // The key assertion: the new instruction is at least AS FAST as the
    // old one. If this fails, Stage 4 integration wouldn't help.
    EXPECT_GT(r.blockscale_tops, r.legacy_tops)
        << "mxf4nvf4.block_scale is slower than f8f6f4 — Project B Stage 4 "
        << "wouldn't deliver a perf win. Re-check the pipelines before "
        << "committing to the integration.";
}

} // namespace
} // namespace imp
