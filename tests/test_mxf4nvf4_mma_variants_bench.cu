#include <gtest/gtest.h>
#include "compute/mxf4nvf4_mma_variants_bench.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace imp {

TEST(MmaVariantsBench, Compare) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    constexpr int WARPS = 170;           // matches RTX 5090 SM count
    constexpr int ITERATIONS = 1 << 20;  // 1M iters per warp

    auto r = bench_mma_variants(WARPS, ITERATIONS, stream);

    std::printf("\n=== sm_120 MMA variant throughput (%d warps × %d iters) ===\n", WARPS, ITERATIONS);
    std::printf("  %-32s %10s %10s\n", "variant", "ms/run", "TOPS");
    for (int i = 0; i < r.count; ++i) {
        if (r.entries[i].tops < 0) {
            std::printf("  %-32s   FAILED  (kernel could not launch — likely PTX rejection)\n",
                        r.entries[i].label);
        } else {
            std::printf("  %-32s %10.2f %10.2f\n", r.entries[i].label, r.entries[i].ms, r.entries[i].tops);
        }
    }
    std::printf("\n");

    // Diagnostic — none of the variants are required to launch successfully.
    // The test PASSES regardless. The output reveals which variants are
    // viable and their relative throughput.
    SUCCEED();

    cudaStreamDestroy(stream);
}

}  // namespace imp
