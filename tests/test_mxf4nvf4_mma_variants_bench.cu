#include <gtest/gtest.h>
#include "bench/mxf4nvf4_mma_variants_bench.h"
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

    // At least one mxf4/nvf4 mma.sync variant must launch: it is imp's core FP4 kernel on
    // sm_120a, so if every variant fails the FP4 tensor-core path itself is broken, not just
    // some datacenter-only variant being rejected.
    ASSERT_GT(r.count, 0) << "bench_mma_variants produced no entries";
    bool any_viable = false;
    for (int i = 0; i < r.count; ++i)
        if (r.entries[i].tops >= 0) any_viable = true;
    EXPECT_TRUE(any_viable) << "no mxf4/nvf4 MMA variant launched on this GPU";

    cudaStreamDestroy(stream);
}

}  // namespace imp
