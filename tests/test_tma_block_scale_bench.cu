#include <gtest/gtest.h>
#include "bench/tma_block_scale_bench.h"
#include <cuda_runtime.h>
#include <cstdio>

// Fused (one CUtensorMap over packed FP4+scale tile) vs separate (two descriptors) TMA
// loads, sm_120a. Original NVFP4 small-M kernel spec hypothesised fused >5% faster; repeated
// measurement REFUTED it (0.95-1.02x, parity or marginally slower). EXPECT_GT(1.05) retired;
// gate is now just that both variants launch. Consistent with sm_120: TMA bulk is empirically
// equivalent or slower than cp.async here.
TEST(TmaBlockScaleBench, BothDescriptorsLaunch) {
    int dev = 0;
    cudaGetDevice(&dev);
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    if (major * 10 + minor < 120) GTEST_SKIP() << "SM120 required (TMA cp.async.bulk.tensor)";

    auto r = imp::bench_tma_block_scale(2048);

    std::printf("\n=== TMA block-scale bench (cp.async.bulk.tensor, SM%d%d) ===\n",
                major, minor);
    std::printf("  separate (2 CUtensorMap descriptors): %.3f ms\n", r.ms_separate);
    std::printf("  fused    (1 CUtensorMap descriptor):  %.3f ms\n", r.ms_fused);

    if (r.ms_fused <= 0.0 || r.ms_separate <= 0.0) {
        FAIL() << "kernel launch failed (ms <= 0) — likely TMA descriptor or"
               << " smem-cap problem. ms_sep=" << r.ms_separate
               << " ms_fused=" << r.ms_fused;
    }

    double speedup = r.ms_separate / r.ms_fused;
    double bw_sep  = (r.bytes_loaded / r.ms_separate) * 1e-9;
    double bw_fuse = (r.bytes_loaded / r.ms_fused)    * 1e-9;
    std::printf("  speedup: %.3fx   bw_separate=%.1f GB/s   bw_fused=%.1f GB/s\n",
                speedup, bw_sep, bw_fuse);
    if (speedup < 1.05) {
        std::printf("  note: fused / separate at parity (speedup < 1.05x). The spec's\n"
                    "  original >5%% assumption is refuted by measurement; kept as an\n"
                    "  informational observation.\n");
    }
    std::printf("\n");
}
