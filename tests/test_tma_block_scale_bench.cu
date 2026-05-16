#include <gtest/gtest.h>
#include "bench/tma_block_scale_bench.h"
#include <cuda_runtime.h>
#include <cstdio>

// Block-scale-aware TMA descriptor microbench (SM120 sm_120a).
//
// Compares fused (one CUtensorMap over packed [FP4 data | UE4M3 scales] tile)
// vs separate (two CUtensorMap descriptors, one per tile) using REAL
// cp.async.bulk.tensor.2d TMA loads. Spec gate: fused must be >5% faster to
// justify the single-descriptor kernel design assumption in
// docs/superpowers/specs/2026-05-10-nvfp4-smallM-kernel-design.md.
TEST(TmaBlockScaleBench, FusedFasterThanSeparate) {
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
    std::printf("\n");

    EXPECT_GT(speedup, 1.05)
        << "fused TMA must be >5% faster to justify single-descriptor kernel design; "
        << "actual speedup=" << speedup << "x — spec assumption needs revision if this fails";
}
