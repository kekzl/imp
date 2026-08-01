#include <gtest/gtest.h>
#include "bench/tma_block_scale_bench.h"
#include <cuda_runtime.h>
#include <cstdio>

// Block-scale-aware TMA descriptor microbench (SM120 sm_120a).
//
// Compares fused (one CUtensorMap over packed [FP4 data | UE4M3 scales] tile)
// vs separate (two CUtensorMap descriptors, one per tile) using REAL
// cp.async.bulk.tensor.2d TMA loads.
//
// History: the original NVFP4 small-M kernel spec
// hypothesised that fused would be >5% faster, justifying the
// single-descriptor design assumption. Repeated measurement on sm_120a
// **refuted** this: fused runs at parity or marginally slower (typically
// 0.95–1.02× speedup over separate). The hard EXPECT_GT(1.05) was therefore
// retired in favour of a logged observation; the test now only asserts that
// both kernels launch successfully (which is the meaningful regression gate).
//
// This is consistent with the broader pattern:
// TMA bulk on sm_120 is empirically equivalent or slower than cp.async for
// our workload sizes; the perceived advantage of fused descriptors over
// separate doesn't materialise either.
//
// Renamed from FusedFasterThanSeparate: that property was refuted (above), and
// the test's actual gate is that both descriptor variants launch — the name now
// says so instead of asserting a speedup the code deliberately does not check.
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
