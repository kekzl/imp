// Phase 1 INT8 IMMA throughput microbench wrapper. Outcome of the experiment:
// docs/plans/2026-05-28-q4k-mmq-kernel-design.md.
//
// Asserts that every IMMA variant launches without ptxas / runtime error and
// prints the TOPS table to stderr. Does *not* gate on the perf ratio — that's
// informational (a Phase 1 finding) and decided manually in the memo. The
// test pass condition is "the hardware accepts the opcode".

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cstdio>

#include "bench/mmq_q4k_imma_bench.h"

namespace imp {

TEST(MmqQ4kImmaBench, AllVariantsLaunch) {
    // 170 SMs × 4 MMA pipes/SM ≈ 680 warps to fill the issue queue.
    // 1360 warps (×8) leaves comfortable headroom; matches the launch shape
    // mxf4nvf4_mma_variants_bench uses for sm_120 saturation.
    constexpr int WARPS = 1360;
    constexpr int ITERATIONS = 1 << 15;  // 32K MMAs per warp

    ImmaBenchResult r = bench_mmq_q4k_imma(WARPS, ITERATIONS, /*stream=*/nullptr);

    ASSERT_GT(r.count, 0);
    std::fprintf(stderr, "\n[q4k-imma Phase 1 microbench, warps=%d iters=%d]\n", WARPS, ITERATIONS);
    std::fprintf(stderr, "  %-26s %12s %12s\n", "variant", "ms/rep", "TOPS");
    double hmma_tops = -1.0;
    double imma_s8_tops = -1.0;
    for (int i = 0; i < r.count; ++i) {
        const auto& e = r.entries[i];
        if (e.tops < 0.0) {
            std::fprintf(stderr, "  %-26s %12.3f %12s  [launch failed]\n", e.label, e.ms, "-");
        } else {
            std::fprintf(stderr, "  %-26s %12.3f %12.2f\n", e.label, e.ms, e.tops);
        }
        // ASSERT every variant launches successfully on sm_120a.
        ASSERT_GE(e.tops, 0.0) << "variant " << e.label << " failed to launch";
        if (std::string(e.label) == "hmma_f32_f16_f16_k16") hmma_tops = e.tops;
        if (std::string(e.label) == "imma_s32_s8_s8_k32") imma_s8_tops = e.tops;
    }

    if (hmma_tops > 0.0 && imma_s8_tops > 0.0) {
        double ratio = imma_s8_tops / hmma_tops;
        std::fprintf(stderr, "  ---\n");
        std::fprintf(stderr, "  IMMA s8 / HMMA f16 TOPS ratio = %.2fx (theoretical peak ≈ 2.00x)\n",
                     ratio);
        std::fprintf(stderr,
                     "  Phase 1 gate: >=1.8x => PROCEED to Phase 2; <1.5x => DEFER. "
                     "Phase 2 = full tile kernel with cp.async + ldmatrix + Q4-sym-s8 reorder.\n");
    }
}

}  // namespace imp
