// Phase 3a FP4 PV microbench wrapper — see bench/fp4_pv_bench.h for context.
//
// Two GTests:
//   - Fp4PvBenchTest.Accuracy: synthetic post-softmax accuracy, no gate
//   - Fp4PvBenchTest.ThroughputBlockscaleBeatsHmma: raw MMA throughput,
//     asserts the 2× theoretical speedup over HMMA m16n8k16 is at least
//     ≥ 2.0× in practice (the "≥ 2.5× for Phase 3b proceed" gate in the
//     design memo is informational here — printed but not enforced, so a
//     marginal result still produces actionable data).

#include <gtest/gtest.h>
#include "bench/fp4_pv_bench.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace imp {
namespace {

TEST(Fp4PvBenchTest, Accuracy) {
    // 2000 rows × 64 head_dim = 128k samples. K=64 = one mxf4nvf4 tile.
    // Deterministic seed so the percentile numbers are reproducible.
    constexpr int N_ROWS = 2000;
    constexpr int K = 64;
    constexpr int HEAD_DIM = 64;

    Fp4PvAccuracyResult r = bench_fp4_pv_accuracy(N_ROWS, K, HEAD_DIM, /*seed=*/42);

    std::printf("\n=== Phase 3a FP4 PV accuracy on synthetic post-softmax data ===\n");
    std::printf("  rows=%d  K=%d  head_dim=%d  (single-level FP4, NO two-level accumulator)\n",
                r.n_rows, r.K, r.head_dim);
    std::printf("  Absolute error |O_fp4 - O_ref|:\n");
    std::printf("    median  %12.4e\n", r.abs_err_median);
    std::printf("    p99     %12.4e\n", r.abs_err_p99);
    std::printf("    max     %12.4e\n", r.abs_err_max);
    std::printf("  Relative error |O_fp4 - O_ref| / |O_ref|:\n");
    std::printf("    median  %7.2f %%\n", r.rel_err_median * 100.0);
    std::printf("    p90     %7.2f %%\n", r.rel_err_p90 * 100.0);
    std::printf("    p99     %7.2f %%\n", r.rel_err_p99 * 100.0);
    std::printf("    max     %7.2f %%\n", r.rel_err_max * 100.0);
    std::printf("  Catastrophic outputs (rel err > 50%%):  %6.2f %% of %d\n",
                r.frac_rel_err_above_50pct * 100.0, r.n_rows * r.head_dim);
    std::printf("\n");
    std::printf("Interpretation (Phase 3a design memo §5):\n");
    std::printf("  - p99 < 5%%   → single-level FP4 PV is viable; skip Phase 3b accumulator.\n");
    std::printf("  - p99 < 50%%  → Phase 3b two-level accumulator is the right next step.\n");
    std::printf("  - p99 > 50%%  → Phase 3 is dead without the residual path.\n\n");

    // No assertion — the bench's job is to PRODUCE the data, not to gate
    // CI on a specific quality threshold.
    EXPECT_GT(r.rel_err_median, 0.0f);
    EXPECT_LE(r.rel_err_median, 1.0f);
}

TEST(Fp4PvBenchTest, ThroughputBlockscaleBeatsHmma) {
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    constexpr int WARPS = 170;            // 1 per SM on RTX 5090
    constexpr int ITERATIONS = 1 << 20;   // 1M MMAs per warp

    Fp4PvThroughputResult r = bench_fp4_pv_throughput(WARPS, ITERATIONS, stream);
    cudaStreamDestroy(stream);

    std::printf("\n=== Phase 3a raw MMA throughput, PV reference ===\n");
    std::printf("  hmma.m16n8k16 (FP16 PV)              %7.2f ms  %7.2f TOPS\n",
                r.hmma_ms, r.hmma_tops);
    std::printf("  mxf4nvf4.block_scale.m16n8k64 (FP4)  %7.2f ms  %7.2f TOPS\n",
                r.blockscale_ms, r.blockscale_tops);
    std::printf("  Speedup (blockscale / hmma):         %5.2fx\n", r.speedup);
    std::printf("  Phase 3b gate (≥ 2.5×): %s\n\n",
                r.speedup >= 2.5 ? "PASS" : "FAIL — re-examine before integrating");

    EXPECT_GT(r.hmma_tops, 0.0);
    EXPECT_GT(r.blockscale_tops, 0.0);
    // The MMA-level ratio is the lower bound on Phase 3b's eventual e2e
    // payoff. If it isn't even ≥ 2.0×, integration won't help.
    EXPECT_GT(r.speedup, 2.0)
        << "mxf4nvf4.m16n8k64 raw-MMA speedup over HMMA m16n8k16 is < 2.0× — "
        << "Phase 3b integration cannot recover the +13 % e2e ceiling from this base.";
}

}  // namespace
}  // namespace imp
