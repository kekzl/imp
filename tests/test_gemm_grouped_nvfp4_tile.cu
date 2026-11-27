// gemm_grouped_nvfp4_tile (own grouped NVFP4 MoE GEMM, ROADMAP row 123) against the independent fp64
// reference (nvfp4_fp64_ref.h) on device-args operands, B as a slab (mode a) and as pointer arrays
// (mode b, bit-identical to a). Timing cases: tests/bench/test_gemm_grouped_nvfp4_tile_bench.cu.

#include "compute/nvfp4_sfatom_quant.h"
#include "compute/gemm_grouped_nvfp4_tile.h"
#include "core/tensor.h"
#include "nvfp4_fp64_ref.h"
#include "compute/nvfp4_sfatom_owned.h"
#include "quant/nvfp4_owned.h"
#include "quant/nvfp4_quant.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "test_gemm_grouped_nvfp4_tile_util.cuh"
namespace imp {
// Tails (7, 1 rows), several M tiles per expert (300), N tail (N=384 = 3 tiles), both B modes. K = 64 / 192
// / 320 / 704 / 2880 end on a half 64-K stage (64: the only stage), 768 on full stages.
class GroupedNvfp4TileK : public GroupedNvfp4TileTest, public ::testing::WithParamInterface<int> {};

TEST_P(GroupedNvfp4TileK, MatchesFp64Reference) {
    Grouped g;
    g.build({130, 7, 300, 64, 1}, 384, GetParam(), s_);
    const nvfp4_ref::GroupedHost h = g.host();
    const size_t out_bytes = size_t(g.M_total) * g.N * sizeof(half);
    void* own = nullptr;
    cudaMalloc(&own, out_bytes);
    std::vector<half> o[2];
    for (bool ptr_mode : {false, true}) {
        std::vector<half>& out = o[ptr_mode ? 1 : 0];
        out.resize(size_t(g.M_total) * g.N);
        cudaMemset(own, 0xff, out_bytes);  // NaN fill: a row the kernel misses fails the check
        ASSERT_TRUE(gemm_grouped_nvfp4_tile(g.ne, g.N, g.K, g.M_total, g.args(own, ptr_mode), s_));
        ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
        cudaMemcpy(out.data(), own, out_bytes, cudaMemcpyDeviceToHost);
        const nvfp4_ref::GroupedErr err = nvfp4_ref::check_grouped(h, out, 1e-2);
        std::fprintf(
            stderr,
            "[grouped tile] K=%d mode %c: checked %zu bad %zu nonfinite %zu max_rel %.3e max_abs %.3e "
            "max|ref| %.4g\n",
            g.K, ptr_mode ? 'b' : 'a', err.checked, err.bad, err.nonfinite, err.max_rel, err.max_abs,
            err.max_ref);
        EXPECT_EQ(err.checked, size_t(g.M_total) * g.N);
        EXPECT_GT(err.max_ref, 0.0);
        EXPECT_EQ(err.bad, 0u) << "mode " << (ptr_mode ? 'b' : 'a');
    }
    size_t mism = 0;
    for (size_t i = 0; i < o[0].size(); ++i)
        mism += __half_as_ushort(o[0][i]) != __half_as_ushort(o[1][i]) ? 1 : 0;
    EXPECT_EQ(mism, 0u) << "mode b differs from mode a";
    cudaFree(own);
    g.release();
}

INSTANTIATE_TEST_SUITE_P(K, GroupedNvfp4TileK, ::testing::Values(64, 192, 320, 768, 704, 2880));

}  // namespace imp
