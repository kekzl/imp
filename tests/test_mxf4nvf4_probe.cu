#include <gtest/gtest.h>
#include "bench/attention_mxf4nvf4_probe.h"
#include <cuda_runtime.h>

namespace imp {
namespace {

// Gate for Project B feasibility: does the SageAttention3-style block-scale
// MMA instruction actually launch and run on sm_120f hardware with CUDA 13.2?
// Compile already demonstrated PTX acceptance — this test covers runtime.
TEST(Mxf4nvf4ProbeTest, BlockScaleMMA_LaunchesCleanly) {
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    bool ok = probe_mxf4nvf4_blockscale(stream);

    cudaStreamDestroy(stream);
    EXPECT_TRUE(ok) << "mma.sync.kind::mxf4nvf4.block_scale failed at runtime "
                    << "on sm_120f — Project B (MXFP4 FMHA hardware block-scale "
                    << "upgrade) is blocked until this is resolved.";
}

// Numerical sanity: A=0 must force D=0 regardless of B and scale factors.
// If this fails, our assumptions about operand encoding or layout are
// wrong and Stage 3 (full integration) would be building on shaky ground.
TEST(Mxf4nvf4ProbeTest, BlockScaleMMA_ZeroAZeroesOutput) {
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    float d[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
    bool ok = probe_mxf4nvf4_allzero_a(stream, d);

    cudaStreamDestroy(stream);
    ASSERT_TRUE(ok) << "Kernel launch or memcpy failed";
    EXPECT_FLOAT_EQ(d[0], 0.0f);
    EXPECT_FLOAT_EQ(d[1], 0.0f);
    EXPECT_FLOAT_EQ(d[2], 0.0f);
    EXPECT_FLOAT_EQ(d[3], 0.0f);
}

}  // namespace
}  // namespace imp
