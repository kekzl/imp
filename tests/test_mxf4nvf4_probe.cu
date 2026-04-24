#include <gtest/gtest.h>
#include "compute/attention_mxf4nvf4_probe.h"
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

} // namespace
} // namespace imp
