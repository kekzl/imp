// tests/test_gemm_grouped_nvfp4_smallM.cu
#include <gtest/gtest.h>
#include "compute/gemm_grouped_nvfp4_smallM.h"
#include <vector>
#include <cuda_runtime.h>

extern "C" void smallM_smoke_single_mma(float*, const uint32_t*, const uint32_t*,
                                        uint32_t, uint32_t, cudaStream_t);

namespace {

bool has_sm120() {
    int dev = 0; cudaGetDevice(&dev);
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    return major * 10 + minor >= 120;
}

TEST(SmallMMmaWrapper, IssuesSingleMma) {
    if (!has_sm120()) GTEST_SKIP() << "SM120 required";

    uint32_t a[4] = {0, 0, 0, 0}, b[2] = {0, 0};
    uint32_t* d_a = nullptr; uint32_t* d_b = nullptr;
    cudaMalloc(&d_a, sizeof(a)); cudaMalloc(&d_b, sizeof(b));
    cudaMemcpy(d_a, a, sizeof(a), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(b), cudaMemcpyHostToDevice);

    float* d_out = nullptr;
    cudaMalloc(&d_out, 4 * sizeof(float));
    float poison[4] = {-99.f, -99.f, -99.f, -99.f};
    cudaMemcpy(d_out, poison, sizeof(poison), cudaMemcpyHostToDevice);

    smallM_smoke_single_mma(d_out, d_a, d_b, 0u, 0u, /*stream*/nullptr);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    float h_out[4];
    cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);
    // All-zero MMA must produce zero (initial acc=0 + zero*zero*any_scale = 0).
    EXPECT_EQ(h_out[0], 0.f); EXPECT_EQ(h_out[1], 0.f);
    EXPECT_EQ(h_out[2], 0.f); EXPECT_EQ(h_out[3], 0.f);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
}

TEST(SmallMMmaWrapper, NonZeroProducesNonZero) {
    if (!has_sm120()) GTEST_SKIP() << "SM120 required";

    // Patterned non-zero FP4 inputs — exact values don't matter, just nonzero.
    uint32_t a[4] = {0x11111111, 0x11111111, 0x11111111, 0x11111111};
    uint32_t b[2] = {0x11111111, 0x11111111};
    // UE4M3 scale ≈ 1.0 — 0x38383838 matches BENCH_PREAMBLE in variants_bench.
    uint32_t sfa = 0x38383838u;
    uint32_t sfb = 0x38383838u;

    uint32_t* d_a = nullptr; uint32_t* d_b = nullptr;
    cudaMalloc(&d_a, sizeof(a)); cudaMalloc(&d_b, sizeof(b));
    cudaMemcpy(d_a, a, sizeof(a), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(b), cudaMemcpyHostToDevice);

    float* d_out = nullptr;
    cudaMalloc(&d_out, 4 * sizeof(float));
    cudaMemset(d_out, 0, 4 * sizeof(float));

    smallM_smoke_single_mma(d_out, d_a, d_b, sfa, sfb, /*stream*/nullptr);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    float h_out[4];
    cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);
    bool any_nonzero = false;
    for (int i = 0; i < 4; ++i) if (h_out[i] != 0.f) any_nonzero = true;
    EXPECT_TRUE(any_nonzero) << "MMA with nonzero inputs produced all zeros";

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
}

TEST(SmallMScheduler, PicksMinimalTile) {
    using imp::detail::pick_m_tile;
    EXPECT_EQ(pick_m_tile(1),   16);
    EXPECT_EQ(pick_m_tile(16),  16);
    EXPECT_EQ(pick_m_tile(17),  32);
    EXPECT_EQ(pick_m_tile(32),  32);
    EXPECT_EQ(pick_m_tile(40),  64);
    EXPECT_EQ(pick_m_tile(64),  64);
    EXPECT_EQ(pick_m_tile(128), 128);
    EXPECT_EQ(pick_m_tile(200), 128);
}

TEST(SmallMScheduler, WorkQueueOrderedByTileSize) {
    using imp::detail::build_work_queue;
    int M_per[] = {32, 100, 8, 0, 200};   // 5 experts; e=3 inactive
    auto q = build_work_queue(5, M_per, 256);
    ASSERT_FALSE(q.empty());

    // First items must be tile_M=128 (from e=4 with M=200, two M-tiles needed)
    EXPECT_EQ(q[0].m_tile_size, 128);
    // Last items must be tile_M=16 (from e=2 with M=8)
    EXPECT_EQ(q.back().m_tile_size, 16);
    // No work for inactive expert e=3
    for (auto& wi : q) EXPECT_NE(wi.expert_id, 3);
}

}  // anonymous namespace
