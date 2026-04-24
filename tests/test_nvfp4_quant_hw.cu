#include <gtest/gtest.h>
#include "compute/nvfp4_quant_hw.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>

namespace imp {
namespace {

// Round-trip with HW-layout scales: validates that the offset formula
// is self-consistent (quant writes scale at offset X, dequant reads the
// same scale at offset X). If the formula is broken on one side only,
// error will be catastrophic. Self-consistent means the round-trip
// works even without a matching MMA.
class Nvfp4QuantHwTest : public ::testing::Test {
protected:
    void run_roundtrip(int batch, int heads, int tokens, int head_dim, float sigma) {
        const int64_t in_elems   = (int64_t)batch * heads * tokens * head_dim;
        const int64_t nvfp4_bytes = in_elems / 2;
        // HW scale layout rounds tokens up to 64 per batch-head block. 8 scales
        // per token (hd=128) or 4 (hd=64). Minimum footprint: 128 bytes per
        // 64-token block per (batch, head).
        const int64_t tokens_rounded = ((tokens + 63) / 64) * 64;
        const int64_t sf_bytes = (int64_t)batch * heads * tokens_rounded * (head_dim / 16);

        // Generate input.
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, sigma);
        std::vector<half> h_input(in_elems);
        double input_ss = 0.0;
        for (int64_t i = 0; i < in_elems; ++i) {
            float v = dist(rng);
            h_input[i] = __float2half(v);
            input_ss += v * v;
        }
        double input_rms = std::sqrt(input_ss / in_elems);

        half*    d_input  = nullptr;
        uint8_t* d_nvfp4  = nullptr;
        uint8_t* d_sf     = nullptr;
        half*    d_output = nullptr;

        ASSERT_EQ(cudaMalloc(&d_input,  in_elems * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_nvfp4,  nvfp4_bytes),             cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_sf,     sf_bytes),                cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_output, in_elems * sizeof(half)), cudaSuccess);

        cudaMemset(d_sf, 0, sf_bytes);
        cudaMemcpy(d_input, h_input.data(), in_elems * sizeof(half), cudaMemcpyHostToDevice);

        // Strides (contiguous, row-major [B, H, T, D]).
        int s_bz = heads * tokens * head_dim;
        int s_h  = tokens * head_dim;
        int s_t  = head_dim;

        int s_bz_out = heads * tokens * head_dim / 2;
        int s_h_out  = tokens * head_dim / 2;
        int s_t_out  = head_dim / 2;

        // Scale strides: HW layout stores (head_dim/16) bytes per token.
        // Kernel advances base by (token_id/64) * 64 * s_t_sf to reach the
        // next 64-token block, so s_t_sf must equal (head_dim/16) bytes/token.
        int s_t_sf  = head_dim / 16;
        int s_h_sf  = tokens_rounded * s_t_sf;
        int s_bz_sf = heads * s_h_sf;

        bool q_ok = nvfp4_quant_hw_fp16(
            d_input, d_nvfp4, d_sf,
            batch, heads, tokens, head_dim,
            s_bz, s_h, s_t,
            s_bz_out, s_h_out, s_t_out,
            s_bz_sf, s_h_sf, s_t_sf,
            0);
        ASSERT_TRUE(q_ok);

        bool dq_ok = nvfp4_dequant_hw_fp16(
            d_nvfp4, d_sf, d_output,
            batch, heads, tokens, head_dim,
            s_bz_out, s_h_out, s_t_out,
            s_bz, s_h, s_t,
            s_bz_sf, s_h_sf, s_t_sf,
            0);
        ASSERT_TRUE(dq_ok);

        cudaDeviceSynchronize();

        std::vector<half> h_output(in_elems);
        cudaMemcpy(h_output.data(), d_output, in_elems * sizeof(half), cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_nvfp4);
        cudaFree(d_sf);
        cudaFree(d_output);

        double err_ss = 0.0;
        for (int64_t i = 0; i < in_elems; ++i) {
            float a = __half2float(h_input[i]);
            float b = __half2float(h_output[i]);
            double d = a - b;
            err_ss += d * d;
        }
        double err_rms = std::sqrt(err_ss / in_elems);
        double rel     = err_rms / input_rms;

        std::cout << "[  INFO    ] HW round-trip (b=" << batch
                  << " h=" << heads << " t=" << tokens << " d=" << head_dim
                  << " σ=" << sigma << "): RMSE " << (rel * 100.0) << "% of input RMS"
                  << std::endl;

        // Self-consistency bound: if the scale layout formula matches on both
        // sides, round-trip should be equivalent to the linear-layout case.
        // Reference from nvfp4_quant_ref: ~9.5% for σ=1 Gaussian.
        EXPECT_LT(rel, 0.15);
    }
};

TEST_F(Nvfp4QuantHwTest, HeadDim128_Single64TokenBlock) {
    // Exactly one 64-token block, single batch/head — simplest case for the
    // layout formula.
    run_roundtrip(/*batch=*/1, /*heads=*/1, /*tokens=*/64, /*head_dim=*/128, 1.0f);
}

TEST_F(Nvfp4QuantHwTest, HeadDim128_MultipleBlocks) {
    // Several 64-token blocks exercise the (token_id / 64) stride in the
    // scale buffer.
    run_roundtrip(1, 1, 256, 128, 1.0f);
}

TEST_F(Nvfp4QuantHwTest, HeadDim128_BatchAndHeads) {
    // Non-trivial batch + head dims.
    run_roundtrip(2, 4, 128, 128, 1.0f);
}

TEST_F(Nvfp4QuantHwTest, HeadDim64_SingleBlock) {
    // head_dim=64 path uses CVT_FP4_ELTS_PER_THREAD=8 (different layout).
    run_roundtrip(1, 1, 64, 64, 1.0f);
}

} // namespace
} // namespace imp
