// gemm_nvfp4_batched: batched-M FP16-output NVFP4 GEMM for spec-verify chunks
// (#998). The verify chunk forward (M = 2..33 rows) previously took the M>1
// prefill dispatch, which on GGUF-with-NVFP4-overlay dequantizes the full
// quantized source per GEMM — measured 52% of the tg window at ctx 2048 on
// Qwen3-14B Q6_K (dequant_q6k_v2 329 ms of a 634 ms window, tg −39% vs
// spec-off). The batched kernel reads each NVFP4 weight row once and reuses
// it across up to 4 activation rows per pass (same MR tiling as the LM-head
// gemv_nvfp4_kpar_batched_fp32).
//
// Reference path: gemm_nvfp4 (dequant → cuBLAS) on identical quantized
// weights — the two paths must agree within FP16 accumulation-order noise.

#include "quant/nvfp4_quant.h"
#include "quant/nvfp4_gemm.h"
#include "core/tensor.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>

namespace imp {
namespace {

class GemmNvfp4Batched : public ::testing::Test {
protected:
    void SetUp() override { cudaStreamCreate(&stream_); }
    void TearDown() override { cudaStreamDestroy(stream_); }

    // Quantize a synthetic FP16 weight [N,K], run gemm_nvfp4_batched vs
    // gemm_nvfp4 on the same activations [M,K], compare outputs [M,N].
    void run_case(int N, int K, int M) {
        std::vector<half> h_w(static_cast<size_t>(N) * K);
        std::vector<half> h_a(static_cast<size_t>(M) * K);
        for (size_t i = 0; i < h_w.size(); ++i)
            h_w[i] = __float2half(((static_cast<int>(i * 17u) % 31) - 15) * 0.01f);
        for (size_t i = 0; i < h_a.size(); ++i)
            h_a[i] = __float2half(((static_cast<int>(i * 23u) % 29) - 14) * 0.02f);

        half *d_w = nullptr, *d_a = nullptr, *d_y = nullptr, *d_y_ref = nullptr;
        cudaMalloc(&d_w, h_w.size() * sizeof(half));
        cudaMalloc(&d_a, h_a.size() * sizeof(half));
        cudaMalloc(&d_y, static_cast<size_t>(M) * N * sizeof(half));
        cudaMalloc(&d_y_ref, static_cast<size_t>(M) * N * sizeof(half));
        cudaMemcpy(d_w, h_w.data(), h_w.size() * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(half), cudaMemcpyHostToDevice);

        int64_t wshape[2] = {N, K};
        Tensor w_t(d_w, QType::F16, 2, wshape, /*on_device=*/true);
        NvFP4QuantResult qr;
        quantize_fp16_to_nvfp4(w_t, qr, stream_);
        cudaStreamSynchronize(stream_);

        cudaMemsetAsync(d_y, 0, static_cast<size_t>(M) * N * sizeof(half), stream_);
        gemm_nvfp4_batched(qr, d_a, d_y, N, K, M, stream_);
        cudaStreamSynchronize(stream_);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        int64_t ashape[2] = {M, K};
        int64_t yshape[2] = {M, N};
        Tensor a_t(d_a, QType::F16, 2, ashape, /*on_device=*/true);
        Tensor y_ref_t(d_y_ref, QType::F16, 2, yshape, /*on_device=*/true);
        cudaMemsetAsync(d_y_ref, 0, static_cast<size_t>(M) * N * sizeof(half), stream_);
        gemm_nvfp4(qr, a_t, y_ref_t, stream_);
        cudaStreamSynchronize(stream_);

        std::vector<half> h_y(static_cast<size_t>(M) * N);
        std::vector<half> h_y_ref(static_cast<size_t>(M) * N);
        cudaMemcpy(h_y.data(), d_y, h_y.size() * sizeof(half), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_y_ref.data(), d_y_ref, h_y_ref.size() * sizeof(half), cudaMemcpyDeviceToHost);

        int n_nan = 0;
        float max_abs_diff = 0.0f;
        for (size_t i = 0; i < h_y.size(); ++i) {
            float a = __half2float(h_y[i]);
            float b = __half2float(h_y_ref[i]);
            if (std::isnan(a))
                ++n_nan;
            float d = std::fabs(a - b);
            if (d > max_abs_diff)
                max_abs_diff = d;
        }
        EXPECT_EQ(n_nan, 0) << "batched GEMM produced NaNs (N=" << N << " K=" << K << " M=" << M << ")";
        EXPECT_LT(max_abs_diff, 0.5f)
            << "gemm_nvfp4_batched diverges from gemm_nvfp4 (N=" << N << " K=" << K << " M=" << M << ")";

        free_nvfp4_result(qr);
        cudaFree(d_w);
        cudaFree(d_a);
        cudaFree(d_y);
        cudaFree(d_y_ref);
    }

    cudaStream_t stream_ = nullptr;
};

// Qwen3-14B verify-chunk shapes: d_model=5120, d_ff=13824. M covers the
// capture buckets {3,5,9,17,33} plus the MR tiling edges (1,2,4,5).
TEST_F(GemmNvfp4Batched, MatchesDequantRefAtAttnProjDims) {
    for (int M : {1, 2, 3, 4, 5, 9}) run_case(5120, 5120, M);
}

TEST_F(GemmNvfp4Batched, MatchesDequantRefAtFfnDims) {
    for (int M : {2, 5, 17, 33}) run_case(13824, 5120, M);
}

TEST_F(GemmNvfp4Batched, MatchesDequantRefAtDownProjDims) {
    for (int M : {5, 9}) run_case(5120, 13824, M);
}

// Odd K edge: micro-block count not divisible by thread count.
TEST_F(GemmNvfp4Batched, MatchesDequantRefAtNarrowDims) {
    run_case(704, 2816, 7);
}

}  // namespace
}  // namespace imp
