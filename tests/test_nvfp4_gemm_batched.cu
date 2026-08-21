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
#include "runtime/process_diag.h"

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

// Accumulate (beta=1) variant for the o/down residual-add GEMMs (#1055):
// y must end as W@x + y_prev, matching cuBLAS beta=1 semantics.
TEST_F(GemmNvfp4Batched, AccumulateAddsIntoOutput) {
    const int N = 1024, K = 2048, M = 3;
    std::vector<half> h_w(static_cast<size_t>(N) * K);
    std::vector<half> h_a(static_cast<size_t>(M) * K);
    std::vector<half> h_res(static_cast<size_t>(M) * N);
    for (size_t i = 0; i < h_w.size(); ++i)
        h_w[i] = __float2half(((static_cast<int>(i * 13u) % 27) - 13) * 0.01f);
    for (size_t i = 0; i < h_a.size(); ++i)
        h_a[i] = __float2half(((static_cast<int>(i * 7u) % 23) - 11) * 0.02f);
    for (size_t i = 0; i < h_res.size(); ++i)
        h_res[i] = __float2half(((static_cast<int>(i * 5u) % 19) - 9) * 0.05f);

    half *d_w = nullptr, *d_a = nullptr, *d_y = nullptr, *d_y_ref = nullptr;
    cudaMalloc(&d_w, h_w.size() * sizeof(half));
    cudaMalloc(&d_a, h_a.size() * sizeof(half));
    cudaMalloc(&d_y, h_res.size() * sizeof(half));
    cudaMalloc(&d_y_ref, h_res.size() * sizeof(half));
    cudaMemcpy(d_w, h_w.data(), h_w.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_res.data(), h_res.size() * sizeof(half), cudaMemcpyHostToDevice);

    int64_t wshape[2] = {N, K};
    Tensor w_t(d_w, QType::F16, 2, wshape, true);
    NvFP4QuantResult qr;
    quantize_fp16_to_nvfp4(w_t, qr, stream_);
    cudaStreamSynchronize(stream_);

    gemm_nvfp4_batched_acc(qr, d_a, d_y, N, K, M, stream_);
    cudaStreamSynchronize(stream_);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    // Reference: plain batched into a zero buffer, then add residual on host.
    cudaMemsetAsync(d_y_ref, 0, h_res.size() * sizeof(half), stream_);
    gemm_nvfp4_batched(qr, d_a, d_y_ref, N, K, M, stream_);
    cudaStreamSynchronize(stream_);

    std::vector<half> h_y(h_res.size()), h_y_ref(h_res.size());
    cudaMemcpy(h_y.data(), d_y, h_y.size() * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y_ref.data(), d_y_ref, h_y_ref.size() * sizeof(half), cudaMemcpyDeviceToHost);

    float max_abs_diff = 0.0f;
    for (size_t i = 0; i < h_y.size(); ++i) {
        float want = __half2float(h_y_ref[i]) + __half2float(h_res[i]);
        float got = __half2float(h_y[i]);
        max_abs_diff = std::max(max_abs_diff, std::fabs(got - want));
    }
    EXPECT_LT(max_abs_diff, 0.05f) << "accumulate variant diverges from ref + residual";

    free_nvfp4_result(qr);
    cudaFree(d_w);
    cudaFree(d_a);
    cudaFree(d_y);
    cudaFree(d_y_ref);
}

// ---------------------------------------------------------------------------
// Bit-parity between the verify chunk and the M=1 decode GEMV
// (speculative.verify_row_parity).
//
// These two paths compute the same projections and, until 2026-08-21, did not
// agree on the answer. Both inner loops are instruction-for-instruction
// identical; they differed only in how wide the K reduction is - decode groups
// the products into 32 partial sums (one per warp lane), the batched verify
// kernel into 128 (one per block thread). Same mathematics, different float
// rounding, and it reached the STOP decision: at speculative.mtp_k=1 on
// Qwen3.8-27B-NVFP4 it truncated answers after ~40 tokens
// (docs/LIMITATIONS.md).
//
// "Close enough" is not the property under test. Row m of the batched result
// must be BIT-identical to a standalone decode GEMV on activation row m, so
// the comparison is on the raw uint16 bits, not a tolerance.
// ---------------------------------------------------------------------------
class Nvfp4VerifyRowParity : public ::testing::Test {
protected:
    void SetUp() override { cudaStreamCreate(&stream_); }
    void TearDown() override {
        process_diag_set_verify_row_parity(false);
        cudaStreamDestroy(stream_);
    }

    // Returns the number of elements that differ in their raw bits.
    int mismatching_bits(int N, int K, int M, bool parity_on) {
        std::vector<half> h_w(static_cast<size_t>(N) * K);
        std::vector<half> h_a(static_cast<size_t>(M) * K);
        // The input has to be able to SHOW the difference. A first version of
        // this test used small evenly-spaced integers scaled by 0.01/0.02, and
        // the 32-wide and 128-wide reductions came out bit-identical on it -
        // the control passed for the wrong reason and would have blessed a
        // no-op kernel. Real activations after RMSNorm span orders of
        // magnitude and cancel; that is what makes the grouping matter, so the
        // data here does too.
        uint32_t rng = 0x9E3779B9u;
        auto next = [&rng]() {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            return rng;
        };
        auto wide = [&next](float lo_exp, float hi_exp) {
            const float u = static_cast<float>(next() & 0xFFFFFF) / 16777216.0f;
            const float sign = (next() & 1u) ? 1.0f : -1.0f;
            return sign * std::pow(2.0f, lo_exp + u * (hi_exp - lo_exp));
        };
        for (size_t i = 0; i < h_w.size(); ++i)
            h_w[i] = __float2half(wide(-6.0f, 0.0f));
        for (size_t i = 0; i < h_a.size(); ++i)
            h_a[i] = __float2half(wide(-8.0f, 3.0f));

        half *d_w = nullptr, *d_a = nullptr, *d_y = nullptr, *d_ref = nullptr;
        cudaMalloc(&d_w, h_w.size() * sizeof(half));
        cudaMalloc(&d_a, h_a.size() * sizeof(half));
        cudaMalloc(&d_y, static_cast<size_t>(M) * N * sizeof(half));
        cudaMalloc(&d_ref, static_cast<size_t>(M) * N * sizeof(half));
        cudaMemcpy(d_w, h_w.data(), h_w.size() * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(half), cudaMemcpyHostToDevice);

        int64_t wshape[2] = {N, K};
        Tensor w_t(d_w, QType::F16, 2, wshape, /*on_device=*/true);
        NvFP4QuantResult qr;
        quantize_fp16_to_nvfp4(w_t, qr, stream_);
        cudaStreamSynchronize(stream_);

        // The reference IS the decode path: one standalone GEMV per activation
        // row, which is what a non-speculative step would run.
        process_diag_set_verify_row_parity(false);
        for (int m = 0; m < M; ++m)
            gemv_nvfp4_kpar(qr, d_a + static_cast<size_t>(m) * K, d_ref + static_cast<size_t>(m) * N, N, K,
                            stream_);

        process_diag_set_verify_row_parity(parity_on);
        cudaMemsetAsync(d_y, 0, static_cast<size_t>(M) * N * sizeof(half), stream_);
        gemm_nvfp4_batched(qr, d_a, d_y, N, K, M, stream_);
        cudaStreamSynchronize(stream_);
        EXPECT_EQ(cudaGetLastError(), cudaSuccess);

        std::vector<uint16_t> y(static_cast<size_t>(M) * N), ref(static_cast<size_t>(M) * N);
        cudaMemcpy(y.data(), d_y, y.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(ref.data(), d_ref, ref.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
        int diff = 0;
        for (size_t i = 0; i < y.size(); ++i)
            if (y[i] != ref[i])
                ++diff;

        free_nvfp4_result(qr);
        cudaFree(d_w);
        cudaFree(d_a);
        cudaFree(d_y);
        cudaFree(d_ref);
        return diff;
    }

    cudaStream_t stream_{};
};

// N=8192,K=8192: mr_blocks=1024 >= 6*170 and n_mb=512 <= 512, so decode takes
// the 32-lane multirow branch. This is the shape class the divergence lives in.
TEST_F(Nvfp4VerifyRowParity, MultirowShapeIsBitIdenticalToDecodeWhenOn) {
    for (int M : {1, 2, 3, 4}) {
        EXPECT_EQ(mismatching_bits(8192, 8192, M, /*parity_on=*/true), 0)
            << "verify row " << M << " must match the decode GEMV bit for bit";
    }
}

// The control that makes the test above mean something: with the knob off the
// same shape DOES diverge. If this ever reads 0, the two paths have converged
// for some other reason and the parity kernel is no longer what is being
// measured.
TEST_F(Nvfp4VerifyRowParity, MultirowShapeDivergesWhenOff) {
    EXPECT_GT(mismatching_bits(8192, 8192, 4, /*parity_on=*/false), 0)
        << "expected the 128-wide kpar reduction to differ from the 32-wide decode one";
}

// The real down-projection shape: n_mb = 17408/16 = 1088 > 512, so use_multirow
// is false and BOTH paths take the 128-wide reduction. Asserted rather than
// assumed, because the whole point of this exercise is that "the same width"
// was inferred once and turned out to be only half the story.
TEST_F(Nvfp4VerifyRowParity, DownProjectionShapeAgreesWithoutTheKnob) {
    EXPECT_EQ(mismatching_bits(5120, 17408, 2, /*parity_on=*/false), 0)
        << "5120x17408 should already reduce K the same way on both paths";
}

// N=1024: mr_blocks=128 < 6*170, so decode takes the 128-wide kpar kernel and
// the existing batched path already agrees with it. The knob must be a no-op
// here rather than routing these shapes somewhere new.
TEST_F(Nvfp4VerifyRowParity, NonMultirowShapeAlreadyAgreesEitherWay) {
    EXPECT_EQ(mismatching_bits(1024, 1024, 4, /*parity_on=*/false), 0);
    EXPECT_EQ(mismatching_bits(1024, 1024, 4, /*parity_on=*/true), 0);
}

}  // namespace
}  // namespace imp
