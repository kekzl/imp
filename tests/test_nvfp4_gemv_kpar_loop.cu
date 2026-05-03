// Synthetic repro attempt for the gemv_nvfp4_kpar M>1 per-row-loop pathology
// that corrupted Gemma-4-26B-A4B NVFP4 legacy-MoE prefill (memo
// llm_compressor_phase2_item2_2026_04_26 + PR #65 fix).
//
// Bug context: src/graph/executor_forward_moe.cu had a manual M loop calling
// gemv_nvfp4_kpar once per row for M>1 prefill on per-expert NVFP4 weights.
// At Gemma-4 expert dims (N=704, K=2816 for gate/up; N=2816, K=704 for down),
// this produced garbage output empirically while the dequant→cuBLAS path
// (gemm_nvfp4) was correct on identical weights. Fix shipped: route M>1 to
// gemm_nvfp4 in the legacy MoE branch.
//
// What this test pins down — NEGATIVE RESULT:
//   The per-row gemv_kpar loop is mathematically equivalent to gemm_nvfp4 in
//   isolation. Both Gemma-4 expert aspect ratios pass with max_abs_diff well
//   below the 0.5 threshold. So the bug is NOT a defect of gemv_nvfp4_kpar
//   itself at these shapes — it must come from something the synthetic case
//   doesn't reproduce: launch-queue interaction at MoE scale (8 experts × 30
//   layers × 3 projections × M kernels), PDL cross-contamination from
//   neighboring MoE-routing kernels, or numerical edge cases triggered by
//   real weight/activation distributions.
//
// Value: this test is a regression gate ensuring per-row gemv_kpar stays
// numerically equivalent to the dequant fallback at the affected shapes —
// any future "fix" that breaks the kernel in isolation will fail here. The
// real bug remains open; restoring native NVFP4 GEMV for legacy MoE prefill
// (avoiding the dequant→FP16 overhead) needs an end-to-end repro that does
// capture the actual launch context.

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

class GemvKparLoopRepro : public ::testing::Test {
protected:
    void SetUp() override { cudaStreamCreate(&stream_); }
    void TearDown() override { cudaStreamDestroy(stream_); }
    cudaStream_t stream_ = nullptr;
};

// Run per-row gemv_nvfp4_kpar M times in a tight loop on a single stream and
// compare to gemm_nvfp4 single-call. Shape matches Gemma-4-26B gate/up
// projection per-expert tile: N=704, K=2816, with M=11 prefill rows.
TEST_F(GemvKparLoopRepro, PerRowLoopMatchesSingleCallAtGemma4GateUpDims) {
    const int N = 704;
    const int K = 2816;
    const int M = 11;

    std::vector<half> h_w(static_cast<size_t>(N) * K);
    std::vector<half> h_a(static_cast<size_t>(M) * K);
    for (size_t i = 0; i < h_w.size(); ++i) {
        h_w[i] = __float2half(((static_cast<int>(i * 17u) % 31) - 15) * 0.01f);
    }
    for (size_t i = 0; i < h_a.size(); ++i) {
        h_a[i] = __float2half(((static_cast<int>(i * 23u) % 29) - 14) * 0.02f);
    }

    half *d_w = nullptr, *d_a = nullptr;
    half *d_y_loop = nullptr, *d_y_single = nullptr;
    cudaMalloc(&d_w, h_w.size() * sizeof(half));
    cudaMalloc(&d_a, h_a.size() * sizeof(half));
    cudaMalloc(&d_y_loop, static_cast<size_t>(M) * N * sizeof(half));
    cudaMalloc(&d_y_single, static_cast<size_t>(M) * N * sizeof(half));
    cudaMemcpy(d_w, h_w.data(), h_w.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(half), cudaMemcpyHostToDevice);

    int64_t wshape[2] = {N, K};
    Tensor w_t(d_w, QType::F16, 2, wshape, /*on_device=*/true);

    NvFP4QuantResult qr;
    quantize_fp16_to_nvfp4(w_t, qr, stream_);
    cudaStreamSynchronize(stream_);

    // Path A: per-row gemv_kpar loop — the suspected bug path.
    cudaMemsetAsync(d_y_loop, 0, static_cast<size_t>(M) * N * sizeof(half), stream_);
    for (int r = 0; r < M; ++r) {
        const half* a_row = d_a + static_cast<size_t>(r) * K;
        half* c_row = d_y_loop + static_cast<size_t>(r) * N;
        gemv_nvfp4_kpar(qr, a_row, c_row, N, K, stream_);
    }
    cudaStreamSynchronize(stream_);

    // Path B: gemm_nvfp4 single-call (dequant → cuBLAS) — the bypass path.
    int64_t ashape[2] = {M, K};
    int64_t yshape[2] = {M, N};
    Tensor a_t(d_a, QType::F16, 2, ashape, /*on_device=*/true);
    Tensor y_single_t(d_y_single, QType::F16, 2, yshape, /*on_device=*/true);
    cudaMemsetAsync(d_y_single, 0, static_cast<size_t>(M) * N * sizeof(half), stream_);
    gemm_nvfp4(qr, a_t, y_single_t, stream_);
    cudaStreamSynchronize(stream_);

    std::vector<half> h_y_loop(static_cast<size_t>(M) * N);
    std::vector<half> h_y_single(static_cast<size_t>(M) * N);
    cudaMemcpy(h_y_loop.data(), d_y_loop, h_y_loop.size() * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y_single.data(), d_y_single, h_y_single.size() * sizeof(half), cudaMemcpyDeviceToHost);

    int n_nan_loop = 0, n_nan_single = 0;
    int n_inf_loop = 0, n_inf_single = 0;
    int n_large_diff = 0;
    float max_abs_diff = 0.0f;
    float sum_sq_loop = 0.0f, sum_sq_single = 0.0f;
    for (size_t i = 0; i < h_y_loop.size(); ++i) {
        float a = __half2float(h_y_loop[i]);
        float b = __half2float(h_y_single[i]);
        if (std::isnan(a))
            ++n_nan_loop;
        if (std::isnan(b))
            ++n_nan_single;
        if (std::isinf(a))
            ++n_inf_loop;
        if (std::isinf(b))
            ++n_inf_single;
        if (!std::isnan(a) && !std::isnan(b)) {
            float d = std::fabs(a - b);
            if (d > 0.5f)
                ++n_large_diff;
            if (d > max_abs_diff)
                max_abs_diff = d;
            sum_sq_loop += a * a;
            sum_sq_single += b * b;
        }
    }

    EXPECT_EQ(n_nan_loop, 0) << "gemv_kpar loop produced NaNs";
    EXPECT_EQ(n_inf_loop, 0) << "gemv_kpar loop produced Infs";
    EXPECT_EQ(n_nan_single, 0) << "gemm_nvfp4 single-call produced NaNs";
    EXPECT_EQ(n_inf_single, 0) << "gemm_nvfp4 single-call produced Infs";
    EXPECT_LT(max_abs_diff, 0.5f) << "Per-row gemv_kpar loop diverges from gemm_nvfp4 single-call at "
                                  << "Gemma-4 gate/up dims (N=" << N << " K=" << K << " M=" << M << ").\n"
                                  << "  n_large_diff=" << n_large_diff << " / " << (M * N) << "\n"
                                  << "  max_abs_diff=" << max_abs_diff << "\n"
                                  << "  ||y_loop||^2=" << sum_sq_loop << "\n"
                                  << "  ||y_single||^2=" << sum_sq_single;

    free_nvfp4_result(qr);
    cudaFree(d_w);
    cudaFree(d_a);
    cudaFree(d_y_loop);
    cudaFree(d_y_single);
}

// Same comparison at the down-projection per-expert dims: N=2816 (d_model),
// K=704 (intermediate). Tests the inverse aspect ratio.
TEST_F(GemvKparLoopRepro, PerRowLoopMatchesSingleCallAtGemma4DownDims) {
    const int N = 2816;
    const int K = 704;
    const int M = 11;

    std::vector<half> h_w(static_cast<size_t>(N) * K);
    std::vector<half> h_a(static_cast<size_t>(M) * K);
    for (size_t i = 0; i < h_w.size(); ++i) {
        h_w[i] = __float2half(((static_cast<int>(i * 13u) % 23) - 11) * 0.015f);
    }
    for (size_t i = 0; i < h_a.size(); ++i) {
        h_a[i] = __float2half(((static_cast<int>(i * 19u) % 27) - 13) * 0.018f);
    }

    half *d_w = nullptr, *d_a = nullptr;
    half *d_y_loop = nullptr, *d_y_single = nullptr;
    cudaMalloc(&d_w, h_w.size() * sizeof(half));
    cudaMalloc(&d_a, h_a.size() * sizeof(half));
    cudaMalloc(&d_y_loop, static_cast<size_t>(M) * N * sizeof(half));
    cudaMalloc(&d_y_single, static_cast<size_t>(M) * N * sizeof(half));
    cudaMemcpy(d_w, h_w.data(), h_w.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(half), cudaMemcpyHostToDevice);

    int64_t wshape[2] = {N, K};
    Tensor w_t(d_w, QType::F16, 2, wshape, /*on_device=*/true);

    NvFP4QuantResult qr;
    quantize_fp16_to_nvfp4(w_t, qr, stream_);
    cudaStreamSynchronize(stream_);

    cudaMemsetAsync(d_y_loop, 0, static_cast<size_t>(M) * N * sizeof(half), stream_);
    cudaMemsetAsync(d_y_single, 0, static_cast<size_t>(M) * N * sizeof(half), stream_);

    for (int r = 0; r < M; ++r) {
        const half* a_row = d_a + static_cast<size_t>(r) * K;
        half* c_row = d_y_loop + static_cast<size_t>(r) * N;
        gemv_nvfp4_kpar(qr, a_row, c_row, N, K, stream_);
    }
    cudaStreamSynchronize(stream_);

    int64_t ashape[2] = {M, K};
    int64_t yshape[2] = {M, N};
    Tensor a_t(d_a, QType::F16, 2, ashape, /*on_device=*/true);
    Tensor y_single_t(d_y_single, QType::F16, 2, yshape, /*on_device=*/true);
    gemm_nvfp4(qr, a_t, y_single_t, stream_);
    cudaStreamSynchronize(stream_);

    std::vector<half> h_y_loop(static_cast<size_t>(M) * N);
    std::vector<half> h_y_single(static_cast<size_t>(M) * N);
    cudaMemcpy(h_y_loop.data(), d_y_loop, h_y_loop.size() * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y_single.data(), d_y_single, h_y_single.size() * sizeof(half), cudaMemcpyDeviceToHost);

    float max_abs_diff = 0.0f;
    int n_large_diff = 0;
    for (size_t i = 0; i < h_y_loop.size(); ++i) {
        float a = __half2float(h_y_loop[i]);
        float b = __half2float(h_y_single[i]);
        if (std::isnan(a) || std::isnan(b))
            continue;
        float d = std::fabs(a - b);
        if (d > 0.5f)
            ++n_large_diff;
        if (d > max_abs_diff)
            max_abs_diff = d;
    }

    EXPECT_LT(max_abs_diff, 0.5f) << "Per-row gemv_kpar loop diverges from gemm_nvfp4 single-call at "
                                  << "Gemma-4 down dims (N=" << N << " K=" << K << " M=" << M << ").\n"
                                  << "  n_large_diff=" << n_large_diff << " / " << (M * N) << "\n"
                                  << "  max_abs_diff=" << max_abs_diff;

    free_nvfp4_result(qr);
    cudaFree(d_w);
    cudaFree(d_a);
    cudaFree(d_y_loop);
    cudaFree(d_y_single);
}

}  // namespace
}  // namespace imp
