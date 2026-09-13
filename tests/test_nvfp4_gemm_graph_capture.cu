// gemm_nvfp4 M>1 fallback dequantizes NVFP4 to FP16 then calls cuBLAS; the dequant scratch
// buffer used to lazy-cudaMalloc on first use, crashing inside a captured CUDA stream. Fix:
// pre-allocated workspace (set_nvfp4_dequant_workspace); missing workspace + capture mode
// now fails loud instead of crashing. Does NOT cover cuBLAS's own graph-safety (algo reselect).

#include "quant/nvfp4_quant.h"
#include "quant/nvfp4_gemm.h"
#include "core/tensor.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cuda_fp16.h>
#include <vector>

namespace imp {
namespace {

class NvFP4GemmGraphCapture : public ::testing::Test {
protected:
    void SetUp() override { cudaStreamCreate(&stream_); }
    void TearDown() override {
        if (stream_)
            cudaStreamDestroy(stream_);
        // Always clear workspace so tests don't pollute each other.
        set_nvfp4_dequant_workspace(nullptr, 0);
        // Drain any sticky CUDA error state from this test (e.g. abandoned
        // captures) so subsequent test suites start with a clean runtime.
        cudaGetLastError();
    }
    cudaStream_t stream_ = nullptr;
};

// Build a tiny NVFP4 weight matrix sized so the dequant workspace fits
// comfortably (well below typical sanity caps). N×K=64×64 FP16 = 8 KiB.
struct TinyNvFP4 {
    half *d_w = nullptr, *d_a = nullptr, *d_c = nullptr;
    NvFP4QuantResult qr{};
    int N = 64, K = 64, M = 4;

    void create(cudaStream_t stream) {
        std::vector<half> h_w(static_cast<size_t>(N) * K);
        std::vector<half> h_a(static_cast<size_t>(M) * K);
        for (size_t i = 0; i < h_w.size(); ++i)
            h_w[i] = __float2half(((static_cast<int>(i * 17u) % 31) - 15) * 0.01f);
        for (size_t i = 0; i < h_a.size(); ++i)
            h_a[i] = __float2half(((static_cast<int>(i * 23u) % 29) - 14) * 0.02f);

        cudaMalloc(&d_w, h_w.size() * sizeof(half));
        cudaMalloc(&d_a, h_a.size() * sizeof(half));
        cudaMalloc(&d_c, static_cast<size_t>(M) * N * sizeof(half));
        cudaMemcpy(d_w, h_w.data(), h_w.size() * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(half), cudaMemcpyHostToDevice);

        int64_t wshape[2] = {N, K};
        Tensor w_t(d_w, QType::F16, 2, wshape, /*on_device=*/true);
        quantize_fp16_to_nvfp4(w_t, qr, stream);
        cudaStreamSynchronize(stream);
    }

    void destroy() {
        free_nvfp4_result(qr);
        if (d_w)
            cudaFree(d_w);
        if (d_a)
            cudaFree(d_a);
        if (d_c)
            cudaFree(d_c);
    }
};

// When a workspace is registered and large enough, gemm_nvfp4 must NOT grow
// the legacy lazy cudaMalloc buffer — the workspace is the graph-safe path
// and proves it's the active branch.
TEST_F(NvFP4GemmGraphCapture, WorkspaceIsPreferredOverLazyAlloc) {
    TinyNvFP4 tw;
    tw.create(stream_);

    // Pre-allocate workspace sized for one full FP16 dequant of the weight.
    size_t ws_bytes = static_cast<size_t>(tw.N) * tw.K * sizeof(half);
    void* ws = nullptr;
    ASSERT_EQ(cudaMalloc(&ws, ws_bytes), cudaSuccess);
    set_nvfp4_dequant_workspace(ws, ws_bytes);

    size_t lazy_before = nvfp4_lazy_dequant_buf_size_for_testing();

    int64_t ashape[2] = {tw.M, tw.K};
    int64_t cshape[2] = {tw.M, tw.N};
    Tensor a_t(tw.d_a, QType::F16, 2, ashape, /*on_device=*/true);
    Tensor c_t(tw.d_c, QType::F16, 2, cshape, /*on_device=*/true);
    cudaMemsetAsync(tw.d_c, 0, static_cast<size_t>(tw.M) * tw.N * sizeof(half), stream_);
    gemm_nvfp4(tw.qr, a_t, c_t, stream_);
    cudaStreamSynchronize(stream_);

    size_t lazy_after = nvfp4_lazy_dequant_buf_size_for_testing();
    EXPECT_EQ(lazy_after, lazy_before)
        << "Workspace was set with " << ws_bytes
        << " bytes (>= needed) yet the legacy lazy buffer grew from " << lazy_before << " to "
        << lazy_after << " — the workspace branch was bypassed";

    // Also check the GEMM produced something (the dequant + cuBLAS path
    // ran end-to-end against the workspace, not just bailed out).
    std::vector<half> h_c(static_cast<size_t>(tw.M) * tw.N);
    cudaMemcpy(h_c.data(), tw.d_c, h_c.size() * sizeof(half), cudaMemcpyDeviceToHost);
    int n_nonzero = 0;
    for (auto v : h_c) {
        if (__half2float(v) != 0.0f)
            ++n_nonzero;
    }
    EXPECT_GT(n_nonzero, 0) << "GEMM via workspace produced all-zero output";

    set_nvfp4_dequant_workspace(nullptr, 0);
    cudaFree(ws);
    tw.destroy();
}

// Without a pre-allocated workspace, gemm_nvfp4 inside stream capture must NOT crash via
// cudaMalloc: the cudaStreamIsCapturing guard logs and returns nullptr, gemm_nvfp4
// early-returns. Verified by checking the runtime stays usable afterwards.
TEST_F(NvFP4GemmGraphCapture, FallbackInsideCaptureWithoutWorkspaceFailsLoud) {
    TinyNvFP4 tw;
    tw.create(stream_);

    // Explicitly clear any previously-set workspace.
    set_nvfp4_dequant_workspace(nullptr, 0);

    ASSERT_EQ(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeRelaxed), cudaSuccess);

    int64_t ashape[2] = {tw.M, tw.K};
    int64_t cshape[2] = {tw.M, tw.N};
    Tensor a_t(tw.d_a, QType::F16, 2, ashape, /*on_device=*/true);
    Tensor c_t(tw.d_c, QType::F16, 2, cshape, /*on_device=*/true);
    // Must fail LOUD: a silent skip records a graph missing this GEMM, later launched with an
    // uninitialized activation buffer (#855 "hybrid crash", misaligned address on Nemotron).
    // The capture-refusal now surfaces as a throw the capturer catches.
    EXPECT_THROW(gemm_nvfp4(tw.qr, a_t, c_t, stream_), std::runtime_error);

    cudaGraph_t graph = nullptr;
    cudaError_t end_err = cudaStreamEndCapture(stream_, &graph);
    if (graph)
        cudaGraphDestroy(graph);
    // We don't assert end_err here — cuBLAS-or-other internal state may have
    // invalidated the capture before our guard fired. The point is that the
    // runtime is still usable below.
    (void)end_err;

    // Drain sticky errors and verify the CUDA runtime is still healthy.
    cudaGetLastError();
    void* probe = nullptr;
    EXPECT_EQ(cudaMalloc(&probe, 16), cudaSuccess) << "Runtime still in error state after failed capture";
    if (probe)
        cudaFree(probe);

    tw.destroy();
}

// Outside capture, with no workspace set, gemm_nvfp4 must continue to lazy-
// allocate (legacy behaviour) — this is the path used by prefill on
// non-graph code today.
TEST_F(NvFP4GemmGraphCapture, FallbackOutsideCaptureUsesLazyAllocAsBefore) {
    TinyNvFP4 tw;
    tw.create(stream_);

    set_nvfp4_dequant_workspace(nullptr, 0);

    int64_t ashape[2] = {tw.M, tw.K};
    int64_t cshape[2] = {tw.M, tw.N};
    Tensor a_t(tw.d_a, QType::F16, 2, ashape, /*on_device=*/true);
    Tensor c_t(tw.d_c, QType::F16, 2, cshape, /*on_device=*/true);
    cudaMemsetAsync(tw.d_c, 0, static_cast<size_t>(tw.M) * tw.N * sizeof(half), stream_);

    gemm_nvfp4(tw.qr, a_t, c_t, stream_);
    cudaStreamSynchronize(stream_);

    std::vector<half> h_c(static_cast<size_t>(tw.M) * tw.N);
    cudaMemcpy(h_c.data(), tw.d_c, h_c.size() * sizeof(half), cudaMemcpyDeviceToHost);
    int n_nonzero = 0;
    for (auto v : h_c) {
        if (__half2float(v) != 0.0f)
            ++n_nonzero;
    }
    EXPECT_GT(n_nonzero, 0) << "Lazy-alloc path produced all-zero output";

    tw.destroy();
}

}  // namespace
}  // namespace imp
