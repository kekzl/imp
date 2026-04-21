#include "compute/weight_dispatch.h"
#include "compute/gemm.h"
#include "core/tensor.h"
#include "graph/weight_handle.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <vector>

using namespace imp;

namespace {

class WeightDispatchFP16Test : public ::testing::Test {
protected:
    void SetUp() override {
        cublasLtCreate(&lt_);
        cudaStreamCreate(&stream_);
        cudaMalloc(&workspace_, 16 * 1024 * 1024);
    }
    void TearDown() override {
        cudaFree(workspace_);
        cudaStreamDestroy(stream_);
        cublasLtDestroy(lt_);
    }

    cublasLtHandle_t lt_;
    cudaStream_t stream_;
    void* workspace_;
};

} // namespace

TEST_F(WeightDispatchFP16Test, MatchesDirectFP16Gemm) {
    // gemm semantics: C = alpha * A @ B^T + beta*C
    // A [M, K], B [N, K] row-major, C [M, N]
    // So: W [M=16, K=64], X [N=32, K=64], Y [M=16, N=32]
    const int M = 16, N = 32, K = 64;
    std::vector<half> h_w(M * K), h_x(N * K);
    for (int i = 0; i < M * K; ++i) h_w[i] = __float2half((i % 7) * 0.01f);
    for (int i = 0; i < N * K; ++i) h_x[i] = __float2half((i % 11) * 0.01f);

    half *d_w, *d_x, *d_y_direct, *d_y_dispatch;
    cudaMalloc(&d_w, M * K * sizeof(half));
    cudaMalloc(&d_x, N * K * sizeof(half));
    cudaMalloc(&d_y_direct, M * N * sizeof(half));
    cudaMalloc(&d_y_dispatch, M * N * sizeof(half));
    cudaMemcpy(d_w, h_w.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), N * K * sizeof(half), cudaMemcpyHostToDevice);

    int64_t wshape[2] = {M, K};
    int64_t xshape[2] = {N, K};  // [N, K] so gemm computes W @ X^T = [M,N]
    int64_t yshape[2] = {M, N};
    Tensor w_t(d_w, DType::FP16, 2, wshape, true);
    Tensor x_t(d_x, DType::FP16, 2, xshape, true);
    Tensor y_direct(d_y_direct, DType::FP16, 2, yshape, true);
    // Call gemm directly — gemm(A, B, C, alpha, beta, stream)
    gemm(w_t, x_t, y_direct, 1.0f, 0.0f, stream_);
    cudaStreamSynchronize(stream_);

    WeightHandle h;
    h.kind = TensorKind::WQ;
    h.primary_tier = StorageTier::FP16;
    h.shape[0] = M; h.shape[1] = K;
    h.payload.fp16.data = d_w;
    Tensor y_disp(d_y_dispatch, DType::FP16, 2, yshape, true);
    gemm_dispatch(lt_, h, x_t, y_disp, 1.0f, 0.0f, workspace_, 16*1024*1024, stream_);
    cudaStreamSynchronize(stream_);

    std::vector<half> h_direct(M * N), h_disp(M * N);
    cudaMemcpy(h_direct.data(), d_y_direct, M*N*sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_disp.data(), d_y_dispatch, M*N*sizeof(half), cudaMemcpyDeviceToHost);
    for (int i = 0; i < M*N; ++i) {
        EXPECT_EQ(__half_as_ushort(h_direct[i]), __half_as_ushort(h_disp[i]))
            << "mismatch at i=" << i;
    }

    cudaFree(d_w); cudaFree(d_x);
    cudaFree(d_y_direct); cudaFree(d_y_dispatch);
}
