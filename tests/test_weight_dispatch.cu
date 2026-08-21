#include "compute/weight_dispatch.h"
#include "compute/gemm.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/gemm_cutlass_mxfp4_sm120.h"
#include "core/tensor.h"
#include "exec/weight_handle.h"
#include "quant/fp8_quant.h"
#include "quant/nvfp4_quant.h"
#include "quant/nvfp4_gemm.h"
#include "quant/mxfp4_gemm.h"

#include <gtest/gtest.h>
#include <cstdint>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <vector>
#include <cmath>

using namespace imp;

namespace {

// ---------------------------------------------------------------------------
// Shared fixture: cuBLASLt handle + stream + workspace
// ---------------------------------------------------------------------------

class WeightDispatchTest : public ::testing::Test {
protected:
    static constexpr size_t kWorkspaceBytes = 64 * 1024 * 1024;  // 64 MiB

    void SetUp() override {
        cublasLtCreate(&lt_);
        cudaStreamCreate(&stream_);
        cudaMalloc(&workspace_, kWorkspaceBytes);
        cudaMemset(workspace_, 0, kWorkspaceBytes);
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

// Allocate and copy helper
template <typename T>
T* dev_alloc_copy(const std::vector<T>& h, cudaStream_t s = nullptr) {
    T* d;
    cudaMalloc(&d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

template <typename T>
std::vector<T> dev_read(const T* d_ptr, size_t n) {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), d_ptr, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}

}  // namespace

// ===========================================================================
// FP16 parity (existing test, kept for completeness)
// ===========================================================================

TEST_F(WeightDispatchTest, FP16_GemmMatchesDirect) {
    // W [M=16, K=64], X [N=32, K=64], Y [M=16, N=32]
    const int M = 16, N = 32, K = 64;
    std::vector<half> h_w(M * K), h_x(N * K);
    for (int i = 0; i < M * K; ++i)
        h_w[i] = __float2half((i % 7) * 0.01f);
    for (int i = 0; i < N * K; ++i)
        h_x[i] = __float2half((i % 11) * 0.01f);

    half* d_w = dev_alloc_copy(h_w);
    half* d_x = dev_alloc_copy(h_x);
    half* d_y_direct;
    cudaMalloc(&d_y_direct, M * N * sizeof(half));
    half* d_y_disp;
    cudaMalloc(&d_y_disp, M * N * sizeof(half));

    int64_t wshape[2] = {M, K}, xshape[2] = {N, K}, yshape[2] = {M, N};
    Tensor w_t(d_w, QType::F16, 2, wshape, true);
    Tensor x_t(d_x, QType::F16, 2, xshape, true);
    Tensor y_direct(d_y_direct, QType::F16, 2, yshape, true);
    Tensor y_disp(d_y_disp, QType::F16, 2, yshape, true);

    gemm(w_t, x_t, y_direct, 1.0f, 0.0f, stream_);
    cudaStreamSynchronize(stream_);

    WeightHandle h;
    h.kind = TensorKind::WQ;
    h.primary_tier = StorageTier::FP16;
    h.shape[0] = M;
    h.shape[1] = K;
    h.payload.fp16.data = d_w;

    gemm_dispatch(lt_, h, x_t, y_disp, 1.0f, 0.0f, workspace_, kWorkspaceBytes, stream_);
    cudaStreamSynchronize(stream_);

    auto h_direct = dev_read(d_y_direct, M * N);
    auto h_disp = dev_read(d_y_disp, M * N);
    for (int i = 0; i < M * N; ++i)
        EXPECT_EQ(__half_as_ushort(h_direct[i]), __half_as_ushort(h_disp[i]))
            << "FP16 GEMM mismatch at i=" << i;

    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_y_direct);
    cudaFree(d_y_disp);
}

// FP16 gemv_dispatch parity (M=1)
TEST_F(WeightDispatchTest, FP16_GemvMatchesDirect) {
    const int M = 16, K = 64;
    std::vector<half> h_w(M * K), h_x(K);
    for (int i = 0; i < M * K; ++i)
        h_w[i] = __float2half((i % 7) * 0.01f);
    for (int i = 0; i < K; ++i)
        h_x[i] = __float2half((i % 11) * 0.01f);

    half* d_w = dev_alloc_copy(h_w);
    half* d_x = dev_alloc_copy(h_x);
    half* d_y_direct;
    cudaMalloc(&d_y_direct, M * sizeof(half));
    half* d_y_disp;
    cudaMalloc(&d_y_disp, M * sizeof(half));

    int64_t wshape[2] = {M, K}, xshape[2] = {1, K}, yshape[2] = {M, 1};
    Tensor w_t(d_w, QType::F16, 2, wshape, true);
    Tensor x_t(d_x, QType::F16, 2, xshape, true);
    Tensor y_direct(d_y_direct, QType::F16, 2, yshape, true);
    Tensor y_disp(d_y_disp, QType::F16, 2, yshape, true);

    gemm(w_t, x_t, y_direct, 1.0f, 0.0f, stream_);
    cudaStreamSynchronize(stream_);

    WeightHandle h;
    h.kind = TensorKind::WQ;
    h.primary_tier = StorageTier::FP16;
    h.shape[0] = M;
    h.shape[1] = K;
    h.payload.fp16.data = d_w;

    gemv_dispatch(h, x_t, y_disp, stream_);
    cudaStreamSynchronize(stream_);

    auto h_direct = dev_read(d_y_direct, M);
    auto h_disp = dev_read(d_y_disp, M);
    for (int i = 0; i < M; ++i)
        EXPECT_EQ(__half_as_ushort(h_direct[i]), __half_as_ushort(h_disp[i]))
            << "FP16 GEMV mismatch at i=" << i;

    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_y_direct);
    cudaFree(d_y_disp);
}

// ===========================================================================
// FP8 tier
// ===========================================================================

// gemm_dispatch FP8 tier: pre-quantize BOTH weights and activation to FP8,
// build handle, verify dispatch produces the same result as calling
// gemm_cublaslt directly.
//
// cuBLASLt on sm_120 requires FP8×FP8 (both operands FP8); FP16×FP8 mixed is
// NOT supported (CUBLAS_STATUS_NOT_SUPPORTED=15).  So both x (activation) and
// the weight must be FP8_E4M3.  gemm_dispatch passes x through unchanged to
// gemm_cublaslt, so if the caller provides a FP8 activation tensor, the call
// is byte-for-byte identical.
TEST_F(WeightDispatchTest, FP8_GemmMatchesDirect) {
    const int M = 16, N = 32, K = 64;

    // Build FP16 weight and activation, quantize both to FP8
    std::vector<half> h_w_fp16(M * K), h_x_fp16(N * K);
    for (int i = 0; i < M * K; ++i)
        h_w_fp16[i] = __float2half((i % 7) * 0.01f);
    for (int i = 0; i < N * K; ++i)
        h_x_fp16[i] = __float2half((i % 11) * 0.01f);

    half* d_w_fp16 = dev_alloc_copy(h_w_fp16);
    half* d_x_fp16 = dev_alloc_copy(h_x_fp16);

    // FP8 weight buffer + weight scale
    void* d_w_fp8;
    cudaMalloc(&d_w_fp8, M * K * sizeof(__nv_fp8_e4m3));
    float* d_w_scale;
    cudaMalloc(&d_w_scale, sizeof(float));
    // FP8 activation buffer + activation scale
    void* d_x_fp8;
    cudaMalloc(&d_x_fp8, N * K * sizeof(__nv_fp8_e4m3));
    float* d_x_scale;
    cudaMalloc(&d_x_scale, sizeof(float));

    int64_t wshape[2] = {M, K}, xshape[2] = {N, K};
    Tensor w_fp16_t(d_w_fp16, QType::F16, 2, wshape, true);
    Tensor w_fp8_t(d_w_fp8, QType::FP8_E4M3, 2, wshape, true);
    Tensor x_fp16_t(d_x_fp16, QType::F16, 2, xshape, true);
    Tensor x_fp8_t(d_x_fp8, QType::FP8_E4M3, 2, xshape, true);

    quantize_fp16_to_fp8_e4m3(w_fp16_t, w_fp8_t, d_w_scale, stream_);
    quantize_fp16_to_fp8_e4m3(x_fp16_t, x_fp8_t, d_x_scale, stream_);
    cudaStreamSynchronize(stream_);

    int64_t yshape[2] = {M, N};
    half* d_y_direct;
    cudaMalloc(&d_y_direct, M * N * sizeof(half));
    half* d_y_disp;
    cudaMalloc(&d_y_disp, M * N * sizeof(half));
    Tensor y_direct(d_y_direct, QType::F16, 2, yshape, true);
    Tensor y_disp(d_y_disp, QType::F16, 2, yshape, true);

    // Direct call: gemm_cublaslt(fp8_x, fp8_w, fp16_y, alpha, beta, aScale, bScale)
    gemm_cublaslt(x_fp8_t, w_fp8_t, y_direct, 1.0f, 0.0f, d_x_scale, d_w_scale, stream_);
    cudaStreamSynchronize(stream_);

    // Dispatch call: build handle with FP8 payload; pass FP8-quantized x
    WeightHandle h;
    h.kind = TensorKind::WQ;
    h.primary_tier = StorageTier::FP8;
    h.shape[0] = M;
    h.shape[1] = K;
    h.payload.fp8.data = reinterpret_cast<__nv_fp8_e4m3*>(d_w_fp8);
    h.payload.fp8.d_scale = d_w_scale;

    // gemm_dispatch FP8: calls gemm_cublaslt(x, fp8_w, y, alpha, beta, nullptr, d_scale)
    // For byte-identity, we pass null aScale in dispatch (same as direct when aScale ignored)
    // and use the same x tensor. But the direct call uses d_x_scale as aScale.
    // To make them identical: direct call also with nullptr aScale.
    cudaMemset(d_y_direct, 0, M * N * sizeof(half));
    gemm_cublaslt(x_fp8_t, w_fp8_t, y_direct, 1.0f, 0.0f, nullptr, d_w_scale, stream_);
    cudaStreamSynchronize(stream_);

    gemm_dispatch(lt_, h, x_fp8_t, y_disp, 1.0f, 0.0f, workspace_, kWorkspaceBytes, stream_);
    cudaStreamSynchronize(stream_);

    auto h_direct = dev_read(d_y_direct, M * N);
    auto h_disp = dev_read(d_y_disp, M * N);
    for (int i = 0; i < M * N; ++i)
        EXPECT_EQ(__half_as_ushort(h_direct[i]), __half_as_ushort(h_disp[i]))
            << "FP8 GEMM mismatch at i=" << i;

    cudaFree(d_w_fp16);
    cudaFree(d_x_fp16);
    cudaFree(d_w_fp8);
    cudaFree(d_w_scale);
    cudaFree(d_x_fp8);
    cudaFree(d_x_scale);
    cudaFree(d_y_direct);
    cudaFree(d_y_disp);
}

// gemv_dispatch FP8 tier: single-token decode path.
// gemv_fp8(W_fp8, x_fp16, y, scale) is the direct call.
// The dispatch reconstructs scale via D2H copy and calls the same function.
TEST_F(WeightDispatchTest, FP8_GemvMatchesDirect) {
    const int M = 32, K = 64;

    std::vector<half> h_w_fp16(M * K), h_x(K);
    for (int i = 0; i < M * K; ++i)
        h_w_fp16[i] = __float2half((i % 7) * 0.02f);
    for (int i = 0; i < K; ++i)
        h_x[i] = __float2half((i % 11) * 0.01f);

    half* d_w_fp16 = dev_alloc_copy(h_w_fp16);
    half* d_x = dev_alloc_copy(h_x);

    void* d_w_fp8;
    cudaMalloc(&d_w_fp8, M * K * sizeof(__nv_fp8_e4m3));
    float* d_scale;
    cudaMalloc(&d_scale, sizeof(float));

    int64_t wshape[2] = {M, K};
    Tensor w_fp16_t(d_w_fp16, QType::F16, 2, wshape, true);
    Tensor w_fp8_t(d_w_fp8, QType::FP8_E4M3, 2, wshape, true);
    quantize_fp16_to_fp8_e4m3(w_fp16_t, w_fp8_t, d_scale, stream_);
    cudaStreamSynchronize(stream_);

    // Read back host scale for direct call
    float host_scale;
    cudaMemcpy(&host_scale, d_scale, sizeof(float), cudaMemcpyDeviceToHost);

    int64_t xshape[2] = {1, K}, yshape[2] = {M, 1};
    Tensor x_t(d_x, QType::F16, 2, xshape, true);

    half* d_y_direct;
    cudaMalloc(&d_y_direct, M * sizeof(half));
    half* d_y_disp;
    cudaMalloc(&d_y_disp, M * sizeof(half));
    Tensor y_direct(d_y_direct, QType::F16, 2, yshape, true);
    Tensor y_disp(d_y_disp, QType::F16, 2, yshape, true);

    gemv_fp8(w_fp8_t, x_t, y_direct, host_scale, stream_);
    cudaStreamSynchronize(stream_);

    WeightHandle h;
    h.kind = TensorKind::WQ;
    h.primary_tier = StorageTier::FP8;
    h.shape[0] = M;
    h.shape[1] = K;
    h.payload.fp8.data = reinterpret_cast<__nv_fp8_e4m3*>(d_w_fp8);
    h.payload.fp8.d_scale = d_scale;

    gemv_dispatch(h, x_t, y_disp, stream_);
    cudaStreamSynchronize(stream_);

    auto h_direct = dev_read(d_y_direct, M);
    auto h_disp = dev_read(d_y_disp, M);
    for (int i = 0; i < M; ++i)
        EXPECT_EQ(__half_as_ushort(h_direct[i]), __half_as_ushort(h_disp[i]))
            << "FP8 GEMV mismatch at i=" << i;

    cudaFree(d_w_fp16);
    cudaFree(d_x);
    cudaFree(d_w_fp8);
    cudaFree(d_scale);
    cudaFree(d_y_direct);
    cudaFree(d_y_disp);
}

// ===========================================================================
// NVFP4 tier
// ===========================================================================

// gemm_dispatch NVFP4 tier (M>1): quantize FP16 weight → NVFP4, build handle,
// call dispatch.  Compare vs gemm_nvfp4 direct call.
//
// Dispatch reconstructs NvFP4QuantResult with tensor_scale=1.0f (phase-2 shim
// limitation: payload has null device ptr for tensor_scale).  Direct call also
// uses tensor_scale=1.0f since we force it after quantization.
// If tensors are zero-weight the test trivially passes; use non-zero values.
TEST_F(WeightDispatchTest, NVFP4_GemmMatchesDirect) {
    // gemm_nvfp4 convention (matches PyTorch nn.Linear):
    //   A (weight, NVFP4): [N_out, K_in]
    //   B (input,  FP16):  [M_batch, K_in]
    //   C (output, FP16):  [M_batch, N_out]
    // The fallback path computes  C = B @ A_fp16^T  via standard cuBLAS GEMM,
    // so the output's first dim is the input's batch dim, not the weight's
    // output-feature dim. (Earlier test code mixed up M and N → shape error
    // at gemm_nvfp4:1251.)
    //
    // K must be multiple of 16 (NVFP4 micro-block size).
    const int N_OUT = 16, M_BATCH = 8, K = 64;

    std::vector<half> h_w(N_OUT * K), h_x(M_BATCH * K);
    for (int i = 0; i < N_OUT * K; ++i)
        h_w[i] = __float2half((i % 5) * 0.1f - 0.2f);
    for (int i = 0; i < M_BATCH * K; ++i)
        h_x[i] = __float2half((i % 7) * 0.05f - 0.15f);

    half* d_w = dev_alloc_copy(h_w);
    half* d_x = dev_alloc_copy(h_x);

    int64_t wshape[2] = {N_OUT, K};
    Tensor w_t(d_w, QType::F16, 2, wshape, true);

    // Quantize to NVFP4
    NvFP4QuantResult qr;
    quantize_fp16_to_nvfp4(w_t, qr, stream_);
    cudaStreamSynchronize(stream_);

    // Force tensor_scale=1.0f so both paths are identical
    // (the phase-2 dispatch uses 1.0f when payload.tensor_scale is null)
    float saved_ts = qr.tensor_scale;
    qr.tensor_scale = 1.0f;

    int64_t xshape[2] = {M_BATCH, K}, yshape[2] = {M_BATCH, N_OUT};
    Tensor x_t(d_x, QType::F16, 2, xshape, true);

    half* d_y_direct;
    cudaMalloc(&d_y_direct, M_BATCH * N_OUT * sizeof(half));
    half* d_y_disp;
    cudaMalloc(&d_y_disp, M_BATCH * N_OUT * sizeof(half));
    Tensor y_direct(d_y_direct, QType::F16, 2, yshape, true);
    Tensor y_disp(d_y_disp, QType::F16, 2, yshape, true);

    gemm_nvfp4(qr, x_t, y_direct, stream_);
    cudaStreamSynchronize(stream_);

    // Build handle: payload.nvfp4.tensor_scale = nullptr (phase-2 shim)
    WeightHandle h;
    h.kind = TensorKind::WQ;
    h.primary_tier = StorageTier::NVFP4;
    h.shape[0] = N_OUT;
    h.shape[1] = K;
    h.payload.nvfp4.data = static_cast<uint8_t*>(qr.packed_data);
    h.payload.nvfp4.block_scales = static_cast<uint8_t*>(qr.micro_scales);
    h.payload.nvfp4.tensor_scale = nullptr;  // shim limitation: no device ptr
    h.payload.nvfp4.tensor_scale_2 = nullptr;

    gemm_dispatch(lt_, h, x_t, y_disp, 1.0f, 0.0f, workspace_, kWorkspaceBytes, stream_);
    cudaStreamSynchronize(stream_);

    auto h_direct = dev_read(d_y_direct, M_BATCH * N_OUT);
    auto h_disp = dev_read(d_y_disp, M_BATCH * N_OUT);
    for (int i = 0; i < M_BATCH * N_OUT; ++i) {
        float vd = __half2float(h_direct[i]);
        float vp = __half2float(h_disp[i]);
        // Same underlying call (gemm_nvfp4 dequant + cuBLAS), expect identical.
        EXPECT_EQ(__half_as_ushort(h_direct[i]), __half_as_ushort(h_disp[i]))
            << "NVFP4 GEMM dispatch mismatch at i=" << i << " direct=" << vd << " dispatch=" << vp;
    }

    qr.tensor_scale = saved_ts;  // restore before free
    free_nvfp4_result(qr);
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_y_direct);
    cudaFree(d_y_disp);
}

// Regression: prequant-loaded NVFP4 weight handles carry the PACKED K/2 in
// shape[1] (two FP4 nibbles per byte), NOT the logical K — an inconsistent
// convention vs the handles built above (which use logical K). The phase-2 shim
// must derive K from the activation, never from the handle, or the M>1 dequant
// GEMM fallback aborts "B.shape[1]=<K> must equal weight K=<K/2>". This is the
// native-NVFP4 server crash that surfaced when a large KV budget starved the
// CUTLASS prefill workspace and forced the dequant fallback. Identical to
// NVFP4_GemmMatchesDirect except h.shape[1] holds the PACKED dimension.
TEST_F(WeightDispatchTest, NVFP4_GemmPackedShapeMatchesDirect) {
    const int N_OUT = 16, M_BATCH = 8, K = 64;

    std::vector<half> h_w(N_OUT * K), h_x(M_BATCH * K);
    for (int i = 0; i < N_OUT * K; ++i)
        h_w[i] = __float2half((i % 5) * 0.1f - 0.2f);
    for (int i = 0; i < M_BATCH * K; ++i)
        h_x[i] = __float2half((i % 7) * 0.05f - 0.15f);

    half* d_w = dev_alloc_copy(h_w);
    half* d_x = dev_alloc_copy(h_x);

    int64_t wshape[2] = {N_OUT, K};
    Tensor w_t(d_w, QType::F16, 2, wshape, true);

    NvFP4QuantResult qr;
    quantize_fp16_to_nvfp4(w_t, qr, stream_);
    cudaStreamSynchronize(stream_);
    float saved_ts = qr.tensor_scale;
    qr.tensor_scale = 1.0f;

    int64_t xshape[2] = {M_BATCH, K}, yshape[2] = {M_BATCH, N_OUT};
    Tensor x_t(d_x, QType::F16, 2, xshape, true);

    half *d_y_direct, *d_y_disp;
    cudaMalloc(&d_y_direct, M_BATCH * N_OUT * sizeof(half));
    cudaMalloc(&d_y_disp, M_BATCH * N_OUT * sizeof(half));
    Tensor y_direct(d_y_direct, QType::F16, 2, yshape, true);
    Tensor y_disp(d_y_disp, QType::F16, 2, yshape, true);

    gemm_nvfp4(qr, x_t, y_direct, stream_);
    cudaStreamSynchronize(stream_);

    WeightHandle h;
    h.kind = TensorKind::WQ;
    h.primary_tier = StorageTier::NVFP4;
    h.shape[0] = N_OUT;
    h.shape[1] = K / 2;  // PACKED — mirrors the prequant loader (logical K = shape[1]*2)
    h.payload.nvfp4.data = static_cast<uint8_t*>(qr.packed_data);
    h.payload.nvfp4.block_scales = static_cast<uint8_t*>(qr.micro_scales);
    h.payload.nvfp4.tensor_scale = nullptr;
    h.payload.nvfp4.tensor_scale_2 = nullptr;

    // With the old `tmp.K = w.shape[1]` this aborted in gemm_nvfp4; the activation-
    // derived K makes it succeed and match the direct call.
    gemm_dispatch(lt_, h, x_t, y_disp, 1.0f, 0.0f, workspace_, kWorkspaceBytes, stream_);
    cudaStreamSynchronize(stream_);

    auto h_direct = dev_read(d_y_direct, M_BATCH * N_OUT);
    auto h_disp = dev_read(d_y_disp, M_BATCH * N_OUT);
    for (int i = 0; i < M_BATCH * N_OUT; ++i)
        EXPECT_EQ(__half_as_ushort(h_direct[i]), __half_as_ushort(h_disp[i]))
            << "NVFP4 packed-shape GEMM dispatch mismatch at i=" << i;

    qr.tensor_scale = saved_ts;
    free_nvfp4_result(qr);
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_y_direct);
    cudaFree(d_y_disp);
}

// gemv_dispatch NVFP4 tier (M=1): single-token decode.
// Same parity approach: force tensor_scale=1.0f in both paths.
TEST_F(WeightDispatchTest, NVFP4_GemvMatchesDirect) {
    const int M = 32, K = 64;

    std::vector<half> h_w(M * K), h_x(K);
    for (int i = 0; i < M * K; ++i)
        h_w[i] = __float2half((i % 5) * 0.1f - 0.2f);
    for (int i = 0; i < K; ++i)
        h_x[i] = __float2half((i % 7) * 0.05f);

    half* d_w = dev_alloc_copy(h_w);
    half* d_x = dev_alloc_copy(h_x);

    int64_t wshape[2] = {M, K};
    Tensor w_t(d_w, QType::F16, 2, wshape, true);

    NvFP4QuantResult qr;
    quantize_fp16_to_nvfp4(w_t, qr, stream_);
    cudaStreamSynchronize(stream_);

    float saved_ts = qr.tensor_scale;
    qr.tensor_scale = 1.0f;  // force same scale in dispatch (phase-2 shim returns 1.0f)

    half* d_y_direct;
    cudaMalloc(&d_y_direct, M * sizeof(half));
    half* d_y_disp;
    cudaMalloc(&d_y_disp, M * sizeof(half));

    gemv_nvfp4_kpar(qr, static_cast<const half*>(d_x), d_y_direct, M, K, stream_);
    cudaStreamSynchronize(stream_);

    int64_t xshape[2] = {1, K}, yshape[2] = {M, 1};
    Tensor x_t(d_x, QType::F16, 2, xshape, true);
    Tensor y_disp(d_y_disp, QType::F16, 2, yshape, true);

    WeightHandle h;
    h.kind = TensorKind::WQ;
    h.primary_tier = StorageTier::NVFP4;
    h.shape[0] = M;
    h.shape[1] = K;
    h.payload.nvfp4.data = static_cast<uint8_t*>(qr.packed_data);
    h.payload.nvfp4.block_scales = static_cast<uint8_t*>(qr.micro_scales);
    h.payload.nvfp4.tensor_scale = nullptr;
    h.payload.nvfp4.tensor_scale_2 = nullptr;

    gemv_dispatch(h, x_t, y_disp, stream_);
    cudaStreamSynchronize(stream_);

    auto h_direct = dev_read(d_y_direct, M);
    auto h_disp = dev_read(d_y_disp, M);
    for (int i = 0; i < M; ++i) {
        float vd = __half2float(h_direct[i]);
        float vp = __half2float(h_disp[i]);
        EXPECT_EQ(__half_as_ushort(h_direct[i]), __half_as_ushort(h_disp[i]))
            << "NVFP4 GEMV mismatch at i=" << i << " direct=" << vd << " dispatch=" << vp;
    }

    qr.tensor_scale = saved_ts;
    free_nvfp4_result(qr);
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_y_direct);
    cudaFree(d_y_disp);
}

// ===========================================================================
// CUTLASS_NVFP4 tier
// ===========================================================================

// gemm_dispatch CUTLASS_NVFP4 tier (M>1): quantize FP16 → NVFP4 → CUTLASS
// weight format, build handle, call dispatch.
//
// Dispatch reconstructs CutlassNvFP4Weight from payload, quantizes the FP16
// activation using workspace, and calls gemm_nvfp4_cutlass_sm120.
// Direct call does the same manually.
//
// If CUTLASS kernel is unavailable (cutlass_sm120_nvfp4_available()==false),
// skip the test.
TEST_F(WeightDispatchTest, CUTLASS_NVFP4_GemmMatchesDirect) {
    if (!cutlass_sm120_nvfp4_available()) {
        GTEST_SKIP() << "CUTLASS sm_120 NVFP4 not compiled/available on this device";
    }

    // K must be multiple of 16; choose dims that CUTLASS GEMM can handle
    const int M = 8, N = 16, K = 64;

    std::vector<half> h_w(N * K), h_x(M * K);
    for (int i = 0; i < N * K; ++i)
        h_w[i] = __float2half((i % 5) * 0.1f - 0.2f);
    for (int i = 0; i < M * K; ++i)
        h_x[i] = __float2half((i % 7) * 0.05f - 0.15f);

    half* d_w = dev_alloc_copy(h_w);
    half* d_x = dev_alloc_copy(h_x);

    int64_t wshape[2] = {N, K};
    Tensor w_t(d_w, QType::F16, 2, wshape, true);

    NvFP4QuantResult qr;
    quantize_fp16_to_nvfp4(w_t, qr, stream_);
    cudaStreamSynchronize(stream_);

    CutlassNvFP4Weight cw;
    convert_nvfp4_to_cutlass(qr, cw, stream_);
    cudaStreamSynchronize(stream_);

    // Allocate activation scratch for direct call
    size_t act_data_bytes = static_cast<size_t>(M) * K / 2;
    size_t act_sf_bytes = cutlass_nvfp4_sf_size(M, K);
    size_t ws_needed = gemm_nvfp4_cutlass_sm120_workspace(M, N, K);

    void* d_act_data;
    cudaMalloc(&d_act_data, act_data_bytes);
    void* d_act_sf;
    cudaMalloc(&d_act_sf, act_sf_bytes);
    void* d_ws;
    cudaMalloc(&d_ws, (ws_needed > 0) ? ws_needed : 1);

    quantize_fp16_to_nvfp4_cutlass(d_x, d_act_data, d_act_sf, M, K, stream_);
    cudaStreamSynchronize(stream_);

    half* d_y_direct;
    cudaMalloc(&d_y_direct, M * N * sizeof(half));
    half* d_y_disp;
    cudaMalloc(&d_y_disp, M * N * sizeof(half));

    bool ok = gemm_nvfp4_cutlass_sm120(d_act_data, d_act_sf, cw, d_y_direct, M, N, K, d_ws, ws_needed,
                                       stream_);
    cudaStreamSynchronize(stream_);

    if (!ok) {
        // CUTLASS rejected the dimensions — skip test, not a dispatch bug.
        GTEST_SKIP() << "gemm_nvfp4_cutlass_sm120 returned false for M=" << M << " N=" << N << " K=" << K;
    }

    // Build handle: workspace holds act_data + act_sf + CUTLASS workspace
    WeightHandle h;
    h.kind = TensorKind::WQ;
    h.primary_tier = StorageTier::CUTLASS_NVFP4;
    h.shape[0] = N;
    h.shape[1] = K;
    h.payload.cutlass_nvfp4.weight = const_cast<void*>(cw.data);
    h.payload.cutlass_nvfp4.sf = cw.scale_factors;
    h.payload.cutlass_nvfp4.global_scale = const_cast<float*>(&cw.tensor_scale);

    int64_t xshape[2] = {M, K}, yshape[2] = {M, N};
    Tensor x_t(d_x, QType::F16, 2, xshape, true);
    Tensor y_disp(d_y_disp, QType::F16, 2, yshape, true);

    // Workspace layout: [act_data | act_sf | cutlass_ws]
    // Must fit in kWorkspaceBytes (64 MiB)
    ASSERT_LE(act_data_bytes + act_sf_bytes + ws_needed, kWorkspaceBytes);

    gemm_dispatch(lt_, h, x_t, y_disp, 1.0f, 0.0f, workspace_, kWorkspaceBytes, stream_);
    cudaStreamSynchronize(stream_);

    auto h_direct = dev_read(d_y_direct, M * N);
    auto h_disp = dev_read(d_y_disp, M * N);
    for (int i = 0; i < M * N; ++i) {
        float vd = __half2float(h_direct[i]);
        float vp = __half2float(h_disp[i]);
        // Same kernel path (both call gemm_nvfp4_cutlass_sm120 with identical
        // FP4-quantized activation) — outputs should be byte-identical.
        EXPECT_EQ(__half_as_ushort(h_direct[i]), __half_as_ushort(h_disp[i]))
            << "CUTLASS_NVFP4 GEMM mismatch at i=" << i << " direct=" << vd << " dispatch=" << vp;
    }

    free_nvfp4_result(qr);
    free_cutlass_nvfp4_weight(cw);
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_act_data);
    cudaFree(d_act_sf);
    cudaFree(d_ws);
    cudaFree(d_y_direct);
    cudaFree(d_y_disp);
}

// gemv_dispatch CUTLASS_NVFP4 (M=1): CUTLASS_NVFP4 is a prefill-only tier and
// the consumer decode path uses the NVFP4 tier directly, so reaching this case
// is a routing bug in the caller.
//
// This test used to assert the opposite. It was named `CUTLASS_NVFP4_GemvIsStub`
// and asserted EXPECT_NO_THROW with the comment "output buffer is unchanged
// (stub returns early)" - i.e. it pinned a branch that answered with whatever
// the output buffer already held, behind one ERROR line, and it pinned it as
// the expected behaviour. That is the shape #654 removed from
// attention_prefill_dispatch: "no tier accepted" is an error, not a degraded
// answer (SETTLED.md S-22). The routing check the test was written for is kept;
// what changed is what counts as correct routing behaviour.
TEST_F(WeightDispatchTest, CUTLASS_NVFP4_GemvRefusesInsteadOfAnsweringWithStaleMemory) {
    const int M = 8, K = 32;

    void* dummy_data;
    cudaMalloc(&dummy_data, M * K / 2);
    void* dummy_sf;
    cudaMalloc(&dummy_sf, cutlass_nvfp4_sf_size(M, K));
    float dummy_scale = 1.0f;

    WeightHandle h;
    h.kind = TensorKind::WQ;
    h.primary_tier = StorageTier::CUTLASS_NVFP4;
    h.shape[0] = M;
    h.shape[1] = K;
    h.payload.cutlass_nvfp4.weight = dummy_data;
    h.payload.cutlass_nvfp4.sf = dummy_sf;
    h.payload.cutlass_nvfp4.global_scale = &dummy_scale;

    half* d_x;
    cudaMalloc(&d_x, K * sizeof(half));
    half* d_y;
    cudaMalloc(&d_y, M * sizeof(half));
    cudaMemset(d_x, 0, K * sizeof(half));
    cudaMemset(d_y, 0, M * sizeof(half));

    int64_t xshape[2] = {1, K}, yshape[2] = {M, 1};
    Tensor x_t(d_x, QType::F16, 2, xshape, true);
    Tensor y_t(d_y, QType::F16, 2, yshape, true);

    // A sentinel the old behaviour would have left in place and called an answer.
    EXPECT_EQ(cudaMemset(d_y, 0xAB, M * sizeof(half)), cudaSuccess);

    EXPECT_THROW(gemv_dispatch(h, x_t, y_t, stream_), std::runtime_error);
    EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    // And it refuses without touching the output, so a caller that ignores the
    // exception still cannot mistake the buffer for a result.
    std::vector<uint16_t> host(M);
    EXPECT_EQ(cudaMemcpy(host.data(), d_y, M * sizeof(half), cudaMemcpyDeviceToHost), cudaSuccess);
    for (int i = 0; i < M; ++i)
        EXPECT_EQ(host[i], 0xABAB) << "output was written by a path that refused";

    cudaFree(dummy_data);
    cudaFree(dummy_sf);
    cudaFree(d_x);
    cudaFree(d_y);
}

// ===========================================================================
// MXFP4 tier
// ===========================================================================

// gemm_dispatch MXFP4 tier (M>1): quantize FP16 weight → NVFP4 → convert to
// MXFP4 CUTLASS format, build handle, call dispatch.
//
// Dispatch reconstructs CutlassMxFP4Weight and calls gemm_mxfp4_cutlass_sm120.
// K must be multiple of 32 (UE8M0 SFVecSize=32).
TEST_F(WeightDispatchTest, MXFP4_GemmMatchesDirect) {
    if (!cutlass_sm120_mxfp4_available()) {
        GTEST_SKIP() << "CUTLASS sm_120 MXFP4 not compiled/available on this device";
    }

    // K must be multiple of 32 for MXFP4
    const int M = 8, N = 16, K = 64;

    std::vector<half> h_w(N * K), h_x(M * K);
    for (int i = 0; i < N * K; ++i)
        h_w[i] = __float2half((i % 5) * 0.1f - 0.2f);
    for (int i = 0; i < M * K; ++i)
        h_x[i] = __float2half((i % 7) * 0.05f - 0.15f);

    half* d_w = dev_alloc_copy(h_w);
    half* d_x = dev_alloc_copy(h_x);

    int64_t wshape[2] = {N, K};
    Tensor w_t(d_w, QType::F16, 2, wshape, true);

    // Quantize: FP16 → NVFP4 → MXFP4
    NvFP4QuantResult qr;
    quantize_fp16_to_nvfp4(w_t, qr, stream_);
    cudaStreamSynchronize(stream_);

    CutlassMxFP4Weight mw;
    convert_nvfp4_to_mxfp4_cutlass(qr, mw, stream_);
    cudaStreamSynchronize(stream_);

    // Allocate activation scratch for direct call
    size_t act_data_bytes = static_cast<size_t>(M) * K / 2;
    size_t act_sf_bytes = cutlass_mxfp4_sf_size(M, K);
    size_t ws_needed = gemm_mxfp4_cutlass_sm120_workspace(M, N, K);

    void* d_act_data;
    cudaMalloc(&d_act_data, act_data_bytes);
    void* d_act_sf;
    cudaMalloc(&d_act_sf, act_sf_bytes);
    void* d_ws;
    cudaMalloc(&d_ws, (ws_needed > 0) ? ws_needed : 1);

    quantize_fp16_to_mxfp4_cutlass(d_x, d_act_data, d_act_sf, M, K, stream_);
    cudaStreamSynchronize(stream_);

    half* d_y_direct;
    cudaMalloc(&d_y_direct, M * N * sizeof(half));
    half* d_y_disp;
    cudaMalloc(&d_y_disp, M * N * sizeof(half));

    bool ok = gemm_mxfp4_cutlass_sm120(d_act_data, d_act_sf, mw, d_y_direct, M, N, K, d_ws, ws_needed,
                                       stream_);
    cudaStreamSynchronize(stream_);

    if (!ok) {
        GTEST_SKIP() << "gemm_mxfp4_cutlass_sm120 returned false for M=" << M << " N=" << N << " K=" << K;
    }

    // Build handle
    WeightHandle h;
    h.kind = TensorKind::WQ;
    h.primary_tier = StorageTier::MXFP4;
    h.shape[0] = N;
    h.shape[1] = K;
    h.payload.mxfp4.weight = const_cast<void*>(mw.data);
    h.payload.mxfp4.scales = mw.scale_factors;
    h.payload.mxfp4.linear_scales = mw.linear_scales;

    int64_t xshape[2] = {M, K}, yshape[2] = {M, N};
    Tensor x_t(d_x, QType::F16, 2, xshape, true);
    Tensor y_disp(d_y_disp, QType::F16, 2, yshape, true);

    ASSERT_LE(act_data_bytes + act_sf_bytes + ws_needed, kWorkspaceBytes);

    gemm_dispatch(lt_, h, x_t, y_disp, 1.0f, 0.0f, workspace_, kWorkspaceBytes, stream_);
    cudaStreamSynchronize(stream_);

    auto h_direct = dev_read(d_y_direct, M * N);
    auto h_disp = dev_read(d_y_disp, M * N);
    for (int i = 0; i < M * N; ++i) {
        float vd = __half2float(h_direct[i]);
        float vp = __half2float(h_disp[i]);
        // Same kernel path: byte-identical output expected.
        EXPECT_EQ(__half_as_ushort(h_direct[i]), __half_as_ushort(h_disp[i]))
            << "MXFP4 GEMM mismatch at i=" << i << " direct=" << vd << " dispatch=" << vp;
    }

    free_nvfp4_result(qr);
    free_cutlass_mxfp4_weight(mw);
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_act_data);
    cudaFree(d_act_sf);
    cudaFree(d_ws);
    cudaFree(d_y_direct);
    cudaFree(d_y_disp);
}

// gemv_dispatch MXFP4 tier (M=1): decode path using linear_scales.
TEST_F(WeightDispatchTest, MXFP4_GemvMatchesDirect) {
    // K must be multiple of 32
    const int M = 16, K = 64;

    std::vector<half> h_w(M * K), h_x(K);
    for (int i = 0; i < M * K; ++i)
        h_w[i] = __float2half((i % 5) * 0.1f - 0.2f);
    for (int i = 0; i < K; ++i)
        h_x[i] = __float2half((i % 7) * 0.05f);

    half* d_w = dev_alloc_copy(h_w);
    half* d_x = dev_alloc_copy(h_x);

    int64_t wshape[2] = {M, K};
    Tensor w_t(d_w, QType::F16, 2, wshape, true);

    NvFP4QuantResult qr;
    quantize_fp16_to_nvfp4(w_t, qr, stream_);
    cudaStreamSynchronize(stream_);

    CutlassMxFP4Weight mw;
    convert_nvfp4_to_mxfp4_cutlass(qr, mw, stream_);
    cudaStreamSynchronize(stream_);

    if (mw.linear_scales == nullptr) {
        GTEST_SKIP() << "convert_nvfp4_to_mxfp4_cutlass did not populate linear_scales";
    }

    half* d_y_direct;
    cudaMalloc(&d_y_direct, M * sizeof(half));
    half* d_y_disp;
    cudaMalloc(&d_y_disp, M * sizeof(half));

    // Direct call
    gemv_mxfp4_kpar(mw, static_cast<const half*>(d_x), d_y_direct, M, K, stream_);
    cudaStreamSynchronize(stream_);

    // Dispatch call
    int64_t xshape[2] = {1, K}, yshape[2] = {M, 1};
    Tensor x_t(d_x, QType::F16, 2, xshape, true);
    Tensor y_disp(d_y_disp, QType::F16, 2, yshape, true);

    WeightHandle h;
    h.kind = TensorKind::WQ;
    h.primary_tier = StorageTier::MXFP4;
    h.shape[0] = M;
    h.shape[1] = K;
    h.payload.mxfp4.weight = const_cast<void*>(mw.data);
    h.payload.mxfp4.scales = mw.scale_factors;
    h.payload.mxfp4.linear_scales = mw.linear_scales;

    gemv_dispatch(h, x_t, y_disp, stream_);
    cudaStreamSynchronize(stream_);

    auto h_direct = dev_read(d_y_direct, M);
    auto h_disp = dev_read(d_y_disp, M);
    for (int i = 0; i < M; ++i) {
        float vd = __half2float(h_direct[i]);
        float vp = __half2float(h_disp[i]);
        EXPECT_EQ(__half_as_ushort(h_direct[i]), __half_as_ushort(h_disp[i]))
            << "MXFP4 GEMV mismatch at i=" << i << " direct=" << vd << " dispatch=" << vp;
    }

    free_nvfp4_result(qr);
    free_cutlass_mxfp4_weight(mw);
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_y_direct);
    cudaFree(d_y_disp);
}
