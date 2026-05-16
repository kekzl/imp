#include "graph/gemm_kernel_registry.h"
#include "compute/gemm.h"
#include "core/tensor.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>

using namespace imp;

// ---------------------------------------------------------------------------
// R5 Slice 1 — GemmKernel registry tests.
//
// Pin the public contract of the registry so future migrations of FP8 /
// NVFP4 / CUTLASS_NVFP4 / MXFP4 tiers don't accidentally reshape the
// dispatch surface.
// ---------------------------------------------------------------------------

namespace {

class GemmKernelRegistryTest : public ::testing::Test {
protected:
    void SetUp() override { cudaStreamCreate(&stream_); }
    void TearDown() override { cudaStreamDestroy(stream_); }
    cudaStream_t stream_ = nullptr;
};

// The FP16 kernel is registered at static-init via the FP16Registration
// struct in gemm_kernel_registry.cu. We expect entries for both
// (m_is_one=true) and (m_is_one=false) FP16 strategies.
TEST_F(GemmKernelRegistryTest, Fp16KernelIsRegisteredAtStaticInit) {
    const auto& reg = GemmKernelRegistry::instance();
    EXPECT_GE(reg.size(), 2u) << "FP16 kernel (M==1 and M>1) expected to be pre-registered";
}

TEST_F(GemmKernelRegistryTest, UnregisteredStrategyReturnsNoMatch) {
    const auto& reg = GemmKernelRegistry::instance();
    // CUTLASS_NVFP4 is not registered yet — Slice 1 has only FP16.
    GemmStrategy unregistered{StorageTier::CUTLASS_NVFP4, QType::F16, /*m_is_one=*/true};
    GemmKernelArgs args{};
    args.stream = stream_;
    EXPECT_EQ(reg.dispatch(unregistered, args), GemmDispatchResult::NoMatch);
}

// End-to-end correctness: register-then-dispatch produces the same output
// as calling gemm() directly. Uses small dims so the test is fast.
TEST_F(GemmKernelRegistryTest, Fp16RegistryDispatchMatchesDirectGemm) {
    constexpr int M = 4;
    constexpr int N = 8;
    constexpr int K = 16;

    // Initialize input + weight with deterministic FP16 values.
    std::vector<__half> h_input(M * K);
    std::vector<__half> h_weight(N * K);
    for (int i = 0; i < M * K; ++i) h_input[i] = __float2half((i % 7) * 0.125f - 0.5f);
    for (int i = 0; i < N * K; ++i) h_weight[i] = __float2half((i % 11) * 0.0625f - 0.25f);

    __half *d_input, *d_weight, *d_out_direct, *d_out_registry;
    cudaMalloc(&d_input, sizeof(__half) * M * K);
    cudaMalloc(&d_weight, sizeof(__half) * N * K);
    cudaMalloc(&d_out_direct, sizeof(__half) * M * N);
    cudaMalloc(&d_out_registry, sizeof(__half) * M * N);
    cudaMemcpy(d_input, h_input.data(), sizeof(__half) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight.data(), sizeof(__half) * N * K, cudaMemcpyHostToDevice);

    int64_t in_shape[2] = {M, K};
    int64_t w_shape[2] = {N, K};
    int64_t out_shape[2] = {M, N};
    Tensor input(d_input, QType::F16, 2, in_shape, /*on_device=*/true);
    Tensor weight(d_weight, QType::F16, 2, w_shape, /*on_device=*/true);
    Tensor out_direct(d_out_direct, QType::F16, 2, out_shape, /*on_device=*/true);
    Tensor out_registry(d_out_registry, QType::F16, 2, out_shape, /*on_device=*/true);

    // Path 1: direct gemm()
    gemm(input, weight, out_direct, /*alpha=*/1.0f, /*beta=*/0.0f, stream_);

    // Path 2: registry dispatch
    GemmKernelArgs args{};
    args.input = &input;
    args.output = &out_registry;
    args.stream = stream_;
    args.weight_payload = &weight;
    GemmStrategy strat{StorageTier::FP16, QType::F16, /*m_is_one=*/false};
    EXPECT_EQ(GemmKernelRegistry::instance().dispatch(strat, args), GemmDispatchResult::Ok);

    cudaStreamSynchronize(stream_);

    // Compare bit-identical (both paths invoke the same gemm() backend).
    std::vector<__half> h_out_direct(M * N), h_out_registry(M * N);
    cudaMemcpy(h_out_direct.data(), d_out_direct, sizeof(__half) * M * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_registry.data(), d_out_registry, sizeof(__half) * M * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < M * N; ++i) {
        EXPECT_EQ(__half_as_ushort(h_out_direct[i]), __half_as_ushort(h_out_registry[i]))
            << "Mismatch at i=" << i << " (direct=" << __half2float(h_out_direct[i])
            << " registry=" << __half2float(h_out_registry[i]) << ")";
    }

    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_out_direct);
    cudaFree(d_out_registry);
}

// Pin the GemmStrategy operator== contract — used by the linear-scan
// lookup in dispatch().
TEST(GemmStrategy, EqualityIsByValue) {
    GemmStrategy a{StorageTier::FP16, QType::F16, /*m_is_one=*/true};
    GemmStrategy b{StorageTier::FP16, QType::F16, /*m_is_one=*/true};
    GemmStrategy c{StorageTier::FP16, QType::F16, /*m_is_one=*/false};
    GemmStrategy d{StorageTier::FP8, QType::F16, /*m_is_one=*/true};
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
    EXPECT_FALSE(a == d);
}

}  // namespace
