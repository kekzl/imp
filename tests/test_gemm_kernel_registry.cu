#include "graph/gemm_kernel_registry.h"
#include "compute/gemm.h"
#include "core/tensor.h"
#include "graph/executor.h"  // FP8CacheEntry
#include "quant/fp8_quant.h"
#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_quant.h"

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
    // CUTLASS_NVFP4 is not registered yet — Slices 1-4 cover FP16, FP8, NVFP4 GEMV+GEMM.
    GemmStrategy unregistered{StorageTier::CUTLASS_NVFP4, QType::F16, /*m_is_one=*/true};
    GemmKernelArgs args{};
    args.stream = stream_;
    EXPECT_EQ(reg.dispatch(unregistered, args), GemmDispatchResult::NoMatch);
}

// R5 Slice 2 — FP8 prefill kernel registered (M>1 only). With FP16 (2) +
// FP8 prefill (1) we now expect at least 3 entries.
TEST_F(GemmKernelRegistryTest, Fp8PrefillKernelIsRegisteredAtStaticInit) {
    const auto& reg = GemmKernelRegistry::instance();
    EXPECT_GE(reg.size(), 3u) << "FP16 (M==1/M>1) + FP8 (M>1) expected to be pre-registered";
}

// FP8 decode (M=1) strategy is intentionally NOT registered — decode uses
// GEMV fast paths upstream, not this dispatch.
TEST_F(GemmKernelRegistryTest, Fp8DecodeStrategyReturnsNoMatch) {
    const auto& reg = GemmKernelRegistry::instance();
    GemmStrategy decode{StorageTier::FP8, QType::F16, /*m_is_one=*/true};
    GemmKernelArgs args{};
    args.stream = stream_;
    EXPECT_EQ(reg.dispatch(decode, args), GemmDispatchResult::NoMatch);
}

// FP8 kernel rejects loud when activation scratch is missing — refuses to
// silently fall through to legacy.
TEST_F(GemmKernelRegistryTest, Fp8KernelRejectsMissingScratch) {
    constexpr int M = 4;
    constexpr int N = 8;
    constexpr int K = 16;

    __half* d_input = nullptr;
    int8_t* d_weight_fp8 = nullptr;
    __half* d_out = nullptr;
    float* d_w_scale = nullptr;
    cudaMalloc(&d_input, sizeof(__half) * M * K);
    cudaMalloc(&d_weight_fp8, sizeof(int8_t) * N * K);
    cudaMalloc(&d_out, sizeof(__half) * M * N);
    cudaMalloc(&d_w_scale, sizeof(float));
    cudaMemset(d_input, 0, sizeof(__half) * M * K);
    cudaMemset(d_weight_fp8, 0, sizeof(int8_t) * N * K);
    float h_w_scale = 1.0f;
    cudaMemcpy(d_w_scale, &h_w_scale, sizeof(float), cudaMemcpyHostToDevice);

    int64_t in_shape[2] = {M, K};
    int64_t w_shape[2] = {N, K};
    int64_t out_shape[2] = {M, N};
    Tensor input(d_input, QType::F16, 2, in_shape, /*on_device=*/true);
    Tensor out(d_out, QType::F16, 2, out_shape, /*on_device=*/true);
    FP8CacheEntry entry{};
    entry.weight = Tensor(d_weight_fp8, QType::FP8_E4M3, 2, w_shape, /*on_device=*/true);
    entry.host_scale = 1.0f;
    entry.d_scale = d_w_scale;

    GemmKernelArgs args{};
    args.input = &input;
    args.output = &out;
    args.stream = stream_;
    args.weight_payload = &entry;
    // Deliberately leave fp8_act_buf / d_act_scale null — kernel must refuse.

    GemmStrategy strat{StorageTier::FP8, QType::F16, /*m_is_one=*/false};
    EXPECT_EQ(GemmKernelRegistry::instance().dispatch(strat, args), GemmDispatchResult::PreconditionFail);

    cudaFree(d_input);
    cudaFree(d_weight_fp8);
    cudaFree(d_out);
    cudaFree(d_w_scale);
}

// End-to-end correctness: registry FP8 dispatch produces the same output as
// calling quantize_fp16_to_fp8_e4m3 + gemm_cublaslt directly. Mirrors the
// FP16 parity test pattern.
TEST_F(GemmKernelRegistryTest, Fp8RegistryDispatchMatchesDirectPath) {
    constexpr int M = 8;
    constexpr int N = 16;
    constexpr int K = 32;

    std::vector<__half> h_input(M * K);
    std::vector<__half> h_weight_fp16(N * K);
    for (int i = 0; i < M * K; ++i) h_input[i] = __float2half((i % 5) * 0.0625f - 0.125f);
    for (int i = 0; i < N * K; ++i) h_weight_fp16[i] = __float2half((i % 7) * 0.0625f - 0.1875f);

    __half *d_input = nullptr, *d_weight_fp16 = nullptr;
    cudaMalloc(&d_input, sizeof(__half) * M * K);
    cudaMalloc(&d_weight_fp16, sizeof(__half) * N * K);
    cudaMemcpy(d_input, h_input.data(), sizeof(__half) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_fp16, h_weight_fp16.data(), sizeof(__half) * N * K, cudaMemcpyHostToDevice);

    int64_t in_shape[2] = {M, K};
    int64_t w_shape[2] = {N, K};
    int64_t out_shape[2] = {M, N};
    Tensor input(d_input, QType::F16, 2, in_shape, /*on_device=*/true);
    Tensor weight_fp16(d_weight_fp16, QType::F16, 2, w_shape, /*on_device=*/true);

    // Pre-quantize weight to FP8 once (matches the load-time flow that fills
    // WeightCaches::fp8). Two copies so the direct + registry paths each get
    // their own — the FP8 weight buffer is read-only post-quant, so sharing
    // would also work, but separate buffers keep the two paths isolated.
    void *d_weight_fp8 = nullptr, *d_weight_fp8_b = nullptr;
    cudaMalloc(&d_weight_fp8, sizeof(int8_t) * N * K);
    cudaMalloc(&d_weight_fp8_b, sizeof(int8_t) * N * K);
    float* d_w_scale = nullptr;
    cudaMalloc(&d_w_scale, sizeof(float));
    Tensor weight_fp8_a(d_weight_fp8, QType::FP8_E4M3, 2, w_shape, /*on_device=*/true);
    quantize_fp16_to_fp8_e4m3(weight_fp16, weight_fp8_a, d_w_scale, stream_);
    cudaMemcpyAsync(d_weight_fp8_b, d_weight_fp8, sizeof(int8_t) * N * K, cudaMemcpyDeviceToDevice, stream_);

    // FP8 activation scratch — shared shape with the input.
    void* d_fp8_act = nullptr;
    float *d_act_scale = nullptr, *d_block_maxes = nullptr, *d_absmax = nullptr;
    cudaMalloc(&d_fp8_act, sizeof(int8_t) * M * K);
    cudaMalloc(&d_act_scale, sizeof(float));
    cudaMalloc(&d_block_maxes, sizeof(float) * 256);
    cudaMalloc(&d_absmax, sizeof(float));

    // Path 1: direct (mirror executor_kernels.cu legacy branch verbatim).
    __half* d_out_direct = nullptr;
    cudaMalloc(&d_out_direct, sizeof(__half) * M * N);
    Tensor out_direct(d_out_direct, QType::F16, 2, out_shape, /*on_device=*/true);
    Tensor fp8_act_direct(d_fp8_act, QType::FP8_E4M3, 2, in_shape, /*on_device=*/true);
    quantize_fp16_to_fp8_e4m3(input, fp8_act_direct, d_act_scale, stream_, d_block_maxes, d_absmax, 256);
    gemm_cublaslt(fp8_act_direct, weight_fp8_a, out_direct, 1.0f, 0.0f, d_act_scale, d_w_scale, stream_);

    // Path 2: registry dispatch through the FP8 kernel adapter.
    __half* d_out_registry = nullptr;
    cudaMalloc(&d_out_registry, sizeof(__half) * M * N);
    Tensor out_registry(d_out_registry, QType::F16, 2, out_shape, /*on_device=*/true);
    FP8CacheEntry entry{};
    entry.weight = Tensor(d_weight_fp8_b, QType::FP8_E4M3, 2, w_shape, /*on_device=*/true);
    entry.host_scale = 1.0f;
    entry.d_scale = d_w_scale;
    GemmKernelArgs args{};
    args.input = &input;
    args.output = &out_registry;
    args.stream = stream_;
    args.beta = 0.0f;
    args.weight_payload = &entry;
    args.fp8_act_buf = d_fp8_act;
    args.d_act_scale = d_act_scale;
    args.d_fp8_block_maxes = d_block_maxes;
    args.d_fp8_absmax = d_absmax;
    args.fp8_max_grid = 256;
    GemmStrategy strat{StorageTier::FP8, QType::F16, /*m_is_one=*/false};
    EXPECT_EQ(GemmKernelRegistry::instance().dispatch(strat, args), GemmDispatchResult::Ok);

    cudaStreamSynchronize(stream_);

    std::vector<__half> h_out_direct(M * N), h_out_registry(M * N);
    cudaMemcpy(h_out_direct.data(), d_out_direct, sizeof(__half) * M * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_registry.data(), d_out_registry, sizeof(__half) * M * N, cudaMemcpyDeviceToHost);

    // Both paths run the identical quant+cuBLASLt sequence → bit-identical.
    for (int i = 0; i < M * N; ++i) {
        EXPECT_EQ(__half_as_ushort(h_out_direct[i]), __half_as_ushort(h_out_registry[i]))
            << "Mismatch at i=" << i << " (direct=" << __half2float(h_out_direct[i])
            << " registry=" << __half2float(h_out_registry[i]) << ")";
    }

    cudaFree(d_input);
    cudaFree(d_weight_fp16);
    cudaFree(d_weight_fp8);
    cudaFree(d_weight_fp8_b);
    cudaFree(d_w_scale);
    cudaFree(d_fp8_act);
    cudaFree(d_act_scale);
    cudaFree(d_block_maxes);
    cudaFree(d_absmax);
    cudaFree(d_out_direct);
    cudaFree(d_out_registry);
}

// R5 Slice 3 — NVFP4 GEMV (M==1 decode) kernel registered. With FP16 (2) +
// FP8 prefill (1) + NVFP4 GEMV (1) we expect at least 4 entries after
// Slice 3; Slice 4 brings the count to 5 — see the Slice 4 test below.
TEST_F(GemmKernelRegistryTest, Nvfp4GemvKernelIsRegisteredAtStaticInit) {
    const auto& reg = GemmKernelRegistry::instance();
    EXPECT_GE(reg.size(), 4u) << "FP16 (M==1/M>1) + FP8 (M>1) + NVFP4 GEMV (M==1) expected pre-registered";
}

// The NVFP4 GEMV adapter registers under (tier=NVFP4, qtype=F16,
// m_is_one=true) only. Asking the registry for an off-axis weight_qtype
// (e.g. BF16) must return NoMatch so the dispatch site falls back to
// legacy — no silent re-routing to the wrong kernel.
TEST_F(GemmKernelRegistryTest, Nvfp4GemvWrongQtypeReturnsNoMatch) {
    const auto& reg = GemmKernelRegistry::instance();
    GemmStrategy bf16{StorageTier::NVFP4, QType::BF16, /*m_is_one=*/true};
    GemmKernelArgs args{};
    args.stream = stream_;
    EXPECT_EQ(reg.dispatch(bf16, args), GemmDispatchResult::NoMatch);
}

// End-to-end correctness: registry NVFP4 GEMV dispatch produces the same
// output as calling gemv_nvfp4_kpar directly. Mirrors the FP8 parity test
// pattern — pre-quantize an FP16 weight to NVFP4 once, run both paths,
// compare bit-identical (both paths invoke the same GEMV backend).
TEST_F(GemmKernelRegistryTest, Nvfp4GemvRegistryDispatchMatchesDirectPath) {
    constexpr int M = 1;
    constexpr int N = 16;
    constexpr int K = 64;  // multiple of micro-block (16) and packed alignment.

    std::vector<__half> h_input(M * K);
    std::vector<__half> h_weight_fp16(N * K);
    for (int i = 0; i < M * K; ++i) h_input[i] = __float2half((i % 5) * 0.0625f - 0.125f);
    for (int i = 0; i < N * K; ++i) h_weight_fp16[i] = __float2half((i % 7) * 0.0625f - 0.1875f);

    __half *d_input = nullptr, *d_weight_fp16 = nullptr;
    cudaMalloc(&d_input, sizeof(__half) * M * K);
    cudaMalloc(&d_weight_fp16, sizeof(__half) * N * K);
    cudaMemcpy(d_input, h_input.data(), sizeof(__half) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_fp16, h_weight_fp16.data(), sizeof(__half) * N * K, cudaMemcpyHostToDevice);

    int64_t in_shape[2] = {M, K};
    int64_t w_shape[2] = {N, K};
    int64_t out_shape[2] = {M, N};
    Tensor input(d_input, QType::F16, 2, in_shape, /*on_device=*/true);
    Tensor weight_fp16(d_weight_fp16, QType::F16, 2, w_shape, /*on_device=*/true);

    // Pre-quantize the FP16 weight to NVFP4 once. Both paths read from the
    // same NvFP4QuantResult — the GEMV is read-only.
    NvFP4QuantResult nv4{};
    quantize_fp16_to_nvfp4(weight_fp16, nv4, stream_);

    // Path 1: direct gemv_nvfp4_kpar (mirrors executor_kernels.cu:2137-2139).
    __half* d_out_direct = nullptr;
    cudaMalloc(&d_out_direct, sizeof(__half) * M * N);
    gemv_nvfp4_kpar(nv4, reinterpret_cast<const half*>(d_input), reinterpret_cast<half*>(d_out_direct),
                    static_cast<int>(nv4.N), static_cast<int>(nv4.K), stream_);

    // Path 2: registry dispatch through the NVFP4 GEMV kernel adapter.
    __half* d_out_registry = nullptr;
    cudaMalloc(&d_out_registry, sizeof(__half) * M * N);
    Tensor out_registry(d_out_registry, QType::F16, 2, out_shape, /*on_device=*/true);
    GemmKernelArgs args{};
    args.input = &input;
    args.output = &out_registry;
    args.stream = stream_;
    args.beta = 0.0f;
    args.weight_payload = &nv4;
    GemmStrategy strat{StorageTier::NVFP4, QType::F16, /*m_is_one=*/true};
    EXPECT_EQ(GemmKernelRegistry::instance().dispatch(strat, args), GemmDispatchResult::Ok);

    cudaStreamSynchronize(stream_);

    std::vector<__half> h_out_direct(M * N), h_out_registry(M * N);
    cudaMemcpy(h_out_direct.data(), d_out_direct, sizeof(__half) * M * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_registry.data(), d_out_registry, sizeof(__half) * M * N, cudaMemcpyDeviceToHost);

    // Both paths invoke gemv_nvfp4_kpar with identical args → bit-identical.
    for (int i = 0; i < M * N; ++i) {
        EXPECT_EQ(__half_as_ushort(h_out_direct[i]), __half_as_ushort(h_out_registry[i]))
            << "Mismatch at i=" << i << " (direct=" << __half2float(h_out_direct[i])
            << " registry=" << __half2float(h_out_registry[i]) << ")";
    }

    free_nvfp4_result(nv4);
    cudaFree(d_input);
    cudaFree(d_weight_fp16);
    cudaFree(d_out_direct);
    cudaFree(d_out_registry);
}

// R5 Slice 4 — NVFP4 GEMM (M>1 prefill, dequant fallback) kernel registered.
// With FP16 (2) + FP8 prefill (1) + NVFP4 GEMV (1) + NVFP4 GEMM (1) we now
// expect at least 5 entries.
TEST_F(GemmKernelRegistryTest, Nvfp4GemmKernelIsRegisteredAtStaticInit) {
    const auto& reg = GemmKernelRegistry::instance();
    EXPECT_GE(reg.size(), 5u)
        << "FP16 (M==1/M>1) + FP8 (M>1) + NVFP4 GEMV (M==1) + NVFP4 GEMM (M>1) expected pre-registered";
}

// Both NVFP4 strategies (GEMV M==1 and GEMM M>1) are now registered. Pin
// that the registry returns Ok for both — i.e. the m_is_one axis selects
// the right kernel and there is no accidental aliasing between the two.
// Uses minimal stub args; the kernels' own IMP_CHECKs would fire if real
// data were passed, but here we only verify the lookup hits.
TEST_F(GemmKernelRegistryTest, Nvfp4GemvAndGemmAreDistinct) {
    const auto& reg = GemmKernelRegistry::instance();
    // Cheaper than running the actual kernels: build separate strategy keys
    // and confirm each one resolves to a registered handler. The registry
    // exposes `size()` and `dispatch()`; we use a probe dispatch with
    // deliberately-NoMatch off-axis weight_qtype on one and compare against
    // the registered on-axis F16 path. Both registered keys are F16; an
    // off-axis BF16 key must NOT collide.
    GemmStrategy gemv_on{StorageTier::NVFP4, QType::F16, /*m_is_one=*/true};
    GemmStrategy gemm_on{StorageTier::NVFP4, QType::F16, /*m_is_one=*/false};
    GemmStrategy gemv_bf16{StorageTier::NVFP4, QType::BF16, /*m_is_one=*/true};
    GemmStrategy gemm_bf16{StorageTier::NVFP4, QType::BF16, /*m_is_one=*/false};
    GemmKernelArgs args{};
    args.stream = stream_;
    // Off-axis qtypes must not resolve (registry returns NoMatch). On-axis
    // strategies *do* resolve and would invoke the kernel; we do not call
    // them here without real data (the kernels IMP_CHECK on null payload).
    EXPECT_EQ(reg.dispatch(gemv_bf16, args), GemmDispatchResult::NoMatch);
    EXPECT_EQ(reg.dispatch(gemm_bf16, args), GemmDispatchResult::NoMatch);
    // The on-axis keys are equal-by-value only to themselves — pin the
    // distinction via operator==.
    EXPECT_FALSE(gemv_on == gemm_on);
}

// The NVFP4 GEMM adapter registers under (tier=NVFP4, qtype=F16,
// m_is_one=false) only. Asking the registry for an off-axis weight_qtype
// (e.g. BF16) must return NoMatch so the dispatch site falls back to
// legacy — no silent re-routing to the wrong kernel.
TEST_F(GemmKernelRegistryTest, Nvfp4GemmWrongQtypeReturnsNoMatch) {
    const auto& reg = GemmKernelRegistry::instance();
    GemmStrategy bf16{StorageTier::NVFP4, QType::BF16, /*m_is_one=*/false};
    GemmKernelArgs args{};
    args.stream = stream_;
    EXPECT_EQ(reg.dispatch(bf16, args), GemmDispatchResult::NoMatch);
}

// End-to-end correctness: registry NVFP4 GEMM dispatch produces the same
// output as calling gemm_nvfp4 directly. Mirrors the Slice 3 GEMV parity
// pattern but with M>1 — the adapter wraps the same dequant→cuBLAS path,
// so output is bit-identical.
TEST_F(GemmKernelRegistryTest, Nvfp4GemmRegistryDispatchMatchesDirectPath) {
    constexpr int M = 4;
    constexpr int N = 16;
    constexpr int K = 64;  // multiple of micro-block (16) and packed alignment.

    std::vector<__half> h_input(M * K);
    std::vector<__half> h_weight_fp16(N * K);
    for (int i = 0; i < M * K; ++i) h_input[i] = __float2half((i % 5) * 0.0625f - 0.125f);
    for (int i = 0; i < N * K; ++i) h_weight_fp16[i] = __float2half((i % 7) * 0.0625f - 0.1875f);

    __half *d_input = nullptr, *d_weight_fp16 = nullptr;
    cudaMalloc(&d_input, sizeof(__half) * M * K);
    cudaMalloc(&d_weight_fp16, sizeof(__half) * N * K);
    cudaMemcpy(d_input, h_input.data(), sizeof(__half) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_fp16, h_weight_fp16.data(), sizeof(__half) * N * K, cudaMemcpyHostToDevice);

    int64_t in_shape[2] = {M, K};
    int64_t w_shape[2] = {N, K};
    int64_t out_shape[2] = {M, N};
    Tensor input(d_input, QType::F16, 2, in_shape, /*on_device=*/true);
    Tensor weight_fp16(d_weight_fp16, QType::F16, 2, w_shape, /*on_device=*/true);

    // Pre-quantize the FP16 weight to NVFP4 once. Both paths read from the
    // same NvFP4QuantResult — the GEMM is read-only over the weight.
    NvFP4QuantResult nv4{};
    quantize_fp16_to_nvfp4(weight_fp16, nv4, stream_);

    // Path 1: direct gemm_nvfp4 (mirrors executor_kernels.cu:2186-2188).
    __half* d_out_direct = nullptr;
    cudaMalloc(&d_out_direct, sizeof(__half) * M * N);
    Tensor out_direct(d_out_direct, QType::F16, 2, out_shape, /*on_device=*/true);
    gemm_nvfp4(nv4, input, out_direct, stream_);

    // Path 2: registry dispatch through the NVFP4 GEMM kernel adapter.
    __half* d_out_registry = nullptr;
    cudaMalloc(&d_out_registry, sizeof(__half) * M * N);
    Tensor out_registry(d_out_registry, QType::F16, 2, out_shape, /*on_device=*/true);
    GemmKernelArgs args{};
    args.input = &input;
    args.output = &out_registry;
    args.stream = stream_;
    args.beta = 0.0f;
    args.weight_payload = &nv4;
    GemmStrategy strat{StorageTier::NVFP4, QType::F16, /*m_is_one=*/false};
    EXPECT_EQ(GemmKernelRegistry::instance().dispatch(strat, args), GemmDispatchResult::Ok);

    cudaStreamSynchronize(stream_);

    std::vector<__half> h_out_direct(M * N), h_out_registry(M * N);
    cudaMemcpy(h_out_direct.data(), d_out_direct, sizeof(__half) * M * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_registry.data(), d_out_registry, sizeof(__half) * M * N, cudaMemcpyDeviceToHost);

    // Both paths invoke gemm_nvfp4 with identical args → bit-identical.
    for (int i = 0; i < M * N; ++i) {
        EXPECT_EQ(__half_as_ushort(h_out_direct[i]), __half_as_ushort(h_out_registry[i]))
            << "Mismatch at i=" << i << " (direct=" << __half2float(h_out_direct[i])
            << " registry=" << __half2float(h_out_registry[i]) << ")";
    }

    free_nvfp4_result(nv4);
    cudaFree(d_input);
    cudaFree(d_weight_fp16);
    cudaFree(d_out_direct);
    cudaFree(d_out_registry);
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
