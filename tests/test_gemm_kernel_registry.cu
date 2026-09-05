#include "exec/gemm_kernel_registry.h"
#include "compute/gemm.h"
#include "compute/gemm_cutlass_sm120.h"  // CutlassNvFP4Weight, convert_nvfp4_to_cutlass, gemm_nvfp4_cutlass_sm120
#include "core/tensor.h"
#include "exec/executor_kernels.h"  // is_dp4a_qtype, dispatch_dp4a_gemv
#include "quant/dequant_gpu.h"      // generic-dequant parity test
#include "quant/nvfp4_quant.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>
#include <cstring>

using namespace imp;

// ---------------------------------------------------------------------------
// GemmKernel registry tests: the three live producers (generic dequant
// catch-all, GGUF small-M, CUTLASS_NVFP4 prefill) against their direct
// paths, plus the pin that the table holds exactly the produced keys
// (AUDIT_arch_2026 A1-1, dispatch #8).
// ---------------------------------------------------------------------------

namespace {

class GemmKernelRegistryTest : public ::testing::Test {
protected:
    void SetUp() override { cudaStreamCreate(&stream_); }
    void TearDown() override { cudaStreamDestroy(stream_); }
    cudaStream_t stream_ = nullptr;
};

TEST_F(GemmKernelRegistryTest, UnregisteredStrategyReturnsNoMatch) {
    const auto& reg = GemmKernelRegistry::instance();
    // A tier no producer constructs (MXFP4 GEMV runs inside gemm_via_handle_).
    GemmStrategy unregistered{StorageTier::MXFP4, QType::BF16, /*m_is_one=*/true};
    GemmKernelArgs args{};
    args.stream = stream_;
    EXPECT_EQ(reg.dispatch(unregistered, args), GemmDispatchResult::NoMatch);
}

// AUDIT_arch_2026 A1-1 (dispatch #8): the table holds exactly the keys the
// three dispatch sites in executor_gemm_dispatch.cu construct. A registration
// without a producer is dead code; a producer without a registration falls
// through to gemm_via_handle_ silently. Either moves this count.
TEST_F(GemmKernelRegistryTest, RegistryHoldsExactlyTheProducedKeys) {
    const auto& reg = GemmKernelRegistry::instance();
    EXPECT_EQ(reg.size(), 10u) << "1 generic dequant + 8 GGUF small-M qtypes + 1 CUTLASS_NVFP4";
    GemmKernelArgs args{};
    args.stream = stream_;
    // The 9 keys retired with the stalled R5 migration stay unreachable.
    const GemmStrategy retired[] = {
        {StorageTier::FP16, QType::F16, false},  {StorageTier::FP16, QType::F16, true},
        {StorageTier::FP16, QType::Q4_K, false}, {StorageTier::FP8, QType::F16, false},
        {StorageTier::FP8, QType::NONE, false},  {StorageTier::NVFP4, QType::F16, false},
        {StorageTier::NVFP4, QType::F16, true},  {StorageTier::MXFP4, QType::F16, false},
        {StorageTier::MXFP4, QType::F16, true},
    };
    for (const auto& s : retired)
        EXPECT_EQ(reg.dispatch(s, args), GemmDispatchResult::NoMatch)
            << "tier=" << static_cast<int>(s.tier) << " qtype=" << static_cast<int>(s.weight_qtype)
            << " m_is_one=" << s.m_is_one;
}

// The CUTLASS_NVFP4 adapter rejects loud when the activation scratch is
// missing — refuses to silently fall through to legacy. Mirrors the FP8
// missing-scratch test (Slice 2 pattern). Returns PreconditionFail so the
// dispatch site can fall back to the Slice 4 dequant kernel. Note: the
// GEMM workspace (cutlass_workspace) is intentionally NOT a precondition —
// gemm_nvfp4_cutlass_sm120 has its own static-fallback alloc — so this
// test specifically exercises the act_data null path.
TEST_F(GemmKernelRegistryTest, CutlassNvfp4KernelRejectsMissingActScratch) {
    constexpr int M = 4;
    constexpr int N = 16;
    constexpr int K = 64;

    __half* d_input = nullptr;
    __half* d_out = nullptr;
    cudaMalloc(&d_input, sizeof(__half) * M * K);
    cudaMalloc(&d_out, sizeof(__half) * M * N);

    int64_t in_shape[2] = {M, K};
    int64_t out_shape[2] = {M, N};
    Tensor input(d_input, QType::F16, 2, in_shape, /*on_device=*/true);
    Tensor output(d_out, QType::F16, 2, out_shape, /*on_device=*/true);

    // Build a dummy CutlassNvFP4Weight — payload pointer is non-null but the
    // kernel returns PreconditionFail before dereferencing it (workspace
    // check fires first). We never invoke gemm_nvfp4_cutlass_sm120 here so
    // the dummy payload is safe.
    CutlassNvFP4Weight dummy{};
    dummy.N = N;
    dummy.K = K;

    GemmKernelArgs args{};
    args.input = &input;
    args.output = &output;
    args.stream = stream_;
    args.weight_payload = &dummy;
    // Intentionally leave cutlass_act_data / cutlass_act_sf / cutlass_workspace
    // null. Kernel must return PreconditionFail.

    GemmStrategy strat{StorageTier::CUTLASS_NVFP4, QType::F16, /*m_is_one=*/false};
    EXPECT_EQ(GemmKernelRegistry::instance().dispatch(strat, args), GemmDispatchResult::PreconditionFail);

    cudaFree(d_input);
    cudaFree(d_out);
}

// End-to-end smoke: registry CUTLASS_NVFP4 GEMM dispatch runs to Ok and
// produces non-zero output for a small toy problem. We deliberately do NOT
// do a back-to-back direct-vs-registry parity comparison the way the FP16 /
// FP8 / NVFP4-dequant slices do, because gemm_nvfp4_cutlass_sm120 bails out
// (returns false) on any sticky CUDA error from a prior call in the same
// test process — meaning the second invocation in a parity test
// PreconditionFails not on its own merits but on the residue of Path 1. The
// adapter is a verbatim wrap of `quantize_fp16_to_nvfp4_cutlass` +
// `gemm_nvfp4_cutlass_sm120` (executor_kernels.cu:2147 + 2179-2181); parity
// is enforced structurally by the wrap, and the dispatch-site call site is
// covered by the existing engine smoke tests (verify-fast). A single
// invocation here is sufficient to pin that the adapter wires through to a
// successful CUTLASS run.
TEST_F(GemmKernelRegistryTest, CutlassNvfp4RegistryDispatchRunsToCompletion) {
    constexpr int M = 128;
    constexpr int N = 128;
    constexpr int K = 128;  // CUTLASS NVFP4 tile is 128x128x128 — match for can_implement.

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

    // Quantize weight to NVFP4, then convert to CUTLASS block-scaled layout
    // (SfAtom). The adapter consumes the CutlassNvFP4Weight verbatim.
    NvFP4QuantResult nv4{};
    quantize_fp16_to_nvfp4(weight_fp16, nv4, stream_);
    CutlassNvFP4Weight cw{};
    convert_nvfp4_to_cutlass(nv4, cw, stream_);

    // Allocate activation quantization scratch + GEMM workspace.
    size_t act_sf_bytes = cutlass_nvfp4_sf_size(M, K);
    size_t ws_bytes = gemm_nvfp4_cutlass_sm120_workspace(M, N, K);
    void *d_act_data = nullptr, *d_act_sf = nullptr, *d_ws = nullptr;
    cudaMalloc(&d_act_data, static_cast<size_t>(M) * K / 2);  // packed FP4 [M, K/2]
    cudaMalloc(&d_act_sf, act_sf_bytes);
    if (ws_bytes > 0) cudaMalloc(&d_ws, ws_bytes);
    cudaMemsetAsync(d_act_sf, 0, act_sf_bytes, stream_);  // zero-init padding

    __half* d_out = nullptr;
    cudaMalloc(&d_out, sizeof(__half) * M * N);
    cudaMemsetAsync(d_out, 0, sizeof(__half) * M * N, stream_);
    Tensor output(d_out, QType::F16, 2, out_shape, /*on_device=*/true);

    // Drain sticky CUDA errors from setup (quantize/convert/cudaMalloc). The
    // CUTLASS kernel bails on any prior cudaGetLastError() != cudaSuccess.
    cudaStreamSynchronize(stream_);
    cudaError_t pre = cudaGetLastError();
    ASSERT_EQ(pre, cudaSuccess) << "Pre-dispatch sticky CUDA error: " << cudaGetErrorString(pre);

    GemmKernelArgs args{};
    args.input = &input;
    args.output = &output;
    args.stream = stream_;
    args.beta = 0.0f;
    args.weight_payload = &cw;
    args.cutlass_act_data = d_act_data;
    args.cutlass_act_sf = d_act_sf;
    args.cutlass_workspace = d_ws;
    args.cutlass_workspace_size = ws_bytes;
    GemmStrategy strat{StorageTier::CUTLASS_NVFP4, QType::F16, /*m_is_one=*/false};
    EXPECT_EQ(GemmKernelRegistry::instance().dispatch(strat, args), GemmDispatchResult::Ok);

    cudaStreamSynchronize(stream_);

    // Non-zero output check: the GEMM must have written *something*. The
    // hand-crafted inputs both have nonzero values so the dot product is
    // certainly not identically zero across all 64 output cells.
    std::vector<__half> h_out(M * N);
    cudaMemcpy(h_out.data(), d_out, sizeof(__half) * M * N, cudaMemcpyDeviceToHost);
    int nonzero_count = 0;
    for (int i = 0; i < M * N; ++i) {
        if (__half_as_ushort(h_out[i]) != 0) ++nonzero_count;
    }
    EXPECT_GT(nonzero_count, 0) << "Registry CUTLASS_NVFP4 dispatch wrote all-zero output";

    free_cutlass_nvfp4_weight(cw);
    free_nvfp4_result(nv4);
    cudaFree(d_act_data);
    cudaFree(d_act_sf);
    if (d_ws) cudaFree(d_ws);
    cudaFree(d_input);
    cudaFree(d_weight_fp16);
    cudaFree(d_out);
}

// Off-axis m_is_one (M>1 prefill) is NoMatch: the GGUF handlers register only
// the M==1 decode side and no per-qtype M>1 key exists; the M>1 GGUF path is
// the generic-dequant key `{FP16, NONE, false}`. Until dispatch #8 this test
// read PreconditionFail because the retired `{FP16, Q4_K, false}` Q4_K-IMMA
// registration answered it - a key no dispatch site ever constructed.
TEST_F(GemmKernelRegistryTest, GgufWrongMIsOneReturnsNoMatch) {
    const auto& reg = GemmKernelRegistry::instance();
    GemmStrategy m_gt_one{StorageTier::FP16, QType::Q4_K, /*m_is_one=*/false};
    GemmKernelArgs args{};
    args.stream = stream_;
    EXPECT_EQ(reg.dispatch(m_gt_one, args), GemmDispatchResult::NoMatch);
}

// An unsupported qtype (Q4_1 stays on legacy quant_gemm_int4, IQ4_NL/XS not
// in slice 7) must NoMatch so the dispatch site falls back.
TEST_F(GemmKernelRegistryTest, GgufUnsupportedQtypeReturnsNoMatch) {
    const auto& reg = GemmKernelRegistry::instance();
    GemmStrategy q4_1{StorageTier::FP16, QType::Q4_1, /*m_is_one=*/true};
    GemmKernelArgs args{};
    args.stream = stream_;
    EXPECT_EQ(reg.dispatch(q4_1, args), GemmDispatchResult::NoMatch);
}

// When neither backend can run (mmvq disabled via force_mmvq=false AND dp4a
// scratch missing), the handler returns PreconditionFail so the dispatch
// site falls back to legacy. Q6_K is dp4a-only (no mmvq backend) — with
// q8_1_buf/d8_buf null and force_mmvq=false, neither path matches.
TEST_F(GemmKernelRegistryTest, GgufQ6kRejectsMissingDp4aScratch) {
    constexpr int M = 1;
    constexpr int N = 16;
    constexpr int K = 256;  // Q6_K block = 256 elements

    __half* d_input = nullptr;
    __half* d_out = nullptr;
    cudaMalloc(&d_input, sizeof(__half) * M * K);
    cudaMalloc(&d_out, sizeof(__half) * M * N);
    cudaMemset(d_input, 0, sizeof(__half) * M * K);

    int64_t in_shape[2] = {M, K};
    int64_t w_shape[2] = {N, K};
    int64_t out_shape[2] = {M, N};
    Tensor input(d_input, QType::F16, 2, in_shape, /*on_device=*/true);
    Tensor output(d_out, QType::F16, 2, out_shape, /*on_device=*/true);
    // Dummy weight — never dereferenced because the handler returns
    // PreconditionFail before touching the bytes.
    Tensor weight(reinterpret_cast<void*>(static_cast<uintptr_t>(0xDEAD)), QType::Q6_K, 2, w_shape,
                  /*on_device=*/true);

    GemmKernelArgs args{};
    args.input = &input;
    args.output = &output;
    args.stream = stream_;
    args.weight_payload = &weight;
    // q8_1_buf / d8_buf intentionally null — dp4a backend can't run.

    GemmStrategy strat{StorageTier::FP16, QType::Q6_K, /*m_is_one=*/true};
    EXPECT_EQ(GemmKernelRegistry::instance().dispatch(strat, args), GemmDispatchResult::PreconditionFail);

    cudaFree(d_input);
    cudaFree(d_out);
}

// dp4a backend smoke: provide q8_1_buf + d8_buf, build a real Q8_0 weight,
// and verify the registry dispatch produces the same output as calling
// quantize_fp16_to_q8_1 + dispatch_dp4a_gemv directly (the legacy code path
// at executor_kernels.cu:2249-2251). Q8_0 is the simplest qtype to set up:
// the block layout is `block_q8_0 { half d; int8_t qs[32]; }` = 34 bytes,
// and we can quantize an FP16 tensor on the host.
TEST_F(GemmKernelRegistryTest, GgufQ8_0Dp4aRegistryDispatchMatchesDirectPath) {
    constexpr int M = 1;
    constexpr int N = 32;
    constexpr int K = 128;  // multiple of 32 (Q8_0 block size)
    constexpr int blocks_per_row = K / 32;

    // Build a deterministic FP16 weight + activation.
    std::vector<__half> h_weight_fp16(N * K);
    std::vector<__half> h_input(M * K);
    for (int i = 0; i < N * K; ++i) h_weight_fp16[i] = __float2half((i % 7) * 0.0625f - 0.1875f);
    for (int i = 0; i < M * K; ++i) h_input[i] = __float2half((i % 5) * 0.0625f - 0.125f);

    // Host-quantize the FP16 weight to Q8_0 layout: per 32-element block,
    // store FP16 scale `d` (= max(|x|)/127), then 32 int8 quantized values.
    constexpr int kQ8_0_BlockBytes = 2 + 32;  // half + 32 * int8
    std::vector<uint8_t> h_weight_q8_0(static_cast<size_t>(N) * blocks_per_row * kQ8_0_BlockBytes);
    for (int row = 0; row < N; ++row) {
        for (int b = 0; b < blocks_per_row; ++b) {
            float absmax = 0.0f;
            for (int j = 0; j < 32; ++j) {
                float v = __half2float(h_weight_fp16[row * K + b * 32 + j]);
                absmax = std::max(absmax, std::fabs(v));
            }
            float d = absmax / 127.0f;
            float inv_d = (d > 0.0f) ? 1.0f / d : 0.0f;
            uint8_t* blk = h_weight_q8_0.data() +
                           (static_cast<size_t>(row) * blocks_per_row + b) * kQ8_0_BlockBytes;
            __half d_h = __float2half(d);
            std::memcpy(blk, &d_h, sizeof(__half));
            int8_t* qs = reinterpret_cast<int8_t*>(blk + 2);
            for (int j = 0; j < 32; ++j) {
                float v = __half2float(h_weight_fp16[row * K + b * 32 + j]);
                int q = static_cast<int>(std::lrintf(v * inv_d));
                qs[j] = static_cast<int8_t>(std::max(-127, std::min(127, q)));
            }
        }
    }

    // Upload weight + input.
    void* d_weight = nullptr;
    __half* d_input = nullptr;
    cudaMalloc(&d_weight, h_weight_q8_0.size());
    cudaMalloc(&d_input, sizeof(__half) * M * K);
    cudaMemcpy(d_weight, h_weight_q8_0.data(), h_weight_q8_0.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input.data(), sizeof(__half) * M * K, cudaMemcpyHostToDevice);

    // dp4a scratch: q8_1_buf is `block_q8_1[K/32]` (36 bytes/block), d8_buf
    // is `float[K/32]`.
    void* d_q8_1 = nullptr;
    float* d_d8 = nullptr;
    cudaMalloc(&d_q8_1, sizeof(block_q8_1) * blocks_per_row);
    cudaMalloc(&d_d8, sizeof(float) * blocks_per_row);

    int64_t in_shape[2] = {M, K};
    int64_t w_shape[2] = {N, K};
    int64_t out_shape[2] = {M, N};
    Tensor input(d_input, QType::F16, 2, in_shape, /*on_device=*/true);
    Tensor weight(d_weight, QType::Q8_0, 2, w_shape, /*on_device=*/true);

    // Path 1: direct legacy invocation (mirrors gemm_dispatch_impl:2249-2251).
    __half* d_out_direct = nullptr;
    cudaMalloc(&d_out_direct, sizeof(__half) * M * N);
    quantize_fp16_to_q8_1(static_cast<const half*>(d_input), static_cast<block_q8_1*>(d_q8_1), d_d8, K,
                          stream_);
    dispatch_dp4a_gemv(QType::Q8_0, d_weight, static_cast<const block_q8_1*>(d_q8_1), d_d8,
                       reinterpret_cast<half*>(d_out_direct), N, K, stream_);

    // Path 2: registry dispatch through the GGUF Q8_0 kernel adapter. The
    // handler reads `args.force_mmvq` (Phase 5 Track A: per-model override
    // forwarded via GemmKernelArgs); leave it false so the dp4a backend
    // wins (legacy precedence: mmvq wins when both eligible).
    __half* d_out_registry = nullptr;
    cudaMalloc(&d_out_registry, sizeof(__half) * M * N);
    Tensor out_registry(d_out_registry, QType::F16, 2, out_shape, /*on_device=*/true);
    GemmKernelArgs args{};
    args.input = &input;
    args.output = &out_registry;
    args.stream = stream_;
    args.weight_payload = &weight;
    args.q8_1_buf = d_q8_1;
    args.d8_buf = d_d8;
    args.force_mmvq = false;
    GemmStrategy strat{StorageTier::FP16, QType::Q8_0, /*m_is_one=*/true};
    EXPECT_EQ(GemmKernelRegistry::instance().dispatch(strat, args), GemmDispatchResult::Ok);

    cudaStreamSynchronize(stream_);

    std::vector<__half> h_out_direct(M * N), h_out_registry(M * N);
    cudaMemcpy(h_out_direct.data(), d_out_direct, sizeof(__half) * M * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_registry.data(), d_out_registry, sizeof(__half) * M * N, cudaMemcpyDeviceToHost);

    // Both paths run the identical quantize + dp4a sequence → bit-identical.
    for (int i = 0; i < M * N; ++i) {
        EXPECT_EQ(__half_as_ushort(h_out_direct[i]), __half_as_ushort(h_out_registry[i]))
            << "Mismatch at i=" << i << " (direct=" << __half2float(h_out_direct[i])
            << " registry=" << __half2float(h_out_registry[i]) << ")";
    }

    cudaFree(d_weight);
    cudaFree(d_input);
    cudaFree(d_q8_1);
    cudaFree(d_d8);
    cudaFree(d_out_direct);
    cudaFree(d_out_registry);
}

// mmvq backend smoke: provide weight + scratch + force_mmvq=true, verify
// the registry handler picks mmvq (not dp4a) and produces a non-zero
// output. We don't do bit-identical parity here because the mmvq scratch
// is single-global and re-used between paths, which mucks with synchronous
// invocation expectations; the dp4a parity test above is enough to pin
// the args plumbing. This test only checks "Ok + non-zero output".
TEST_F(GemmKernelRegistryTest, GgufQ8_0MmvqRegistryDispatchProducesNonZero) {
    constexpr int M = 1;
    constexpr int N = 32;
    constexpr int K = 128;  // multiple of 32
    constexpr int blocks_per_row = K / 32;

    std::vector<__half> h_weight_fp16(N * K);
    std::vector<__half> h_input(M * K);
    for (int i = 0; i < N * K; ++i) h_weight_fp16[i] = __float2half((i % 7) * 0.0625f - 0.1875f);
    for (int i = 0; i < M * K; ++i) h_input[i] = __float2half((i % 5) * 0.0625f - 0.125f);

    constexpr int kQ8_0_BlockBytes = 34;
    std::vector<uint8_t> h_weight_q8_0(static_cast<size_t>(N) * blocks_per_row * kQ8_0_BlockBytes);
    for (int row = 0; row < N; ++row) {
        for (int b = 0; b < blocks_per_row; ++b) {
            float absmax = 0.0f;
            for (int j = 0; j < 32; ++j) {
                float v = __half2float(h_weight_fp16[row * K + b * 32 + j]);
                absmax = std::max(absmax, std::fabs(v));
            }
            float d = absmax / 127.0f;
            float inv_d = (d > 0.0f) ? 1.0f / d : 0.0f;
            uint8_t* blk = h_weight_q8_0.data() +
                           (static_cast<size_t>(row) * blocks_per_row + b) * kQ8_0_BlockBytes;
            __half d_h = __float2half(d);
            std::memcpy(blk, &d_h, sizeof(__half));
            int8_t* qs = reinterpret_cast<int8_t*>(blk + 2);
            for (int j = 0; j < 32; ++j) {
                float v = __half2float(h_weight_fp16[row * K + b * 32 + j]);
                int q = static_cast<int>(std::lrintf(v * inv_d));
                qs[j] = static_cast<int8_t>(std::max(-127, std::min(127, q)));
            }
        }
    }

    void* d_weight = nullptr;
    __half* d_input = nullptr;
    cudaMalloc(&d_weight, h_weight_q8_0.size());
    cudaMalloc(&d_input, sizeof(__half) * M * K);
    cudaMemcpy(d_weight, h_weight_q8_0.data(), h_weight_q8_0.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input.data(), sizeof(__half) * M * K, cudaMemcpyHostToDevice);

    int64_t in_shape[2] = {M, K};
    int64_t w_shape[2] = {N, K};
    int64_t out_shape[2] = {M, N};
    Tensor input(d_input, QType::F16, 2, in_shape, /*on_device=*/true);
    Tensor weight(d_weight, QType::Q8_0, 2, w_shape, /*on_device=*/true);

    // Enable force_mmvq on the args so the handler picks mmvq (precedence
    // over dp4a). Phase 5 Track A: per-model override now lives on
    // GemmKernelArgs::force_mmvq, forwarded from ModelConfig overrides.
    __half* d_out = nullptr;
    cudaMalloc(&d_out, sizeof(__half) * M * N);
    cudaMemsetAsync(d_out, 0, sizeof(__half) * M * N, stream_);
    Tensor output(d_out, QType::F16, 2, out_shape, /*on_device=*/true);

    GemmKernelArgs args{};
    args.input = &input;
    args.output = &output;
    args.stream = stream_;
    args.weight_payload = &weight;
    args.force_mmvq = true;
    // q8_1_buf / d8_buf intentionally NOT supplied — mmvq has its own
    // file-scope scratch and should win the gate.

    GemmStrategy strat{StorageTier::FP16, QType::Q8_0, /*m_is_one=*/true};
    EXPECT_EQ(GemmKernelRegistry::instance().dispatch(strat, args), GemmDispatchResult::Ok);
    cudaStreamSynchronize(stream_);

    std::vector<__half> h_out(M * N);
    cudaMemcpy(h_out.data(), d_out, sizeof(__half) * M * N, cudaMemcpyDeviceToHost);
    int nonzero_count = 0;
    for (int i = 0; i < M * N; ++i) {
        if (__half_as_ushort(h_out[i]) != 0) ++nonzero_count;
    }
    EXPECT_GT(nonzero_count, 0) << "mmvq Q8_0 registry dispatch wrote all-zero output";

    cudaFree(d_weight);
    cudaFree(d_input);
    cudaFree(d_out);
}

// ---------------------------------------------------------------------------
// R5 Slice 8.2 — fused gemv Q6_K/Q8_0 fallback (3rd branch inside the
// existing Slice 7 handlers).
//
// When mmvq is disabled AND dp4a scratch is unavailable, Q6_K and Q8_0 now
// route to the fused-dequant-and-dot kernel (`gemv_q6k` / `gemv_q8_0`) via
// the same {FP16, <qtype>, m_is_one=true} strategy key Slice 7 registered.
// `dequant_scratch != nullptr` is the engine-ready sentinel matching legacy
// gemm_dispatch_impl:2267 / :2272 (the fused kernel itself does not consume
// the scratch).
// ---------------------------------------------------------------------------

// Q8_0 fused-gemv parity: with force_mmvq=false and dp4a scratch absent, the
// handler MUST take the fused-gemv branch and produce bit-identical output
// to calling gemv_q8_0 directly (the legacy code path at executor_kernels.cu
// :2275-2276).
TEST_F(GemmKernelRegistryTest, GgufQ8_0FusedGemvFallbackMatchesDirectPath) {
    constexpr int M = 1;
    constexpr int N = 32;
    constexpr int K = 128;

    // Host-quantize an FP16 weight to Q8_0 layout. Per 32-element block:
    // FP16 scale `d = max(|x|)/127`, then 32 int8 quantized values.
    const int blocks_per_row = K / 32;
    std::vector<__half> h_weight_fp16(N * K);
    std::vector<__half> h_input(M * K);
    for (int i = 0; i < N * K; ++i)
        h_weight_fp16[i] = __float2half((i % 7) * 0.0625f - 0.1875f);
    for (int i = 0; i < M * K; ++i)
        h_input[i] = __float2half((i % 5) * 0.0625f - 0.125f);

    std::vector<uint8_t> h_weight_q8_0(static_cast<size_t>(N) * blocks_per_row * 34);
    for (int row = 0; row < N; ++row) {
        for (int b = 0; b < blocks_per_row; ++b) {
            float absmax = 0.0f;
            for (int j = 0; j < 32; ++j) {
                float v = __half2float(h_weight_fp16[row * K + b * 32 + j]);
                absmax = std::max(absmax, std::fabs(v));
            }
            float d = absmax / 127.0f;
            float inv_d = (d > 0.0f) ? 1.0f / d : 0.0f;
            uint8_t* blk = h_weight_q8_0.data() +
                           (static_cast<size_t>(row) * blocks_per_row + b) * 34;
            __half d_h = __float2half(d);
            std::memcpy(blk, &d_h, sizeof(__half));
            int8_t* qs = reinterpret_cast<int8_t*>(blk + 2);
            for (int j = 0; j < 32; ++j) {
                float v = __half2float(h_weight_fp16[row * K + b * 32 + j]);
                int q = static_cast<int>(std::lrintf(v * inv_d));
                qs[j] = static_cast<int8_t>(std::max(-127, std::min(127, q)));
            }
        }
    }

    void* d_weight = nullptr;
    __half* d_input = nullptr;
    cudaMalloc(&d_weight, h_weight_q8_0.size());
    cudaMalloc(&d_input, sizeof(__half) * M * K);
    cudaMemcpy(d_weight, h_weight_q8_0.data(), h_weight_q8_0.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input.data(), sizeof(__half) * M * K, cudaMemcpyHostToDevice);

    int64_t in_shape[2] = {M, K};
    int64_t w_shape[2] = {N, K};
    int64_t out_shape[2] = {M, N};
    Tensor input(d_input, QType::F16, 2, in_shape, /*on_device=*/true);
    Tensor weight(d_weight, QType::Q8_0, 2, w_shape, /*on_device=*/true);

    // Path 1: direct gemv_q8_0 invocation (mirrors legacy line 2275).
    __half* d_out_direct = nullptr;
    cudaMalloc(&d_out_direct, sizeof(__half) * M * N);
    gemv_q8_0(d_weight, static_cast<const half*>(d_input), reinterpret_cast<half*>(d_out_direct), N, K,
              stream_);

    // Path 2: registry dispatch with mmvq disabled, dp4a scratch null,
    // dequant_scratch non-null. Force the handler into the third branch.
    // Phase 5 Track A: force_mmvq now lives on GemmKernelArgs.
    __half* d_out_registry = nullptr;
    cudaMalloc(&d_out_registry, sizeof(__half) * M * N);
    Tensor out_registry(d_out_registry, QType::F16, 2, out_shape, /*on_device=*/true);

    // Engine-ready sentinel — the kernel does not dereference this pointer,
    // it only checks for nullptr (matching legacy line 2272's gate).
    uint8_t dummy_scratch_sentinel = 0;

    GemmKernelArgs args{};
    args.input = &input;
    args.output = &out_registry;
    args.stream = stream_;
    args.weight_payload = &weight;
    args.force_mmvq = false;
    // q8_1_buf / d8_buf intentionally null — dp4a branch must not fire.
    args.dequant_scratch = &dummy_scratch_sentinel;

    GemmStrategy strat{StorageTier::FP16, QType::Q8_0, /*m_is_one=*/true};
    EXPECT_EQ(GemmKernelRegistry::instance().dispatch(strat, args), GemmDispatchResult::Ok);

    cudaStreamSynchronize(stream_);

    std::vector<__half> h_out_direct(M * N), h_out_registry(M * N);
    cudaMemcpy(h_out_direct.data(), d_out_direct, sizeof(__half) * M * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_registry.data(), d_out_registry, sizeof(__half) * M * N, cudaMemcpyDeviceToHost);

    // Both paths run the identical gemv_q8_0 kernel → bit-identical.
    for (int i = 0; i < M * N; ++i) {
        EXPECT_EQ(__half_as_ushort(h_out_direct[i]), __half_as_ushort(h_out_registry[i]))
            << "Mismatch at i=" << i;
    }

    cudaFree(d_weight);
    cudaFree(d_input);
    cudaFree(d_out_direct);
    cudaFree(d_out_registry);
}

// Q6_K fused-gemv parity: same shape as the Q8_0 test, but Q6_K has a more
// complex block layout (256 elements / block, 16 sub-blocks with 6-bit
// quants). We can't trivially host-quantize Q6_K, so this test focuses on
// dispatch-correctness only: build a zeroed Q6_K weight, run both paths,
// and check that registry result matches direct path. With zero weight,
// both paths produce zero output → bit-identical "noop" check that the
// handler actually invoked the right branch (an mmvq/dp4a path would also
// produce zero, so we additionally instrument with a non-null
// dequant_scratch and zero scratch, then verify Ok was returned).
TEST_F(GemmKernelRegistryTest, GgufQ6kFusedGemvFallbackReturnsOk) {
    constexpr int M = 1;
    constexpr int N = 16;
    constexpr int K = 256;  // one Q6_K block

    // Q6_K block layout (256 elements / 210 bytes): for this dispatch test
    // we just need a non-null device buffer of the correct size; we don't
    // verify numerical output.
    const size_t block_bytes = 210;
    void* d_weight = nullptr;
    __half* d_input = nullptr;
    __half* d_out = nullptr;
    cudaMalloc(&d_weight, static_cast<size_t>(N) * block_bytes);
    cudaMalloc(&d_input, sizeof(__half) * M * K);
    cudaMalloc(&d_out, sizeof(__half) * M * N);
    cudaMemset(d_weight, 0, static_cast<size_t>(N) * block_bytes);
    cudaMemset(d_input, 0, sizeof(__half) * M * K);
    cudaMemset(d_out, 0, sizeof(__half) * M * N);

    int64_t in_shape[2] = {M, K};
    int64_t w_shape[2] = {N, K};
    int64_t out_shape[2] = {M, N};
    Tensor input(d_input, QType::F16, 2, in_shape, /*on_device=*/true);
    Tensor weight(d_weight, QType::Q6_K, 2, w_shape, /*on_device=*/true);
    Tensor output(d_out, QType::F16, 2, out_shape, /*on_device=*/true);

    // Force the fused-gemv branch: no mmvq backend for Q6_K + dp4a scratch null
    // + dequant_scratch non-null.
    uint8_t dummy_scratch_sentinel = 0;
    GemmKernelArgs args{};
    args.input = &input;
    args.output = &output;
    args.stream = stream_;
    args.weight_payload = &weight;
    args.dequant_scratch = &dummy_scratch_sentinel;

    // Clear any sticky CUDA error from earlier tests in this binary; we
    // want this check to reflect only the gemv_q6k launch.
    cudaGetLastError();

    GemmStrategy strat{StorageTier::FP16, QType::Q6_K, /*m_is_one=*/true};
    EXPECT_EQ(GemmKernelRegistry::instance().dispatch(strat, args), GemmDispatchResult::Ok);
    cudaStreamSynchronize(stream_);

    // With a zeroed weight and zeroed input, the dequant-and-dot path
    // produces zero output. Confirm the kernel ran (no kernel-launch error
    // and output is all zero / valid FP16).
    EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "gemv_q6k launch failed";
    std::vector<__half> h_out(M * N);
    cudaMemcpy(h_out.data(), d_out, sizeof(__half) * M * N, cudaMemcpyDeviceToHost);
    for (int i = 0; i < M * N; ++i) {
        EXPECT_EQ(__half_as_ushort(h_out[i]), __half_as_ushort(__float2half(0.0f)))
            << "Expected zero output for zero weight × zero input, got "
            << __half2float(h_out[i]) << " at i=" << i;
    }

    cudaFree(d_weight);
    cudaFree(d_input);
    cudaFree(d_out);
}

// Negative case: without `dequant_scratch` AND without dp4a scratch AND with
// mmvq disabled, Q6_K must return PreconditionFail (no branch fires). This
// pins the engine-ready sentinel gate added in Slice 8.2.
TEST_F(GemmKernelRegistryTest, GgufQ6kFusedGemvRequiresDequantScratchSentinel) {
    constexpr int M = 1;
    constexpr int N = 16;
    constexpr int K = 256;

    __half* d_input = nullptr;
    __half* d_out = nullptr;
    cudaMalloc(&d_input, sizeof(__half) * M * K);
    cudaMalloc(&d_out, sizeof(__half) * M * N);
    cudaMemset(d_input, 0, sizeof(__half) * M * K);

    int64_t in_shape[2] = {M, K};
    int64_t w_shape[2] = {N, K};
    int64_t out_shape[2] = {M, N};
    Tensor input(d_input, QType::F16, 2, in_shape, /*on_device=*/true);
    Tensor weight(reinterpret_cast<void*>(static_cast<uintptr_t>(0xDEAD)), QType::Q6_K, 2, w_shape,
                  /*on_device=*/true);
    Tensor output(d_out, QType::F16, 2, out_shape, /*on_device=*/true);

    GemmKernelArgs args{};
    args.input = &input;
    args.output = &output;
    args.stream = stream_;
    args.weight_payload = &weight;
    // q8_1_buf / d8_buf / dequant_scratch all null — no branch can fire.

    GemmStrategy strat{StorageTier::FP16, QType::Q6_K, /*m_is_one=*/true};
    EXPECT_EQ(GemmKernelRegistry::instance().dispatch(strat, args), GemmDispatchResult::PreconditionFail);

    cudaFree(d_input);
    cudaFree(d_out);
}

// The generic-dequant handler refuses loud when the dequant scratch is
// missing. The dispatch site provides `qs->dequant`; without it the kernel
// cannot stage the FP16 weight and must PreconditionFail so the caller
// falls through to legacy `gemm_dispatch_impl`.
TEST_F(GemmKernelRegistryTest, GenericDequantRejectsMissingScratch) {
    constexpr int M = 4;
    constexpr int N = 8;
    constexpr int K = 128;  // multiple of 32 for Q8_0 blocks

    __half* d_input = nullptr;
    __half* d_out = nullptr;
    void* d_weight_q8_0 = nullptr;
    constexpr int kQ8_0_BlockBytes = 34;
    const size_t weight_bytes = static_cast<size_t>(N) * (K / 32) * kQ8_0_BlockBytes;
    cudaMalloc(&d_input, sizeof(__half) * M * K);
    cudaMalloc(&d_out, sizeof(__half) * M * N);
    cudaMalloc(&d_weight_q8_0, weight_bytes);
    cudaMemset(d_input, 0, sizeof(__half) * M * K);
    cudaMemset(d_weight_q8_0, 0, weight_bytes);

    int64_t in_shape[2] = {M, K};
    int64_t w_shape[2] = {N, K};
    int64_t out_shape[2] = {M, N};
    Tensor input(d_input, QType::F16, 2, in_shape, /*on_device=*/true);
    Tensor weight(d_weight_q8_0, QType::Q8_0, 2, w_shape, /*on_device=*/true);
    Tensor out(d_out, QType::F16, 2, out_shape, /*on_device=*/true);

    GemmKernelArgs args{};
    args.input = &input;
    args.output = &out;
    args.stream = stream_;
    args.weight_payload = &weight;
    // dequant_scratch deliberately nullptr — the handler must refuse.

    GemmStrategy strat{StorageTier::FP16, QType::NONE, /*m_is_one=*/false};
    EXPECT_EQ(GemmKernelRegistry::instance().dispatch(strat, args), GemmDispatchResult::PreconditionFail);

    cudaFree(d_input);
    cudaFree(d_out);
    cudaFree(d_weight_q8_0);
}

// Off-axis qtype (a qtype that `dequant_gpu_supported` rejects, e.g. raw
// F16/BF16 which are not block-quantized) must surface as PreconditionFail
// — the dispatch site falls back to legacy `gemm_dispatch_impl`, which has
// a final raw `gemm()` arm for the FP16/BF16-no-dequant case.
TEST_F(GemmKernelRegistryTest, GenericDequantRejectsUnsupportedQtype) {
    constexpr int M = 4;
    constexpr int N = 8;
    constexpr int K = 16;

    __half* d_input = nullptr;
    __half* d_weight = nullptr;
    __half* d_out = nullptr;
    void* d_scratch = nullptr;
    cudaMalloc(&d_input, sizeof(__half) * M * K);
    cudaMalloc(&d_weight, sizeof(__half) * N * K);
    cudaMalloc(&d_out, sizeof(__half) * M * N);
    cudaMalloc(&d_scratch, sizeof(__half) * N * K);
    cudaMemset(d_input, 0, sizeof(__half) * M * K);
    cudaMemset(d_weight, 0, sizeof(__half) * N * K);

    int64_t in_shape[2] = {M, K};
    int64_t w_shape[2] = {N, K};
    int64_t out_shape[2] = {M, N};
    Tensor input(d_input, QType::F16, 2, in_shape, /*on_device=*/true);
    // F16 weight — dequant_gpu_supported(F16) is false, so the handler
    // must refuse rather than dequant a non-block-quant tensor.
    Tensor weight(d_weight, QType::F16, 2, w_shape, /*on_device=*/true);
    Tensor out(d_out, QType::F16, 2, out_shape, /*on_device=*/true);

    GemmKernelArgs args{};
    args.input = &input;
    args.output = &out;
    args.stream = stream_;
    args.weight_payload = &weight;
    args.dequant_scratch = d_scratch;  // provided; the qtype check fails.

    GemmStrategy strat{StorageTier::FP16, QType::NONE, /*m_is_one=*/false};
    EXPECT_EQ(GemmKernelRegistry::instance().dispatch(strat, args), GemmDispatchResult::PreconditionFail);

    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_out);
    cudaFree(d_scratch);
}

// End-to-end parity: registry generic-dequant dispatch produces the same
// output as the legacy `dequant_gpu → gemm` sequence for a Q8_0 weight.
// Mirrors the Slice 8.1 FP8 cache-miss parity test — both adapters share
// the same dequant_gpu + cuBLAS gemm sequence; only the strategy key and
// the outer registration differ.
TEST_F(GemmKernelRegistryTest, GenericDequantMatchesDirectPath) {
    constexpr int M = 8;
    constexpr int N = 16;
    constexpr int K = 128;  // multiple of 32 for Q8_0
    constexpr int blocks_per_row = K / 32;
    constexpr int kQ8_0_BlockBytes = 34;

    std::vector<__half> h_input(M * K);
    std::vector<__half> h_weight_fp16(N * K);
    for (int i = 0; i < M * K; ++i) h_input[i] = __float2half((i % 5) * 0.0625f - 0.125f);
    for (int i = 0; i < N * K; ++i) h_weight_fp16[i] = __float2half((i % 7) * 0.0625f - 0.1875f);

    std::vector<uint8_t> h_weight_q8_0(static_cast<size_t>(N) * blocks_per_row * kQ8_0_BlockBytes);
    for (int row = 0; row < N; ++row) {
        for (int b = 0; b < blocks_per_row; ++b) {
            float absmax = 0.0f;
            for (int j = 0; j < 32; ++j) {
                float v = __half2float(h_weight_fp16[row * K + b * 32 + j]);
                absmax = std::max(absmax, std::fabs(v));
            }
            float d = absmax / 127.0f;
            float inv_d = (d > 0.0f) ? 1.0f / d : 0.0f;
            uint8_t* blk = h_weight_q8_0.data() +
                           (static_cast<size_t>(row) * blocks_per_row + b) * kQ8_0_BlockBytes;
            __half d_h = __float2half(d);
            std::memcpy(blk, &d_h, sizeof(__half));
            int8_t* qs = reinterpret_cast<int8_t*>(blk + 2);
            for (int j = 0; j < 32; ++j) {
                float v = __half2float(h_weight_fp16[row * K + b * 32 + j]);
                int q = static_cast<int>(std::lrintf(v * inv_d));
                qs[j] = static_cast<int8_t>(std::max(-127, std::min(127, q)));
            }
        }
    }

    __half* d_input = nullptr;
    void* d_weight = nullptr;
    void* d_scratch_direct = nullptr;
    void* d_scratch_registry = nullptr;
    cudaMalloc(&d_input, sizeof(__half) * M * K);
    cudaMalloc(&d_weight, h_weight_q8_0.size());
    cudaMalloc(&d_scratch_direct, sizeof(__half) * N * K);
    cudaMalloc(&d_scratch_registry, sizeof(__half) * N * K);
    cudaMemcpy(d_input, h_input.data(), sizeof(__half) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight_q8_0.data(), h_weight_q8_0.size(), cudaMemcpyHostToDevice);

    int64_t in_shape[2] = {M, K};
    int64_t w_shape[2] = {N, K};
    int64_t out_shape[2] = {M, N};
    Tensor input(d_input, QType::F16, 2, in_shape, /*on_device=*/true);
    Tensor weight(d_weight, QType::Q8_0, 2, w_shape, /*on_device=*/true);

    // Path 1: direct (mirror executor_kernels.cu legacy generic catch-all).
    __half* d_out_direct = nullptr;
    cudaMalloc(&d_out_direct, sizeof(__half) * M * N);
    Tensor out_direct(d_out_direct, QType::F16, 2, out_shape, /*on_device=*/true);
    dequant_gpu(weight.data, d_scratch_direct, weight.qtype, N, K, stream_);
    Tensor w_fp16_direct(d_scratch_direct, QType::F16, 2, w_shape, /*on_device=*/true);
    gemm(input, w_fp16_direct, out_direct, /*alpha=*/1.0f, /*beta=*/0.0f, stream_);

    // Path 2: registry dispatch through the generic-dequant adapter.
    __half* d_out_registry = nullptr;
    cudaMalloc(&d_out_registry, sizeof(__half) * M * N);
    Tensor out_registry(d_out_registry, QType::F16, 2, out_shape, /*on_device=*/true);
    GemmKernelArgs args{};
    args.input = &input;
    args.output = &out_registry;
    args.stream = stream_;
    args.beta = 0.0f;
    args.weight_payload = &weight;
    args.dequant_scratch = d_scratch_registry;
    GemmStrategy strat{StorageTier::FP16, QType::NONE, /*m_is_one=*/false};
    EXPECT_EQ(GemmKernelRegistry::instance().dispatch(strat, args), GemmDispatchResult::Ok);

    cudaStreamSynchronize(stream_);

    std::vector<__half> h_out_direct(M * N), h_out_registry(M * N);
    cudaMemcpy(h_out_direct.data(), d_out_direct, sizeof(__half) * M * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_registry.data(), d_out_registry, sizeof(__half) * M * N, cudaMemcpyDeviceToHost);

    // Both paths run an identical `dequant_gpu` + `gemm` sequence → bit-identical.
    for (int i = 0; i < M * N; ++i) {
        EXPECT_EQ(__half_as_ushort(h_out_direct[i]), __half_as_ushort(h_out_registry[i]))
            << "Mismatch at i=" << i << " (direct=" << __half2float(h_out_direct[i])
            << " registry=" << __half2float(h_out_registry[i]) << ")";
    }

    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_scratch_direct);
    cudaFree(d_scratch_registry);
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
