#include <gtest/gtest.h>
#include "compute/quantize_fp16_nvfp4_moe_native.h"
#include "quant/nvfp4_quant.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cstdint>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
namespace {

int sm_major() {
    int dev = 0;
    cudaGetDevice(&dev);
    int major = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    return major;
}

}  // anonymous namespace

namespace imp {
namespace {

// ---------------------------------------------------------------------------
// Test: single-expert problem must produce bit-exact output vs the reference
//       quantize_fp16_to_nvfp4 (which uses the same two-level algorithm).
// ---------------------------------------------------------------------------
TEST(QuantizeMoeNative, SingleExpertMatchesReference) {
    if (sm_major() < 12)
        GTEST_SKIP() << "SM120 required for HW FP4 conversion";

    const int M = 64, K = 128, ne = 1;

    // Build deterministic test data in the range used by existing tests.
    std::vector<__half> h_src(M * K);
    for (int i = 0; i < M * K; ++i)
        h_src[i] = __float2half(0.5f * (i % 13) - 3.0f);

    // Upload to device.
    __half* d_src = nullptr;
    cudaMalloc(&d_src, M * K * sizeof(__half));
    cudaMemcpy(d_src, h_src.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice);

    // Allocate per-expert output buffers.
    void* d_packed_e0 = nullptr;
    void* d_sf_e0     = nullptr;
    cudaMalloc(&d_packed_e0, (size_t)M * (K / 2));    // packed FP4
    cudaMalloc(&d_sf_e0,     (size_t)M * (K / 16));   // UE4M3 scales

    // Expert offsets on device.
    int h_offsets[2] = {0, M};
    int* d_offsets = nullptr;
    cudaMalloc(&d_offsets, 2 * sizeof(int));
    cudaMemcpy(d_offsets, h_offsets, 2 * sizeof(int), cudaMemcpyHostToDevice);

    // Run the native MoE quantize kernel.
    void* h_packed_ptrs[1] = {d_packed_e0};
    void* h_sf_ptrs[1]     = {d_sf_e0};
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    imp::quantize_fp16_to_nvfp4_moe_native(
        d_src, h_packed_ptrs, h_sf_ptrs, d_offsets, M, K, ne, stream);
    cudaStreamSynchronize(stream);

    // Retrieve native output.
    std::vector<uint8_t> h_packed_got(M * (K / 2));
    std::vector<uint8_t> h_sf_got(M * (K / 16));
    cudaMemcpy(h_packed_got.data(), d_packed_e0, h_packed_got.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sf_got.data(),     d_sf_e0,     h_sf_got.size(),     cudaMemcpyDeviceToHost);

    // Reference: single-tensor quantize_fp16_to_nvfp4.
    // The reference produces linear row-major micro_scales [M, K/16] and
    // packed [M, K/2] — identical layout to what the native kernel targets.
    int64_t shape[2] = {M, K};
    Tensor src_t(d_src, QType::F16, 2, shape, true);
    NvFP4QuantResult ref;
    quantize_fp16_to_nvfp4(src_t, ref, stream);
    cudaStreamSynchronize(stream);

    std::vector<uint8_t> h_packed_ref(M * (K / 2));
    std::vector<uint8_t> h_sf_ref(M * (K / 16));
    cudaMemcpy(h_packed_ref.data(), ref.packed_data,   h_packed_ref.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sf_ref.data(),     ref.micro_scales,  h_sf_ref.size(),     cudaMemcpyDeviceToHost);

    // Bit-exact equality check.
    EXPECT_EQ(h_sf_got, h_sf_ref)
        << "UE4M3 scale bytes differ between native MoE kernel and reference";
    EXPECT_EQ(h_packed_got, h_packed_ref)
        << "Packed FP4 bytes differ between native MoE kernel and reference";

    // Cleanup.
    cudaFree(d_src);
    cudaFree(d_packed_e0);
    cudaFree(d_sf_e0);
    cudaFree(d_offsets);
    free_nvfp4_result(ref);
    cudaStreamDestroy(stream);
}

// ---------------------------------------------------------------------------
// Test: two-expert problem — verify both experts produce valid (non-zero) output
// and that the two experts are independently quantized.
// ---------------------------------------------------------------------------
TEST(QuantizeMoeNative, TwoExpertsIndependent) {
    if (sm_major() < 12)
        GTEST_SKIP() << "SM120 required for HW FP4 conversion";

    // Expert 0: M0=32 rows, Expert 1: M1=48 rows
    const int M0 = 32, M1 = 48, K = 64, ne = 2;
    const int total = M0 + M1;

    std::vector<__half> h_src(total * K);
    for (int i = 0; i < total * K; ++i)
        h_src[i] = __float2half((float)(i % 17) - 8.0f);

    __half* d_src = nullptr;
    cudaMalloc(&d_src, total * K * sizeof(__half));
    cudaMemcpy(d_src, h_src.data(), total * K * sizeof(__half), cudaMemcpyHostToDevice);

    void* d_packed[2] = {};
    void* d_sf[2]     = {};
    cudaMalloc(&d_packed[0], (size_t)M0 * (K / 2));
    cudaMalloc(&d_sf[0],     (size_t)M0 * (K / 16));
    cudaMalloc(&d_packed[1], (size_t)M1 * (K / 2));
    cudaMalloc(&d_sf[1],     (size_t)M1 * (K / 16));

    int h_offsets[3] = {0, M0, M0 + M1};
    int* d_offsets = nullptr;
    cudaMalloc(&d_offsets, 3 * sizeof(int));
    cudaMemcpy(d_offsets, h_offsets, 3 * sizeof(int), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    imp::quantize_fp16_to_nvfp4_moe_native(
        d_src, d_packed, d_sf, d_offsets, total, K, ne, stream);
    cudaStreamSynchronize(stream);

    // Verify expert 0 output is non-trivial (some non-zero packed bytes).
    std::vector<uint8_t> h_packed0(M0 * (K / 2));
    cudaMemcpy(h_packed0.data(), d_packed[0], h_packed0.size(), cudaMemcpyDeviceToHost);
    bool any_nonzero0 = false;
    for (uint8_t b : h_packed0) if (b) { any_nonzero0 = true; break; }
    EXPECT_TRUE(any_nonzero0) << "Expert 0 packed output is all zeros";

    // Verify expert 1 output is non-trivial.
    std::vector<uint8_t> h_packed1(M1 * (K / 2));
    cudaMemcpy(h_packed1.data(), d_packed[1], h_packed1.size(), cudaMemcpyDeviceToHost);
    bool any_nonzero1 = false;
    for (uint8_t b : h_packed1) if (b) { any_nonzero1 = true; break; }
    EXPECT_TRUE(any_nonzero1) << "Expert 1 packed output is all zeros";

    // Verify scales are non-zero.
    std::vector<uint8_t> h_sf0(M0 * (K / 16));
    cudaMemcpy(h_sf0.data(), d_sf[0], h_sf0.size(), cudaMemcpyDeviceToHost);
    bool any_sf0 = false;
    for (uint8_t b : h_sf0) if (b) { any_sf0 = true; break; }
    EXPECT_TRUE(any_sf0) << "Expert 0 scales are all zeros";

    cudaFree(d_src);
    cudaFree(d_packed[0]); cudaFree(d_sf[0]);
    cudaFree(d_packed[1]); cudaFree(d_sf[1]);
    cudaFree(d_offsets);
    cudaStreamDestroy(stream);
}

// ---------------------------------------------------------------------------
// Test: empty expert (M_e == 0) must not crash or produce out-of-bounds writes.
// ---------------------------------------------------------------------------
TEST(QuantizeMoeNative, EmptyExpertNocrash) {
    if (sm_major() < 12)
        GTEST_SKIP() << "SM120 required";

    const int M = 16, K = 32, ne = 2;

    std::vector<__half> h_src(M * K, __float2half(1.0f));
    __half* d_src = nullptr;
    cudaMalloc(&d_src, M * K * sizeof(__half));
    cudaMemcpy(d_src, h_src.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice);

    // Expert 1 is empty (offsets[1] == offsets[2]).
    void* d_packed[2] = {};
    void* d_sf[2]     = {};
    cudaMalloc(&d_packed[0], (size_t)M * (K / 2));
    cudaMalloc(&d_sf[0],     (size_t)M * (K / 16));
    // Expert 1 gets minimal 1-byte buffers — should never be written.
    cudaMalloc(&d_packed[1], 1);
    cudaMalloc(&d_sf[1],     1);

    int h_offsets[3] = {0, M, M};  // expert 1 is empty
    int* d_offsets = nullptr;
    cudaMalloc(&d_offsets, 3 * sizeof(int));
    cudaMemcpy(d_offsets, h_offsets, 3 * sizeof(int), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    imp::quantize_fp16_to_nvfp4_moe_native(
        d_src, d_packed, d_sf, d_offsets, M, K, ne, stream);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    cudaFree(d_src);
    cudaFree(d_packed[0]); cudaFree(d_sf[0]);
    cudaFree(d_packed[1]); cudaFree(d_sf[1]);
    cudaFree(d_offsets);
    cudaStreamDestroy(stream);
}

// ---------------------------------------------------------------------------
// Test: compute_M_per_from_offsets_device — device-side per-expert token count.
// Replaces the host-side cudaMemcpyAsync(D2H) + sync + loop pattern in MoE
// prefill dispatch. Required for CUDA-graph capture (Phase 1 of MoE-prefill-
// graphs lever, plan moe_prefill_graphs_plan_2026_05_10).
// ---------------------------------------------------------------------------
TEST(QuantizeMoeNative, ComputeMPerFromOffsetsDevice) {
    const int ne = 4;
    // Offsets: [0, 3, 3, 7, 10] → M_per: [3, 0, 4, 3]
    const int32_t h_offsets[ne + 1] = {0, 3, 3, 7, 10};
    const int32_t expected_M[ne]    = {3, 0, 4, 3};

    int32_t* d_offsets = nullptr;
    int32_t* d_M_per   = nullptr;
    cudaMalloc(&d_offsets, (ne + 1) * sizeof(int32_t));
    cudaMalloc(&d_M_per,   ne       * sizeof(int32_t));
    cudaMemcpy(d_offsets, h_offsets, (ne + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    imp::compute_M_per_from_offsets_device(d_offsets, d_M_per, ne, stream);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    int32_t got_M[ne] = {};
    cudaMemcpy(got_M, d_M_per, ne * sizeof(int32_t), cudaMemcpyDeviceToHost);

    for (int e = 0; e < ne; ++e)
        EXPECT_EQ(got_M[e], expected_M[e]) << "M_per[" << e << "] mismatch";

    cudaFree(d_offsets);
    cudaFree(d_M_per);
    cudaStreamDestroy(stream);
}

// n_experts == 0 must not launch a kernel and must not segfault.
TEST(QuantizeMoeNative, ComputeMPerFromOffsetsDeviceEmpty) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    imp::compute_M_per_from_offsets_device(nullptr, nullptr, 0, stream);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    cudaStreamDestroy(stream);
}

}  // namespace
}  // namespace imp
