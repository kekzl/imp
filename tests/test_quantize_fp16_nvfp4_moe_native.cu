#include <gtest/gtest.h>
#include "compute/quantize_fp16_nvfp4_moe_native.h"
#include "compute/gemm_cutlass_sm120.h"  // cutlass_nvfp4_sf_size (host reference)
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

// ---------------------------------------------------------------------------
// Test: compact_alpha_active — order-preserving stream compaction of d_alpha
// to active-only experts. Phase 2 of moe_prefill_graphs_plan_2026_05_10.
// ---------------------------------------------------------------------------
TEST(QuantizeMoeNative, CompactAlphaActiveBasic) {
    const int ne = 4;
    const float    h_alpha[ne]   = {1.0f, 2.0f, 3.0f, 4.0f};
    const int32_t  h_M_per[ne]   = {5, 0, 3, 0};
    // Expected: na=2, alpha_compact=[1.0, 3.0] (active experts in source order).
    const float    expected_compact[2] = {1.0f, 3.0f};
    const int32_t  expected_na = 2;

    float*   d_alpha   = nullptr;
    int32_t* d_M_per   = nullptr;
    float*   d_compact = nullptr;
    int32_t* d_na      = nullptr;
    cudaMalloc(&d_alpha,   ne * sizeof(float));
    cudaMalloc(&d_M_per,   ne * sizeof(int32_t));
    cudaMalloc(&d_compact, ne * sizeof(float));
    cudaMalloc(&d_na,           sizeof(int32_t));
    cudaMemcpy(d_alpha, h_alpha, ne * sizeof(float),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_M_per, h_M_per, ne * sizeof(int32_t), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    imp::compact_alpha_active(d_alpha, d_M_per, d_compact, d_na, ne, stream);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    int32_t got_na = -1;
    cudaMemcpy(&got_na, d_na, sizeof(int32_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(got_na, expected_na);

    float got_compact[2] = {};
    cudaMemcpy(got_compact, d_compact, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_FLOAT_EQ(got_compact[0], expected_compact[0]);
    EXPECT_FLOAT_EQ(got_compact[1], expected_compact[1]);

    cudaFree(d_alpha); cudaFree(d_M_per);
    cudaFree(d_compact); cudaFree(d_na);
    cudaStreamDestroy(stream);
}

// All experts active: alpha_compact == alpha, na == n_experts.
TEST(QuantizeMoeNative, CompactAlphaActiveAllActive) {
    const int ne = 3;
    const float   h_alpha[ne] = {0.5f, 1.5f, 2.5f};
    const int32_t h_M[ne]     = {2, 4, 1};

    float*   d_alpha   = nullptr;
    int32_t* d_M       = nullptr;
    float*   d_compact = nullptr;
    int32_t* d_na      = nullptr;
    cudaMalloc(&d_alpha,   ne * sizeof(float));
    cudaMalloc(&d_M,       ne * sizeof(int32_t));
    cudaMalloc(&d_compact, ne * sizeof(float));
    cudaMalloc(&d_na,           sizeof(int32_t));
    cudaMemcpy(d_alpha, h_alpha, ne * sizeof(float),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_M,     h_M,     ne * sizeof(int32_t), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    imp::compact_alpha_active(d_alpha, d_M, d_compact, d_na, ne, stream);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    int32_t got_na = -1;
    cudaMemcpy(&got_na, d_na, sizeof(int32_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(got_na, ne);

    float got[ne] = {};
    cudaMemcpy(got, d_compact, ne * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < ne; ++i)
        EXPECT_FLOAT_EQ(got[i], h_alpha[i]);

    cudaFree(d_alpha); cudaFree(d_M); cudaFree(d_compact); cudaFree(d_na);
    cudaStreamDestroy(stream);
}

// No active experts: na=0; compact buffer untouched.
TEST(QuantizeMoeNative, CompactAlphaActiveNoneActive) {
    const int ne = 4;
    const float   h_alpha[ne] = {1.0f, 2.0f, 3.0f, 4.0f};
    const int32_t h_M[ne]     = {0, 0, 0, 0};

    float*   d_alpha   = nullptr;
    int32_t* d_M       = nullptr;
    float*   d_compact = nullptr;
    int32_t* d_na      = nullptr;
    cudaMalloc(&d_alpha,   ne * sizeof(float));
    cudaMalloc(&d_M,       ne * sizeof(int32_t));
    cudaMalloc(&d_compact, ne * sizeof(float));
    cudaMalloc(&d_na,           sizeof(int32_t));
    cudaMemcpy(d_alpha, h_alpha, ne * sizeof(float),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_M,     h_M,     ne * sizeof(int32_t), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    imp::compact_alpha_active(d_alpha, d_M, d_compact, d_na, ne, stream);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    int32_t got_na = -1;
    cudaMemcpy(&got_na, d_na, sizeof(int32_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(got_na, 0);

    cudaFree(d_alpha); cudaFree(d_M); cudaFree(d_compact); cudaFree(d_na);
    cudaStreamDestroy(stream);
}

// ---------------------------------------------------------------------------
// Test: compute_sfa_offsets_device — exclusive prefix sum of SfAtom-padded SFA
// byte sizes must match host cutlass_nvfp4_sf_size formula. Phase 3a of
// moe_prefill_graphs_plan_2026_05_10.
// ---------------------------------------------------------------------------
TEST(QuantizeMoeNative, ComputeSfaOffsetsDeviceMatchesHost) {
    // Mixed M values exercise the ceil(M/128) padding edge: 0, exactly 128,
    // just-under-tile (127), small (1), and a couple of multi-tile values.
    const int ne = 6;
    const int K  = 256;  // n_k_tiles = ceil(256/64) = 4
    const int32_t h_M[ne] = {0, 1, 127, 128, 129, 256};

    int32_t* d_M = nullptr;
    int64_t* d_offsets = nullptr;
    cudaMalloc(&d_M,      ne       * sizeof(int32_t));
    cudaMalloc(&d_offsets,(ne + 1) * sizeof(int64_t));
    cudaMemcpy(d_M, h_M, ne * sizeof(int32_t), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    imp::compute_sfa_offsets_device(d_M, d_offsets, ne, K, stream);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    int64_t got[ne + 1] = {};
    cudaMemcpy(got, d_offsets, (ne + 1) * sizeof(int64_t), cudaMemcpyDeviceToHost);

    // Reference: exclusive prefix sum using host cutlass_nvfp4_sf_size.
    int64_t expected[ne + 1] = {};
    for (int e = 0; e < ne; ++e) {
        expected[e + 1] = expected[e] +
                          static_cast<int64_t>(imp::cutlass_nvfp4_sf_size(h_M[e], K));
    }

    for (int e = 0; e <= ne; ++e) {
        EXPECT_EQ(got[e], expected[e])
            << "sfa offset[" << e << "] mismatch (M_per=[0,1,127,128,129,256], K=" << K << ")";
    }

    cudaFree(d_M);
    cudaFree(d_offsets);
    cudaStreamDestroy(stream);
}

// build_sfa_bases_device must write base + d_sfa_offsets[e] per expert.
// Phase 3c-full Step 2a foundation.
TEST(QuantizeMoeNative, BuildSfaBasesDevice) {
    const int ne = 4;
    // Simulate base SFA slab via a known device pointer (any aligned alloc).
    void* d_base = nullptr;
    cudaMalloc(&d_base, 65536);  // 64 KiB sentinel buffer (only addresses matter)
    const int64_t h_offsets[ne + 1] = {0, 512, 1024, 1024, 2560};  // bytes
    int64_t*  d_offsets = nullptr;
    uint8_t** d_bases   = nullptr;
    cudaMalloc(&d_offsets, (ne + 1) * sizeof(int64_t));
    cudaMalloc(&d_bases,   ne       * sizeof(uint8_t*));
    cudaMemcpy(d_offsets, h_offsets, (ne + 1) * sizeof(int64_t), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    imp::build_sfa_bases_device(d_bases, d_base, d_offsets, ne, stream);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    uint8_t* got[ne] = {};
    cudaMemcpy(got, d_bases, ne * sizeof(uint8_t*), cudaMemcpyDeviceToHost);
    for (int e = 0; e < ne; ++e) {
        uint8_t* expected = static_cast<uint8_t*>(d_base) + h_offsets[e];
        EXPECT_EQ(got[e], expected) << "base[" << e << "] mismatch";
    }

    cudaFree(d_base);
    cudaFree(d_offsets);
    cudaFree(d_bases);
    cudaStreamDestroy(stream);
}

// n_experts == 0: trailing slot 0 must be 0, no kernel work.
TEST(QuantizeMoeNative, ComputeSfaOffsetsDeviceEmpty) {
    int64_t* d_offsets = nullptr;
    cudaMalloc(&d_offsets, sizeof(int64_t));
    int64_t init = 999;
    cudaMemcpy(d_offsets, &init, sizeof(int64_t), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    imp::compute_sfa_offsets_device(nullptr, d_offsets, 0, 256, stream);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    int64_t got = -1;
    cudaMemcpy(&got, d_offsets, sizeof(int64_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(got, 0);

    cudaFree(d_offsets);
    cudaStreamDestroy(stream);
}

// n_experts == 0: na written as 0, no kernel launch needed.
TEST(QuantizeMoeNative, CompactAlphaActiveEmpty) {
    int32_t* d_na = nullptr;
    cudaMalloc(&d_na, sizeof(int32_t));
    int32_t init = 999;
    cudaMemcpy(d_na, &init, sizeof(int32_t), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    imp::compact_alpha_active(nullptr, nullptr, nullptr, d_na, 0, stream);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    int32_t got = -1;
    cudaMemcpy(&got, d_na, sizeof(int32_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(got, 0);

    cudaFree(d_na);
    cudaStreamDestroy(stream);
}

}  // namespace
}  // namespace imp
