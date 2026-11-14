#include <gtest/gtest.h>
#include "compute/quantize_fp16_nvfp4_moe_native.h"
#include "compute/gemm_cutlass_sm120.h"  // cutlass_nvfp4_sf_size (host reference)
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

namespace imp {
namespace {

// compute_M_per_from_offsets_device: device-side per-expert token count, replacing the
// host D2H memcpy+sync+loop pattern. Required for CUDA-graph capture (MoE-prefill-graphs
// Phase 1, plan moe_prefill_graphs_plan_2026_05_10).
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

// compact_alpha_active: order-preserving stream compaction of d_alpha to active-only experts
// (MoE-prefill-graphs Phase 2, plan moe_prefill_graphs_plan_2026_05_10).
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

// compute_sfa_offsets_device: exclusive prefix sum of SfAtom-padded SFA byte sizes must match
// the host cutlass_nvfp4_sf_size formula (MoE-prefill-graphs Phase 3a).
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

// 512 experts (Qwen3.8-Flash-Next): the one-expert-per-thread scans capped at 256 and
// refused the layer. Both single-block scans now chunk experts per thread.
TEST(QuantizeMoeNative, CompactAlphaAndSfaOffsetsWide512) {
    const int ne = 512;
    const int K = 2560;
    std::vector<float> h_alpha(ne);
    std::vector<int32_t> h_M(ne);
    for (int e = 0; e < ne; ++e) {
        h_alpha[e] = 1.0f + static_cast<float>(e);
        h_M[e] = (e % 3 == 0) ? 0 : (e * 7) % 300;  // a third inactive, M up to 299
    }
    float* d_alpha = nullptr;
    int32_t* d_M = nullptr;
    float* d_compact = nullptr;
    int32_t* d_na = nullptr;
    int64_t* d_offsets = nullptr;
    cudaMalloc(&d_alpha, ne * sizeof(float));
    cudaMalloc(&d_M, ne * sizeof(int32_t));
    cudaMalloc(&d_compact, ne * sizeof(float));
    cudaMalloc(&d_na, sizeof(int32_t));
    cudaMalloc(&d_offsets, (ne + 1) * sizeof(int64_t));
    cudaMemcpy(d_alpha, h_alpha.data(), ne * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M.data(), ne * sizeof(int32_t), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    imp::compact_alpha_active(d_alpha, d_M, d_compact, d_na, ne, stream);
    imp::compute_sfa_offsets_device(d_M, d_offsets, ne, K, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<float> exp_compact;
    std::vector<int64_t> exp_off(ne + 1, 0);
    for (int e = 0; e < ne; ++e) {
        if (h_M[e] > 0)
            exp_compact.push_back(h_alpha[e]);
        exp_off[e + 1] = exp_off[e] + static_cast<int64_t>(imp::cutlass_nvfp4_sf_size(h_M[e], K));
    }
    int32_t got_na = -1;
    cudaMemcpy(&got_na, d_na, sizeof(int32_t), cudaMemcpyDeviceToHost);
    ASSERT_EQ(got_na, static_cast<int32_t>(exp_compact.size()));
    std::vector<float> got_compact(exp_compact.size());
    cudaMemcpy(got_compact.data(), d_compact, got_compact.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < exp_compact.size(); ++i)
        EXPECT_FLOAT_EQ(got_compact[i], exp_compact[i]) << "compact[" << i << "]";
    std::vector<int64_t> got_off(ne + 1);
    cudaMemcpy(got_off.data(), d_offsets, (ne + 1) * sizeof(int64_t), cudaMemcpyDeviceToHost);
    for (int e = 0; e <= ne; ++e)
        EXPECT_EQ(got_off[e], exp_off[e]) << "sfa offset[" << e << "]";

    cudaFree(d_alpha);
    cudaFree(d_M);
    cudaFree(d_compact);
    cudaFree(d_na);
    cudaFree(d_offsets);
    cudaStreamDestroy(stream);
}

// compute_sfa_offsets_device with d_sfa_bases_out: bases[e] == base + offsets[e] from the same launch.
TEST(QuantizeMoeNative, ComputeSfaOffsetsDeviceWritesBases) {
    const int ne = 6;
    const int K = 256;
    const int32_t h_M[ne] = {0, 1, 127, 128, 129, 256};
    int32_t* d_M = nullptr;
    int64_t* d_offsets = nullptr;
    uint8_t** d_bases = nullptr;
    void* d_base = nullptr;
    cudaMalloc(&d_M, ne * sizeof(int32_t));
    cudaMalloc(&d_offsets, (ne + 1) * sizeof(int64_t));
    cudaMalloc(&d_bases, ne * sizeof(uint8_t*));
    cudaMalloc(&d_base, 65536);  // only addresses matter
    cudaMemset(d_bases, 0, ne * sizeof(uint8_t*));
    cudaMemcpy(d_M, h_M, ne * sizeof(int32_t), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    imp::compute_sfa_offsets_device(d_M, d_offsets, ne, K, stream, d_bases, d_base);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    int64_t off[ne + 1] = {};
    uint8_t* got[ne] = {};
    cudaMemcpy(off, d_offsets, (ne + 1) * sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(got, d_bases, ne * sizeof(uint8_t*), cudaMemcpyDeviceToHost);
    int64_t expected = 0;
    for (int e = 0; e < ne; ++e) {
        EXPECT_EQ(off[e], expected) << "offset[" << e << "]";
        EXPECT_EQ(got[e], static_cast<uint8_t*>(d_base) + expected) << "base[" << e << "]";
        expected += static_cast<int64_t>(imp::cutlass_nvfp4_sf_size(h_M[e], K));
    }

    cudaFree(d_M);
    cudaFree(d_offsets);
    cudaFree(d_bases);
    cudaFree(d_base);
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
