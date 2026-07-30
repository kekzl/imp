// Correctness test for CUTLASS 3.x NVFP4 BlockScaled Grouped GEMM (SM120).
// Compares the grouped dispatch against per-expert single-GEMM (sm120 2.x/3.x
// non-grouped) on synthetic NVFP4 expert weights + FP16 activations.

#include <gtest/gtest.h>
#include "compute/gemm_cutlass_grouped_3x.h"
#include "scoped_engine_arena.h"
#include "compute/gemm_cutlass_sm120.h"
#include "core/tensor.h"
#include "quant/nvfp4_quant.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <memory>
#include <vector>
#include <cmath>
#include <random>

namespace imp {
namespace {

class CutlassGrouped3xNvfp4Test : public ::testing::Test {
protected:
    void SetUp() override {
        // The grouped path takes its staging + workspace from the T2 arena
        // (A7 step 8), which Engine::init opens in production. No Engine here.
        arena_ = std::make_unique<ScopedEngineArena>();
        cudaStreamCreate(&stream_);
        int dev = 0;
        cudaGetDevice(&dev);
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        sm_ = major * 10 + minor;
    }
    void TearDown() override {
        cudaStreamDestroy(stream_);
        gemm_grouped_3x_nvfp4_cleanup();
        arena_.reset();
    }
    std::unique_ptr<ScopedEngineArena> arena_;
    cudaStream_t stream_ = nullptr;
    int sm_ = 0;
};

// Build synthetic per-expert NVFP4 weight via NvFP4QuantResult → CUTLASS conversion.
// Returns owning CutlassNvFP4Weight + backing NVFP4 result (for lifetime).
struct SyntheticExpert {
    std::vector<half> weight_fp16;  // [N, K] reference weights
    NvFP4QuantResult nvfp4{};
    CutlassNvFP4Weight cutlass_w{};
};

static void make_expert(SyntheticExpert& e, int N, int K, float wscale, uint64_t seed, cudaStream_t stream) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-wscale, wscale);
    e.weight_fp16.resize(static_cast<size_t>(N) * K);
    for (auto& v : e.weight_fp16)
        v = __float2half(dist(gen));

    // Upload FP16 weight to device, NVFP4-quantize it.
    void* d_w_fp16 = nullptr;
    cudaMalloc(&d_w_fp16, e.weight_fp16.size() * sizeof(half));
    cudaMemcpy(d_w_fp16, e.weight_fp16.data(), e.weight_fp16.size() * sizeof(half), cudaMemcpyHostToDevice);
    int64_t w_shape[2] = {N, K};
    Tensor w_input(d_w_fp16, QType::F16, 2, w_shape, true);
    quantize_fp16_to_nvfp4(w_input, e.nvfp4, stream);
    cudaFree(d_w_fp16);

    convert_nvfp4_to_cutlass(e.nvfp4, e.cutlass_w, stream);
    cudaStreamSynchronize(stream);
}

static void free_expert(SyntheticExpert& e) {
    free_cutlass_nvfp4_weight(e.cutlass_w);
    free_nvfp4_result(e.nvfp4);
}

TEST_F(CutlassGrouped3xNvfp4Test, GroupedMatchesPerExpertSingle) {
    if (sm_ < 120) {
        GTEST_SKIP() << "SM120 required";
    }
    if (!cutlass_sm120_nvfp4_available()) {
        GTEST_SKIP() << "CUTLASS NVFP4 disabled";
    }
    if (!cutlass_grouped_3x_nvfp4_available()) {
        GTEST_SKIP() << "CUTLASS 3x grouped NVFP4 disabled";
    }

    const int ne = 4;
    const int N = 256;
    const int K = 256;
    const std::vector<int> M_per = {32, 16, 48, 64};
    int M_total = 0;
    for (int m : M_per)
        M_total += m;

    // ----- 4 synthetic experts -----
    std::vector<SyntheticExpert> experts(ne);
    for (int i = 0; i < ne; ++i) {
        make_expert(experts[i], N, K, /*wscale=*/0.5f, /*seed=*/100 + i, stream_);
    }

    // ----- FP16 activations [M_total, K] -----
    std::mt19937 agen(42);
    std::uniform_real_distribution<float> adist(-1.0f, 1.0f);
    std::vector<half> h_A(static_cast<size_t>(M_total) * K);
    for (auto& v : h_A)
        v = __float2half(adist(agen));
    void* d_A_fp16 = nullptr;
    cudaMalloc(&d_A_fp16, h_A.size() * sizeof(half));
    cudaMemcpy(d_A_fp16, h_A.data(), h_A.size() * sizeof(half), cudaMemcpyHostToDevice);

    // ----- Per-expert NVFP4 quantization of A (into per-expert buffers) -----
    std::vector<void*> dA_packed(ne, nullptr);
    std::vector<void*> dA_sf(ne, nullptr);
    std::vector<size_t> sfa_sizes(ne);
    int row_offset = 0;
    for (int i = 0; i < ne; ++i) {
        int M_i = M_per[i];
        size_t packed_bytes = static_cast<size_t>(M_i) * K / 2;
        sfa_sizes[i] = cutlass_nvfp4_sf_size(M_i, K);
        cudaMalloc(&dA_packed[i], packed_bytes);
        cudaMalloc(&dA_sf[i], sfa_sizes[i]);
        const half* a_src = reinterpret_cast<const half*>(d_A_fp16) + row_offset * K;
        quantize_fp16_to_nvfp4_cutlass(a_src, dA_packed[i], dA_sf[i], M_i, K, stream_);
        row_offset += M_i;
    }
    cudaStreamSynchronize(stream_);

    // ----- Reference: per-expert single GEMM via gemm_nvfp4_cutlass_sm120 -----
    std::vector<half> ref_out(static_cast<size_t>(M_total) * N, __float2half(0.f));
    void* d_ref_out = nullptr;
    cudaMalloc(&d_ref_out, ref_out.size() * sizeof(half));
    size_t ref_row_off = 0;
    for (int i = 0; i < ne; ++i) {
        int M_i = M_per[i];
        half* d_out_i = reinterpret_cast<half*>(d_ref_out) + ref_row_off * N;
        bool ok = gemm_nvfp4_cutlass_sm120(dA_packed[i], dA_sf[i], experts[i].cutlass_w, d_out_i, M_i, N, K,
                                           nullptr, 0, stream_);
        ASSERT_TRUE(ok) << "per-expert reference GEMM failed on expert " << i;
        ref_row_off += M_i;
    }
    cudaStreamSynchronize(stream_);
    cudaMemcpy(ref_out.data(), d_ref_out, ref_out.size() * sizeof(half), cudaMemcpyDeviceToHost);

    // ----- Grouped dispatch -----
    void* d_grp_out = nullptr;
    cudaMalloc(&d_grp_out, ref_out.size() * sizeof(half));
    cudaMemset(d_grp_out, 0, ref_out.size() * sizeof(half));

    std::vector<const void*> hA(ne), hSFA(ne), hB(ne), hSFB(ne);
    std::vector<void*> hD(ne);
    std::vector<float> hAlpha(ne);
    size_t grp_row_off = 0;
    for (int i = 0; i < ne; ++i) {
        hA[i] = dA_packed[i];
        hSFA[i] = dA_sf[i];
        hB[i] = experts[i].cutlass_w.data;
        hSFB[i] = experts[i].cutlass_w.scale_factors;
        hD[i] = reinterpret_cast<half*>(d_grp_out) + grp_row_off * N;
        hAlpha[i] = experts[i].cutlass_w.tensor_scale;
        grp_row_off += M_per[i];
    }

    bool ok = gemm_grouped_cutlass_3x_nvfp4(ne, M_per.data(), N, K, hA.data(), hSFA.data(), hB.data(),
                                            hSFB.data(), hD.data(), hAlpha.data(), stream_);
    ASSERT_TRUE(ok) << "grouped dispatch failed";
    cudaStreamSynchronize(stream_);

    std::vector<half> grp_out(ref_out.size());
    cudaMemcpy(grp_out.data(), d_grp_out, grp_out.size() * sizeof(half), cudaMemcpyDeviceToHost);

    // ----- Compare: grouped output must equal per-expert reference (exact, same kernel logic) -----
    int mismatches = 0;
    double max_abs_err = 0.0;
    double sum_abs_err = 0.0;
    for (size_t i = 0; i < ref_out.size(); ++i) {
        float a = __half2float(ref_out[i]);
        float b = __half2float(grp_out[i]);
        float err = std::fabs(a - b);
        if (err > 1e-2f * (1.0f + std::fabs(a)))
            mismatches++;
        max_abs_err = std::max<double>(max_abs_err, err);
        sum_abs_err += err;
    }
    double mean_abs_err = sum_abs_err / static_cast<double>(ref_out.size());
    EXPECT_LT(mismatches, static_cast<int>(ref_out.size()) / 100)
        << "too many mismatches: " << mismatches << " / " << ref_out.size() << " max_err=" << max_abs_err
        << " mean_err=" << mean_abs_err;

    // ----- Cleanup -----
    for (int i = 0; i < ne; ++i) {
        cudaFree(dA_packed[i]);
        cudaFree(dA_sf[i]);
        free_expert(experts[i]);
    }
    cudaFree(d_A_fp16);
    cudaFree(d_ref_out);
    cudaFree(d_grp_out);
}

// ---------------------------------------------------------------------------
// Phase 3b: device-args wrapper must produce bit-identical output to the
// host-args wrapper on the same per-expert problem (both call the same
// underlying CUTLASS adapter; only the staging buffer build differs).
// ---------------------------------------------------------------------------
TEST_F(CutlassGrouped3xNvfp4Test, DeviceArgsMatchesHostArgs) {
    if (sm_ < 120) {
        GTEST_SKIP() << "SM120 required";
    }
    if (!cutlass_sm120_nvfp4_available()) {
        GTEST_SKIP() << "CUTLASS NVFP4 disabled";
    }
    if (!cutlass_grouped_3x_nvfp4_available()) {
        GTEST_SKIP() << "CUTLASS 3x grouped NVFP4 disabled";
    }

    const int ne = 4;
    const int N  = 256;
    const int K  = 256;
    const std::vector<int> M_per = {32, 16, 48, 64};
    int M_total = 0;
    for (int m : M_per) M_total += m;

    // ----- 4 synthetic experts with CONTIGUOUS B/SFB (so b_expert_stride_*
    //       are constant — required by the device-args base+stride layout). -----
    std::vector<SyntheticExpert> experts(ne);
    for (int i = 0; i < ne; ++i) {
        make_expert(experts[i], N, K, /*wscale=*/0.5f, /*seed=*/200 + i, stream_);
    }
    // Stack per-expert B/SFB into a single contiguous slab.
    const size_t b_packed_per_expert = static_cast<size_t>(N) * K / 2;
    const size_t sfb_per_expert      = cutlass_nvfp4_sf_size(N, K);
    void* d_B_packed_slab = nullptr;
    void* d_B_sf_slab     = nullptr;
    cudaMalloc(&d_B_packed_slab, b_packed_per_expert * ne);
    cudaMalloc(&d_B_sf_slab,     sfb_per_expert * ne);
    for (int i = 0; i < ne; ++i) {
        cudaMemcpyAsync(static_cast<char*>(d_B_packed_slab) + i * b_packed_per_expert,
                        experts[i].cutlass_w.data, b_packed_per_expert,
                        cudaMemcpyDeviceToDevice, stream_);
        cudaMemcpyAsync(static_cast<char*>(d_B_sf_slab) + i * sfb_per_expert,
                        experts[i].cutlass_w.scale_factors, sfb_per_expert,
                        cudaMemcpyDeviceToDevice, stream_);
    }
    cudaStreamSynchronize(stream_);

    // ----- FP16 activations [M_total, K], quantize to a CONTIGUOUS A slab -----
    std::mt19937 agen(43);
    std::uniform_real_distribution<float> adist(-1.0f, 1.0f);
    std::vector<half> h_A(static_cast<size_t>(M_total) * K);
    for (auto& v : h_A) v = __float2half(adist(agen));
    void* d_A_fp16 = nullptr;
    cudaMalloc(&d_A_fp16, h_A.size() * sizeof(half));
    cudaMemcpy(d_A_fp16, h_A.data(), h_A.size() * sizeof(half), cudaMemcpyHostToDevice);

    // Contiguous packed A slab: total_packed = sum(M_per * K / 2).
    // Contiguous SFA slab:      total_sfa    = sum(cutlass_nvfp4_sf_size(M_per, K)).
    size_t total_packed = 0;
    std::vector<int>     h_offsets(ne + 1, 0);
    std::vector<int64_t> h_sfa_offsets(ne + 1, 0);
    for (int e = 0; e < ne; ++e) {
        total_packed     += static_cast<size_t>(M_per[e]) * K / 2;
        h_offsets[e + 1]      = h_offsets[e] + M_per[e];
        h_sfa_offsets[e + 1]  = h_sfa_offsets[e] +
                                static_cast<int64_t>(cutlass_nvfp4_sf_size(M_per[e], K));
    }
    void* d_A_packed_slab = nullptr;
    void* d_A_sf_slab     = nullptr;
    cudaMalloc(&d_A_packed_slab, total_packed);
    cudaMalloc(&d_A_sf_slab,     h_sfa_offsets[ne]);
    // Quantize each expert's chunk into the contiguous slab.
    int row_offset = 0;
    for (int i = 0; i < ne; ++i) {
        const half* a_src =
            reinterpret_cast<const half*>(d_A_fp16) + static_cast<size_t>(row_offset) * K;
        void* dst_packed =
            static_cast<char*>(d_A_packed_slab) + static_cast<size_t>(row_offset) * K / 2;
        void* dst_sf =
            static_cast<char*>(d_A_sf_slab) + h_sfa_offsets[i];
        quantize_fp16_to_nvfp4_cutlass(a_src, dst_packed, dst_sf, M_per[i], K, stream_);
        row_offset += M_per[i];
    }
    cudaStreamSynchronize(stream_);

    // ----- Reference: host-args wrapper -----
    void* d_ref_out = nullptr;
    cudaMalloc(&d_ref_out, static_cast<size_t>(M_total) * N * sizeof(half));
    cudaMemset(d_ref_out, 0, static_cast<size_t>(M_total) * N * sizeof(half));

    std::vector<const void*> hA(ne), hSFA(ne), hB(ne), hSFB(ne);
    std::vector<void*> hD(ne);
    std::vector<float> hAlpha(ne);
    int dst_row = 0;
    for (int e = 0; e < ne; ++e) {
        hA[e]   = static_cast<const char*>(d_A_packed_slab) +
                  static_cast<size_t>(h_offsets[e]) * K / 2;
        hSFA[e] = static_cast<const char*>(d_A_sf_slab) + h_sfa_offsets[e];
        hB[e]   = static_cast<const char*>(d_B_packed_slab) + e * b_packed_per_expert;
        hSFB[e] = static_cast<const char*>(d_B_sf_slab)     + e * sfb_per_expert;
        hD[e]   = static_cast<char*>(d_ref_out) +
                  static_cast<size_t>(dst_row) * N * sizeof(half);
        hAlpha[e] = experts[e].cutlass_w.tensor_scale;
        dst_row += M_per[e];
    }
    ASSERT_TRUE(gemm_grouped_cutlass_3x_nvfp4(ne, M_per.data(), N, K, hA.data(), hSFA.data(),
                                              hB.data(), hSFB.data(), hD.data(), hAlpha.data(),
                                              stream_))
        << "host-args wrapper failed";
    cudaStreamSynchronize(stream_);
    std::vector<half> ref_out(static_cast<size_t>(M_total) * N);
    cudaMemcpy(ref_out.data(), d_ref_out, ref_out.size() * sizeof(half), cudaMemcpyDeviceToHost);

    // ----- Device-args wrapper: same inputs, device-resident metadata -----
    void* d_dev_out = nullptr;
    cudaMalloc(&d_dev_out, static_cast<size_t>(M_total) * N * sizeof(half));
    cudaMemset(d_dev_out, 0, static_cast<size_t>(M_total) * N * sizeof(half));

    int32_t* d_M_per   = nullptr;
    int32_t* d_offsets = nullptr;
    int64_t* d_sfa_offsets = nullptr;
    float*   d_alpha   = nullptr;
    cudaMalloc(&d_M_per,       ne       * sizeof(int32_t));
    cudaMalloc(&d_offsets,     (ne + 1) * sizeof(int32_t));
    cudaMalloc(&d_sfa_offsets, (ne + 1) * sizeof(int64_t));
    cudaMalloc(&d_alpha,       ne       * sizeof(float));
    cudaMemcpy(d_M_per,       M_per.data(),         ne * sizeof(int32_t),       cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets,     h_offsets.data(),     (ne + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sfa_offsets, h_sfa_offsets.data(), (ne + 1) * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha,       hAlpha.data(),        ne * sizeof(float),         cudaMemcpyHostToDevice);

    // Mode (a): contiguous B/SFB slab + per-expert byte stride.
    GroupedNvfp4DeviceArgs dargs{};
    dargs.d_M_per                = d_M_per;
    dargs.d_expert_offsets       = d_offsets;
    dargs.d_sfa_offsets          = d_sfa_offsets;
    dargs.d_alpha                = d_alpha;
    dargs.base_A_packed          = d_A_packed_slab;
    dargs.base_A_sf              = d_A_sf_slab;
    dargs.base_B_packed          = d_B_packed_slab;
    dargs.b_expert_stride_packed = static_cast<int64_t>(b_packed_per_expert);
    dargs.base_B_sf              = d_B_sf_slab;
    dargs.b_expert_stride_sf     = static_cast<int64_t>(sfb_per_expert);
    dargs.d_B_ptrs               = nullptr;  // mode (a) — base + stride
    dargs.d_SFB_ptrs             = nullptr;
    dargs.base_D                 = d_dev_out;

    ASSERT_TRUE(gemm_grouped_cutlass_3x_nvfp4_device_args(ne, N, K, dargs, stream_))
        << "device-args wrapper failed";
    cudaStreamSynchronize(stream_);
    std::vector<half> dev_out(static_cast<size_t>(M_total) * N);
    cudaMemcpy(dev_out.data(), d_dev_out, dev_out.size() * sizeof(half), cudaMemcpyDeviceToHost);

    // ----- Compare: both wrappers use the same CUTLASS adapter, so bit-exact -----
    int mismatches = 0;
    double max_abs_err = 0.0;
    for (size_t i = 0; i < ref_out.size(); ++i) {
        float a = __half2float(ref_out[i]);
        float b = __half2float(dev_out[i]);
        float err = std::fabs(a - b);
        if (err > 0.0f) mismatches++;
        max_abs_err = std::max<double>(max_abs_err, err);
    }
    EXPECT_EQ(mismatches, 0)
        << "device-args (mode a) output differs from host-args (" << mismatches << " / "
        << ref_out.size() << " mismatches, max_err=" << max_abs_err << ")";

    // ----- Mode (b): per-expert pointer arrays. Builds device-resident ptr
    //       arrays that point into the SAME slab — both modes must yield the
    //       same output. -----
    void* d_dev_out_b = nullptr;
    cudaMalloc(&d_dev_out_b, static_cast<size_t>(M_total) * N * sizeof(half));
    cudaMemset(d_dev_out_b, 0, static_cast<size_t>(M_total) * N * sizeof(half));

    std::vector<const void*> h_B_ptrs(ne), h_SFB_ptrs(ne);
    for (int e = 0; e < ne; ++e) {
        h_B_ptrs[e]   = static_cast<const char*>(d_B_packed_slab) + e * b_packed_per_expert;
        h_SFB_ptrs[e] = static_cast<const char*>(d_B_sf_slab)     + e * sfb_per_expert;
    }
    const void** d_B_ptrs   = nullptr;
    const void** d_SFB_ptrs = nullptr;
    cudaMalloc(&d_B_ptrs,   ne * sizeof(const void*));
    cudaMalloc(&d_SFB_ptrs, ne * sizeof(const void*));
    cudaMemcpy(d_B_ptrs,   h_B_ptrs.data(),   ne * sizeof(const void*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_SFB_ptrs, h_SFB_ptrs.data(), ne * sizeof(const void*), cudaMemcpyHostToDevice);

    GroupedNvfp4DeviceArgs dargs_b = dargs;
    dargs_b.base_B_packed          = nullptr;  // ignored when d_B_ptrs set
    dargs_b.b_expert_stride_packed = 0;
    dargs_b.base_B_sf              = nullptr;
    dargs_b.b_expert_stride_sf     = 0;
    dargs_b.d_B_ptrs               = d_B_ptrs;
    dargs_b.d_SFB_ptrs             = d_SFB_ptrs;
    dargs_b.base_D                 = d_dev_out_b;

    ASSERT_TRUE(gemm_grouped_cutlass_3x_nvfp4_device_args(ne, N, K, dargs_b, stream_))
        << "device-args wrapper (mode b) failed";
    cudaStreamSynchronize(stream_);
    std::vector<half> dev_out_b(static_cast<size_t>(M_total) * N);
    cudaMemcpy(dev_out_b.data(), d_dev_out_b, dev_out_b.size() * sizeof(half), cudaMemcpyDeviceToHost);

    int mismatches_b = 0;
    double max_abs_err_b = 0.0;
    for (size_t i = 0; i < ref_out.size(); ++i) {
        float a = __half2float(ref_out[i]);
        float b = __half2float(dev_out_b[i]);
        float err = std::fabs(a - b);
        if (err > 0.0f) mismatches_b++;
        max_abs_err_b = std::max<double>(max_abs_err_b, err);
    }
    EXPECT_EQ(mismatches_b, 0)
        << "device-args (mode b) output differs from host-args (" << mismatches_b << " / "
        << ref_out.size() << " mismatches, max_err=" << max_abs_err_b << ")";

    // ----- Cleanup -----
    for (int i = 0; i < ne; ++i) free_expert(experts[i]);
    cudaFree(d_A_fp16);
    cudaFree(d_A_packed_slab);
    cudaFree(d_A_sf_slab);
    cudaFree(d_B_packed_slab);
    cudaFree(d_B_sf_slab);
    cudaFree(d_ref_out);
    cudaFree(d_dev_out);
    cudaFree(d_dev_out_b);
    cudaFree(d_M_per);
    cudaFree(d_offsets);
    cudaFree(d_sfa_offsets);
    cudaFree(d_alpha);
    cudaFree(d_B_ptrs);
    cudaFree(d_SFB_ptrs);
}

}  // namespace
}  // namespace imp
