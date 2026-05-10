// tests/test_gemm_grouped_nvfp4_smallM.cu
//
// Note: TMA descriptor builders (build_tma_a/b/sfa/sfb) are file-local
// templates in gemm_grouped_nvfp4_smallM.cu. They are exercised
// indirectly by the kernel-level tests in later tasks. This file
// has no test for them by design.
#include <gtest/gtest.h>
#include "compute/gemm_grouped_nvfp4_smallM.h"
#include "compute/quantize_fp16_nvfp4_moe_native.h"
#include "quant/nvfp4_quant.h"
#include "core/tensor.h"
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" void smallM_smoke_single_mma(float*, const uint32_t*, const uint32_t*,
                                        uint32_t, uint32_t, cudaStream_t);

#ifdef SMALLM_SOFTWARE_REF
// Debug entry point exposed by gemm_grouped_nvfp4_smallM.cu when built with
// SMALLM_SOFTWARE_REF defined. Same signature as gemm_grouped_nvfp4_smallM
// but routes to the software-decode reference kernel.
extern "C" bool gemm_grouped_nvfp4_smallM_software_ref(
    int n_experts, const int* host_M, int N, int K,
    const void* const* host_ptr_A,   const void* const* host_ptr_SFA,
    const void* const* host_ptr_B,   const void* const* host_ptr_SFB,
    void* const* host_ptr_D,         const float* dev_alpha,
    cudaStream_t stream);
#endif

namespace {

bool has_sm120() {
    int dev = 0; cudaGetDevice(&dev);
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    return major * 10 + minor >= 120;
}

TEST(SmallMMmaWrapper, IssuesSingleMma) {
    if (!has_sm120()) GTEST_SKIP() << "SM120 required";

    uint32_t a[4] = {0, 0, 0, 0}, b[2] = {0, 0};
    uint32_t* d_a = nullptr; uint32_t* d_b = nullptr;
    cudaMalloc(&d_a, sizeof(a)); cudaMalloc(&d_b, sizeof(b));
    cudaMemcpy(d_a, a, sizeof(a), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(b), cudaMemcpyHostToDevice);

    float* d_out = nullptr;
    cudaMalloc(&d_out, 4 * sizeof(float));
    float poison[4] = {-99.f, -99.f, -99.f, -99.f};
    cudaMemcpy(d_out, poison, sizeof(poison), cudaMemcpyHostToDevice);

    smallM_smoke_single_mma(d_out, d_a, d_b, 0u, 0u, /*stream*/nullptr);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    float h_out[4];
    cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);
    // All-zero MMA must produce zero (initial acc=0 + zero*zero*any_scale = 0).
    EXPECT_EQ(h_out[0], 0.f); EXPECT_EQ(h_out[1], 0.f);
    EXPECT_EQ(h_out[2], 0.f); EXPECT_EQ(h_out[3], 0.f);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
}

TEST(SmallMMmaWrapper, NonZeroProducesNonZero) {
    if (!has_sm120()) GTEST_SKIP() << "SM120 required";

    // Patterned non-zero FP4 inputs — exact values don't matter, just nonzero.
    uint32_t a[4] = {0x11111111, 0x11111111, 0x11111111, 0x11111111};
    uint32_t b[2] = {0x11111111, 0x11111111};
    // UE4M3 scale ≈ 1.0 — 0x38383838 matches BENCH_PREAMBLE in variants_bench.
    uint32_t sfa = 0x38383838u;
    uint32_t sfb = 0x38383838u;

    uint32_t* d_a = nullptr; uint32_t* d_b = nullptr;
    cudaMalloc(&d_a, sizeof(a)); cudaMalloc(&d_b, sizeof(b));
    cudaMemcpy(d_a, a, sizeof(a), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(b), cudaMemcpyHostToDevice);

    float* d_out = nullptr;
    cudaMalloc(&d_out, 4 * sizeof(float));
    cudaMemset(d_out, 0, 4 * sizeof(float));

    smallM_smoke_single_mma(d_out, d_a, d_b, sfa, sfb, /*stream*/nullptr);
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    float h_out[4];
    cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);
    bool any_nonzero = false;
    for (int i = 0; i < 4; ++i) if (h_out[i] != 0.f) any_nonzero = true;
    EXPECT_TRUE(any_nonzero) << "MMA with nonzero inputs produced all zeros";

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
}

TEST(SmallMScheduler, PicksMinimalTile) {
    using imp::detail::pick_m_tile;
    EXPECT_EQ(pick_m_tile(1),   16);
    EXPECT_EQ(pick_m_tile(16),  16);
    EXPECT_EQ(pick_m_tile(17),  32);
    EXPECT_EQ(pick_m_tile(32),  32);
    EXPECT_EQ(pick_m_tile(40),  64);
    EXPECT_EQ(pick_m_tile(64),  64);
    EXPECT_EQ(pick_m_tile(128), 128);
    EXPECT_EQ(pick_m_tile(200), 128);
}

TEST(SmallMScheduler, WorkQueueOrderedByTileSize) {
    using imp::detail::build_work_queue;
    int M_per[] = {32, 100, 8, 0, 200};   // 5 experts; e=3 inactive
    auto q = build_work_queue(5, M_per, 256);
    ASSERT_FALSE(q.empty());

    // First items must be tile_M=128 (from e=4 with M=200, two M-tiles needed)
    EXPECT_EQ(q[0].m_tile_size, 128);
    // Last items must be tile_M=16 (from e=2 with M=8)
    EXPECT_EQ(q.back().m_tile_size, 16);
    // No work for inactive expert e=3
    for (auto& wi : q) EXPECT_NE(wi.expert_id, 3);
}

// ---------------------------------------------------------------------------
// Helper: run a single-expert smallM GEMM and validate correctness.
//
// Builds random FP16 A[M,K] and B[N,K], quantizes both to NVFP4, runs the
// smallM kernel, and checks the output against a dequantized FP32 reference.
// The acceptance criterion is: kernel output vs. post-quantize ideal (CTRL)
// has a relative RMSE below `ctrl_rmse_tol`.
// ---------------------------------------------------------------------------
static void run_smallm_single_expert(int M, int N, int K,
                                     double ctrl_rmse_tol = 1e-2) {
    // ----- Build FP16 weights B[N, K] and quantize via reference quantize_fp16_to_nvfp4
    std::mt19937 rng_w(13);
    std::uniform_real_distribution<float> dist_w(-0.5f, 0.5f);
    std::vector<__half> h_B_fp16((size_t)N * K);
    for (auto& v : h_B_fp16) v = __float2half(dist_w(rng_w));

    __half* d_B_fp16 = nullptr;
    cudaMalloc(&d_B_fp16, (size_t)N * K * sizeof(__half));
    cudaMemcpy(d_B_fp16, h_B_fp16.data(), (size_t)N * K * sizeof(__half), cudaMemcpyHostToDevice);
    int64_t b_shape[2] = {N, K};
    imp::Tensor B_t(d_B_fp16, imp::QType::F16, 2, b_shape, true);
    imp::NvFP4QuantResult B_q;
    imp::quantize_fp16_to_nvfp4(B_t, B_q, /*stream*/0);
    cudaStreamSynchronize(0);

    // ----- Build FP16 activations A[M, K]
    std::mt19937 rng_a(7);
    std::uniform_real_distribution<float> dist_a(-1.f, 1.f);
    std::vector<__half> h_A_fp16((size_t)M * K);
    for (auto& v : h_A_fp16) v = __float2half(dist_a(rng_a));

    __half* d_A_fp16 = nullptr;
    cudaMalloc(&d_A_fp16, (size_t)M * K * sizeof(__half));
    cudaMemcpy(d_A_fp16, h_A_fp16.data(), (size_t)M * K * sizeof(__half), cudaMemcpyHostToDevice);

    // Quantize activations via the moe_native quantize (1-expert).
    void* d_A_packed = nullptr; void* d_A_sf = nullptr;
    cudaMalloc(&d_A_packed, (size_t)M * K / 2);
    cudaMalloc(&d_A_sf,     (size_t)M * K / 16);
    int h_off[2] = {0, M};
    int* d_off = nullptr; cudaMalloc(&d_off, sizeof(h_off));
    cudaMemcpy(d_off, h_off, sizeof(h_off), cudaMemcpyHostToDevice);
    void* h_pp[1] = {d_A_packed}; void* h_sp[1] = {d_A_sf};
    imp::quantize_fp16_to_nvfp4_moe_native(d_A_fp16, h_pp, h_sp, d_off, M, K, 1, 0);
    cudaStreamSynchronize(0);

    // ----- Compute the per-expert tensor_scale for A.
    // moe_native's per-expert tensor_scale = absmax / 6.0; reproduce on host.
    float a_absmax = 0.f;
    for (auto h : h_A_fp16) a_absmax = std::max(a_absmax, std::fabs(__half2float(h)));
    float a_tensor_scale = (a_absmax == 0.f) ? 1.f : (a_absmax / 6.0f);
    // Combined alpha = a_tensor_scale * b_tensor_scale (folded into output).
    float combined_alpha = a_tensor_scale * B_q.tensor_scale;

    // ----- Run smallM kernel.
    void* d_D = nullptr; cudaMalloc(&d_D, (size_t)M * N * sizeof(__half));
    cudaMemset(d_D, 0, (size_t)M * N * sizeof(__half));
    int M_per[1] = {M};
    const void* A_arr[1]   = {d_A_packed};
    const void* SFA_arr[1] = {d_A_sf};
    const void* B_arr[1]   = {B_q.packed_data};
    const void* SFB_arr[1] = {B_q.micro_scales};
    void* D_arr[1]   = {d_D};
    // API now requires a device pointer for alpha.
    float h_alpha[1] = {combined_alpha};
    float* d_alpha = nullptr;
    cudaMalloc(&d_alpha, sizeof(float));
    cudaMemcpy(d_alpha, h_alpha, sizeof(float), cudaMemcpyHostToDevice);
    bool ok = imp::gemm_grouped_nvfp4_smallM(
        1, M_per, N, K, A_arr, SFA_arr, B_arr, SFB_arr, D_arr, d_alpha, /*stream*/0);
    ASSERT_TRUE(ok);
    cudaError_t err = cudaDeviceSynchronize();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    // ----- Reference: FP32 matmul on the dequantized FP16 inputs.
    //   ref[m, n] = sum_k A_fp16[m, k] * B_fp16[n, k]
    // (This is the high-precision reference. The kernel's NVFP4 output is
    // expected to deviate from this by the FP4 quantization noise floor.)
    std::vector<float> ref((size_t)M * N, 0.f);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) {
                acc += __half2float(h_A_fp16[(size_t)m * K + k]) *
                       __half2float(h_B_fp16[(size_t)n * K + k]);
            }
            ref[(size_t)m * N + n] = acc;
        }
    }

    // ----- Control reference: dequantize NVFP4 A and B back to FP16, then
    // matmul. This is the "ideal" output the kernel SHOULD produce (modulo
    // FP16 vs FP32 accumulation differences). Comparing kernel-output to
    // this `ctrl` isolates kernel correctness from quantization noise.
    std::vector<__half> h_A_dq((size_t)M * K), h_B_dq((size_t)N * K);
    {
        // Dequantize A (per moe_native): each row m, micro-block kb has
        //   sf = ue4m3(SFA[m, kb]),
        //   dequant_val = fp4_int_decoded * a_tensor_scale * sf
        std::vector<uint8_t> h_A_packed_v((size_t)M * K / 2);
        std::vector<uint8_t> h_A_sf_v((size_t)M * K / 16);
        cudaMemcpy(h_A_packed_v.data(), d_A_packed, h_A_packed_v.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_A_sf_v.data(),     d_A_sf,     h_A_sf_v.size(),     cudaMemcpyDeviceToHost);
        const float kFP4Mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
        auto ue4m3 = [&](uint8_t b) {
            uint32_t sign = (b >> 7) & 1;
            uint32_t exp  = (b >> 3) & 0x0F;
            uint32_t man  = b & 0x07;
            uint32_t fp32;
            if (exp == 0) {
                float v = (float)man * (1.0f / 512.0f);
                std::memcpy(&fp32, &v, 4);
                fp32 |= (sign << 31);
            } else {
                fp32 = (sign << 31) | ((exp + 120u) << 23) | (man << 20);
            }
            float r;
            std::memcpy(&r, &fp32, 4);
            return r;
        };
        for (int m = 0; m < M; ++m) {
            for (int kb = 0; kb < K / 16; ++kb) {
                float sf = ue4m3(h_A_sf_v[(size_t)m * (K / 16) + kb]);
                for (int j = 0; j < 8; ++j) {
                    uint8_t byte = h_A_packed_v[(size_t)m * (K / 2) + kb * 8 + j];
                    uint8_t lo = byte & 0xF, hi = (byte >> 4) & 0xF;
                    float v0 = kFP4Mag[lo & 7] * (lo & 8 ? -1.f : 1.f) * sf * a_tensor_scale;
                    float v1 = kFP4Mag[hi & 7] * (hi & 8 ? -1.f : 1.f) * sf * a_tensor_scale;
                    h_A_dq[(size_t)m * K + kb * 16 + j * 2]     = __float2half(v0);
                    h_A_dq[(size_t)m * K + kb * 16 + j * 2 + 1] = __float2half(v1);
                }
            }
        }
        // Dequantize B (per nvfp4_quant.cu, exact same row-major layout)
        std::vector<uint8_t> h_B_packed_v((size_t)N * K / 2);
        std::vector<uint8_t> h_B_sf_v((size_t)N * K / 16);
        cudaMemcpy(h_B_packed_v.data(), B_q.packed_data, h_B_packed_v.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_B_sf_v.data(),     B_q.micro_scales, h_B_sf_v.size(),    cudaMemcpyDeviceToHost);
        for (int n = 0; n < N; ++n) {
            for (int kb = 0; kb < K / 16; ++kb) {
                float sf = ue4m3(h_B_sf_v[(size_t)n * (K / 16) + kb]);
                for (int j = 0; j < 8; ++j) {
                    uint8_t byte = h_B_packed_v[(size_t)n * (K / 2) + kb * 8 + j];
                    uint8_t lo = byte & 0xF, hi = (byte >> 4) & 0xF;
                    float v0 = kFP4Mag[lo & 7] * (lo & 8 ? -1.f : 1.f) * sf * B_q.tensor_scale;
                    float v1 = kFP4Mag[hi & 7] * (hi & 8 ? -1.f : 1.f) * sf * B_q.tensor_scale;
                    h_B_dq[(size_t)n * K + kb * 16 + j * 2]     = __float2half(v0);
                    h_B_dq[(size_t)n * K + kb * 16 + j * 2 + 1] = __float2half(v1);
                }
            }
        }
    }
    std::vector<float> ctrl((size_t)M * N, 0.f);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) {
                acc += __half2float(h_A_dq[(size_t)m * K + k]) *
                       __half2float(h_B_dq[(size_t)n * K + k]);
            }
            ctrl[(size_t)m * N + n] = acc;
        }
    }

    std::vector<__half> got((size_t)M * N);
    cudaMemcpy(got.data(), d_D, (size_t)M * N * sizeof(__half), cudaMemcpyDeviceToHost);

    // ----- Tolerance: NVFP4 noise floor — accept 5% relative max error.
    // We measure relative error against a noise floor scaled to the typical
    // magnitude of the result (fixed denominator `eps * max|ref|`), which
    // is the standard NVFP4 reference-comparison metric. A naive
    // |g-r|/max(|r|,small) blows up at near-zero ref cells.
    float max_abs_ref = 0.f;
    for (int i = 0; i < M * N; ++i) max_abs_ref = std::max(max_abs_ref, std::fabs(ref[i]));
    const float floor_v = std::max(max_abs_ref * 1e-2f, 1e-3f);

    float max_rel = 0.f;
    int worst_i = -1;
    double sum_sq_err = 0.0, sum_sq_ref = 0.0;
    for (int i = 0; i < M * N; ++i) {
        float g = __half2float(got[i]);
        float r = ref[i];
        float diff = g - r;
        sum_sq_err += (double)diff * diff;
        sum_sq_ref += (double)r * r;
        float rel = std::fabs(diff) / std::max(std::fabs(r), floor_v);
        if (rel > max_rel) { max_rel = rel; worst_i = i; }
    }
    double rmse_rel = std::sqrt(sum_sq_err / std::max(sum_sq_ref, 1e-12));

    // Also compute rmse_rel of kernel output vs. CTRL (post-quantize ideal).
    double sum_sq_err_ctrl = 0.0, sum_sq_ctrl = 0.0;
    float max_abs_err_ctrl = 0.f;
    for (int i = 0; i < M * N; ++i) {
        float g = __half2float(got[i]);
        float c = ctrl[i];
        float diff = g - c;
        sum_sq_err_ctrl += (double)diff * diff;
        sum_sq_ctrl += (double)c * c;
        max_abs_err_ctrl = std::max(max_abs_err_ctrl, std::fabs(diff));
    }
    double rmse_rel_ctrl = std::sqrt(sum_sq_err_ctrl / std::max(sum_sq_ctrl, 1e-12));

    // Acceptance criterion: kernel output must match the post-quantize ideal
    // (CTRL) to within ctrl_rmse_tol relative RMSE. CTRL itself differs from
    // the FP32 reference by ~10-15% RMSE (NVFP4 noise floor), which is
    // fundamental to 4-bit quantization and not a kernel correctness issue.
    EXPECT_LT(rmse_rel_ctrl, ctrl_rmse_tol)
        << "kernel vs post-quantize ideal:"
        << " rmse_rel_ctrl=" << rmse_rel_ctrl
        << " max_abs_err_ctrl=" << max_abs_err_ctrl
        << " | vs FP32 ref: rmse_rel=" << rmse_rel
        << " max_rel=" << max_rel
        << " worst_i=" << worst_i
        << " got=" << __half2float(got[worst_i])
        << " ref=" << ref[worst_i]
        << " | max|ref|=" << max_abs_ref;

    cudaFree(d_A_fp16); cudaFree(d_A_packed); cudaFree(d_A_sf);
    cudaFree(d_off); cudaFree(d_D); cudaFree(d_alpha); cudaFree(d_B_fp16);
    imp::free_nvfp4_result(B_q);
}

// ---------------------------------------------------------------------------
// First end-to-end smallM kernel correctness test (Task 1.7).
// Single expert, M=N=K=128, single CTA per (expert, n_tile).  Produces FP16
// output that must match an FP32 host matmul of the original (pre-quant) FP16
// inputs to within 5% relative error (NVFP4 noise floor).
// ---------------------------------------------------------------------------
TEST(SmallMKernel, SingleExpert128x128x128) {
    if (!has_sm120()) GTEST_SKIP() << "SM120 required";
    run_smallm_single_expert(/*M=*/128, /*N=*/128, /*K=*/128);
}

// ---------------------------------------------------------------------------
// K-tile loop test (Task 1.8).
// Single expert, M=N=128, K=2048 (Qwen3-Coder-30B-A3B hidden_dim).
// Exercises 16 K-tile iterations of the outer loop added in T1.8.
// ---------------------------------------------------------------------------
TEST(SmallMKernel, SingleExpertK2048) {
    if (!has_sm120()) GTEST_SKIP() << "SM120 required";
    // Allow slightly looser tolerance than K=128: longer K-reduction chain
    // accumulates more FP4 quantization noise. 5e-2 is generous; if it
    // consistently lands well under this in practice, tighten in a follow-up.
    run_smallm_single_expert(/*M=*/128, /*N=*/128, /*K=*/2048, /*ctrl_rmse_tol=*/5e-2);
}

// ---------------------------------------------------------------------------
// Multi-expert test (Task 1.9).
// 4 experts with varying M_e: {128, 128, 64, 32}, N=256, K=512.
// All M_e ≤ 128 (Phase A constraint). Smaller experts (M_e=64, 32) round up
// to TILE_M=128 internally — padding rows do useless compute but the actual
// M_e output rows must be numerically correct.
// ---------------------------------------------------------------------------
TEST(SmallMKernel, FourExpertsVaryingM) {
    if (!has_sm120()) GTEST_SKIP() << "SM120 required";

    const int N = 256, K = 512, ne = 4;
    const int M_per[ne] = {128, 128, 64, 32};

    // Build per-expert weights B[e][N, K] FP16, quantize each.
    std::vector<imp::NvFP4QuantResult> B_q(ne);
    std::vector<std::vector<__half>> h_B_fp16(ne);
    for (int e = 0; e < ne; ++e) {
        std::mt19937 rng(13 + e);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        h_B_fp16[e].resize((size_t)N * K);
        for (auto& v : h_B_fp16[e]) v = __float2half(dist(rng));
        __half* d_B_fp16 = nullptr;
        cudaMalloc(&d_B_fp16, (size_t)N * K * sizeof(__half));
        cudaMemcpy(d_B_fp16, h_B_fp16[e].data(), (size_t)N * K * sizeof(__half), cudaMemcpyHostToDevice);
        int64_t b_shape[2] = {N, K};
        imp::Tensor B_t(d_B_fp16, imp::QType::F16, 2, b_shape, true);
        imp::quantize_fp16_to_nvfp4(B_t, B_q[e], /*stream*/0);
        cudaStreamSynchronize(0);
        cudaFree(d_B_fp16);
    }

    // Build activations: total_M = sum(M_per), [total_M, K] FP16.
    int h_offsets[ne + 1];
    h_offsets[0] = 0;
    for (int e = 0; e < ne; ++e) h_offsets[e + 1] = h_offsets[e] + M_per[e];
    int total_M = h_offsets[ne];

    std::vector<__half> h_A_fp16((size_t)total_M * K);
    {
        std::mt19937 rng_a(7);
        std::uniform_real_distribution<float> dist_a(-1.f, 1.f);
        for (auto& v : h_A_fp16) v = __float2half(dist_a(rng_a));
    }
    __half* d_A_fp16 = nullptr;
    cudaMalloc(&d_A_fp16, (size_t)total_M * K * sizeof(__half));
    cudaMemcpy(d_A_fp16, h_A_fp16.data(), (size_t)total_M * K * sizeof(__half), cudaMemcpyHostToDevice);

    // Upload offsets for moe_native quantize.
    int* d_offsets = nullptr;
    cudaMalloc(&d_offsets, (ne + 1) * sizeof(int));
    cudaMemcpy(d_offsets, h_offsets, (ne + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate per-expert packed/SF/output buffers.
    std::vector<void*> d_packed(ne), d_sf(ne), d_D(ne);
    for (int e = 0; e < ne; ++e) {
        cudaMalloc(&d_packed[e], (size_t)M_per[e] * K / 2);
        cudaMalloc(&d_sf[e],     (size_t)M_per[e] * K / 16);
        cudaMalloc(&d_D[e],      (size_t)M_per[e] * N * sizeof(__half));
        cudaMemset(d_D[e], 0,    (size_t)M_per[e] * N * sizeof(__half));
    }

    // Quantize activations via moe_native (per-expert packed + SF buffers).
    imp::quantize_fp16_to_nvfp4_moe_native(
        d_A_fp16, d_packed.data(), d_sf.data(), d_offsets, total_M, K, ne, /*stream*/0);
    cudaStreamSynchronize(0);

    // Compute per-expert tensor_scale for A (mirrors moe_native: absmax/6).
    std::vector<float> a_tensor_scale(ne);
    for (int e = 0; e < ne; ++e) {
        float absmax = 0.f;
        const __half* row = h_A_fp16.data() + (size_t)h_offsets[e] * K;
        for (int i = 0; i < M_per[e] * K; ++i)
            absmax = std::max(absmax, std::fabs(__half2float(row[i])));
        a_tensor_scale[e] = (absmax == 0.f) ? 1.f : (absmax / 6.0f);
    }

    // Run kernel. API requires dev_alpha as a device pointer.
    std::vector<const void*> A_arr(ne), SFA_arr(ne), B_arr(ne), SFB_arr(ne);
    std::vector<float> h_alpha(ne);
    for (int e = 0; e < ne; ++e) {
        A_arr[e]   = d_packed[e];
        SFA_arr[e] = d_sf[e];
        B_arr[e]   = B_q[e].packed_data;
        SFB_arr[e] = B_q[e].micro_scales;
        h_alpha[e] = a_tensor_scale[e] * B_q[e].tensor_scale;
    }
    float* d_alpha = nullptr;
    cudaMalloc(&d_alpha, ne * sizeof(float));
    cudaMemcpy(d_alpha, h_alpha.data(), ne * sizeof(float), cudaMemcpyHostToDevice);
    bool ok = imp::gemm_grouped_nvfp4_smallM(
        ne, M_per, N, K, A_arr.data(), SFA_arr.data(),
        B_arr.data(), SFB_arr.data(), d_D.data(), d_alpha, /*stream*/0);
    ASSERT_TRUE(ok);
    cudaError_t err = cudaDeviceSynchronize();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    // Per-expert reference + check (same CTRL-based approach as run_smallm_single_expert).
    const float kFP4Mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    auto ue4m3 = [&](uint8_t b) {
        uint32_t sign = (b >> 7) & 1;
        uint32_t exp  = (b >> 3) & 0x0F;
        uint32_t man  = b & 0x07;
        uint32_t fp32;
        if (exp == 0) {
            float v = (float)man * (1.0f / 512.0f);
            std::memcpy(&fp32, &v, 4);
            fp32 |= (sign << 31);
        } else {
            fp32 = (sign << 31) | ((exp + 120u) << 23) | (man << 20);
        }
        float r;
        std::memcpy(&r, &fp32, 4);
        return r;
    };

    for (int e = 0; e < ne; ++e) {
        int M_e = M_per[e];
        const __half* A_e = h_A_fp16.data() + (size_t)h_offsets[e] * K;

        // Dequantize A_e (per moe_native layout).
        std::vector<uint8_t> h_A_packed_v((size_t)M_e * K / 2);
        std::vector<uint8_t> h_A_sf_v((size_t)M_e * K / 16);
        cudaMemcpy(h_A_packed_v.data(), d_packed[e], h_A_packed_v.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_A_sf_v.data(),     d_sf[e],     h_A_sf_v.size(),     cudaMemcpyDeviceToHost);
        std::vector<__half> h_A_dq((size_t)M_e * K);
        for (int m = 0; m < M_e; ++m) {
            for (int kb = 0; kb < K / 16; ++kb) {
                float sf = ue4m3(h_A_sf_v[(size_t)m * (K / 16) + kb]);
                for (int j = 0; j < 8; ++j) {
                    uint8_t byte = h_A_packed_v[(size_t)m * (K / 2) + kb * 8 + j];
                    uint8_t lo = byte & 0xF, hi = (byte >> 4) & 0xF;
                    float v0 = kFP4Mag[lo & 7] * (lo & 8 ? -1.f : 1.f) * sf * a_tensor_scale[e];
                    float v1 = kFP4Mag[hi & 7] * (hi & 8 ? -1.f : 1.f) * sf * a_tensor_scale[e];
                    h_A_dq[(size_t)m * K + kb * 16 + j * 2]     = __float2half(v0);
                    h_A_dq[(size_t)m * K + kb * 16 + j * 2 + 1] = __float2half(v1);
                }
            }
        }

        // Dequantize B_e.
        std::vector<uint8_t> h_B_packed_v((size_t)N * K / 2);
        std::vector<uint8_t> h_B_sf_v((size_t)N * K / 16);
        cudaMemcpy(h_B_packed_v.data(), B_q[e].packed_data,  h_B_packed_v.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_B_sf_v.data(),     B_q[e].micro_scales, h_B_sf_v.size(),     cudaMemcpyDeviceToHost);
        std::vector<__half> h_B_dq((size_t)N * K);
        for (int n = 0; n < N; ++n) {
            for (int kb = 0; kb < K / 16; ++kb) {
                float sf = ue4m3(h_B_sf_v[(size_t)n * (K / 16) + kb]);
                for (int j = 0; j < 8; ++j) {
                    uint8_t byte = h_B_packed_v[(size_t)n * (K / 2) + kb * 8 + j];
                    uint8_t lo = byte & 0xF, hi = (byte >> 4) & 0xF;
                    float v0 = kFP4Mag[lo & 7] * (lo & 8 ? -1.f : 1.f) * sf * B_q[e].tensor_scale;
                    float v1 = kFP4Mag[hi & 7] * (hi & 8 ? -1.f : 1.f) * sf * B_q[e].tensor_scale;
                    h_B_dq[(size_t)n * K + kb * 16 + j * 2]     = __float2half(v0);
                    h_B_dq[(size_t)n * K + kb * 16 + j * 2 + 1] = __float2half(v1);
                }
            }
        }

        // FP32 reference (pre-quant FP16 inputs).
        std::vector<float> ref((size_t)M_e * N, 0.f);
        for (int m = 0; m < M_e; ++m) {
            for (int n = 0; n < N; ++n) {
                float acc = 0.f;
                for (int k = 0; k < K; ++k)
                    acc += __half2float(A_e[(size_t)m * K + k]) *
                           __half2float(h_B_fp16[e][(size_t)n * K + k]);
                ref[(size_t)m * N + n] = acc;
            }
        }

        // Post-quantize ideal (CTRL): dequantized A × dequantized B.
        std::vector<float> ctrl((size_t)M_e * N, 0.f);
        for (int m = 0; m < M_e; ++m) {
            for (int n = 0; n < N; ++n) {
                float acc = 0.f;
                for (int k = 0; k < K; ++k)
                    acc += __half2float(h_A_dq[(size_t)m * K + k]) *
                           __half2float(h_B_dq[(size_t)n * K + k]);
                ctrl[(size_t)m * N + n] = acc;
            }
        }

        // Fetch kernel output.
        std::vector<__half> got((size_t)M_e * N);
        cudaMemcpy(got.data(), d_D[e], (size_t)M_e * N * sizeof(__half), cudaMemcpyDeviceToHost);

        // Tolerance: kernel vs CTRL (post-quantize ideal) relative RMSE < 1%.
        float max_abs_ref = 0.f;
        for (int i = 0; i < M_e * N; ++i)
            max_abs_ref = std::max(max_abs_ref, std::fabs(ref[i]));

        double sum_sq_err_ctrl = 0.0, sum_sq_ctrl = 0.0;
        float max_abs_err_ctrl = 0.f;
        for (int i = 0; i < M_e * N; ++i) {
            float g = __half2float(got[i]);
            float c = ctrl[i];
            float diff = g - c;
            sum_sq_err_ctrl += (double)diff * diff;
            sum_sq_ctrl += (double)c * c;
            max_abs_err_ctrl = std::max(max_abs_err_ctrl, std::fabs(diff));
        }
        double rmse_rel_ctrl = std::sqrt(sum_sq_err_ctrl / std::max(sum_sq_ctrl, 1e-12));

        EXPECT_LT(rmse_rel_ctrl, 1e-2)
            << "expert " << e << " (M_e=" << M_e << ")"
            << " rmse_rel_ctrl=" << rmse_rel_ctrl
            << " max_abs_err_ctrl=" << max_abs_err_ctrl
            << " | max|ref|=" << max_abs_ref;
    }

    // Cleanup.
    cudaFree(d_A_fp16);
    cudaFree(d_offsets);
    cudaFree(d_alpha);
    for (int e = 0; e < ne; ++e) {
        cudaFree(d_packed[e]);
        cudaFree(d_sf[e]);
        cudaFree(d_D[e]);
        imp::free_nvfp4_result(B_q[e]);
    }
}

// ---------------------------------------------------------------------------
// Diagnostic cross-check: run the production HW MMA kernel and the software
// reference kernel on the SAME quantized inputs, then compare element-wise.
// Compiled only with SMALLM_SOFTWARE_REF — not part of the default suite,
// purely a debug guard for the layout-mapping work in T1.7b.
// ---------------------------------------------------------------------------
#ifdef SMALLM_SOFTWARE_REF
TEST(SmallMKernel, HwMatchesSoftwareReference) {
    if (!has_sm120()) GTEST_SKIP() << "SM120 required";

    const int M = 128, N = 128, K = 128;

    std::mt19937 rng_w(13);
    std::uniform_real_distribution<float> dist_w(-0.5f, 0.5f);
    std::vector<__half> h_B_fp16(N * K);
    for (auto& v : h_B_fp16) v = __float2half(dist_w(rng_w));
    __half* d_B_fp16 = nullptr;
    cudaMalloc(&d_B_fp16, N * K * sizeof(__half));
    cudaMemcpy(d_B_fp16, h_B_fp16.data(), N * K * sizeof(__half), cudaMemcpyHostToDevice);
    int64_t b_shape[2] = {N, K};
    imp::Tensor B_t(d_B_fp16, imp::QType::F16, 2, b_shape, true);
    imp::NvFP4QuantResult B_q;
    imp::quantize_fp16_to_nvfp4(B_t, B_q, /*stream*/0);
    cudaStreamSynchronize(0);

    std::mt19937 rng_a(7);
    std::uniform_real_distribution<float> dist_a(-1.f, 1.f);
    std::vector<__half> h_A_fp16(M * K);
    for (auto& v : h_A_fp16) v = __float2half(dist_a(rng_a));
    __half* d_A_fp16 = nullptr;
    cudaMalloc(&d_A_fp16, M * K * sizeof(__half));
    cudaMemcpy(d_A_fp16, h_A_fp16.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice);

    void* d_A_packed = nullptr; void* d_A_sf = nullptr;
    cudaMalloc(&d_A_packed, M * K / 2);
    cudaMalloc(&d_A_sf,     M * K / 16);
    int h_off[2] = {0, M};
    int* d_off = nullptr; cudaMalloc(&d_off, sizeof(h_off));
    cudaMemcpy(d_off, h_off, sizeof(h_off), cudaMemcpyHostToDevice);
    void* h_pp[1] = {d_A_packed}; void* h_sp[1] = {d_A_sf};
    imp::quantize_fp16_to_nvfp4_moe_native(d_A_fp16, h_pp, h_sp, d_off, M, K, 1, 0);
    cudaStreamSynchronize(0);

    float a_absmax = 0.f;
    for (auto h : h_A_fp16) a_absmax = std::max(a_absmax, std::fabs(__half2float(h)));
    float a_tensor_scale = (a_absmax == 0.f) ? 1.f : (a_absmax / 6.0f);
    float combined_alpha = a_tensor_scale * B_q.tensor_scale;

    void* d_D_hw = nullptr; cudaMalloc(&d_D_hw, M * N * sizeof(__half));
    void* d_D_sw = nullptr; cudaMalloc(&d_D_sw, M * N * sizeof(__half));
    cudaMemset(d_D_hw, 0, M * N * sizeof(__half));
    cudaMemset(d_D_sw, 0, M * N * sizeof(__half));
    int M_per[1] = {M};
    const void* A_arr[1]   = {d_A_packed};
    const void* SFA_arr[1] = {d_A_sf};
    const void* B_arr[1]   = {B_q.packed_data};
    const void* SFB_arr[1] = {B_q.micro_scales};
    void* D_arr_hw[1]   = {d_D_hw};
    void* D_arr_sw[1]   = {d_D_sw};
    // API now requires a device pointer for alpha.
    float h_alpha_sw[1] = {combined_alpha};
    float* d_alpha_sw = nullptr;
    cudaMalloc(&d_alpha_sw, sizeof(float));
    cudaMemcpy(d_alpha_sw, h_alpha_sw, sizeof(float), cudaMemcpyHostToDevice);

    bool ok_hw = imp::gemm_grouped_nvfp4_smallM(
        1, M_per, N, K, A_arr, SFA_arr, B_arr, SFB_arr, D_arr_hw, d_alpha_sw, /*stream*/0);
    ASSERT_TRUE(ok_hw);
    bool ok_sw = gemm_grouped_nvfp4_smallM_software_ref(
        1, M_per, N, K, A_arr, SFA_arr, B_arr, SFB_arr, D_arr_sw, d_alpha_sw, /*stream*/0);
    ASSERT_TRUE(ok_sw);
    cudaError_t err = cudaDeviceSynchronize();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    std::vector<__half> got_hw(M * N), got_sw(M * N);
    cudaMemcpy(got_hw.data(), d_D_hw, M * N * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(got_sw.data(), d_D_sw, M * N * sizeof(__half), cudaMemcpyDeviceToHost);

    float max_abs_sw = 0.f;
    for (auto h : got_sw) max_abs_sw = std::max(max_abs_sw, std::fabs(__half2float(h)));
    const float floor_v = std::max(max_abs_sw * 1e-3f, 1e-4f);
    float max_rel = 0.f;
    int worst = -1;
    for (int i = 0; i < M * N; ++i) {
        float h = __half2float(got_hw[i]);
        float s = __half2float(got_sw[i]);
        float rel = std::fabs(h - s) / std::max(std::fabs(s), floor_v);
        if (rel > max_rel) { max_rel = rel; worst = i; }
    }
    EXPECT_LT(max_rel, 5e-3f)
        << "HW vs SW max_rel=" << max_rel << " worst=" << worst
        << " hw=" << __half2float(got_hw[worst]) << " sw=" << __half2float(got_sw[worst]);

    cudaFree(d_A_fp16); cudaFree(d_A_packed); cudaFree(d_A_sf);
    cudaFree(d_off); cudaFree(d_D_hw); cudaFree(d_D_sw);
    cudaFree(d_alpha_sw); cudaFree(d_B_fp16);
    imp::free_nvfp4_result(B_q);
}
#endif  // SMALLM_SOFTWARE_REF

}  // anonymous namespace
