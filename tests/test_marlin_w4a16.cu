// =============================================================================
// test_marlin_w4a16.cu — the vendored Marlin FP4 W4A16 GEMM: correctness + bench
// =============================================================================
//
// Correctness: repack + scale processing + kernel against the same host
// dequant walk the other NVFP4 GEMM tests use, across the M range the decode
// dispatch serves (1..64) and both tile families (N%64 / N%128 shapes).
// Bench: the batched-decode class shape (M=32, N=5120, K=5120) where CUTLASS
// measures 41.4 us and vLLM's Marlin proves ~24% less GEMM time for the class
// (BENCHMARKS.md, "The 1.58x concurrency gap"). Reported, plus a loose floor
// assert so a broken config selection fails loud.
//
// GPU required — skips cleanly without one.
// =============================================================================

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "quant/marlin/marlin_w4a16.h"
#include "quant/nvfp4_quant.h"
#include "compute/gemm_cutlass_sm120.h"
#include "core/tensor.h"

namespace {

bool gpu_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

float host_dequant(const std::vector<uint8_t>& packed, const std::vector<uint8_t>& scales, float tensor_scale,
                   int n, int k, int K) {
    static const float lut[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    uint8_t byte = packed[(size_t)n * (K / 2) + k / 2];
    uint8_t nib = (k & 1) ? (byte >> 4) : (byte & 0xF);
    uint8_t se = scales[(size_t)n * (K / 16) + k / 16];
    int sign = (se >> 7) ? -1 : 1;
    int exp = (se >> 3) & 0xF;
    int mant = se & 0x7;
    float sf = (exp == 0) ? sign * (mant / 8.0f) * std::pow(2.0f, -6.0f)
                          : sign * (1.0f + mant / 8.0f) * std::pow(2.0f, exp - 7);
    return lut[nib] * sf * tensor_scale;
}

struct QuantWeight {
    imp::NvFP4QuantResult q{};
    std::vector<uint8_t> packed, scales;

    void build(int N, int K, unsigned seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<__half> w_h((size_t)N * K);
        for (auto& v : w_h)
            v = __float2half(dist(rng));
        void* d_w = nullptr;
        ASSERT_EQ(cudaMalloc(&d_w, w_h.size() * sizeof(__half)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_w, w_h.data(), w_h.size() * sizeof(__half), cudaMemcpyHostToDevice),
                  cudaSuccess);
        int64_t shp[2] = {N, K};
        imp::Tensor wt(d_w, imp::QType::F16, 2, shp, /*on_device=*/true);
        imp::quantize_fp16_to_nvfp4(wt, q, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_NE(q.packed_data, nullptr);
        cudaFree(d_w);
        packed.resize((size_t)N * K / 2);
        scales.resize((size_t)N * K / 16);
        ASSERT_EQ(cudaMemcpy(packed.data(), q.packed_data, packed.size(), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(scales.data(), q.micro_scales, scales.size(), cudaMemcpyDeviceToHost),
                  cudaSuccess);
    }
};

class MarlinW4A16Test : public ::testing::Test {
protected:
    void SetUp() override {
        if (!gpu_available())
            GTEST_SKIP() << "no CUDA device";
    }
};

void run_correctness(int M, int N, int K, unsigned seed) {
    SCOPED_TRACE(testing::Message() << "M=" << M << " N=" << N << " K=" << K);
    QuantWeight w;
    w.build(N, K, seed);

    imp::marlin_w4a16::MarlinWeight mw;
    ASSERT_TRUE(
        imp::marlin_w4a16::prepare(w.q.packed_data, w.q.micro_scales, w.q.tensor_scale, N, K, mw, nullptr));

    std::mt19937 rng(seed + 100);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<__half> x_h((size_t)M * K);
    for (auto& v : x_h)
        v = __float2half(dist(rng));
    void *d_x = nullptr, *d_y = nullptr, *d_locks = nullptr, *d_ctmp = nullptr;
    ASSERT_EQ(cudaMalloc(&d_x, x_h.size() * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, (size_t)M * N * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_locks, imp::marlin_w4a16::workspace_bytes()), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_locks, 0, imp::marlin_w4a16::workspace_bytes()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_ctmp, imp::marlin_w4a16::c_tmp_bytes(M)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_x, x_h.data(), x_h.size() * sizeof(__half), cudaMemcpyHostToDevice), cudaSuccess);

    ASSERT_TRUE(imp::marlin_w4a16::gemm(mw, static_cast<const half*>(d_x), static_cast<half*>(d_y), M,
                                        /*lda=*/K, static_cast<int*>(d_locks), static_cast<float*>(d_ctmp),
                                        nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<__half> y_h((size_t)M * N);
    ASSERT_EQ(cudaMemcpy(y_h.data(), d_y, y_h.size() * sizeof(__half), cudaMemcpyDeviceToHost), cudaSuccess);

    double max_rel = 0.0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double ref = 0.0;
            for (int k = 0; k < K; ++k)
                ref += host_dequant(w.packed, w.scales, w.q.tensor_scale, n, k, K) *
                       __half2float(x_h[(size_t)m * K + k]);
            double got = __half2float(y_h[(size_t)m * N + n]);
            double rel = std::abs(got - ref) / std::max(1.0, std::abs(ref));
            max_rel = std::max(max_rel, rel);
        }
    }
    EXPECT_LT(max_rel, 2e-2) << "max relative error " << max_rel;

    imp::marlin_w4a16::release(mw);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_locks);
    cudaFree(d_ctmp);
}

TEST_F(MarlinW4A16Test, MatchesHostReferenceAcrossM) {
    // K=512: both thread_k families divide; N=256: both thread_n families.
    for (int M : {1, 5, 8, 16, 17, 32})
        run_correctness(M, 256, 512, 11 + M);
}

TEST_F(MarlinW4A16Test, MatchesHostReferenceNarrowN) {
    // N=64: only the thread_n=64 config divides — exercises the fallback.
    run_correctness(9, 64, 256, 3);
}

TEST_F(MarlinW4A16Test, MatchesHostReferenceLargeM) {
    // Above the decode class: the split loop with thread_m_blocks up to 4.
    run_correctness(64, 256, 512, 21);
    run_correctness(100, 256, 512, 23);
}

TEST_F(MarlinW4A16Test, DecodeShapeBench) {
    const int N = 5120, K = 5120;
    QuantWeight w;
    w.build(N, K, 7);
    imp::marlin_w4a16::MarlinWeight mw;
    ASSERT_TRUE(
        imp::marlin_w4a16::prepare(w.q.packed_data, w.q.micro_scales, w.q.tensor_scale, N, K, mw, nullptr));

    const int max_m = 2048;
    void *d_x = nullptr, *d_y = nullptr, *d_locks = nullptr, *d_ctmp = nullptr;
    ASSERT_EQ(cudaMalloc(&d_x, (size_t)max_m * K * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_x, 0x3c, (size_t)max_m * K * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, (size_t)max_m * N * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_locks, imp::marlin_w4a16::workspace_bytes()), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_locks, 0, imp::marlin_w4a16::workspace_bytes()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_ctmp, imp::marlin_w4a16::c_tmp_bytes(max_m)), cudaSuccess);

    auto bench = [&](int M) -> float {
        const int reps = 200, warmup = 20;
        for (int i = 0; i < warmup; i++)
            imp::marlin_w4a16::gemm(mw, static_cast<const half*>(d_x), static_cast<half*>(d_y), M, K,
                                    static_cast<int*>(d_locks), static_cast<float*>(d_ctmp), nullptr);
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        cudaEventRecord(t0);
        for (int i = 0; i < reps; i++)
            imp::marlin_w4a16::gemm(mw, static_cast<const half*>(d_x), static_cast<half*>(d_y), M, K,
                                    static_cast<int*>(d_locks), static_cast<float*>(d_ctmp), nullptr);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t0, t1);
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        return ms * 1000.0f / reps;
    };

    for (int M : {1, 8, 16, 32, 64, 256, 512, 2048}) {
        float us = bench(M);
        printf("[marlin] M=%4d N=%d K=%d: %.1f us (%.0f GB/s weight stream)\n", M, N, K, us,
               (double)N * K / 2 / us / 1e3);
    }

    // The kernel exists to beat the CUTLASS 41.4 us on this shape; a broken
    // config selection (e.g. falling into a starved single-config path) shows
    // up far above that. Loose floor: 60 us.
    float us32 = bench(32);
    EXPECT_LT(us32, 60.0f) << "M=32 decode-class shape at " << us32 << " us";

    imp::marlin_w4a16::release(mw);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_locks);
    cudaFree(d_ctmp);
}

// Marlin vs the shipping CUTLASS block-scaled path on the same shape, same M
// sweep. The CUTLASS arm includes the activation quantize the real dispatch
// pays before every call (W4A16 reads FP16 activations directly). This is the
// measurement the storage-tier decision rests on: Marlin data is a full second
// copy of the 4-bit bytes, so it can only replace, not join, the resident
// layout per weight.
TEST_F(MarlinW4A16Test, CutlassComparisonBench) {
    const int N = 5120, K = 5120;
    QuantWeight w;
    w.build(N, K, 7);

    imp::marlin_w4a16::MarlinWeight mw;
    ASSERT_TRUE(
        imp::marlin_w4a16::prepare(w.q.packed_data, w.q.micro_scales, w.q.tensor_scale, N, K, mw, nullptr));
    imp::CutlassNvFP4Weight cw;
    imp::convert_nvfp4_to_cutlass(w.q, cw, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_NE(cw.scale_factors, nullptr);

    const int max_m = 4096;
    void *d_x = nullptr, *d_y = nullptr, *d_locks = nullptr, *d_ctmp = nullptr;
    void *d_xq = nullptr, *d_xsf = nullptr, *d_cutlass_ws = nullptr;
    ASSERT_EQ(cudaMalloc(&d_x, (size_t)max_m * K * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_x, 0x3c, (size_t)max_m * K * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, (size_t)max_m * N * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_locks, imp::marlin_w4a16::workspace_bytes()), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_locks, 0, imp::marlin_w4a16::workspace_bytes()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_ctmp, imp::marlin_w4a16::c_tmp_bytes(max_m)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_xq, (size_t)max_m * K / 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_xsf, imp::cutlass_nvfp4_sf_size(max_m, K)), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_xsf, 0, imp::cutlass_nvfp4_sf_size(max_m, K)), cudaSuccess);
    size_t ws_size = imp::gemm_nvfp4_cutlass_sm120_workspace(max_m, N, K);
    ASSERT_EQ(cudaMalloc(&d_cutlass_ws, ws_size ? ws_size : 16), cudaSuccess);

    auto time_us = [&](int reps, auto&& fn) -> float {
        for (int i = 0; i < reps / 10 + 1; i++)
            fn();
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        cudaEventRecord(t0);
        for (int i = 0; i < reps; i++)
            fn();
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t0, t1);
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        return ms * 1000.0f / reps;
    };

    printf("[cmp] %6s %12s %12s %12s\n", "M", "marlin_us", "cutlass_us", "ct_gemm_only");
    for (int M : {1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}) {
        const int reps = M <= 64 ? 200 : 50;
        float marlin_us = time_us(reps, [&] {
            imp::marlin_w4a16::gemm(mw, static_cast<const half*>(d_x), static_cast<half*>(d_y), M, K,
                                    static_cast<int*>(d_locks), static_cast<float*>(d_ctmp), nullptr);
        });
        bool ct_ok = true;
        float cutlass_us = time_us(reps, [&] {
            imp::quantize_fp16_to_nvfp4_cutlass(d_x, d_xq, d_xsf, M, K, nullptr);
            ct_ok = ct_ok && imp::gemm_nvfp4_cutlass_sm120(d_xq, d_xsf, cw, d_y, M, N, K, d_cutlass_ws,
                                                           ws_size, nullptr);
        });
        float ct_gemm_us = time_us(reps, [&] {
            ct_ok = ct_ok && imp::gemm_nvfp4_cutlass_sm120(d_xq, d_xsf, cw, d_y, M, N, K, d_cutlass_ws,
                                                           ws_size, nullptr);
        });
        printf("[cmp] %6d %12.1f %12s %12s\n", M, marlin_us,
               ct_ok ? std::to_string(cutlass_us).substr(0, 6).c_str() : "FAIL",
               ct_ok ? std::to_string(ct_gemm_us).substr(0, 6).c_str() : "FAIL");
    }

    imp::marlin_w4a16::release(mw);
    imp::free_cutlass_nvfp4_weight(cw);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_locks);
    cudaFree(d_ctmp);
    cudaFree(d_xq);
    cudaFree(d_xsf);
    cudaFree(d_cutlass_ws);
}

}  // namespace
