// cuBLASLt stand-ins for the former cuBLAS v2 calls (compute/gemm_lt_ex.h) and the callers that had no
// test of their own: gemm_moe_batched, gemm_kv_batched, gemm() FP16 -> FP32 and FP32 (TF32).

#include "compute/gemm.h"
#include "compute/gemm_grouped.h"
#include "compute/gemm_lt_ex.h"
#include "core/fp_bits.h"
#include "core/tensor.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

namespace imp {
namespace {

std::vector<uint16_t> random_half(size_t n, unsigned seed) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<uint16_t> v(n);
    for (auto& x : v)
        x = float_to_half(u(g));
    return v;
}

template <typename T>
T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    EXPECT_EQ(cudaMalloc(&d, h.size() * sizeof(T)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    return d;
}

template <typename T>
std::vector<T> download(const T* d, size_t n) {
    std::vector<T> h(n);
    EXPECT_EQ(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    return h;
}

// Row-major C[M, N] = A[M, K] @ B[N, K]^T in double.
std::vector<double> ref_nt(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, int M, int N,
                           int K, size_t a_off = 0, size_t b_off = 0) {
    std::vector<double> c(static_cast<size_t>(M) * N, 0.0);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double s = 0.0;
            for (int k = 0; k < K; k++)
                s += static_cast<double>(half_to_float(a[a_off + static_cast<size_t>(m) * K + k])) *
                     half_to_float(b[b_off + static_cast<size_t>(n) * K + k]);
            c[static_cast<size_t>(m) * N + n] = s;
        }
    return c;
}

// Column-major batch with a shared A (stride 0), an odd lda and FP32 output: the GQA attention shape.
TEST(LtGemmEx, StridedBatchSharedAOddLd) {
    constexpr int m = 37, n = 19, k = 40, lda = 41, batch = 3;
    const auto a = random_half(static_cast<size_t>(lda) * k, 1);
    const auto b = random_half(static_cast<size_t>(k) * n * batch, 2);
    uint16_t* da = upload(a);
    uint16_t* db = upload(b);
    float* dc = nullptr;
    ASSERT_EQ(cudaMalloc(&dc, sizeof(float) * m * n * batch), cudaSuccess);
    const float one = 1.0f, zero = 0.0f;
    ASSERT_EQ(lt_gemm_strided_batched(CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, da, CUDA_R_16F, lda, 0, db,
                                      CUDA_R_16F, k, static_cast<long long>(k) * n, &zero, dc, CUDA_R_32F, m,
                                      static_cast<long long>(m) * n, batch, CUBLAS_COMPUTE_32F, nullptr),
              CUBLAS_STATUS_SUCCESS);
    const auto c = download(dc, static_cast<size_t>(m) * n * batch);
    double worst = 0.0;
    for (int bt = 0; bt < batch; bt++)
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++) {
                double s = 0.0;
                for (int t = 0; t < k; t++)
                    s += static_cast<double>(half_to_float(a[static_cast<size_t>(t) * lda + i])) *
                         half_to_float(b[static_cast<size_t>(bt) * k * n + static_cast<size_t>(j) * k + t]);
                worst = std::max(
                    worst,
                    std::fabs(c[static_cast<size_t>(bt) * m * n + static_cast<size_t>(j) * m + i] - s));
            }
    EXPECT_LT(worst, 1e-3);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}

// MoE expert GEMMs: experts 0 and 1 share M 3 (one pointer-array batch), expert 2 is empty, expert 3
// alone (host pointers).
TEST(LtGemmEx, MoeBatchedGroupsAndSingleExpert) {
    constexpr int K = 64, N = 48, ne = 4;
    const std::vector<int32_t> offsets = {0, 3, 6, 6, 10};
    const int rows = offsets.back();
    const auto a = random_half(static_cast<size_t>(rows) * K, 3);
    std::vector<std::vector<uint16_t>> w(ne);
    std::vector<uint16_t*> dw(ne);
    for (int e = 0; e < ne; e++) {
        w[e] = random_half(static_cast<size_t>(N) * K, 10 + e);
        dw[e] = upload(w[e]);
    }
    uint16_t* da = upload(a);
    uint16_t* dc = nullptr;
    ASSERT_EQ(cudaMalloc(&dc, sizeof(uint16_t) * rows * N), cudaSuccess);
    std::vector<const void*> b_ptrs(dw.begin(), dw.end());
    gemm_moe_batched(da, dc, offsets.data(), b_ptrs.data(), K, N, QType::F16, ne, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto c = download(dc, static_cast<size_t>(rows) * N);
    double worst = 0.0;
    for (int e = 0; e < ne; e++) {
        const int M = offsets[e + 1] - offsets[e];
        const auto r = ref_nt(a, w[e], M, N, K, static_cast<size_t>(offsets[e]) * K);
        for (int i = 0; i < M * N; i++)
            worst = std::max(worst,
                             std::fabs(half_to_float(c[static_cast<size_t>(offsets[e]) * N + i]) - r[i]));
    }
    EXPECT_LT(worst, 2e-2) << "fp16 output, |ref| <= ~8";
    for (auto* p : dw)
        cudaFree(p);
    cudaFree(da);
    cudaFree(dc);
}

// FP8 E4M3 experts with per-expert activation scales: every active expert its own group.
TEST(LtGemmEx, MoeBatchedFp8PerExpertScale) {
    constexpr int K = 64, N = 32, ne = 3;
    const std::vector<int32_t> offsets = {0, 4, 4, 9};
    const std::vector<float> scales = {0.5f, 1.0f, 2.0f};
    const int rows = offsets.back();
    std::mt19937 g(9);
    std::uniform_int_distribution<int> code(0x28, 0x40);  // |v| in [0.25, 2], random sign below
    auto fp8 = [&](size_t n) {
        std::vector<uint8_t> v(n);
        for (auto& x : v)
            x = static_cast<uint8_t>(code(g) | ((g() & 1) << 7));
        return v;
    };
    const auto a = fp8(static_cast<size_t>(rows) * K);
    std::vector<std::vector<uint8_t>> w(ne);
    std::vector<uint8_t*> dw(ne);
    for (int e = 0; e < ne; e++) {
        w[e] = fp8(static_cast<size_t>(N) * K);
        dw[e] = upload(w[e]);
    }
    uint8_t* da = upload(a);
    uint16_t* dc = nullptr;
    ASSERT_EQ(cudaMalloc(&dc, sizeof(uint16_t) * rows * N), cudaSuccess);
    std::vector<const void*> b_ptrs(dw.begin(), dw.end());
    gemm_moe_batched(da, dc, offsets.data(), b_ptrs.data(), K, N, QType::FP8_E4M3, ne, nullptr, nullptr,
                     QType::F16, scales.data());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto c = download(dc, static_cast<size_t>(rows) * N);
    double worst = 0.0, ref_max = 0.0;
    for (int e = 0; e < ne; e++)
        for (int m = offsets[e]; m < offsets[e + 1]; m++)
            for (int n = 0; n < N; n++) {
                double s = 0.0;
                for (int k = 0; k < K; k++)
                    s += static_cast<double>(e4m3_to_float(a[static_cast<size_t>(m) * K + k])) *
                         e4m3_to_float(w[e][static_cast<size_t>(n) * K + k]);
                s *= scales[e];
                ref_max = std::max(ref_max, std::fabs(s));
                worst = std::max(worst, std::fabs(half_to_float(c[static_cast<size_t>(m) * N + n]) - s));
            }
    EXPECT_LT(worst, 2e-3 * ref_max) << "max |ref| " << ref_max;
    for (auto* p : dw)
        cudaFree(p);
    cudaFree(da);
    cudaFree(dc);
}

// K and V projections in one strided batch; V's view sits M x N + 16 elements behind K's.
TEST(LtGemmEx, KvBatched) {
    constexpr int M = 5, K = 32, N = 24;
    const auto x = random_half(static_cast<size_t>(M) * K, 4);
    const auto wkv = random_half(static_cast<size_t>(2) * N * K, 5);
    uint16_t* dx = upload(x);
    uint16_t* dw = upload(wkv);
    uint16_t* dout = nullptr;
    const size_t gap = static_cast<size_t>(M) * N + 16;
    ASSERT_EQ(cudaMalloc(&dout, sizeof(uint16_t) * 2 * gap), cudaSuccess);
    int64_t xs[2] = {M, K}, ws[2] = {2 * N, K}, os[2] = {M, N};
    Tensor tx(dx, QType::F16, 2, xs, true), tw(dw, QType::F16, 2, ws, true);
    Tensor tk(dout, QType::F16, 2, os, true), tv(dout + gap, QType::F16, 2, os, true);
    gemm_kv_batched(tx, tw, tk, tv, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto out = download(dout, 2 * gap);
    const auto rk = ref_nt(x, wkv, M, N, K), rv = ref_nt(x, wkv, M, N, K, 0, static_cast<size_t>(N) * K);
    double worst = 0.0;
    for (int i = 0; i < M * N; i++) {
        worst = std::max(worst, std::fabs(half_to_float(out[i]) - rk[i]));
        worst = std::max(worst, std::fabs(half_to_float(out[gap + i]) - rv[i]));
    }
    EXPECT_LT(worst, 1e-2);
    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dout);
}

// gemm(): FP16 x FP16 -> FP32 output (its own descriptor) and FP32 x FP32 (TF32).
TEST(LtGemmEx, GemmF16ToF32AndF32) {
    constexpr int M = 7, N = 33, K = 96;
    const auto a = random_half(static_cast<size_t>(M) * K, 6), b = random_half(static_cast<size_t>(N) * K, 7);
    const auto r = ref_nt(a, b, M, N, K);
    uint16_t* da = upload(a);
    uint16_t* db = upload(b);
    float* dc = nullptr;
    ASSERT_EQ(cudaMalloc(&dc, sizeof(float) * M * N), cudaSuccess);
    int64_t as[2] = {M, K}, bs[2] = {N, K}, cs[2] = {M, N};
    Tensor ta(da, QType::F16, 2, as, true), tb(db, QType::F16, 2, bs, true), tc(dc, QType::F32, 2, cs, true);
    gemm(ta, tb, tc, 1.0f, 0.0f, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto c = download(dc, static_cast<size_t>(M) * N);
    double worst = 0.0;
    for (int i = 0; i < M * N; i++)
        worst = std::max(worst, std::fabs(c[i] - r[i]));
    EXPECT_LT(worst, 1e-4) << "FP16 -> FP32";

    std::vector<float> af(a.size()), bf(b.size());
    for (size_t i = 0; i < a.size(); i++)
        af[i] = half_to_float(a[i]);
    for (size_t i = 0; i < b.size(); i++)
        bf[i] = half_to_float(b[i]);
    float* daf = upload(af);
    float* dbf = upload(bf);
    Tensor taf(daf, QType::F32, 2, as, true), tbf(dbf, QType::F32, 2, bs, true);
    gemm(taf, tbf, tc, 1.0f, 0.0f, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    c = download(dc, static_cast<size_t>(M) * N);
    worst = 0.0;
    for (int i = 0; i < M * N; i++)
        worst = std::max(worst, std::fabs(c[i] - r[i]));
    EXPECT_LT(worst, 2e-2) << "FP32 runs TF32 (10-bit mantissa), K 96";
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    cudaFree(daf);
    cudaFree(dbf);
}

}  // namespace
}  // namespace imp
