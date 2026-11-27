// The drafters' shared norm primitives (audit D-14): head_norm_rope_fp16 (DSpark, per-row positions,
// fp16 weights) and its _seq form (Inkling's draft, pos0 + row, fp32 weights) are one kernel; rms_norm_rows
// serves Inkling (fp32 weights) and DSpark's fc norm (fp16 weights). Equal inputs give equal bits.

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "compute/head_norm_rope.h"
#include "compute/rms_norm_rows.h"

using namespace imp;

namespace {

uint32_t g_seed = 12345u;
float rnd() {
    g_seed = g_seed * 1664525u + 1013904223u;
    return static_cast<float>(g_seed >> 8) / 16777216.0f * 2.0f - 1.0f;
}

template <typename T>
T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    EXPECT_EQ(cudaMalloc(&p, h.size() * sizeof(T)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    return p;
}

template <typename T>
std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    EXPECT_EQ(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    return h;
}

bool same_bits(const void* a, const void* b, size_t bytes) { return std::memcmp(a, b, bytes) == 0; }

}  // namespace

// Inkling's draft calls: positions pos0 + *d_pos0 + row, fp32 weights holding fp16 values; DSpark's: the
// per-row table, the same weights as fp16. Both arms, with and without the norm.
TEST(DrafterNorms, HeadNormRopeSeqEqualsPerRowPositions) {
    constexpr int kRows = 5, kHeads = 3, kStride = kHeads * kHeadNormRopeDim + 64;
    std::vector<half> x(static_cast<size_t>(kRows) * kStride);
    for (auto& v : x)
        v = __float2half(rnd() * 4.0f);
    std::vector<half> wh(kHeadNormRopeDim);
    std::vector<float> wf(kHeadNormRopeDim);
    for (int i = 0; i < kHeadNormRopeDim; ++i) {
        wh[static_cast<size_t>(i)] = __float2half(1.0f + rnd() * 0.5f);
        wf[static_cast<size_t>(i)] = __half2float(wh[static_cast<size_t>(i)]);
    }
    constexpr int kPos0 = 1000, kBase = 37;
    std::vector<int> rows(kRows);
    for (int r = 0; r < kRows; ++r)
        rows[static_cast<size_t>(r)] = kPos0 + kBase + r;
    const float log2_theta = std::log2(1000000.0f), eps = 1e-6f;
    half* xa = dev(x);
    half* xb = dev(x);
    half* d_wh = dev(wh);
    float* d_wf = dev(wf);
    int* d_rows = dev(rows);
    int* d_base = dev(std::vector<int>{kBase});
    for (const bool norm : {true, false}) {
        head_norm_rope_fp16(xa, kRows, kStride, kHeads, norm ? d_wh : nullptr, d_rows, log2_theta, eps,
                            nullptr);
        head_norm_rope_fp16_seq(xb, kRows, kStride, kHeads, norm ? d_wf : nullptr, kPos0, d_base, log2_theta,
                                eps, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto a = host(xa, x.size()), b = host(xb, x.size());
        EXPECT_TRUE(same_bits(a.data(), b.data(), a.size() * sizeof(half))) << (norm ? "norm" : "rope only");
        EXPECT_FALSE(same_bits(a.data(), x.data(), a.size() * sizeof(half)));
    }
    for (void* p : {static_cast<void*>(xa), static_cast<void*>(xb), static_cast<void*>(d_wh),
                    static_cast<void*>(d_wf), static_cast<void*>(d_rows), static_cast<void*>(d_base)})
        cudaFree(p);
}

// fp16 and fp32 weights of the same values give the same bits, in place (Inkling's embedding norm) as
// out of place, and the rows match a double reference.
TEST(DrafterNorms, RmsNormRowsWeightTypesAgreeAndMatchReference) {
    constexpr int kRows = 4, kH = 2304;
    std::vector<float> x(static_cast<size_t>(kRows) * kH);
    for (auto& v : x)
        v = rnd() * 50.0f;
    std::vector<half> wh(kH);
    std::vector<float> wf(kH);
    for (int i = 0; i < kH; ++i) {
        wh[static_cast<size_t>(i)] = __float2half(0.5f + rnd());
        wf[static_cast<size_t>(i)] = __half2float(wh[static_cast<size_t>(i)]);
    }
    const float eps = 1e-6f;
    float* d_x = dev(x);
    float* d_inplace = dev(x);
    half* d_wh = dev(wh);
    float* d_wf = dev(wf);
    float *of_h = nullptr, *of_f = nullptr;
    half *oh_h = nullptr, *oh_f = nullptr;
    ASSERT_EQ(cudaMalloc(&of_h, x.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&of_f, x.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&oh_h, x.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&oh_f, x.size() * sizeof(half)), cudaSuccess);
    rms_norm_rows(d_x, d_wh, of_h, oh_h, kRows, kH, eps, nullptr);
    rms_norm_rows(d_x, d_wf, of_f, oh_f, kRows, kH, eps, nullptr);
    rms_norm_rows(d_inplace, d_wf, d_inplace, nullptr, kRows, kH, eps, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto fh = host(of_h, x.size()), ff = host(of_f, x.size()), fi = host(d_inplace, x.size());
    const auto hh = host(oh_h, x.size()), hf = host(oh_f, x.size());
    EXPECT_TRUE(same_bits(fh.data(), ff.data(), fh.size() * sizeof(float)));
    EXPECT_TRUE(same_bits(ff.data(), fi.data(), ff.size() * sizeof(float)));
    EXPECT_TRUE(same_bits(hh.data(), hf.data(), hh.size() * sizeof(half)));
    for (int r = 0; r < kRows; ++r) {
        double ss = 0.0;
        for (int i = 0; i < kH; ++i)
            ss += static_cast<double>(x[static_cast<size_t>(r) * kH + i]) *
                  x[static_cast<size_t>(r) * kH + i];
        const double inv = 1.0 / std::sqrt(ss / kH + eps);
        for (int i = 0; i < kH; i += 97) {
            const size_t k = static_cast<size_t>(r) * kH + i;
            const double ref = x[k] * inv * wf[static_cast<size_t>(i)];
            EXPECT_NEAR(ff[k], ref, 1e-5 * std::fabs(ref) + 1e-6) << "row " << r << " col " << i;
        }
    }
    for (void* p : {static_cast<void*>(d_x), static_cast<void*>(d_inplace), static_cast<void*>(d_wh),
                    static_cast<void*>(d_wf), static_cast<void*>(of_h), static_cast<void*>(of_f),
                    static_cast<void*>(oh_h), static_cast<void*>(oh_f)})
        cudaFree(p);
}
