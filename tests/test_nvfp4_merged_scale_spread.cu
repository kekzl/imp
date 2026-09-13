// Sibling projections in one small-M launch (FFN gate|up, GDN in|z, q|k|v) keep separate
// weight_global_scale per compressed-tensors tensor (3.7x measured spread); requantizing
// onto a shared scale is the bug this guards. Test gives siblings a 4x amax spread on purpose.
// MUTANT (must fail): alias W1.q.tensor_scale to W0.q.tensor_scale -> sibling 1's block
// comes out 4x small.

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "core/tensor.h"
#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_quant.h"

namespace {

bool gpu_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

// Plain packed NVFP4 element per OCP spec (low nibble=even k, E2M1 + UE4M3 micro-scale per
// 16 values), same formula as test_nvfp4_compressed_tensors_ref.cu but independent of the
// device decoder so a paired bug cannot hide.
float host_dequant_spec(const std::vector<uint8_t>& packed, const std::vector<uint8_t>& scales, int n, int k,
                        int K) {
    static const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const uint8_t byte = packed[static_cast<size_t>(n) * (K / 2) + k / 2];
    const uint8_t nib = (k & 1) ? (byte >> 4) : (byte & 0x0F);
    const float w = (nib & 0x08) ? -mag[nib & 0x07] : mag[nib & 0x07];
    const uint8_t se = scales[static_cast<size_t>(n) * (K / 16) + k / 16];
    const int exp = (se >> 3) & 0x0F;
    const int man = se & 0x07;
    const float sf = (exp == 0) ? (static_cast<float>(man) / 8.0f) * std::ldexp(1.0f, -6)
                                : (1.0f + static_cast<float>(man) / 8.0f) * std::ldexp(1.0f, exp - 7);
    return w * sf;
}

// imp's own quantizer produces the packed pair plus the per-tensor scale; the
// spread comes from the SOURCE amax, exactly as it does in a checkpoint.
struct DeviceQuant {
    imp::NvFP4QuantResult q{};
    std::vector<uint8_t> packed_h, scales_h;
    void quantize(const std::vector<__half>& src, int rows, int K) {
        void* d = nullptr;
        ASSERT_EQ(cudaMalloc(&d, src.size() * sizeof(__half)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d, src.data(), src.size() * sizeof(__half), cudaMemcpyHostToDevice),
                  cudaSuccess);
        int64_t shp[2] = {rows, K};
        imp::Tensor t(d, imp::QType::F16, 2, shp, /*on_device=*/true);
        imp::quantize_fp16_to_nvfp4(t, q, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_NE(q.packed_data, nullptr);
        cudaFree(d);
        packed_h.resize(static_cast<size_t>(rows) * K / 2);
        scales_h.resize(static_cast<size_t>(rows) * K / 16);
        ASSERT_EQ(cudaMemcpy(packed_h.data(), q.packed_data, packed_h.size(), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(scales_h.data(), q.micro_scales, scales_h.size(), cudaMemcpyDeviceToHost),
                  cudaSuccess);
    }
};

class NvFP4MergedScaleSpread : public ::testing::Test {
protected:
    void SetUp() override {
        if (!gpu_available())
            GTEST_SKIP() << "no CUDA device";
    }
};

}  // namespace

TEST_F(NvFP4MergedScaleSpread, SiblingsUseTheirOwnGlobalScale) {
    // N per sibling x 2 = 5120 = 80 n-tiles, the single-stripe threshold the
    // multi-sibling launch requires; K = 256 is one pipeline stage.
    constexpr int M = 8, N = 2560, K = 256;
    constexpr float kSpread = 4.0f;

    std::mt19937 rng(1960);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<__half> w0_h(static_cast<size_t>(N) * K), w1_h(static_cast<size_t>(N) * K),
        x_h(static_cast<size_t>(M) * K);
    for (size_t i = 0; i < w0_h.size(); ++i) {
        const float v = dist(rng);
        w0_h[i] = __float2half(v);
        w1_h[i] = __float2half(kSpread * v);  // same shape, 4x the amax
    }
    for (auto& v : x_h)
        v = __float2half(dist(rng));

    DeviceQuant W0, W1, X;
    W0.quantize(w0_h, N, K);
    W1.quantize(w1_h, N, K);
    X.quantize(x_h, M, K);

    // Without a real spread the test proves nothing, so pin it.
    ASSERT_GT(W1.q.tensor_scale, W0.q.tensor_scale * 3.0f)
        << "fixture lost the amax spread: ts0=" << W0.q.tensor_scale << " ts1=" << W1.q.tensor_scale;

    void *d_y0 = nullptr, *d_y1 = nullptr;
    ASSERT_EQ(cudaMalloc(&d_y0, static_cast<size_t>(M) * N * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y1, static_cast<size_t>(M) * N * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_y0, 0, static_cast<size_t>(M) * N * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_y1, 0, static_cast<size_t>(M) * N * sizeof(__half)), cudaSuccess);

    const imp::SmallMV2Sibling sib[2] = {{&W0.q, static_cast<half*>(d_y0), N},
                                         {&W1.q, static_cast<half*>(d_y1), N}};
    ASSERT_TRUE(imp::gemm_nvfp4_smallm_v2_multi_a4(sib, 2, X.q, M, K, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<__half> y0(static_cast<size_t>(M) * N), y1(static_cast<size_t>(M) * N);
    ASSERT_EQ(cudaMemcpy(y0.data(), d_y0, y0.size() * sizeof(__half), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(y1.data(), d_y1, y1.size() * sizeof(__half), cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(d_y0);
    cudaFree(d_y1);

    // Each block against a reference built from that block's OWN tensor scale.
    const DeviceQuant* w[2] = {&W0, &W1};
    const std::vector<__half>* y[2] = {&y0, &y1};
    for (int s = 0; s < 2; ++s) {
        const float ts = w[s]->q.tensor_scale * X.q.tensor_scale;
        double max_rel = 0.0;
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                double ref = 0.0;
                for (int k = 0; k < K; ++k)
                    ref += static_cast<double>(host_dequant_spec(w[s]->packed_h, w[s]->scales_h, n, k, K)) *
                           host_dequant_spec(X.packed_h, X.scales_h, m, k, K);
                ref *= ts;
                const double got = __half2float((*y[s])[static_cast<size_t>(m) * N + n]);
                max_rel = std::max(max_rel, std::abs(got - ref) / std::max(1.0, std::abs(ref)));
            }
        }
        // Same envelope as test_nvfp4_smallm_v2: exact FP4xUE4M3 products, FP32
        // accumulate, one FP16 round at the end.
        EXPECT_LT(max_rel, 2e-3) << "sibling " << s << " (tensor_scale " << w[s]->q.tensor_scale
                                 << ") diverges from its own-scale reference: max rel " << max_rel;
    }

    // The failure this exists for is silent, so state the separation the two
    // blocks must show: sibling 1 is 4x sibling 0, element for element.
    double max_ratio_err = 0.0;
    for (size_t i = 0; i < y0.size(); ++i) {
        const double a = __half2float(y0[i]);
        if (std::abs(a) < 0.5)
            continue;  // ratio of two near-zero FP16 values says nothing
        max_ratio_err = std::max(max_ratio_err, std::abs(__half2float(y1[i]) / a - kSpread) / kSpread);
    }
    EXPECT_LT(max_ratio_err, 5e-2) << "the two blocks did not separate by the amax spread; a shared "
                                      "global scale would collapse them onto each other";
}
