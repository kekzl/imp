// NVFP4 quantize vs an INDEPENDENT fp64 golden on adversarial weights (TEST_AUDIT retired
// risk #2); prior tests only round-tripped through imp's own code on benign Gaussian data.
// Targets the Gemma mode-2 collapse (#514/#516): an outlier inflates the per-tensor scale
// until normal micro-blocks underflow the 1/512 FP8-E4M3 floor and dequant toward zero,
// ending in NaN logits at decode step 2. Golden built in numpy from the format definition
// (not via imp); covers Gaussian, 64x/512x outliers and all-tiny underflow, with a hard
// no-NaN/Inf gate on every distribution's output.

#include <gtest/gtest.h>
#include "quant/nvfp4_quant.h"
#include "quant/nvfp4_gemm.h"
#include "core/tensor.h"
#include "refs/nvfp4_outlier_golden.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace imp {
namespace {

// Bit-exact mirror of tests/refs/gen_nvfp4_outlier_golden.py::lcg_fill.
// f32 multiply-only transforms (no libm) -> identical f16 bits across langs.
void lcg_fill(std::vector<half>& out, uint32_t seed, float amp, float outlier_mult,
              uint32_t outlier_period, bool tiny) {
    uint32_t x = seed;
    const float inv = 1.0f / 8192.0f;
    const float tinyf = 1.0f / 16384.0f;
    for (auto& h : out) {
        x = x * 1664525u + 1013904223u;
        int32_t v = (int32_t)((x >> 8) & 0x3FFFu) - 8192;
        float f = (float)v * inv;
        float val = f * f * f * amp;
        if (outlier_period > 1 && (x % outlier_period) == 0u)
            val *= outlier_mult;
        if (tiny)
            val *= tinyf;
        h = __float2half(val);
    }
}

struct Dist {
    const char* name;
    int N, K;
    float amp, outlier_mult;
    uint32_t outlier_period;
    bool tiny;
    uint32_t w_seed, x_seed;
    const int* dq_idx;
    const double* dq_val;
    const int* gemv_idx;
    const double* gemv_val;
};

// Matches CONFIGS in the generator: w_seed = 0x2468 + i*7, x_seed = 0xACE0 + i*7.
std::vector<Dist> all_dists() {
    using namespace imp_refs;
    return {
        {"gaussian", 64, 256, 1.0f, 1.0f, 1, false, 0x2468 + 0 * 7, 0xACE0 + 0 * 7, gaussian_dq_spot_idx,
         gaussian_dq_spot_val, gaussian_gemv_spot_idx, gaussian_gemv_spot_val},
        {"gemma_outlier_64x", 64, 256, 1.0f, 64.0f, 256, false, 0x2468 + 1 * 7, 0xACE0 + 1 * 7,
         gemma_outlier_64x_dq_spot_idx, gemma_outlier_64x_dq_spot_val, gemma_outlier_64x_gemv_spot_idx,
         gemma_outlier_64x_gemv_spot_val},
        {"extreme_outlier_512x", 64, 256, 1.0f, 512.0f, 4096, false, 0x2468 + 2 * 7, 0xACE0 + 2 * 7,
         extreme_outlier_512x_dq_spot_idx, extreme_outlier_512x_dq_spot_val,
         extreme_outlier_512x_gemv_spot_idx, extreme_outlier_512x_gemv_spot_val},
        {"all_tiny", 64, 256, 1.0f, 1.0f, 1, true, 0x2468 + 3 * 7, 0xACE0 + 3 * 7, all_tiny_dq_spot_idx,
         all_tiny_dq_spot_val, all_tiny_gemv_spot_idx, all_tiny_gemv_spot_val},
    };
}

constexpr int kNSpots = 48;
constexpr float kNvfp4RelTol = 1e-1f;  // NVFP4 single-op class tolerance (README §2)

// Relative error against an fp64 reference; denom floored at 1 so tiny refs
// (the all_tiny GEMV ~1e-4) are judged on absolute scale, not blown up.
float rel_err(double got, double ref) {
    double denom = std::max(1.0, std::fabs(ref));
    return (float)(std::fabs(got - ref) / denom);
}

struct SpotStats {
    float max_rel = 0.0f;
    int above_tol = 0;
};

class NVFP4OutlierTest : public ::testing::Test {
protected:
    void SetUp() override { ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess); }
    void TearDown() override { cudaStreamDestroy(stream_); }
    cudaStream_t stream_ = nullptr;
};

// Quantize a distribution's weight tensor with imp's REAL production
// quantizer, dequant it back, and run the REAL decode GEMV. Returns the
// host dequant [N*K] and GEMV [N] outputs.
void run_pipeline(cudaStream_t stream, const Dist& d, std::vector<float>& dq_out,
                  std::vector<float>& gemv_out) {
    const int N = d.N, K = d.K;

    std::vector<half> h_w((size_t)N * K);
    lcg_fill(h_w, d.w_seed, d.amp, d.outlier_mult, d.outlier_period, d.tiny);
    std::vector<half> h_x((size_t)K);
    lcg_fill(h_x, d.x_seed, 1.0f, 1.0f, 1, false);

    half* d_w = nullptr;
    half* d_x = nullptr;
    ASSERT_EQ(cudaMalloc(&d_w, (size_t)N * K * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, (size_t)K * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_w, h_w.data(), (size_t)N * K * sizeof(half), cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_x, h_x.data(), (size_t)K * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);

    int64_t shape[] = {N, K};
    Tensor w(d_w, QType::F16, 2, shape, true);

    // REAL production quantizer (the mode-2 conversion calls this exact path).
    NvFP4QuantResult q;
    quantize_fp16_to_nvfp4(w, q, stream);

    // Dequant back to FP16 via the production dequant kernel.
    half* d_dq = nullptr;
    ASSERT_EQ(cudaMalloc(&d_dq, (size_t)N * K * sizeof(half)), cudaSuccess);
    dequantize_nvfp4_to_fp16(q, d_dq, stream);

    // REAL decode GEMV: y[N] = dequant(W)[N,K] @ x[K].
    half* d_y = nullptr;
    ASSERT_EQ(cudaMalloc(&d_y, (size_t)N * sizeof(half)), cudaSuccess);
    gemv_nvfp4_kpar(q, d_x, d_y, N, K, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<half> h_dq((size_t)N * K), h_y((size_t)N);
    ASSERT_EQ(cudaMemcpy(h_dq.data(), d_dq, (size_t)N * K * sizeof(half), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_y.data(), d_y, (size_t)N * sizeof(half), cudaMemcpyDeviceToHost), cudaSuccess);

    dq_out.resize((size_t)N * K);
    for (size_t i = 0; i < dq_out.size(); i++)
        dq_out[i] = __half2float(h_dq[i]);
    gemv_out.resize((size_t)N);
    for (size_t i = 0; i < gemv_out.size(); i++)
        gemv_out[i] = __half2float(h_y[i]);

    free_nvfp4_result(q);
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_dq);
    cudaFree(d_y);
}

// The Gemma-class guard: NO element of the dequant or GEMV output may be
// non-finite, on ANY distribution. Value error is tolerated (1e-1 class);
// a single NaN/Inf is the failure that turned into <pad> argmax.
int count_nonfinite(const std::vector<float>& v) {
    int n = 0;
    for (float x : v)
        if (!std::isfinite(x))
            n++;
    return n;
}

SpotStats check_spots(const std::vector<float>& got, const int* idx, const double* ref, const char* what,
                      const char* dist) {
    SpotStats s;
    for (int k = 0; k < kNSpots; k++) {
        float e = rel_err((double)got[idx[k]], ref[k]);
        s.max_rel = std::max(s.max_rel, e);
        if (e > kNvfp4RelTol) {
            s.above_tol++;
            // Print the worst offenders for the record.
            if (s.above_tol <= 6)
                printf("  [%s/%s] spot %d idx %d: got %.6g ref %.6g rel %.4g\n", dist, what, k, idx[k],
                       (double)got[idx[k]], ref[k], e);
        }
    }
    return s;
}

}  // namespace

// Each distribution: quantize->dequant + GEMV vs independent fp64 golden,
// plus the hard no-NaN/Inf guard.
TEST_F(NVFP4OutlierTest, QuantDequantGemvVsIndependentRef) {
    for (const Dist& d : all_dists()) {
        std::vector<float> dq, gemv;
        run_pipeline(stream_, d, dq, gemv);

        // (1) Hard guard: no non-finite output, ANY distribution. This is the
        //     direct catch for the Gemma collapse class (NaN -> <pad>).
        int dq_nf = count_nonfinite(dq);
        int gemv_nf = count_nonfinite(gemv);
        EXPECT_EQ(dq_nf, 0) << "[" << d.name << "] " << dq_nf << " non-finite dequant elements";
        EXPECT_EQ(gemv_nf, 0) << "[" << d.name << "] " << gemv_nf << " non-finite GEMV elements";

        // (2) Spot values vs the independent fp64 reference at NVFP4 1e-1 rel.
        SpotStats sdq = check_spots(dq, d.dq_idx, d.dq_val, "dequant", d.name);
        SpotStats sgv = check_spots(gemv, d.gemv_idx, d.gemv_val, "gemv", d.name);

        printf("[%-20s] dequant: max_rel=%.4g above_tol=%d/%d | gemv: max_rel=%.4g above_tol=%d/%d\n",
               d.name, sdq.max_rel, sdq.above_tol, kNSpots, sgv.max_rel, sgv.above_tol, kNSpots);

        EXPECT_EQ(sdq.above_tol, 0) << "[" << d.name << "] dequant: " << sdq.above_tol
                                    << " spots exceed " << kNvfp4RelTol << " rel (max " << sdq.max_rel
                                    << ") vs independent fp64 NVFP4 reference";
        EXPECT_EQ(sgv.above_tol, 0) << "[" << d.name << "] gemv: " << sgv.above_tol << " spots exceed "
                                    << kNvfp4RelTol << " rel (max " << sgv.max_rel
                                    << ") vs independent fp64 GEMV reference";
    }
}

// Extreme-outlier floor collapse stays FINITE at the raw dequant: normal blocks underflow
// the 1/512 micro-scale floor onto a coarse grid, but the Gemma NaN is a downstream
// forward-pass artifact, not a raw dequant one. Floor-collapse VALUE behavior reproduces in
// isolation; the NaN-logit endpoint needs multi-layer accumulation and does not.
TEST_F(NVFP4OutlierTest, ExtremeOutlierFloorIsFiniteAndCoarse) {
    Dist d = all_dists()[2];  // extreme_outlier_512x
    std::vector<float> dq, gemv;
    run_pipeline(stream_, d, dq, gemv);

    EXPECT_EQ(count_nonfinite(dq), 0) << "extreme-outlier dequant must stay finite";
    EXPECT_EQ(count_nonfinite(gemv), 0) << "extreme-outlier GEMV must stay finite";

    // Floor makes normal blocks quantize on a tensor_scale/512 * {E2M1} grid: global_absmax
    // ~393.5, tensor_scale ~65.6, coarsest nonzero step ~65.6*(1/512)*0.5 = 0.064
    // (golden values are multiples of ~0.064).
    SpotStats sdq = check_spots(dq, d.dq_idx, d.dq_val, "dequant", d.name);
    EXPECT_EQ(sdq.above_tol, 0) << "floor-collapse dequant grid diverges from independent ref (max "
                                << sdq.max_rel << ")";
}

}  // namespace imp
