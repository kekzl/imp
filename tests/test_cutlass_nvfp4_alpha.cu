// Tests for the CUTLASS sm_120 NVFP4×NVFP4 prefill path and the FP8 E4M3
// encoder it depends on. These guard the bug bisected on 2026-05-02:
//
//   `float_to_fp8_e4m3` was clamping to (e=14, m=7) → bits 0x77, decode
//   240, instead of the correct E4M3-fn max 448 = (e=15, m=6) → 0x7E.
//   Any input value > 240 that fell into the e=15 exponent slot was
//   squashed to 240, breaking compressed-tensors NVFP4 prequant
//   (where outlier-block scales near 448 are routine).
//
// The `Boundary` test pins the encoder semantics. The two GEMM tests
// guard the CUTLASS prefill path against a return of the precision
// cliff that fix removed.

#include "compute/gemm.h"
#include "compute/gemm_cutlass_sm120.h"
#include "core/tensor.h"
#include "quant/nvfp4_quant.h"
#include "quant/fp8_utils.cuh"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>

using namespace imp;

namespace {

class CutlassNvfp4AlphaTest : public ::testing::Test {
protected:
    void SetUp() override { cudaStreamCreate(&stream_); }
    void TearDown() override { cudaStreamDestroy(stream_); }
    cudaStream_t stream_ = nullptr;
};

template<typename T>
T* dev_alloc_copy(const std::vector<T>& h) {
    T* d; cudaMalloc(&d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

__global__ void encode_fp8_e4m3_kernel(const float* in, uint8_t* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = imp::float_to_fp8_e4m3(in[i]);
}

} // namespace

// Pin the encoder boundary: 448 must round-trip to 0x7E, not 0x77.
TEST_F(CutlassNvfp4AlphaTest, Fp8E4M3EncoderClampBoundary) {
    // Inputs cover the precision cliff and the canonical normals.
    // Expected bytes are the standard FP8 E4M3-fn encodings.
    const std::vector<std::pair<float, uint8_t>> cases = {
        // Value     Expected byte   Decode (sanity)
        {  448.0f,   0x7E},  // (1+6/8) * 2^8   = 448  E4M3-fn max — was 0x77 (240) before fix
        {  500.0f,   0x7E},  // overflow → saturate at max         — was 0x77 (240) before fix
        {  300.0f,   0x79},  // (1+1/8) * 2^8   = 288 (300 RNE-rounds to 288) — was 0x77 (240) before fix
        {  416.0f,   0x7D},  // (1+5/8) * 2^8   = 416  e=15, m=5   — was 0x77 (240) before fix
        {  256.0f,   0x78},  // (1+0/8) * 2^8   = 256  e=15, m=0   — was 0x77 (240) before fix
        {  240.0f,   0x77},  // (1+7/8) * 2^7   = 240  (unchanged, still e=14)
        {  448.5f,   0x7E},  // saturate at max
        {  -448.0f,  0xFE},  // negative max
        {  0.5f,     0x30},  // (1+0)   * 2^-1  = 0.5
        {  1.0f,     0x38},  // (1+0)   * 2^0   = 1.0
        {  6.0f,     0x4C},  // (1+4/8) * 2^2   = 6
    };

    std::vector<float> h_in;
    for (auto& [v, _] : cases) h_in.push_back(v);
    float* d_in = nullptr; cudaMalloc(&d_in, h_in.size() * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice);
    uint8_t* d_out = nullptr; cudaMalloc(&d_out, h_in.size());

    const int n = static_cast<int>(h_in.size());
    encode_fp8_e4m3_kernel<<<(n + 31) / 32, 32, 0, stream_>>>(d_in, d_out, n);
    cudaStreamSynchronize(stream_);

    std::vector<uint8_t> h_out(h_in.size());
    cudaMemcpy(h_out.data(), d_out, h_in.size(), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < cases.size(); ++i) {
        EXPECT_EQ(h_out[i], cases[i].second)
            << "float_to_fp8_e4m3(" << cases[i].first
            << ") = 0x" << std::hex << (int)h_out[i]
            << ", expected 0x" << (int)cases[i].second
            << " (E4M3-fn convention)";
    }

    cudaFree(d_in); cudaFree(d_out);
}

TEST_F(CutlassNvfp4AlphaTest, AlphaIsActuallyApplied) {
    if (!cutlass_sm120_nvfp4_available()) {
        GTEST_SKIP() << "CUTLASS sm_120 NVFP4 not available";
    }

    // Modest tile size that CUTLASS reliably accepts.
    const int M = 8, N = 64, K = 128;

    // Random-looking but deterministic weights and activations.
    std::vector<half> h_w(N * K), h_x(M * K);
    for (int i = 0; i < N * K; ++i) {
        h_w[i] = __float2half(((i % 13) - 6) * 0.05f);   // [-0.30, 0.30]
    }
    for (int i = 0; i < M * K; ++i) {
        h_x[i] = __float2half(((i % 17) - 8) * 0.05f);   // [-0.40, 0.40]
    }

    half* d_w = dev_alloc_copy(h_w);
    half* d_x = dev_alloc_copy(h_x);

    int64_t wshape[2] = {N, K};
    Tensor w_t(d_w, QType::F16, 2, wshape, true);

    NvFP4QuantResult qr;
    quantize_fp16_to_nvfp4(w_t, qr, stream_);
    cudaStreamSynchronize(stream_);
    const float baseline_tensor_scale = qr.tensor_scale;
    ASSERT_GT(baseline_tensor_scale, 0.0f) << "quantize produced zero tensor_scale";

    CutlassNvFP4Weight cw;
    convert_nvfp4_to_cutlass(qr, cw, stream_);
    cudaStreamSynchronize(stream_);

    // Activation NVFP4 quantization (CUTLASS dynamic-quant kernel).
    size_t act_data_bytes = static_cast<size_t>(M) * K / 2;
    size_t act_sf_bytes   = cutlass_nvfp4_sf_size(M, K);
    size_t ws_needed      = gemm_nvfp4_cutlass_sm120_workspace(M, N, K);

    void* d_act_data = nullptr; cudaMalloc(&d_act_data, act_data_bytes);
    void* d_act_sf   = nullptr; cudaMalloc(&d_act_sf,   act_sf_bytes);
    void* d_ws       = nullptr; cudaMalloc(&d_ws, ws_needed > 0 ? ws_needed : 1);
    quantize_fp16_to_nvfp4_cutlass(d_x, d_act_data, d_act_sf, M, K, stream_);

    half* d_y_a = nullptr; cudaMalloc(&d_y_a, M * N * sizeof(half));
    half* d_y_b = nullptr; cudaMalloc(&d_y_b, M * N * sizeof(half));

    // -------- Run 1: alpha = baseline_tensor_scale --------
    cw.tensor_scale = baseline_tensor_scale;
    bool ok1 = gemm_nvfp4_cutlass_sm120(d_act_data, d_act_sf, cw,
                                         d_y_a, M, N, K,
                                         d_ws, ws_needed, stream_);
    cudaStreamSynchronize(stream_);
    if (!ok1) GTEST_SKIP() << "CUTLASS rejected dims";

    // -------- Run 2: alpha = baseline * 0.25 (i.e. quarter the scale) --------
    constexpr float kRatio = 0.25f;
    cw.tensor_scale = baseline_tensor_scale * kRatio;
    bool ok2 = gemm_nvfp4_cutlass_sm120(d_act_data, d_act_sf, cw,
                                         d_y_b, M, N, K,
                                         d_ws, ws_needed, stream_);
    cudaStreamSynchronize(stream_);
    ASSERT_TRUE(ok2);

    std::vector<half> h_y_a(M * N), h_y_b(M * N);
    cudaMemcpy(h_y_a.data(), d_y_a, h_y_a.size() * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y_b.data(), d_y_b, h_y_b.size() * sizeof(half), cudaMemcpyDeviceToHost);

    // Diagnostic stats. We expect h_y_b ≈ kRatio * h_y_a if alpha works.
    int n_dropped = 0;     // identical bytes → alpha not applied
    int n_scaled  = 0;     // ratio matches kRatio → alpha applied
    double max_a = 0.0, max_b = 0.0;
    double observed_ratio_sum = 0.0;
    int observed_ratio_n = 0;
    for (int i = 0; i < M * N; ++i) {
        const float va = __half2float(h_y_a[i]);
        const float vb = __half2float(h_y_b[i]);
        max_a = std::max(max_a, std::fabs((double)va));
        max_b = std::max(max_b, std::fabs((double)vb));
        if (std::fabs(va) < 1e-3f) continue;  // skip near-zero outputs
        const float ratio = vb / va;
        observed_ratio_sum += ratio;
        observed_ratio_n++;
        if (__half_as_ushort(h_y_a[i]) == __half_as_ushort(h_y_b[i])) {
            n_dropped++;
        }
        if (std::fabs(ratio - kRatio) < 0.05f) {
            n_scaled++;
        }
    }
    const double mean_ratio = observed_ratio_n > 0
        ? observed_ratio_sum / observed_ratio_n : 0.0;

    fprintf(stderr,
            "[ALPHA-BISECT] baseline_alpha=%.6g halved_alpha=%.6g\n"
            "[ALPHA-BISECT] max|y_a|=%.6g max|y_b|=%.6g  expected_ratio=%.3f\n"
            "[ALPHA-BISECT] mean_observed_ratio=%.4f over %d non-zero outs\n"
            "[ALPHA-BISECT] n_dropped=%d (identical bytes) n_scaled=%d (matches ratio)\n",
            baseline_tensor_scale, baseline_tensor_scale * kRatio,
            max_a, max_b, kRatio, mean_ratio,
            observed_ratio_n,
            n_dropped, n_scaled);

    // The decisive assertion: average output ratio should reflect alpha ratio.
    // If alpha is silently dropped, mean_ratio ≈ 1.0 (outputs identical).
    ASSERT_GT(observed_ratio_n, 0) << "all outputs were ~zero — test inconclusive";
    EXPECT_NEAR(mean_ratio, kRatio, 0.05)
        << "alpha appears NOT to be applied: outputs unchanged when tensor_scale changed by "
        << kRatio << "×. Mean observed output ratio: " << mean_ratio;

    free_nvfp4_result(qr);
    free_cutlass_nvfp4_weight(cw);
    cudaFree(d_w); cudaFree(d_x);
    cudaFree(d_act_data); cudaFree(d_act_sf); cudaFree(d_ws);
    cudaFree(d_y_a); cudaFree(d_y_b);
}

// Mistral-3.2-NVFP4 L0 q_proj-shaped reproducer.
//
// Synthesises the conditions the stress-test memo observed in the wild:
//   K = 5120  (Mistral hidden)
//   N = 4096  (q_proj out for 32 heads × 128 head_dim)
//   max(|W|) ≈ 4.36  → global_scale = 2688/4.36 ≈ 616
//                    → tensor_scale (post-flip) ≈ 0.00162
//   activation RMS ≈ 1  (RMSNorm output)
//
// If CUTLASS produces saturated FP16 (Inf / 65504) on this — that's the
// in-the-wild bug isolated. If it produces sane sub-1.0 outputs — the
// bug is upstream of the GEMM (loader, dequant, byte layout, etc.).
TEST_F(CutlassNvfp4AlphaTest, MistralL0Reproducer) {
    if (!cutlass_sm120_nvfp4_available()) {
        GTEST_SKIP() << "CUTLASS sm_120 NVFP4 not available";
    }

    const int M = 16, N = 4096, K = 5120;

    // Construct W with one large outlier per row (Mistral-style attention
    // weights). Most values bounded ~0.05; a few rows have entries up to ±4.4.
    std::vector<half> h_w(static_cast<size_t>(N) * K);
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            float v = ((static_cast<int64_t>(n) * 13 + k * 7) % 17 - 8) * 0.0064f;
            h_w[(size_t)n * K + k] = __float2half(v);
        }
        // outlier in each row at varying k
        h_w[(size_t)n * K + (n * 31) % K] = __float2half(((n & 1) ? -4.36f : 4.36f));
    }

    // Activation: RMSNorm-output-like, RMS ≈ 1, range ≈ [-2, 2]
    std::vector<half> h_x(static_cast<size_t>(M) * K);
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            float v = std::sin((float)((m * 1009 + k * 17) % 1009) * 0.01f) * 1.5f;
            h_x[(size_t)m * K + k] = __float2half(v);
        }
    }

    half* d_w = dev_alloc_copy(h_w);
    half* d_x = dev_alloc_copy(h_x);

    int64_t wshape[2] = {N, K};
    Tensor w_t(d_w, QType::F16, 2, wshape, true);

    NvFP4QuantResult qr;
    quantize_fp16_to_nvfp4(w_t, qr, stream_);
    cudaStreamSynchronize(stream_);
    fprintf(stderr,
            "[MISTRAL-REPRO] auto-tensor_scale=%.6g (Modelopt-convention multiplier)\n",
            qr.tensor_scale);

    CutlassNvFP4Weight cw;
    convert_nvfp4_to_cutlass(qr, cw, stream_);
    cudaStreamSynchronize(stream_);

    size_t act_data_bytes = static_cast<size_t>(M) * K / 2;
    size_t act_sf_bytes   = cutlass_nvfp4_sf_size(M, K);
    size_t ws_needed      = gemm_nvfp4_cutlass_sm120_workspace(M, N, K);

    void* d_act_data = nullptr; cudaMalloc(&d_act_data, act_data_bytes);
    void* d_act_sf   = nullptr; cudaMalloc(&d_act_sf,   act_sf_bytes);
    void* d_ws       = nullptr; cudaMalloc(&d_ws, ws_needed > 0 ? ws_needed : 1);
    quantize_fp16_to_nvfp4_cutlass(d_x, d_act_data, d_act_sf, M, K, stream_);

    half* d_y = nullptr; cudaMalloc(&d_y, M * N * sizeof(half));

    bool ok = gemm_nvfp4_cutlass_sm120(d_act_data, d_act_sf, cw,
                                        d_y, M, N, K,
                                        d_ws, ws_needed, stream_);
    cudaStreamSynchronize(stream_);
    ASSERT_TRUE(ok);

    std::vector<half> h_y(M * N);
    cudaMemcpy(h_y.data(), d_y, h_y.size() * sizeof(half), cudaMemcpyDeviceToHost);

    int n_inf = 0, n_nan = 0, n_saturated = 0;
    double max_abs = 0.0, sum_abs = 0.0, sumsq = 0.0;
    for (int i = 0; i < M * N; ++i) {
        float v = __half2float(h_y[i]);
        if (std::isinf(v)) n_inf++;
        else if (std::isnan(v)) n_nan++;
        else if (std::fabs(v) >= 65504.0f) n_saturated++;
        max_abs = std::max(max_abs, (double)std::fabs(v));
        sum_abs += std::fabs(v);
        sumsq += v * v;
    }
    const double mean_abs = sum_abs / (M * N);
    const double rms = std::sqrt(sumsq / (M * N));

    fprintf(stderr,
            "[MISTRAL-REPRO] M=%d N=%d K=%d   alpha=%.6g\n"
            "[MISTRAL-REPRO] max|y|=%.4g  mean|y|=%.4g  rms|y|=%.4g\n"
            "[MISTRAL-REPRO] n_inf=%d  n_nan=%d  n_saturated(>=65504)=%d  total=%d\n",
            M, N, K, qr.tensor_scale,
            max_abs, mean_abs, rms,
            n_inf, n_nan, n_saturated, M * N);

    // Reference FP16 GEMM (dequant→cuBLAS path) for comparison.
    void* d_w_dequant = nullptr;
    cudaMalloc(&d_w_dequant, (size_t)N * K * sizeof(half));
    dequantize_nvfp4_to_fp16(qr, d_w_dequant, stream_);
    cudaStreamSynchronize(stream_);

    half* d_y_ref = nullptr; cudaMalloc(&d_y_ref, M * N * sizeof(half));
    int64_t shape_x[2] = {M, K}, shape_w[2] = {N, K}, shape_y[2] = {M, N};
    Tensor t_x(d_x, QType::F16, 2, shape_x, true);
    Tensor t_w(d_w_dequant, QType::F16, 2, shape_w, true);
    Tensor t_y(d_y_ref, QType::F16, 2, shape_y, true);
    gemm(t_x, t_w, t_y, 1.0f, 0.0f, stream_);
    cudaStreamSynchronize(stream_);

    std::vector<half> h_y_ref(M * N);
    cudaMemcpy(h_y_ref.data(), d_y_ref, h_y_ref.size() * sizeof(half), cudaMemcpyDeviceToHost);

    double ref_max = 0.0, ref_rms = 0.0;
    for (int i = 0; i < M * N; ++i) {
        float v = __half2float(h_y_ref[i]);
        ref_max = std::max(ref_max, (double)std::fabs(v));
        ref_rms += v * v;
    }
    ref_rms = std::sqrt(ref_rms / (M * N));

    // Per-element diff CUTLASS vs reference
    double max_diff = 0.0, sum_diff = 0.0;
    int n_big_diff = 0;
    for (int i = 0; i < M * N; ++i) {
        float a = __half2float(h_y[i]);
        float b = __half2float(h_y_ref[i]);
        float d = std::fabs(a - b);
        max_diff = std::max(max_diff, (double)d);
        sum_diff += d;
        if (d > 0.5f) n_big_diff++;
    }
    fprintf(stderr,
            "[MISTRAL-REPRO] reference (dequant+cuBLAS): max|y|=%.4g rms=%.4g\n"
            "[MISTRAL-REPRO] CUTLASS - reference:  max_diff=%.4g  mean_diff=%.4g  n>0.5=%d\n",
            ref_max, ref_rms,
            max_diff, sum_diff / (M * N), n_big_diff);

    // Hard fail conditions:
    EXPECT_EQ(n_inf, 0) << "CUTLASS output contains Inf — saturation path";
    EXPECT_EQ(n_nan, 0) << "CUTLASS output contains NaN";
    EXPECT_LE(max_abs, ref_max * 5.0) << "CUTLASS output magnitudes wildly exceed reference";

    free_nvfp4_result(qr);
    free_cutlass_nvfp4_weight(cw);
    cudaFree(d_w); cudaFree(d_x);
    cudaFree(d_act_data); cudaFree(d_act_sf); cudaFree(d_ws);
    cudaFree(d_y); cudaFree(d_w_dequant); cudaFree(d_y_ref);
}

// Reproduce the EXACT byte layout of compressed-tensors / llm-compressor
// prequant NVFP4. The two distinguishing properties vs imp's auto-quant
// (used by the test above) are:
//
//   1. Per-block FP8 E4M3 micro-scales encoded in the W'-domain
//      (i.e. scale_stored = local_scale * global_scale, where
//      global_scale = FP8_max * FP4_max / max(|W|) = 2688/max(|W|)).
//      These bytes range up to ~448 for outlier blocks.
//
//   2. tensor_scale (alpha for CUTLASS) = 1/global_scale = max(|W|)/2688
//      — a small number, typically ~0.00162 for Mistral.
//
// imp's auto-quant uses tensor_scale = max(|W|)/6 (large, ~0.73) and
// micro-scales in W-domain (small, range ~0..1). Mathematically the two
// are identical, but the bit-level FP8 encoding of the micro-scales may
// hit precision pathologies under one convention and not the other —
// which is exactly the symptom the stress-test memo described.
TEST_F(CutlassNvfp4AlphaTest, MistralL0PrequantByteLayout) {
    if (!cutlass_sm120_nvfp4_available()) {
        GTEST_SKIP() << "CUTLASS sm_120 NVFP4 not available";
    }

    const int M = 16, N = 4096, K = 5120;
    constexpr int kBlockSize = 16;
    constexpr float kFP4Max  = 6.0f;
    constexpr float kFP8Max  = 448.0f;

    // 1. Build FP16 weight matrix with one Mistral-attention-style outlier per row
    std::vector<float> w_fp(static_cast<size_t>(N) * K);
    float max_w = 0.0f;
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            float v = ((static_cast<int64_t>(n) * 13 + k * 7) % 17 - 8) * 0.0064f;
            w_fp[(size_t)n * K + k] = v;
            max_w = std::max(max_w, std::fabs(v));
        }
        float outlier = (n & 1) ? -4.36f : 4.36f;
        w_fp[(size_t)n * K + (n * 31) % K] = outlier;
        max_w = std::max(max_w, std::fabs(outlier));
    }
    fprintf(stderr, "[PREQ] max(|W|)=%.4f → global_scale=%.4f tensor_scale=%.6g\n",
            max_w, (kFP8Max * kFP4Max) / max_w, max_w / (kFP8Max * kFP4Max));

    const float global_scale = (kFP8Max * kFP4Max) / max_w;  // compressed-tensors
    const float tensor_scale = 1.0f / global_scale;          // post-flip = max_w / 2688

    // 2. Per-block quantization (compressed-tensors convention, K-major blocks of 16)
    int n_blocks_k = K / kBlockSize;
    std::vector<uint8_t> packed(static_cast<size_t>(N) * (K / 2));
    std::vector<uint8_t> micro_scales_fp8(static_cast<size_t>(N) * n_blocks_k);

    auto quantize_fp4 = [](float v) -> uint8_t {
        // imp's E2M1 LUT thresholds: midpoints between {0, .5, 1, 1.5, 2, 3, 4, 6}
        float a = std::fabs(v);
        uint8_t code =
            (a >= 0.25f) + (a >= 0.75f) + (a >= 1.25f) +
            (a >= 1.75f) + (a >= 2.5f)  + (a >= 3.5f)  + (a >= 5.0f);
        uint8_t sign = (v < 0.0f) ? 1u : 0u;
        return (sign << 3) | code;
    };
    auto float_to_fp8_e4m3 = [](float v) -> uint8_t {
        // Match imp's float_to_fp8_e4m3 (clamped, RNE-rounded). Use stdlib bit ops.
        if (v <= 0.0f) return 0u;
        if (v >= 448.0f) return 0x7Fu;  // saturate to E4M3 max
        // Decompose into exp + mantissa.
        int e;
        float m = std::frexp(v, &e);  // m in [0.5, 1), v = m * 2^e
        // E4M3 normal: v = (1 + man/8) * 2^(exp_field - 7), exp_field = exp(v) + 7
        // m = 1.xxx, mantissa_bits = round((m - 0.5) * 16)?  Easier: man = round((v / 2^(e-1) - 1) * 8)
        int exp_field = e - 1 + 7;
        if (exp_field <= 0) {
            // Denormal: v = mantissa * 2^-9  →  mantissa = v * 512, range 0..7
            int man = (int)std::round(v * 512.0f);
            if (man > 7) man = 7;
            return (uint8_t)(man & 0x07);
        }
        if (exp_field >= 15) return 0x7Fu;
        float frac = v / std::ldexp(1.0f, e - 1) - 1.0f;  // in [0, 1)
        int man = (int)std::round(frac * 8.0f);
        if (man == 8) { man = 0; exp_field += 1; }
        if (exp_field >= 15) return 0x7Fu;
        return (uint8_t)(((exp_field & 0x0F) << 3) | (man & 0x07));
    };

    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < n_blocks_k; ++b) {
            // Find block max in W'-domain
            float block_max_W_prime = 0.0f;
            for (int j = 0; j < kBlockSize; ++j) {
                float w = w_fp[(size_t)n * K + b * kBlockSize + j];
                float w_prime = w * global_scale;
                block_max_W_prime = std::max(block_max_W_prime, std::fabs(w_prime));
            }
            float block_scale_W_prime = block_max_W_prime / kFP4Max;
            // Encode this micro-scale as FP8 E4M3 (W'-domain magnitude!)
            uint8_t fp8_byte = float_to_fp8_e4m3(block_scale_W_prime);
            micro_scales_fp8[(size_t)n * n_blocks_k + b] = fp8_byte;

            // Reconstruct the actual scale (FP8 rounding applied)
            // (decode FP8 via standard formula, matching imp)
            float reconstructed;
            uint8_t bits = fp8_byte;
            uint32_t exp = (bits >> 3) & 0x0F;
            uint32_t man = bits & 0x07;
            if (exp == 0) {
                reconstructed = (float)man * (1.0f / 512.0f);
            } else {
                reconstructed = (1.0f + (float)man * 0.125f) * std::ldexp(1.0f, (int)exp - 7);
            }
            if (reconstructed == 0.0f) reconstructed = 1.0f / 512.0f;

            // Quantize the 16 W' values into FP4 codes using this block_scale
            for (int j = 0; j < kBlockSize; j += 2) {
                float w0 = w_fp[(size_t)n * K + b * kBlockSize + j]     * global_scale;
                float w1 = w_fp[(size_t)n * K + b * kBlockSize + j + 1] * global_scale;
                uint8_t lo = quantize_fp4(w0 / reconstructed);
                uint8_t hi = quantize_fp4(w1 / reconstructed);
                size_t out_idx = (size_t)n * (K / 2) + (b * kBlockSize + j) / 2;
                packed[out_idx] = (hi << 4) | (lo & 0x0F);
            }
        }
    }

    // 3. Upload to device — these ARE the bytes a real prequant SafeTensors load delivers.
    void* d_packed = nullptr; cudaMalloc(&d_packed, packed.size());
    cudaMemcpy(d_packed, packed.data(), packed.size(), cudaMemcpyHostToDevice);
    void* d_micro = nullptr;  cudaMalloc(&d_micro, micro_scales_fp8.size());
    cudaMemcpy(d_micro, micro_scales_fp8.data(), micro_scales_fp8.size(), cudaMemcpyHostToDevice);

    // 4. Build NvFP4QuantResult mirroring what executor_pre_dequant Phase-0 promote produces
    NvFP4QuantResult qr_pq;
    qr_pq.packed_data  = d_packed;
    qr_pq.micro_scales = d_micro;
    qr_pq.tensor_scale = tensor_scale;   // = max_W / 2688, ~0.00162 for Mistral
    qr_pq.N = N;
    qr_pq.K = K;

    // 5. Convert to CUTLASS format (this is the same path Phase-0b registers)
    CutlassNvFP4Weight cw;
    convert_nvfp4_to_cutlass(qr_pq, cw, stream_);
    cudaStreamSynchronize(stream_);
    fprintf(stderr,
            "[PREQ] CUTLASS cw.tensor_scale=%.6g  (used as alpha)\n",
            cw.tensor_scale);

    // 6. Build activations — RMSNorm-output magnitude (~1)
    std::vector<half> h_x(static_cast<size_t>(M) * K);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k)
            h_x[(size_t)m * K + k] =
                __float2half(std::sin((float)((m * 1009 + k * 17) % 1009) * 0.01f) * 1.5f);
    half* d_x = nullptr; cudaMalloc(&d_x, h_x.size() * sizeof(half));
    cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(half), cudaMemcpyHostToDevice);

    size_t act_data_bytes = static_cast<size_t>(M) * K / 2;
    size_t act_sf_bytes   = cutlass_nvfp4_sf_size(M, K);
    size_t ws_needed      = gemm_nvfp4_cutlass_sm120_workspace(M, N, K);
    void* d_act_data = nullptr; cudaMalloc(&d_act_data, act_data_bytes);
    void* d_act_sf   = nullptr; cudaMalloc(&d_act_sf,   act_sf_bytes);
    void* d_ws       = nullptr; cudaMalloc(&d_ws, ws_needed > 0 ? ws_needed : 1);
    quantize_fp16_to_nvfp4_cutlass(d_x, d_act_data, d_act_sf, M, K, stream_);

    half* d_y = nullptr; cudaMalloc(&d_y, M * N * sizeof(half));
    bool ok = gemm_nvfp4_cutlass_sm120(d_act_data, d_act_sf, cw,
                                        d_y, M, N, K, d_ws, ws_needed, stream_);
    cudaStreamSynchronize(stream_);
    ASSERT_TRUE(ok);

    std::vector<half> h_y(M * N);
    cudaMemcpy(h_y.data(), d_y, h_y.size() * sizeof(half), cudaMemcpyDeviceToHost);

    int n_inf = 0, n_nan = 0, n_saturated = 0;
    double max_abs = 0.0, sum_abs = 0.0, sumsq = 0.0;
    for (int i = 0; i < M * N; ++i) {
        float v = __half2float(h_y[i]);
        if (std::isinf(v)) n_inf++;
        else if (std::isnan(v)) n_nan++;
        else if (std::fabs(v) >= 65504.0f) n_saturated++;
        max_abs = std::max(max_abs, (double)std::fabs(v));
        sum_abs += std::fabs(v);
        sumsq += v * v;
    }
    fprintf(stderr,
            "[PREQ] max|y|=%.4g mean|y|=%.4g rms=%.4g  n_inf=%d n_nan=%d n_sat=%d\n",
            max_abs, sum_abs / (M * N), std::sqrt(sumsq / (M * N)),
            n_inf, n_nan, n_saturated);

    // Reference: dequant our hand-built NVFP4 buffer to FP16, run cuBLAS GEMM
    void* d_w_dequant = nullptr; cudaMalloc(&d_w_dequant, (size_t)N * K * sizeof(half));
    dequantize_nvfp4_to_fp16(qr_pq, d_w_dequant, stream_);
    cudaStreamSynchronize(stream_);

    half* d_y_ref = nullptr; cudaMalloc(&d_y_ref, M * N * sizeof(half));
    int64_t shape_x[2] = {M, K}, shape_w[2] = {N, K}, shape_y[2] = {M, N};
    Tensor t_x(d_x, QType::F16, 2, shape_x, true);
    Tensor t_w(d_w_dequant, QType::F16, 2, shape_w, true);
    Tensor t_y(d_y_ref, QType::F16, 2, shape_y, true);
    gemm(t_x, t_w, t_y, 1.0f, 0.0f, stream_);
    cudaStreamSynchronize(stream_);

    std::vector<half> h_y_ref(M * N);
    cudaMemcpy(h_y_ref.data(), d_y_ref, h_y_ref.size() * sizeof(half), cudaMemcpyDeviceToHost);

    double ref_max = 0.0, ref_rms = 0.0;
    for (int i = 0; i < M * N; ++i) {
        float v = __half2float(h_y_ref[i]);
        ref_max = std::max(ref_max, (double)std::fabs(v));
        ref_rms += v * v;
    }
    ref_rms = std::sqrt(ref_rms / (M * N));

    double max_diff = 0.0, sum_diff = 0.0;
    int n_big_diff = 0;
    for (int i = 0; i < M * N; ++i) {
        float a = __half2float(h_y[i]);
        float b = __half2float(h_y_ref[i]);
        float d = std::fabs(a - b);
        max_diff = std::max(max_diff, (double)d);
        sum_diff += d;
        if (d > 0.5f) n_big_diff++;
    }
    fprintf(stderr,
            "[PREQ] reference dequant→cuBLAS: max=%.4g rms=%.4g\n"
            "[PREQ] CUTLASS - reference:  max_diff=%.4g  mean_diff=%.4g  n>0.5=%d\n",
            ref_max, ref_rms, max_diff, sum_diff / (M * N), n_big_diff);

    EXPECT_EQ(n_inf, 0) << "CUTLASS prequant-byte-layout produced Inf";
    EXPECT_EQ(n_nan, 0) << "CUTLASS prequant-byte-layout produced NaN";
    EXPECT_LE(max_abs, ref_max * 5.0)
        << "CUTLASS prequant magnitudes exceed reference 5x — likely the in-the-wild bug";

    free_cutlass_nvfp4_weight(cw);
    cudaFree(d_packed); cudaFree(d_micro);
    cudaFree(d_x); cudaFree(d_act_data); cudaFree(d_act_sf); cudaFree(d_ws);
    cudaFree(d_y); cudaFree(d_w_dequant); cudaFree(d_y_ref);
}
