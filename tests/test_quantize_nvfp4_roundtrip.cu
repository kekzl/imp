// The export round trip imp-quantize never had: quantize a tiny model the way
// the tool does, decode the written bytes by the format's own rule, and put a
// number on what came back.
//
// tests/test_quantize_checkpoint_out.cpp covers the FORMAT rules (which key
// holds the reciprocal, which tensors share a scale) and asserts nothing
// numeric; tests/test_awq_calibration.cpp proves the transform's algebra on
// exact arithmetic. Between the two sat the case that matters: a checkpoint
// whose bytes are structurally perfect and numerically wrong. A reciprocal
// written the wrong way round is off by absmax^2/36 and still loads.
//
// Second half: the unit-offset norm fold (Qwen3.5 / 3.8). The stored delta is
// (1 + g)/s - 1, so the gain the loader reconstructs is (1 + g') and the
// consumer's columns carry s. This asserts the product survives BF16 storage,
// including on the channel that made the clamp necessary.

#include "../tools/imp-quantize/checkpoint_out.h"
#include "../tools/imp-quantize/quant_report.h"

#include "core/tensor.h"
#include "quant/awq_norm_fold.h"
#include "quant/awq_transform.h"
#include "quant/nvfp4_quant.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace imp {
namespace {

// core/fp_bits.h is host-only (its static_asserts call std::ldexp, which nvcc
// will not fold), so the two conversions this file needs are written out. The
// bf16 rule is the same round-to-nearest-even on the discarded 16 bits.
float half_to_f32(uint16_t h) {
    __half_raw raw;
    raw.x = h;
    return __half2float(__half(raw));
}

uint16_t f32_to_half(float f) { return __half_raw(__float2half(f)).x; }

float bf16_to_f32(uint16_t b) {
    const uint32_t bits = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

uint16_t f32_to_bf16(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return static_cast<uint16_t>((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
}

struct Packed {
    std::vector<unsigned char> packed;  // [N, K/2]
    std::vector<unsigned char> micro;   // [N, K/16]
    float tensor_scale = 1.0f;
    std::vector<uint16_t> dequantized;  // [N, K] FP16, imp's own decode
};

// What the exporter does to one matrix: absmax/6 as the tensor scale, the real
// kernel, and everything copied back to the host.
Packed quantize_like_the_exporter(const std::vector<uint16_t>& h_fp16, int64_t N, int64_t K) {
    Packed out;
    const size_t bytes = static_cast<size_t>(N) * static_cast<size_t>(K) * 2;
    void* d_in = nullptr;
    void* d_deq = nullptr;
    EXPECT_EQ(cudaMalloc(&d_in, bytes), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&d_deq, bytes), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d_in, h_fp16.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    int64_t shape[2] = {N, K};
    Tensor in(d_in, QType::F16, 2, shape, /*on_device=*/true);
    const float scale = quantize::export_tensor_scale(quantize::fp16_absmax(h_fp16.data(), h_fp16.size()));
    NvFP4QuantResult q;
    quantize_fp16_to_nvfp4_with_scale(in, scale, q);
    dequantize_nvfp4_to_fp16(q, d_deq);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    out.tensor_scale = q.tensor_scale;
    out.packed.resize(static_cast<size_t>(N) * static_cast<size_t>(K / 2));
    out.micro.resize(static_cast<size_t>(N) * static_cast<size_t>(K / 16));
    out.dequantized.resize(static_cast<size_t>(N) * static_cast<size_t>(K));
    EXPECT_EQ(cudaMemcpy(out.packed.data(), q.packed_data, out.packed.size(), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(cudaMemcpy(out.micro.data(), q.micro_scales, out.micro.size(), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(cudaMemcpy(out.dequantized.data(), d_deq, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
    free_nvfp4_result(q);
    cudaFree(d_in);
    cudaFree(d_deq);
    return out;
}

}  // namespace

// Two layers, K = 256: quantize -> decode by the format -> bound the error.
// The bound is loose on purpose. It is not a quality claim about NVFP4; it is
// the line between "this checkpoint is the model" and "this checkpoint is
// noise", and every layout bug lands far on the wrong side of it.
TEST(NvFP4ExportRoundTrip, TinyModelDecodesWithinTheGridBound) {
    constexpr int64_t N = 64, K = 256;
    std::mt19937 rng(20260909);
    std::normal_distribution<float> nd(0.0f, 0.02f);

    for (int layer = 0; layer < 2; layer++) {
        std::vector<uint16_t> w(static_cast<size_t>(N * K));
        for (auto& v : w)
            v = f32_to_half(nd(rng));
        // One loud channel per layer, which is what the micro-scales exist for.
        for (int64_t i = 0; i < N; i++)
            w[static_cast<size_t>(i * K + 7)] = f32_to_half(0.4f);

        const Packed q = quantize_like_the_exporter(w, N, K);
        const quantize::TensorError e = quantize::nvfp4_tensor_error("layer" + std::to_string(layer) + ".w",
                                                                     w.data(), q.packed.data(),
                                                                     q.micro.data(), q.tensor_scale, N, K);

        // FP4 E2M1's widest gap is 4 -> 6 against a block absmax of 6, so half
        // a step is 1/6 of the block's own scale; the FP8 micro-scale adds its
        // own 2^-3 relative. A wrong reciprocal or a swapped nibble order lands
        // orders of magnitude past this.
        EXPECT_LT(e.max_rel, 0.30) << "layer " << layer;
        EXPECT_GT(e.max_rel, 0.0) << "layer " << layer
                                  << ": a 4-bit grid is not lossless, so an exact "
                                     "zero means the comparison did not happen";
        EXPECT_LT(std::sqrt(e.mse), 0.05 * 0.4) << "layer " << layer;
    }
}

// The fold the qwen3_5 family needs, through the quantizer rather than on
// paper: the divided norm and the scaled columns must reproduce the layer the
// checkpoint started as.
TEST(NvFP4ExportRoundTrip, OffsetFoldedNormReproducesTheEffectiveGain) {
    constexpr int64_t N = 32, K = 256;
    constexpr float kTol = 0.01f;
    std::mt19937 rng(4242);
    std::normal_distribution<float> nd(0.0f, 0.05f);

    std::vector<uint16_t> w(static_cast<size_t>(N * K));
    for (auto& v : w)
        v = f32_to_half(nd(rng));

    // A unit-offset norm: stored deltas near 0, gains near 1, plus the real
    // worst channel of Qwen3.8-27B (gain 2^-8) that BF16 cannot fold at s = 2.
    std::vector<uint16_t> norm(static_cast<size_t>(K));
    for (int64_t j = 0; j < K; j++)
        norm[static_cast<size_t>(j)] = f32_to_bf16(nd(rng));
    norm[3] = f32_to_bf16(-0.99609375f);

    std::vector<float> gain_before(static_cast<size_t>(K));
    for (int64_t j = 0; j < K; j++)
        gain_before[static_cast<size_t>(j)] = awq_norm_gain(bf16_to_f32(norm[static_cast<size_t>(j)]),
                                                            NormOffset::Unit);

    // AWQ-shaped divisors: a decade of spread, both sides of 1.
    std::vector<float> div(static_cast<size_t>(K));
    for (int64_t j = 0; j < K; j++)
        div[static_cast<size_t>(j)] = 0.5f + 0.25f * static_cast<float>(j % 7);

    std::vector<float> source(static_cast<size_t>(K));
    for (int64_t j = 0; j < K; j++)
        source[static_cast<size_t>(j)] = bf16_to_f32(norm[static_cast<size_t>(j)]);
    NormFoldReport rep;
    awq_clamp_norm_divisors(source.data(), source.size(), NormOffset::Unit, "BF16", kTol, div, rep);
    EXPECT_GE(rep.clamped, 1u) << "the 2^-8 gain channel must have been clamped";

    std::vector<uint16_t> folded = norm;
    ASSERT_TRUE(awq_apply_vector_div(reinterpret_cast<unsigned char*>(folded.data()), static_cast<size_t>(K),
                                     "BF16", div, NormOffset::Unit));

    // Every channel's effective gain must survive the fold, including the one
    // the clamp had to protect: (1 + g') * s == (1 + g).
    for (int64_t j = 0; j < K; j++) {
        const float after = awq_norm_gain(bf16_to_f32(folded[static_cast<size_t>(j)]), NormOffset::Unit) *
                            div[static_cast<size_t>(j)];
        const float want = gain_before[static_cast<size_t>(j)];
        EXPECT_NEAR(after, want, kTol * std::fabs(want) + 1e-7f) << "channel " << j;
    }

    // And the consumer, quantized with the columns the plan scaled, must still
    // compute the same layer. The tolerance is the quantizer's, not the fold's:
    // the fold contributes at most kTol per channel.
    std::vector<uint16_t> scaled = w;
    awq_apply_matrix(scaled, N, K, /*row_div=*/{}, /*col_scale=*/div);
    const Packed q = quantize_like_the_exporter(scaled, N, K);

    std::vector<float> x(static_cast<size_t>(K));
    for (int64_t j = 0; j < K; j++)
        x[static_cast<size_t>(j)] = 0.5f + 0.1f * static_cast<float>(j % 11);

    for (int64_t i = 0; i < N; i++) {
        double ref = 0.0, got = 0.0, term_mag = 0.0;
        for (int64_t j = 0; j < K; j++) {
            const size_t idx = static_cast<size_t>(i * K + j);
            const double xin = double(x[static_cast<size_t>(j)]);
            const double src = double(half_to_f32(w[idx])) * xin *
                               double(gain_before[static_cast<size_t>(j)]);
            ref += src;
            got += double(half_to_f32(q.dequantized[idx])) * xin *
                   double(awq_norm_gain(bf16_to_f32(folded[static_cast<size_t>(j)]), NormOffset::Unit));
            term_mag += std::fabs(src);
        }
        // Everything the quantizer rounds is inside these terms, so the bound
        // is over their magnitudes: dividing by |y| would measure how much the
        // dot product cancels instead.
        EXPECT_LT(std::fabs(got - ref), 0.20 * term_mag) << "row " << i;
    }
}

}  // namespace imp
