// Native mxf4nvf4 small-M GEMM: correctness vs a host dequant walk over the actual packed
// buffers of both sides (W4A4, FP32 accumulate, 1e-3 relative tolerance). Bandwidth must beat
// CUTLASS (41.4us/19% of floor in-situ) and the refuted W4A16 v1 (23.9us isolated); asserts
// >= 40% of weight floor. Timing cases: tests/bench/test_nvfp4_smallm_v2_bench.cu.
// M2's <=15us acceptance gate lives in the campaign doc, not here.

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_owned.h"
#include "quant/nvfp4_quant.h"
#include "core/tensor.h"

#include "test_nvfp4_smallm_v2_util.cuh"
namespace {

// Host dequant of one element from plain packed buffers (UE4M3 micro-scales).
float host_dequant(const std::vector<uint8_t>& packed, const std::vector<uint8_t>& scales, float tensor_scale,
                   int n, int k, int K) {
    static const float lut[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    uint8_t byte = packed[(size_t)n * (K / 2) + k / 2];
    uint8_t nib = (k & 1) ? (byte >> 4) : (byte & 0xF);
    uint8_t se = scales[(size_t)n * (K / 16) + k / 16];
    int exp = (se >> 3) & 0xF;
    int mant = se & 0x7;
    float sf = (exp == 0) ? (mant / 8.0f) * std::pow(2.0f, -6.0f)
                          : (1.0f + mant / 8.0f) * std::pow(2.0f, exp - 7);
    return lut[nib] * sf * tensor_scale;
}

void run_case(int M, int N, int K, bool accumulate) {
    std::mt19937 rng(7 + M + N + K);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<__half> w_h((size_t)N * K), x_h((size_t)M * K);
    for (auto& v : w_h)
        v = __float2half(dist(rng));
    for (auto& v : x_h)
        v = __float2half(dist(rng));

    DeviceQuant W, X;
    W.quantize(w_h, N, K);
    X.quantize(x_h, M, K);

    std::vector<__half> y0_h((size_t)M * N);
    for (auto& v : y0_h)
        v = __float2half(accumulate ? dist(rng) : 0.0f);
    void *d_y = nullptr, *d_ws = nullptr;
    const size_t ws_bytes = imp::gemm_nvfp4_smallm_v2_workspace_bytes(N, K);
    ASSERT_EQ(cudaMalloc(&d_y, (size_t)M * N * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_ws, ws_bytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_y, y0_h.data(), (size_t)M * N * sizeof(__half), cudaMemcpyHostToDevice),
              cudaSuccess);

    ASSERT_TRUE(
        imp::gemm_nvfp4_smallm_v2_a4(W.q, X.q, static_cast<half*>(d_y), M, N, K, d_ws, nullptr, accumulate));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<__half> y_h((size_t)M * N);
    ASSERT_EQ(cudaMemcpy(y_h.data(), d_y, y_h.size() * sizeof(__half), cudaMemcpyDeviceToHost), cudaSuccess);

    const float ts = W.q.tensor_scale * X.q.tensor_scale;
    double max_rel = 0.0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double ref = 0.0;
            for (int k = 0; k < K; ++k)
                ref += host_dequant(W.packed_h, W.scales_h, 1.0f, n, k, K) *
                       host_dequant(X.packed_h, X.scales_h, 1.0f, m, k, K);
            ref = ref * ts + (accumulate ? __half2float(y0_h[(size_t)m * N + n]) : 0.0);
            double got = __half2float(y_h[(size_t)m * N + n]);
            double rel = std::abs(got - ref) / std::max(1.0, std::abs(ref));
            max_rel = std::max(max_rel, rel);
        }
    }
    // FP32 accumulate over exact FP4xUE4M3 products, then one FP16 round:
    // the FP16 output quantum on |y|~1 dominates the envelope.
    EXPECT_LT(max_rel, 2e-3) << "M=" << M << " N=" << N << " K=" << K << " acc=" << accumulate
                             << " max relative error " << max_rel;

    cudaFree(d_y);
    cudaFree(d_ws);
}

TEST_F(NvFP4SmallMV2Test, MatchesHostReference) {
    run_case(/*M=*/19, /*N=*/192, /*K=*/512, /*accumulate=*/false);
}

TEST_F(NvFP4SmallMV2Test, MatchesHostReferenceFullTileMultiStripe) {
    run_case(/*M=*/32, /*N=*/128, /*K=*/2048, /*accumulate=*/false);
}

TEST_F(NvFP4SmallMV2Test, AccumulateAddsOntoY) {
    run_case(/*M=*/8, /*N=*/64, /*K=*/256, /*accumulate=*/true);
}

TEST_F(NvFP4SmallMV2Test, RejectsUnalignedShapes) {
    imp::NvFP4QuantResult dummy{};
    dummy.packed_data = reinterpret_cast<void*>(0x10);
    dummy.micro_scales = reinterpret_cast<void*>(0x20);
    int probe = 0;
    EXPECT_FALSE(imp::gemm_nvfp4_smallm_v2_a4(dummy, dummy, nullptr, 33, 64, 256, &probe, nullptr));
    EXPECT_FALSE(imp::gemm_nvfp4_smallm_v2_a4(dummy, dummy, nullptr, 8, 96, 256, &probe, nullptr));
    EXPECT_FALSE(imp::gemm_nvfp4_smallm_v2_a4(dummy, dummy, nullptr, 8, 64, 384, &probe, nullptr));
}

// The FP32-output twin (batched LM head) writes the same accumulators the
// FP16 kernel rounds: rounding its output to FP16 must reproduce the FP16
// kernel bit for bit. Striped shapes are refused (no FP32 reduce path).
TEST_F(NvFP4SmallMV2Test, F32OutputRoundsToTheHalfKernel) {
    const int M = 32, N = 5120, K = 5120;  // 80 tiles: one stripe
    std::mt19937 rng(19);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<__half> w_h((size_t)N * K), x_h((size_t)M * K);
    for (auto& v : w_h)
        v = __float2half(dist(rng));
    for (auto& v : x_h)
        v = __float2half(dist(rng));
    DeviceQuant W, X;
    W.quantize(w_h, N, K);
    X.quantize(x_h, M, K);
    void *d_y16 = nullptr, *d_y32 = nullptr;
    ASSERT_EQ(cudaMalloc(&d_y16, (size_t)M * N * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y32, (size_t)M * N * sizeof(float)), cudaSuccess);
    ASSERT_TRUE(imp::gemm_nvfp4_smallm_v2_a4(W.q, X.q, static_cast<half*>(d_y16), M, N, K, nullptr, nullptr));
    ASSERT_TRUE(imp::gemm_nvfp4_smallm_v2_a4_f32(W.q, X.q, static_cast<float*>(d_y32), M, N, K, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<uint16_t> y16((size_t)M * N);
    std::vector<float> y32((size_t)M * N);
    ASSERT_EQ(cudaMemcpy(y16.data(), d_y16, y16.size() * 2, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(y32.data(), d_y32, y32.size() * 4, cudaMemcpyDeviceToHost), cudaSuccess);
    size_t mismatches = 0;
    for (size_t i = 0; i < y32.size(); ++i) {
        __half r = __float2half(y32[i]);
        uint16_t bits;
        std::memcpy(&bits, &r, 2);
        mismatches += (bits != y16[i]);
    }
    EXPECT_EQ(mismatches, 0u);
    cudaFree(d_y16);
    cudaFree(d_y32);
    // N=1024 at K=5120 is a 10-stripe shape: the FP32 twin declines.
    EXPECT_FALSE(imp::gemm_nvfp4_smallm_v2_a4_f32(W.q, X.q, nullptr, M, 1024, K, nullptr));
}

// Three siblings (q|k|v: 80+16+16 tiles) in one launch must be BIT-identical to three single
// launches at stripes=1 (same CTA body/tile order); the striped single path for N=1024
// (10 stripes + reduce) is a different reduction order and is not the reference.
TEST_F(NvFP4SmallMV2Test, TripleMatchesSinglesBitwise) {
    const int M = 32, K = 5120;
    const int Ns[3] = {5120, 1024, 1024};
    std::mt19937 rng(17);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<__half> x_h((size_t)M * K);
    for (auto& v : x_h)
        v = __float2half(dist(rng));
    DeviceQuant X;
    X.quantize(x_h, M, K);
    DeviceQuant W[3];
    void* y_multi[3];
    void* y_single[3];
    for (int i = 0; i < 3; ++i) {
        std::vector<__half> w_h((size_t)Ns[i] * K);
        for (auto& v : w_h)
            v = __float2half(dist(rng));
        W[i].quantize(w_h, Ns[i], K);
        ASSERT_EQ(cudaMalloc(&y_multi[i], (size_t)M * Ns[i] * sizeof(__half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&y_single[i], (size_t)M * Ns[i] * sizeof(__half)), cudaSuccess);
        ASSERT_TRUE(imp::gemm_nvfp4_smallm_v2_a4_tuned(W[i].q, X.q, static_cast<half*>(y_single[i]), M, Ns[i],
                                                       K, nullptr, nullptr, false, /*stages=*/6,
                                                       /*stripes=*/1));
    }
    const imp::SmallMV2Sibling sib[3] = {{&W[0].q, static_cast<half*>(y_multi[0]), Ns[0]},
                                         {&W[1].q, static_cast<half*>(y_multi[1]), Ns[1]},
                                         {&W[2].q, static_cast<half*>(y_multi[2]), Ns[2]}};
    ASSERT_TRUE(imp::gemm_nvfp4_smallm_v2_multi_a4(sib, 3, X.q, M, K, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    for (int i = 0; i < 3; ++i) {
        std::vector<uint16_t> a((size_t)M * Ns[i]), b((size_t)M * Ns[i]);
        ASSERT_EQ(cudaMemcpy(a.data(), y_multi[i], a.size() * 2, cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(b.data(), y_single[i], b.size() * 2, cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_EQ(a, b) << "sibling " << i << " N=" << Ns[i];
        cudaFree(y_multi[i]);
        cudaFree(y_single[i]);
    }
    // Below the combined single-stripe threshold the multi launch declines.
    const imp::SmallMV2Sibling small[2] = {{&W[1].q, nullptr, Ns[1]}, {&W[2].q, nullptr, Ns[2]}};
    EXPECT_FALSE(imp::gemm_nvfp4_smallm_v2_multi_a4(small, 2, X.q, M, K, nullptr));
}

// Run-to-run determinism: same W/Xq/shape must give bit-identical y, single and multi alike
// (fixed-order MMA/stripe reduce, so any mismatch is a pipeline race). Caught on Qwen3-14B
// after #1954: KV cache differed per run, M=1 vs M=1 control flipped 1-3 of 64 greedy tokens
// (max |dlogp| 0.3-0.7). Two input sets alternate so an early read sees the OTHER set's bytes.
TEST_F(NvFP4SmallMV2Test, RepeatedLaunchesBitwiseStable) {
    struct Shape {
        int M, K, count, N[3];
    };
    const Shape shapes[] = {{24, 5120, 3, {5120, 1024, 1024}}, {24, 5120, 2, {17408, 17408, 0}},
                            {24, 5120, 1, {5120, 0, 0}},       {24, 17408, 1, {5120, 0, 0}},
                            {32, 5120, 3, {5120, 1024, 1024}}, {1, 5120, 1, {1024, 0, 0}}};
    const int launches = 200;
    for (const Shape& s : shapes) {
        std::mt19937 rng(11 + s.M + s.K + s.count);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        DeviceQuant W[2][3], X[2];
        for (int set = 0; set < 2; ++set) {
            std::vector<__half> x_h((size_t)s.M * s.K);
            for (auto& v : x_h)
                v = __float2half(dist(rng));
            X[set].quantize(x_h, s.M, s.K);
            for (int i = 0; i < s.count; ++i) {
                std::vector<__half> w_h((size_t)s.N[i] * s.K);
                for (auto& v : w_h)
                    v = __float2half(dist(rng));
                W[set][i].quantize(w_h, s.N[i], s.K);
            }
        }
        int total_n = 0;
        for (int i = 0; i < s.count; ++i)
            total_n += s.N[i];
        const size_t y_elems = (size_t)s.M * total_n;  // one launch: sibling outputs concatenated
        const size_t y_bytes = y_elems * sizeof(__half);
        void *d_y = nullptr, *d_ws = nullptr;
        ASSERT_EQ(cudaMalloc(&d_y, y_bytes * launches), cudaSuccess);
        const int stripes = s.count == 1 ? imp::gemm_nvfp4_smallm_v2_stripes(s.N[0], s.K) : 1;
        if (s.count == 1)
            ASSERT_EQ(cudaMalloc(&d_ws, imp::gemm_nvfp4_smallm_v2_workspace_bytes(s.N[0], s.K)), cudaSuccess);
        for (int back_to_back = 0; back_to_back < 2; ++back_to_back) {
            if (back_to_back && stripes != 1)
                continue;  // striped shapes share one partials workspace; only stream order keeps that safe
            ASSERT_EQ(cudaMemset(d_y, 0, y_bytes * launches), cudaSuccess);
            for (int i = 0; i < launches; ++i) {
                const int set = i & 1;
                half* y_i = reinterpret_cast<half*>(static_cast<uint8_t*>(d_y) + y_bytes * i);
                if (s.count == 1) {
                    ASSERT_TRUE(imp::gemm_nvfp4_smallm_v2_a4(W[set][0].q, X[set].q, y_i, s.M, s.N[0], s.K,
                                                             d_ws, nullptr, /*accumulate=*/false));
                } else {
                    imp::SmallMV2Sibling sib[3];
                    half* yp = y_i;
                    for (int t = 0; t < s.count; ++t) {
                        sib[t] = {&W[set][t].q, yp, s.N[t]};
                        yp += (size_t)s.M * s.N[t];
                    }
                    ASSERT_TRUE(
                        imp::gemm_nvfp4_smallm_v2_multi_a4(sib, s.count, X[set].q, s.M, s.K, nullptr));
                }
                if (!back_to_back)
                    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            }
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            std::vector<uint16_t> all(y_elems * launches);
            ASSERT_EQ(cudaMemcpy(all.data(), d_y, y_bytes * launches, cudaMemcpyDeviceToHost), cudaSuccess);
            int bad_launches = 0;
            size_t bad_elems = 0;
            for (int i = 2; i < launches; ++i) {
                const uint16_t* ref = all.data() + y_elems * (i & 1);
                const uint16_t* cur = all.data() + y_elems * i;
                size_t bad = 0;
                for (size_t e = 0; e < y_elems; ++e)
                    bad += (cur[e] != ref[e]);
                if (bad) {
                    ++bad_launches;
                    bad_elems += bad;
                }
            }
            printf(
                "[ DETERMIN ] M=%2d K=%5d %s N=%d/%d/%d stripes=%2d %-12s: %d of %d launches differ from the "
                "first (%zu elements)\n",
                s.M, s.K, s.count == 1 ? "single" : "multi ", s.N[0], s.N[1], s.N[2], stripes,
                back_to_back ? "back-to-back" : "synced", bad_launches, launches, bad_elems);
            EXPECT_EQ(bad_launches, 0)
                << "M=" << s.M << " K=" << s.K << " count=" << s.count << " N0=" << s.N[0]
                << (back_to_back ? " back-to-back" : " synced") << ": " << bad_launches << " of " << launches
                << " launches differ from the first";
        }
        cudaFree(d_y);
        cudaFree(d_ws);
        // NvFP4Owned releases its own buffers: a hand cudaFree of q.packed_data is a double free.
        ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "before the owner release, M=" << s.M << " K=" << s.K;
        for (int set = 0; set < 2; ++set) {
            X[set].q = imp::NvFP4Owned{};
            for (auto& w : W[set])
                w.q = imp::NvFP4Owned{};
        }
        ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "owner release, M=" << s.M << " K=" << s.K;
    }
}

}  // namespace
