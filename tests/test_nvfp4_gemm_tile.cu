// gemm_nvfp4_tile (own NVFP4 prefill GEMM, ROADMAP row 123) against CUTLASS on the same operand
// bytes, plus a timing table vs CUTLASS and cuBLASLt on Qwen3.8-27B shapes (informational).

#include "compute/gemm.h"
#include "compute/gemm_cutlass_sm120.h"
#include "core/tensor.h"
#include "quant/nvfp4_gemm_tile.h"
#include "quant/nvfp4_quant.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace imp;

namespace {

struct Operands {
    int M = 0, N = 0, K = 0;
    NvFP4QuantResult qr;
    CutlassNvFP4Weight cw;
    void* act = nullptr;
    void* act_sf = nullptr;
    void* ws = nullptr;
    size_t ws_bytes = 0;
    half* y_ref = nullptr;
    half* y_own = nullptr;
};

__global__ void fill_kernel(half* p, size_t n, uint32_t seed, float amp) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
        h ^= h >> 13;
        h *= 0x5bd1e995u;
        h ^= h >> 15;
        p[i] = __float2half(amp * ((static_cast<float>(h & 0xffff) / 32768.0f) - 1.0f));
    }
}

void make_operands(Operands& o, int M, int N, int K, cudaStream_t s) {
    o.M = M;
    o.N = N;
    o.K = K;
    half* w = nullptr;
    half* x = nullptr;
    ASSERT_EQ(cudaMalloc(&w, size_t(N) * K * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&x, size_t(M) * K * sizeof(half)), cudaSuccess);
    fill_kernel<<<1024, 256, 0, s>>>(w, size_t(N) * K, 0x1234u, 0.08f);
    fill_kernel<<<1024, 256, 0, s>>>(x, size_t(M) * K, 0x9876u, 2.0f);
    int64_t wshape[2] = {N, K};
    Tensor wt(w, QType::F16, 2, wshape, true);
    quantize_fp16_to_nvfp4(wt, o.qr, s);
    convert_nvfp4_to_cutlass(o.qr, o.cw, s);
    ASSERT_EQ(cudaMalloc(&o.act, size_t(M) * K / 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&o.act_sf, cutlass_nvfp4_sf_size(M, K)), cudaSuccess);
    cudaMemsetAsync(o.act_sf, 0, cutlass_nvfp4_sf_size(M, K), s);
    quantize_fp16_to_nvfp4_cutlass(x, o.act, o.act_sf, M, K, s);
    // Max of the CUTLASS workspace and the stream-K need (4 KiB flags + 170 x 64 KiB partials).
    o.ws_bytes = std::max<size_t>(gemm_nvfp4_cutlass_sm120_workspace(M, N, K), 4096 + (size_t(170) << 16));
    ASSERT_EQ(cudaMalloc(&o.ws, std::max<size_t>(o.ws_bytes, 1)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&o.y_ref, size_t(M) * N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&o.y_own, size_t(M) * N * sizeof(half)), cudaSuccess);
    cudaStreamSynchronize(s);
    cudaFree(w);
    cudaFree(x);
}

void free_operands(Operands& o) {
    free_cutlass_nvfp4_weight(o.cw);
    cudaFree(o.qr.packed_data);
    cudaFree(o.qr.micro_scales);
    cudaFree(o.act);
    cudaFree(o.act_sf);
    cudaFree(o.ws);
    cudaFree(o.y_ref);
    cudaFree(o.y_own);
}

bool run_cutlass(const Operands& o, cudaStream_t s) {
    return gemm_nvfp4_cutlass_sm120(o.act, o.act_sf, o.cw, o.y_ref, o.M, o.N, o.K, o.ws, o.ws_bytes, s);
}

// Variants: 0 TMA 128x128 x4 + producer warp, 1 TMA 128x256 x3, 2 sync 128x128 BK256x2, -1 auto.
constexpr int kCheckedVariants = 4;  // 3 = stream-K 128x128
constexpr int kVariantBk[kCheckedVariants] = {128, 128, 256, 128};

class Nvfp4GemmTileTest : public ::testing::TestWithParam<std::tuple<int, int, int>> {
protected:
    void SetUp() override { cudaStreamCreate(&s_); }
    void TearDown() override { cudaStreamDestroy(s_); }
    cudaStream_t s_ = nullptr;
};

}  // namespace

// Same operand bytes, fp32 accumulate in both: outputs agree to fp16 rounding plus
// accumulation order (rel. to the output scale).
TEST_P(Nvfp4GemmTileTest, MatchesCutlass) {
    if (!cutlass_sm120_nvfp4_available())
        GTEST_SKIP() << "CUTLASS sm_120 NVFP4 not available";
    const auto [M, N, K] = GetParam();
    Operands o;
    make_operands(o, M, N, K, s_);
    ASSERT_TRUE(run_cutlass(o, s_));
    std::vector<half> ref(size_t(M) * N), own(size_t(M) * N);
    cudaMemcpy(ref.data(), o.y_ref, ref.size() * sizeof(half), cudaMemcpyDeviceToHost);
    for (int v = 0; v < kCheckedVariants; ++v) {
        if (K % kVariantBk[v] != 0)
            continue;
        cudaMemsetAsync(o.y_own, 0xff, ref.size() * sizeof(half), s_);
        ASSERT_TRUE(gemm_nvfp4_tile(o.act, o.act_sf, o.cw.data, o.cw.scale_factors, o.cw.tensor_scale,
                                    o.y_own, M, N, K, s_, v, o.ws, o.ws_bytes));
        ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
        cudaMemcpy(own.data(), o.y_own, own.size() * sizeof(half), cudaMemcpyDeviceToHost);
        double max_ref = 0.0, max_diff = 0.0;
        size_t bad = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            const double r = __half2float(ref[i]);
            const double x = __half2float(own[i]);
            if (!std::isfinite(x))
                ++bad;
            max_ref = std::max(max_ref, std::fabs(r));
            max_diff = std::max(max_diff, std::fabs(r - x));
        }
        std::fprintf(stderr, "[tile v%d] M=%d N=%d K=%d max|ref|=%.4g max|diff|=%.4g\n", v, M, N, K, max_ref,
                     max_diff);
        EXPECT_EQ(bad, 0u) << "variant " << v;
        EXPECT_GT(max_ref, 0.0);
        EXPECT_LE(max_diff, 2e-3 * max_ref + 1e-3) << "variant " << v;
    }
    free_operands(o);
}

INSTANTIATE_TEST_SUITE_P(Shapes, Nvfp4GemmTileTest,
                         ::testing::Values(std::make_tuple(128, 128, 128), std::make_tuple(200, 384, 512),
                                           std::make_tuple(33, 136, 1024), std::make_tuple(512, 1024, 5120),
                                           std::make_tuple(1000, 2056, 256)));

// Timing on Qwen3.8-27B projection shapes; 4 weight copies rotate so W is not L2-resident.
TEST(Nvfp4GemmTileBench, VsCutlassAndCublasLt) {
    if (!cutlass_sm120_nvfp4_available())
        GTEST_SKIP() << "CUTLASS sm_120 NVFP4 not available";
    cudaStream_t s;
    cudaStreamCreate(&s);
    const int shapes[][2] = {{17408, 5120}, {5120, 17408}, {10240, 5120},
                             {6144, 5120},  {5120, 6144},  {1024, 5120}};
    const int ms[] = {512, 2048};
    constexpr int kVariants = 5;  // variants 0..3, then -1 (auto, the dispatch default)
    std::fprintf(stderr, "%6s %6s %6s %8s %8s %8s %8s %8s %8s %8s %7s\n", "M", "N", "K", "cutlass", "lt",
                 "t128", "t256", "sync", "sk", "auto", "best_TF");
    for (const auto& nk : shapes) {
        for (int M : ms) {
            Operands o;
            make_operands(o, M, nk[0], nk[1], s);
            constexpr int kCopies = 4;
            const size_t wb = size_t(o.N) * o.K / 2;
            const size_t sb = o.cw.sf_bytes;
            std::vector<void*> wd(kCopies), wsf(kCopies);
            for (int c = 0; c < kCopies; ++c) {
                cudaMalloc(&wd[c], wb);
                cudaMalloc(&wsf[c], sb);
                cudaMemcpyAsync(wd[c], o.cw.data, wb, cudaMemcpyDeviceToDevice, s);
                cudaMemcpyAsync(wsf[c], o.cw.scale_factors, sb, cudaMemcpyDeviceToDevice, s);
            }
            auto time_us = [&](auto&& fn) {
                cudaEvent_t e0, e1;
                cudaEventCreate(&e0);
                cudaEventCreate(&e1);
                for (int i = 0; i < 40; ++i)
                    fn(i % kCopies);
                constexpr int kIters = 40;
                cudaEventRecord(e0, s);
                for (int i = 0; i < kIters; ++i)
                    fn(i % kCopies);
                cudaEventRecord(e1, s);
                cudaEventSynchronize(e1);
                float ms_total = 0.0f;
                cudaEventElapsedTime(&ms_total, e0, e1);
                cudaEventDestroy(e0);
                cudaEventDestroy(e1);
                return 1000.0 * ms_total / kIters;
            };
            CutlassNvFP4Weight cw = o.cw;
            const double t_ct = time_us([&](int c) {
                cw.data = wd[c];
                cw.scale_factors = wsf[c];
                (void)gemm_nvfp4_cutlass_sm120(o.act, o.act_sf, cw, o.y_ref, o.M, o.N, o.K, o.ws, o.ws_bytes,
                                               s);
            });
            const double t_lt = time_us([&](int c) {
                (void)gemm_nvfp4_cublaslt(o.act, o.act_sf, wd[c], wsf[c], o.cw.tensor_scale, o.y_ref, o.M,
                                          o.N, o.K, s);
            });
            double t_v[kVariants];
            for (int v = 0; v < kVariants; ++v)
                t_v[v] = time_us([&](int c) {
                    (void)gemm_nvfp4_tile(o.act, o.act_sf, wd[c], wsf[c], o.cw.tensor_scale, o.y_own, o.M,
                                          o.N, o.K, s, v < 4 ? v : -1, o.ws, o.ws_bytes);
                });
            ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
            double best = 1e30;
            for (int v = 0; v < kVariants; ++v)
                if (v != 2)  // sync is the alignment fallback, not a contender
                    best = std::min(best, t_v[v]);
            const double tf = 2.0 * o.M * o.N * double(o.K) / (best * 1e-6) / 1e12;
            std::fprintf(stderr, "%6d %6d %6d %8.1f %8.1f", o.M, o.N, o.K, t_ct, t_lt);
            for (double t : t_v)
                std::fprintf(stderr, " %8.1f", t);
            std::fprintf(stderr, " %7.0f\n", tf);
            for (int c = 0; c < kCopies; ++c) {
                cudaFree(wd[c]);
                cudaFree(wsf[c]);
            }
            free_operands(o);
        }
    }
    cudaStreamDestroy(s);
}
