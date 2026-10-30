// gemm_mxfp8_tile (own MXFP8 prefill GEMM, ROADMAP row 123) against a host double reference decoded from the
// same operand bytes, plain SfAtom weights and the FP8 row-scale view, plus a timing table on the
// Qwen3.6-35B GDN projection shapes (informational).

#include "quant/fp8_quant.h"
#include "quant/mxfp8_sfatom.h"
#include "quant/nvfp4_gemm_tile.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace imp;

namespace {

__global__ void fill_kernel(half* p, size_t n, uint32_t seed, float amp) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
        h ^= h >> 13;
        h *= 0x5bd1e995u;
        h ^= h >> 15;
        p[i] = __float2half(amp * ((static_cast<float>(h & 0xffff) / 32768.0f) - 1.0f));
    }
}

struct Operands {
    int M = 0, N = 0, K = 0;
    CutlassMxFP8Weight cw;
    void* act = nullptr;
    void* act_sf = nullptr;
    float* row_scales = nullptr;
    void* ws = nullptr;
    size_t ws_bytes = 0;
    half* y_own = nullptr;
};

// rows = true: W as the FP8 row-scale view (E4M3 rows, SFB = power of two, col_scale residual).
void make_operands(Operands& o, int M, int N, int K, bool rows, cudaStream_t s) {
    o.M = M;
    o.N = N;
    o.K = K;
    half* w = nullptr;
    half* x = nullptr;
    ASSERT_EQ(cudaMalloc(&w, size_t(N) * K * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&x, size_t(M) * K * sizeof(half)), cudaSuccess);
    fill_kernel<<<1024, 256, 0, s>>>(w, size_t(N) * K, 0x1234u, 0.08f);
    fill_kernel<<<1024, 256, 0, s>>>(x, size_t(M) * K, 0x9876u, 2.0f);
    o.cw.N = N;
    o.cw.K = K;
    o.cw.data_bytes = size_t(N) * K;
    o.cw.sf_bytes = cutlass_mxfp8_sf_size(N, K);
    ASSERT_EQ(cudaMalloc(&o.cw.data, o.cw.data_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&o.cw.scale_factors, o.cw.sf_bytes), cudaSuccess);
    if (rows) {
        float* col = nullptr;
        ASSERT_EQ(cudaMalloc(&o.row_scales, size_t(N) * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&col, size_t(N) * sizeof(float)), cudaSuccess);
        quantize_fp8_rows_async(w, o.cw.data, N, K, o.row_scales, s);
        mxfp8_sf_from_row_scales(o.row_scales, o.cw.scale_factors, col, N, K, s);
        o.cw.col_scale = col;
    } else {
        quantize_fp16_to_mxfp8_cutlass(w, o.cw.data, o.cw.scale_factors, N, K, s);
    }
    ASSERT_EQ(cudaMalloc(&o.act, size_t(M) * K), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&o.act_sf, cutlass_mxfp8_sf_size(M, K)), cudaSuccess);
    quantize_fp16_to_mxfp8_cutlass(x, o.act, o.act_sf, M, K, s);
    o.ws_bytes = gemm_tile_streamk_workspace_bytes();
    ASSERT_EQ(cudaMalloc(&o.ws, o.ws_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&o.y_own, size_t(M) * N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
    cudaFree(w);
    cudaFree(x);
}

void free_operands(Operands& o) {
    for (void* p : {o.cw.data, o.cw.scale_factors, static_cast<void*>(const_cast<float*>(o.cw.col_scale)),
                    static_cast<void*>(o.row_scales), o.act, o.act_sf, o.ws, static_cast<void*>(o.y_own)})
        cudaFree(p);
}

float e4m3_to_float(uint8_t b) {
    const int e = (b >> 3) & 0xF, m = b & 7;
    const float v = e == 0 ? std::ldexp(m / 8.0f, -6) : std::ldexp(1.0f + m / 8.0f, e - 7);
    return (b & 0x80) ? -v : v;
}

// E4M3 [rows, K] + SfAtom UE8M0 (128 rows x 4 K-groups per 512-byte atom) -> float rows on the host.
std::vector<float> decode_mx8(const void* data, const void* sf, int rows, int K) {
    std::vector<uint8_t> q(size_t(rows) * K), e(cutlass_mxfp8_sf_size(rows, K));
    cudaMemcpy(q.data(), data, q.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(e.data(), sf, e.size(), cudaMemcpyDeviceToHost);
    const int n_k_tiles = (K + 127) / 128;
    std::vector<float> out(q.size());
    for (int r = 0; r < rows; ++r)
        for (int k = 0; k < K; ++k) {
            const int g = k / 32, rl = r % 128;
            const int off = ((r / 128) * n_k_tiles + g / 4) * 512 + (rl % 32) * 16 + (rl / 32) * 4 + g % 4;
            out[size_t(r) * K + k] = e4m3_to_float(q[size_t(r) * K + k]) * std::ldexp(1.0f, e[off] - 127);
        }
    return out;
}

// D = A W^T (times col_scale) in double from the decoded operand bytes.
std::vector<double> host_reference(const Operands& o) {
    const std::vector<float> a = decode_mx8(o.act, o.act_sf, o.M, o.K);
    const std::vector<float> w = decode_mx8(o.cw.data, o.cw.scale_factors, o.N, o.K);
    std::vector<float> col(o.N, 1.0f);
    if (o.cw.col_scale)
        cudaMemcpy(col.data(), o.cw.col_scale, col.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<double> ref(size_t(o.M) * o.N);
    for (int m = 0; m < o.M; ++m)
        for (int n = 0; n < o.N; ++n) {
            const float* ar = &a[size_t(m) * o.K];
            const float* wr = &w[size_t(n) * o.K];
            double acc = 0.0;
            for (int k = 0; k < o.K; ++k)
                acc += double(ar[k]) * wr[k];
            ref[size_t(m) * o.N + n] = acc * col[n];
        }
    return ref;
}

constexpr int kVariants = 5;  // 0 128x128 PW, 1 128x256, 2 sync, 3 stream-K, then -1 (auto)
constexpr int kVariantBk[kVariants] = {64, 64, 128, 64, 64};

class Mxfp8GemmTileTest : public ::testing::TestWithParam<std::tuple<int, int, int, bool>> {
protected:
    void SetUp() override { cudaStreamCreate(&s_); }
    void TearDown() override { cudaStreamDestroy(s_); }
    cudaStream_t s_ = nullptr;
};

}  // namespace

// Exact products of the operand bytes, fp32 accumulate in the kernel: fp16 rounding plus accumulation order.
TEST_P(Mxfp8GemmTileTest, MatchesHostReference) {
    const auto [M, N, K, rows] = GetParam();
    Operands o;
    make_operands(o, M, N, K, rows, s_);
    const std::vector<double> ref = host_reference(o);
    std::vector<half> own(size_t(M) * N);
    for (int v = 0; v < kVariants; ++v) {
        if (K % kVariantBk[v] != 0)
            continue;
        cudaMemsetAsync(o.y_own, 0xff, ref.size() * sizeof(half), s_);
        ASSERT_TRUE(gemm_mxfp8_tile(o.act, o.act_sf, o.cw.data, o.cw.scale_factors, o.cw.col_scale, o.y_own,
                                    M, N, K, s_, v < 4 ? v : -1, o.ws, o.ws_bytes));
        ASSERT_EQ(cudaStreamSynchronize(s_), cudaSuccess);
        cudaMemcpy(own.data(), o.y_own, own.size() * sizeof(half), cudaMemcpyDeviceToHost);
        double max_ref = 0.0, max_diff = 0.0;
        size_t bad = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            const double r = ref[i];
            const double x = __half2float(own[i]);
            if (!std::isfinite(x))
                ++bad;
            max_ref = std::max(max_ref, std::fabs(r));
            max_diff = std::max(max_diff, std::fabs(r - x));
        }
        std::fprintf(stderr, "[mx8 v%d] M=%d N=%d K=%d rows=%d max|ref|=%.4g max|diff|=%.4g\n", v, M, N, K,
                     int(rows), max_ref, max_diff);
        EXPECT_EQ(bad, 0u) << "variant " << v;
        EXPECT_GT(max_ref, 0.0);
        EXPECT_LE(max_diff, 2e-3 * max_ref + 1e-3) << "variant " << v;
    }
    free_operands(o);
}

INSTANTIATE_TEST_SUITE_P(
    Shapes, Mxfp8GemmTileTest,
    ::testing::Values(std::make_tuple(128, 128, 128, false), std::make_tuple(200, 384, 576, false),
                      std::make_tuple(33, 136, 1024, false), std::make_tuple(512, 1024, 2048, false),
                      std::make_tuple(1000, 2056, 256, false), std::make_tuple(96, 384, 1024, true),
                      std::make_tuple(300, 2056, 576, true),
                      std::make_tuple(1536, 2304, 512, true)));  // 216 tiles: hybrid stream-K

// Timing on the Qwen3.6-35B GDN projections (qkv, z, out) in the FP8 row-scale view, the prefill
// default, per variant; 4 weight copies rotate out of L2.
TEST(Mxfp8GemmTileBench, Variants) {
    cudaStream_t s;
    cudaStreamCreate(&s);
    const int shapes[][2] = {{8192, 2048}, {4096, 2048}, {2048, 4096}};
    const int ms[] = {512, 2048, 4096};
    std::fprintf(stderr, "%6s %6s %6s %8s %8s %8s %8s %8s %7s\n", "M", "N", "K", "t128", "t256", "sync", "sk",
                 "auto", "auto_TF");
    for (const auto& nk : shapes) {
        for (int M : ms) {
            Operands o;
            make_operands(o, M, nk[0], nk[1], true, s);
            constexpr int kCopies = 4;
            std::vector<void*> wd(kCopies), wsf(kCopies);
            for (int c = 0; c < kCopies; ++c) {
                cudaMalloc(&wd[c], o.cw.data_bytes);
                cudaMalloc(&wsf[c], o.cw.sf_bytes);
                cudaMemcpyAsync(wd[c], o.cw.data, o.cw.data_bytes, cudaMemcpyDeviceToDevice, s);
                cudaMemcpyAsync(wsf[c], o.cw.scale_factors, o.cw.sf_bytes, cudaMemcpyDeviceToDevice, s);
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
            double t_v[kVariants];
            for (int v = 0; v < kVariants; ++v)
                t_v[v] = time_us([&](int c) {
                    (void)gemm_mxfp8_tile(o.act, o.act_sf, wd[c], wsf[c], o.cw.col_scale, o.y_own, o.M, o.N,
                                          o.K, s, v < 4 ? v : -1, o.ws, o.ws_bytes);
                });
            ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
            const double tf = 2.0 * o.M * o.N * double(o.K) / (t_v[4] * 1e-6) / 1e12;
            std::fprintf(stderr, "%6d %6d %6d", o.M, o.N, o.K);
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
