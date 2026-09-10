// The M=1 GDN fused input GEMV (quant/nvfp4_gemv_gdn_input.cu) against the
// four launches it replaces at Qwen3.8-27B dims (K=5120: in_proj 10240 rows,
// gate 6144, alpha/beta 48 each). in_proj takes the multirow kernel at these
// rows, the gate the 128-thread K-par kernel, alpha/beta the gemv_fp16 sum:
// all four reproduced bit-for-bit. Plus an fp32 host reference for every
// segment and the K-shape declines.

#include "quant/nvfp4_quant.h"
#include "quant/nvfp4_gemm.h"
#include "compute/gemm.h"
#include "core/tensor.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <vector>

namespace imp {
namespace {

constexpr int kK = 5120;
constexpr int kInRows = 10240;
constexpr int kGateRows = 6144;
constexpr int kAbRows = 48;

class NvFP4GemvGdnInput : public ::testing::Test {
protected:
    void SetUp() override {
        if (cudaSetDevice(0) != cudaSuccess)
            GTEST_SKIP() << "No CUDA device";
        cudaStreamCreate(&stream_);
    }
    void TearDown() override {
        for (void* p : allocs_)
            cudaFree(p);
        if (stream_)
            cudaStreamDestroy(stream_);
    }

    std::vector<half> host_weight(int N, int K, int seed) {
        std::vector<half> h(static_cast<size_t>(N) * K);
        for (size_t i = 0; i < h.size(); ++i)
            h[i] = __float2half(((static_cast<int>(i * 17u + seed) % 31) - 15) * 0.01f);
        return h;
    }

    half* upload(const std::vector<half>& h) {
        half* d = nullptr;
        EXPECT_EQ(cudaMalloc(&d, h.size() * sizeof(half)), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(d, h.data(), h.size() * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);
        allocs_.push_back(d);
        return d;
    }

    half* device_out(int n) {
        half* d = nullptr;
        EXPECT_EQ(cudaMalloc(&d, static_cast<size_t>(n) * sizeof(half)), cudaSuccess);
        cudaMemset(d, 0, static_cast<size_t>(n) * sizeof(half));
        allocs_.push_back(d);
        return d;
    }

    void quantize(half* d_w, int N, int K, NvFP4QuantResult& qr) {
        int64_t wshape[2] = {N, K};
        Tensor w_t(d_w, QType::F16, 2, wshape, true);
        quantize_fp16_to_nvfp4(w_t, qr, stream_);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    }

    std::vector<half> download(const half* d, int n) {
        EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        std::vector<half> h(static_cast<size_t>(n));
        EXPECT_EQ(cudaMemcpy(h.data(), d, h.size() * sizeof(half), cudaMemcpyDeviceToHost), cudaSuccess);
        return h;
    }

    static int count_diff_bits(const std::vector<half>& a, const std::vector<half>& b) {
        int n = 0;
        for (size_t i = 0; i < a.size(); ++i)
            if (__half_as_ushort(a[i]) != __half_as_ushort(b[i]))
                ++n;
        return n;
    }

    // fp32 dot of the DEQUANTIZED weight row (what the NVFP4 kernels compute)
    // or of the FP16 row, against the host x.
    static void expect_close(const char* what, const std::vector<half>& got, const std::vector<float>& ref,
                             float tol) {
        ASSERT_EQ(got.size(), ref.size());
        int bad = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const float g = __half2float(got[i]);
            if (std::isnan(g) || std::fabs(g - ref[i]) > tol + 0.02f * std::fabs(ref[i]))
                ++bad;
        }
        EXPECT_EQ(bad, 0) << what << ": " << bad << " of " << got.size() << " outputs off the reference";
    }

    cudaStream_t stream_ = nullptr;
    std::vector<void*> allocs_;
};

TEST_F(NvFP4GemvGdnInput, MatchesTheFourLaunchesAtQwen38Dims) {
    auto h_in = host_weight(kInRows, kK, 1);
    auto h_gate = host_weight(kGateRows, kK, 2);
    auto h_a = host_weight(kAbRows, kK, 3);
    auto h_b = host_weight(kAbRows, kK, 4);
    std::vector<half> h_x(kK);
    for (int i = 0; i < kK; ++i)
        h_x[i] = __float2half(((i * 23) % 29 - 14) * 0.02f);

    half* d_in = upload(h_in);
    half* d_gate = upload(h_gate);
    half* d_a = upload(h_a);
    half* d_b = upload(h_b);
    half* d_x = upload(h_x);
    NvFP4QuantResult q_in, q_gate;
    quantize(d_in, kInRows, kK, q_in);
    quantize(d_gate, kGateRows, kK, q_gate);

    // Reference launches: the kernels the executor dispatched before.
    half* r_in = device_out(kInRows);
    half* r_gate = device_out(kGateRows);
    half* r_a = device_out(kAbRows);
    half* r_b = device_out(kAbRows);
    gemv_nvfp4_kpar(q_in, d_x, r_in, kInRows, kK, stream_);
    gemv_nvfp4_kpar(q_gate, d_x, r_gate, kGateRows, kK, stream_);
    {
        int64_t ab_shape[2] = {kAbRows, kK};
        int64_t x_shape[1] = {kK};
        int64_t y_shape[1] = {kAbRows};
        Tensor wa(d_a, QType::F16, 2, ab_shape, true), wb(d_b, QType::F16, 2, ab_shape, true);
        Tensor x_t(d_x, QType::F16, 1, x_shape, true);
        Tensor ya(r_a, QType::F16, 1, y_shape, true), yb(r_b, QType::F16, 1, y_shape, true);
        gemv(wa, x_t, ya, stream_);
        gemv(wb, x_t, yb, stream_);
    }

    half* f_in = device_out(kInRows);
    half* f_gate = device_out(kGateRows);
    half* f_a = device_out(kAbRows);
    half* f_b = device_out(kAbRows);
    ASSERT_TRUE(gemv_nvfp4_gdn_input_fused(q_in, q_gate, d_a, d_b, kAbRows, d_x, f_in, f_gate, f_a, f_b, kK,
                                           stream_));

    auto g_in = download(f_in, kInRows), e_in = download(r_in, kInRows);
    auto g_gate = download(f_gate, kGateRows), e_gate = download(r_gate, kGateRows);
    auto g_a = download(f_a, kAbRows), e_a = download(r_a, kAbRows);
    auto g_b = download(f_b, kAbRows), e_b = download(r_b, kAbRows);
    // in_proj rows >= 6 x SMs x 8 take the multirow kernel: same warp_k_loop order.
    EXPECT_EQ(count_diff_bits(g_in, e_in), 0) << "in_proj is not bit-identical to gemv_nvfp4_kpar (multirow)";
    EXPECT_EQ(count_diff_bits(g_a, e_a), 0) << "alpha is not bit-identical to gemv_fp16_kernel";
    EXPECT_EQ(count_diff_bits(g_b, e_b), 0) << "beta is not bit-identical to gemv_fp16_kernel";
    // 6144 gate rows take the 128-thread K-par kernel in gemv_nvfp4_kpar; the
    // fused kernel reproduces its loop stride and reduction order.
    EXPECT_EQ(count_diff_bits(g_gate, e_gate), 0) << "gate is not bit-identical to gemv_nvfp4_kpar (K-par)";

    // fp32 host references (dequantized NVFP4 rows for in_proj / gate).
    auto ref_nvfp4 = [&](const NvFP4QuantResult& q, int N) {
        std::vector<half> deq(static_cast<size_t>(N) * kK);
        half* d_deq = nullptr;
        EXPECT_EQ(cudaMalloc(&d_deq, deq.size() * sizeof(half)), cudaSuccess);
        allocs_.push_back(d_deq);
        dequantize_nvfp4_to_fp16(q, d_deq, stream_);
        deq = download(d_deq, N * kK);
        std::vector<float> ref(N, 0.0f);
        for (int r = 0; r < N; ++r) {
            double s = 0.0;
            for (int k = 0; k < kK; ++k)
                s += static_cast<double>(__half2float(deq[static_cast<size_t>(r) * kK + k])) *
                     __half2float(h_x[k]);
            ref[r] = static_cast<float>(s);
        }
        return ref;
    };
    auto ref_fp16 = [&](const std::vector<half>& w, int N) {
        std::vector<float> ref(N, 0.0f);
        for (int r = 0; r < N; ++r) {
            double s = 0.0;
            for (int k = 0; k < kK; ++k)
                s += static_cast<double>(__half2float(w[static_cast<size_t>(r) * kK + k])) *
                     __half2float(h_x[k]);
            ref[r] = static_cast<float>(s);
        }
        return ref;
    };
    expect_close("in_proj", g_in, ref_nvfp4(q_in, kInRows), 0.05f);
    expect_close("gate", g_gate, ref_nvfp4(q_gate, kGateRows), 0.05f);
    expect_close("alpha", g_a, ref_fp16(h_a, kAbRows), 0.05f);
    expect_close("beta", g_b, ref_fp16(h_b, kAbRows), 0.05f);

    free_nvfp4_result(q_in);
    free_nvfp4_result(q_gate);
}

TEST_F(NvFP4GemvGdnInput, DeclinesShapesTheKernelCannotServe) {
    NvFP4QuantResult q_in, q_gate;
    q_in.K = kK;
    q_gate.K = kK;
    half* dummy = device_out(16);
    // K mismatch between the input and a weight.
    EXPECT_FALSE(gemv_nvfp4_gdn_input_fused(q_in, q_gate, dummy, dummy, 1, dummy, dummy, dummy, dummy, dummy,
                                            kK + 16, stream_));
    // K > 8192 (n_mb > 512): the multirow form declines.
    q_in.K = 16384;
    q_gate.K = 16384;
    EXPECT_FALSE(gemv_nvfp4_gdn_input_fused(q_in, q_gate, dummy, dummy, 1, dummy, dummy, dummy, dummy, dummy,
                                            16384, stream_));
    // K % 16 != 0.
    q_in.K = 5128;
    q_gate.K = 5128;
    EXPECT_FALSE(gemv_nvfp4_gdn_input_fused(q_in, q_gate, dummy, dummy, 1, dummy, dummy, dummy, dummy, dummy,
                                            5128, stream_));
    EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

}  // namespace
}  // namespace imp
