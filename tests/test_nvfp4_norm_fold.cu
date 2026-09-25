// Norm fold (quant/nvfp4_gemm.h NvFP4NormFoldOut/In): residual GEMV + folded consumer against
// residual GEMV + rmsnorm() + plain consumer, at Qwen3.8-27B widths (d 5120).
#include "compute/layernorm.h"
#include "core/tensor.h"
#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_quant.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

namespace imp {
namespace {

constexpr int kD = 5120;
constexpr int kInner = 6144;  // producer K (GDN out_proj width)
constexpr int kFfn = 2048;    // consumer rows per matrix
constexpr float kEps = 1e-6f;

class NvFP4NormFold : public ::testing::Test {
protected:
    void SetUp() override {
        if (cudaSetDevice(0) != cudaSuccess)
            GTEST_SKIP() << "No CUDA device";
        cudaStreamCreate(&stream_);
    }
    void TearDown() override {
        for (auto& q : quants_)
            free_nvfp4_result(q);
        for (void* p : bufs_)
            cudaFree(p);
        if (stream_)
            cudaStreamDestroy(stream_);
    }
    template <typename T>
    T* alloc(size_t n) {
        void* p = nullptr;
        EXPECT_EQ(cudaMalloc(&p, n * sizeof(T)), cudaSuccess);
        cudaMemset(p, 0, n * sizeof(T));
        bufs_.push_back(p);
        return static_cast<T*>(p);
    }
    half* upload(const std::vector<float>& v) {
        std::vector<half> h(v.size());
        for (size_t i = 0; i < v.size(); ++i)
            h[i] = __float2half(v[i]);
        half* d = alloc<half>(v.size());
        EXPECT_EQ(cudaMemcpy(d, h.data(), h.size() * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);
        return d;
    }
    NvFP4QuantResult weight(int N, int K, unsigned seed) {
        std::vector<float> w(static_cast<size_t>(N) * K);
        for (size_t i = 0; i < w.size(); ++i)
            w[i] = static_cast<float>(static_cast<int>((i * 2654435761u + seed) % 61u) - 30) * 0.004f;
        half* d_w = upload(w);
        int64_t shape[2] = {N, K};
        Tensor t(d_w, QType::F16, 2, shape, true);
        NvFP4QuantResult q;
        quantize_fp16_to_nvfp4(t, q, stream_);
        EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        quants_.push_back(q);
        return q;
    }
    template <typename T>
    std::vector<float> fetch(const T* d, size_t n) {
        EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        std::vector<T> h(n);
        EXPECT_EQ(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
        std::vector<float> f(n);
        for (size_t i = 0; i < n; ++i) {
            if constexpr (std::is_same_v<T, half>)
                f[i] = __half2float(h[i]);
            else
                f[i] = h[i];
        }
        return f;
    }
    static double rel_l2(const std::vector<float>& got, const std::vector<float>& ref) {
        double num = 0, den = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            num += (got[i] - ref[i]) * static_cast<double>(got[i] - ref[i]);
            den += ref[i] * static_cast<double>(ref[i]);
        }
        return std::sqrt(num / den);
    }

    void run(float weight_offset) {
        const NvFP4QuantResult w_out = weight(kD, kInner, 1);
        const NvFP4QuantResult w_g = weight(kFfn, kD, 2);
        const NvFP4QuantResult w_u = weight(kFfn, kD, 3);
        const NvFP4QuantResult w_lm = weight(kFfn, kD, 4);
        std::vector<float> y(kInner), res(kD), gamma(kD);
        for (int i = 0; i < kInner; ++i)
            y[i] = static_cast<float>((i * 23) % 29 - 14) * 0.05f;
        // Residual stream with a few large channels, as real hidden states carry.
        for (int i = 0; i < kD; ++i)
            res[i] = static_cast<float>((i * 7) % 13 - 6) * 0.3f + (i % 997 == 0 ? 900.0f : 0.0f);
        for (int i = 0; i < kD; ++i)
            gamma[i] = static_cast<float>((i * 11) % 17) * 0.1f - weight_offset * 0.5f;
        const half* d_y = upload(y);
        const half* d_gamma = upload(gamma);
        half* h_ref = upload(res);
        half* h_fold = upload(res);
        half* no_ref = alloc<half>(kD);
        half* no_fold = alloc<half>(kD);
        unsigned long long* ssq = alloc<unsigned long long>(1);
        half *g_ref = alloc<half>(kFfn), *u_ref = alloc<half>(kFfn);
        half *g_fold = alloc<half>(kFfn), *u_fold = alloc<half>(kFfn);
        float *lm_ref = alloc<float>(kFfn), *lm_fold = alloc<float>(kFfn);

        int64_t hs[2] = {1, kD}, gs[1] = {kD};
        Tensor th(h_ref, QType::F16, 2, hs, true), tno(no_ref, QType::F16, 2, hs, true);
        Tensor tg(const_cast<half*>(d_gamma), QType::F16, 1, gs, true);
        gemv_nvfp4_residual(w_out, d_y, h_ref, h_ref, kD, kInner, stream_);
        rmsnorm(th, tg, tno, kEps, stream_, weight_offset);
        gemv_nvfp4_gate_up_fused(w_g, w_u, no_ref, g_ref, u_ref, kFfn, kD, stream_);
        gemv_nvfp4_kpar_fp32(w_lm, no_ref, lm_ref, kFfn, kD, stream_);

        NvFP4NormFoldOut out{d_gamma, weight_offset, no_fold, ssq};
        NvFP4NormFoldIn in{ssq, static_cast<float>(kD), kEps};
        gemv_nvfp4_residual(w_out, d_y, h_fold, h_fold, kD, kInner, stream_, out);
        gemv_nvfp4_gate_up_fused(w_g, w_u, no_fold, g_fold, u_fold, kFfn, kD, stream_, in);
        gemv_nvfp4_kpar_fp32(w_lm, no_fold, lm_fold, kFfn, kD, stream_, in);

        const auto hr = fetch(h_ref, kD), hf = fetch(h_fold, kD);
        for (int i = 0; i < kD; ++i)
            ASSERT_EQ(hr[i], hf[i]) << "residual stream differs at " << i;
        const auto gr = fetch(g_ref, kFfn), gf = fetch(g_fold, kFfn);
        const auto ur = fetch(u_ref, kFfn), uf = fetch(u_fold, kFfn);
        const auto lr = fetch(lm_ref, kFfn), lf = fetch(lm_fold, kFfn);
        EXPECT_LT(rel_l2(gf, gr), 2e-3) << "gate";
        EXPECT_LT(rel_l2(uf, ur), 2e-3) << "up";
        EXPECT_LT(rel_l2(lf, lr), 2e-3) << "lm head";
        // The fold is not a no-op: the consumer read the prescaled vector, not the norm output.
        EXPECT_NE(fetch(no_fold, kD), fetch(no_ref, kD));
    }

    cudaStream_t stream_ = nullptr;
    std::vector<void*> bufs_;
    std::vector<NvFP4QuantResult> quants_;
};

TEST_F(NvFP4NormFold, MatchesRmsnormThenGemv) { run(0.0f); }
TEST_F(NvFP4NormFold, MatchesRmsnormThenGemvWithWeightOffset) { run(1.0f); }

}  // namespace
}  // namespace imp
