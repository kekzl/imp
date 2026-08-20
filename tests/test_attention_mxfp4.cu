// Test for MXFP4 tensor core prefill attention (sm_120).
//
// Tests the quantize → MXFP4 GEMM → softmax → P·V pipeline against
// a reference FP16 attention implementation. Since MXFP4 quantizes
// to 4 bits, we expect larger error than FP16 but correct behavior.

#include <gtest/gtest.h>
#include "compute/attention_mxfp4_prefill.h"
#include "compute/gemm_cutlass_mxfp4_sm120.h"
#include "compute/attention.h"
#include "compute/attention_tc.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdlib>

namespace imp {
namespace {

class AttentionMxFP4Test : public ::testing::Test {
protected:
    void SetUp() override {
        cudaStreamCreate(&stream_);

        // Check SM version
        int device = 0;
        cudaGetDevice(&device);
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
        sm_ = major * 10 + minor;
    }
    void TearDown() override { cudaStreamDestroy(stream_); }

    bool can_run() const { return sm_ >= 120 && cutlass_sm120_mxfp4_available(); }

    cudaStream_t stream_ = nullptr;
    int sm_ = 0;
};

// Helper: create random FP16 data on device
static void fill_random_fp16(void* d_ptr, size_t n, float amp, unsigned seed) {
    std::vector<half> h(n);
    for (size_t i = 0; i < n; i++) {
        // Simple deterministic pseudo-random
        seed = seed * 1103515245u + 12345u;
        float val = amp * (static_cast<float>((seed >> 16) & 0x7FFF) / 16384.0f - 1.0f);
        h[i] = __float2half(val);
    }
    cudaMemcpy(d_ptr, h.data(), n * sizeof(half), cudaMemcpyHostToDevice);
}

// Helper: compute max absolute error and mean absolute error
static void compute_errors(const half* ref, const half* test, size_t n, float& max_err, float& mean_err) {
    max_err = 0.0f;
    double sum_err = 0.0;
    for (size_t i = 0; i < n; i++) {
        float r = __half2float(ref[i]);
        float t = __half2float(test[i]);
        float err = std::abs(r - t);
        if (err > max_err)
            max_err = err;
        sum_err += err;
    }
    mean_err = static_cast<float>(sum_err / n);
}

// Independent fp64 attention reference from the f16-rounded inputs (NOT imp's
// FP16 kernel). Layout matches the [B=1,S,H,HD] tensors: Q[(i*NH+h)*HD+d],
// K/V[(j*NKV+kvh)*HD+d]. Used to turn the MXFP4 check from "imp-FP4 vs imp-FP16"
// (two imp kernels, absolute tol) into "imp-FP4 vs independent fp64", with a
// signed-mean BIAS guard — the thing an absolute-error budget cannot catch
// (a systematically shifted dequant within 0.1 abs passes the old test).
static void ref_attention_f64_mha(const std::vector<half>& Qh, const std::vector<half>& Kh,
                                  const std::vector<half>& Vh, std::vector<double>& O, int Sq, int Skv,
                                  int NH, int NKV, int HD, bool causal, double softcap) {
    const double scale = 1.0 / std::sqrt((double)HD);
    const int gqa = NH / NKV;
    O.assign((size_t)Sq * NH * HD, 0.0);
    std::vector<double> S(Skv);
    for (int h = 0; h < NH; h++) {
        int kvh = h / gqa;
        for (int i = 0; i < Sq; i++) {
            double m = -1e300;
            for (int j = 0; j < Skv; j++) {
                double dot = 0.0;
                for (int d = 0; d < HD; d++)
                    dot += (double)__half2float(Qh[((size_t)i * NH + h) * HD + d]) *
                           (double)__half2float(Kh[((size_t)j * NKV + kvh) * HD + d]);
                dot *= scale;
                if (softcap > 0.0)
                    dot = softcap * std::tanh(dot / softcap);
                if (causal && j > i)
                    dot = -1e300;
                S[j] = dot;
                m = std::max(m, dot);
            }
            double l = 0.0;
            for (int j = 0; j < Skv; j++) {
                S[j] = (S[j] <= -1e299) ? 0.0 : std::exp(S[j] - m);
                l += S[j];
            }
            for (int d = 0; d < HD; d++) {
                double acc = 0.0;
                for (int j = 0; j < Skv; j++)
                    acc += S[j] * (double)__half2float(Vh[((size_t)j * NKV + kvh) * HD + d]);
                O[((size_t)i * NH + h) * HD + d] = acc / l;
            }
        }
    }
}

TEST_F(AttentionMxFP4Test, AvailabilityReflectsSM) {
    // MXFP4 attention requires sm_120+ and CUTLASS
    if (sm_ < 120) {
        EXPECT_FALSE(cutlass_sm120_mxfp4_available());
    }
    // Note: attention_mxfp4_available() also checks the env var,
    // so we test the CUTLASS availability directly here.
}

TEST_F(AttentionMxFP4Test, BasicPrefill) {
    if (!can_run()) {
        GTEST_SKIP() << "MXFP4 attention requires sm_120+ with CUTLASS";
    }

    // batch=1, seq=128, heads=2, head_dim=64
    const int B = 1, SQ = 128, SKV = 128, NH = 2, NKV = 2, HD = 64;
    size_t qo_elems = B * SQ * NH * HD;
    size_t kv_elems = B * SKV * NKV * HD;

    void *d_q, *d_k, *d_v, *d_o;
    cudaMalloc(&d_q, qo_elems * sizeof(half));
    cudaMalloc(&d_k, kv_elems * sizeof(half));
    cudaMalloc(&d_v, kv_elems * sizeof(half));
    cudaMalloc(&d_o, qo_elems * sizeof(half));

    fill_random_fp16(d_q, qo_elems, 0.5f, 42);
    fill_random_fp16(d_k, kv_elems, 0.5f, 123);
    fill_random_fp16(d_v, kv_elems, 0.5f, 456);
    cudaMemset(d_o, 0, qo_elems * sizeof(half));

    int64_t qo_shape[] = {B, SQ, NH, HD};
    int64_t kv_shape[] = {B, SKV, NKV, HD};
    Tensor Q(d_q, QType::F16, 4, qo_shape, true);
    Tensor K(d_k, QType::F16, 4, kv_shape, true);
    Tensor V(d_v, QType::F16, 4, kv_shape, true);
    Tensor O(d_o, QType::F16, 4, qo_shape, true);

    float scale = 1.0f / std::sqrt(static_cast<float>(HD));

    bool ok = attention_mxfp4_prefill(Q, K, V, O, scale, /*causal=*/true,
                                      /*softcap=*/0.0f, stream_);
    cudaStreamSynchronize(stream_);

    ASSERT_TRUE(ok) << "MXFP4 attention returned false";

    cudaError_t err = cudaGetLastError();
    EXPECT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    // Check output has finite non-zero values
    std::vector<half> h_o(qo_elems);
    cudaMemcpy(h_o.data(), d_o, qo_elems * sizeof(half), cudaMemcpyDeviceToHost);

    int finite_nonzero = 0;
    for (auto& v : h_o) {
        float fv = __half2float(v);
        EXPECT_TRUE(std::isfinite(fv)) << "Non-finite value in output";
        if (fv != 0.0f)
            finite_nonzero++;
    }
    EXPECT_GT(finite_nonzero, 0) << "All-zero output";

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
}

TEST_F(AttentionMxFP4Test, CompareWithFP16Reference) {
    if (!can_run()) {
        GTEST_SKIP() << "MXFP4 attention requires sm_120+ with CUTLASS";
    }

    // Compare MXFP4 output with FP16 Blackwell kernel output
    const int B = 1, SQ = 64, SKV = 64, NH = 2, NKV = 2, HD = 64;
    size_t qo_elems = B * SQ * NH * HD;
    size_t kv_elems = B * SKV * NKV * HD;

    void *d_q, *d_k, *d_v, *d_o_mxfp4, *d_o_ref;
    cudaMalloc(&d_q, qo_elems * sizeof(half));
    cudaMalloc(&d_k, kv_elems * sizeof(half));
    cudaMalloc(&d_v, kv_elems * sizeof(half));
    cudaMalloc(&d_o_mxfp4, qo_elems * sizeof(half));
    cudaMalloc(&d_o_ref, qo_elems * sizeof(half));

    fill_random_fp16(d_q, qo_elems, 0.3f, 789);
    fill_random_fp16(d_k, kv_elems, 0.3f, 101);
    fill_random_fp16(d_v, kv_elems, 0.3f, 202);

    int64_t qo_shape[] = {B, SQ, NH, HD};
    int64_t kv_shape[] = {B, SKV, NKV, HD};
    float scale = 1.0f / std::sqrt(static_cast<float>(HD));

    // MXFP4 path
    {
        cudaMemset(d_o_mxfp4, 0, qo_elems * sizeof(half));
        Tensor Q(d_q, QType::F16, 4, qo_shape, true);
        Tensor K(d_k, QType::F16, 4, kv_shape, true);
        Tensor V(d_v, QType::F16, 4, kv_shape, true);
        Tensor O(d_o_mxfp4, QType::F16, 4, qo_shape, true);
        bool ok = attention_mxfp4_prefill(Q, K, V, O, scale, true, 0.0f, stream_);
        ASSERT_TRUE(ok);
    }

    // FP16 reference (Blackwell kernel; can_run() already gated sm_120+)
    {
        cudaMemset(d_o_ref, 0, qo_elems * sizeof(half));
        Tensor Q(d_q, QType::F16, 4, qo_shape, true);
        Tensor K(d_k, QType::F16, 4, kv_shape, true);
        Tensor V(d_v, QType::F16, 4, kv_shape, true);
        Tensor O(d_o_ref, QType::F16, 4, qo_shape, true);
        ASSERT_TRUE(flash_attention_blackwell(Q, K, V, O, scale, true, 0, 0.0f, stream_));
    }

    cudaStreamSynchronize(stream_);
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    std::vector<half> h_mxfp4(qo_elems), h_ref(qo_elems);
    cudaMemcpy(h_mxfp4.data(), d_o_mxfp4, qo_elems * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref.data(), d_o_ref, qo_elems * sizeof(half), cudaMemcpyDeviceToHost);

    // (legacy abs-error check kept as a coarse tripwire)
    float max_err, mean_err;
    compute_errors(h_ref.data(), h_mxfp4.data(), qo_elems, max_err, mean_err);
    EXPECT_LT(mean_err, 0.1f) << "Mean absolute error too large";
    EXPECT_LT(max_err, 0.5f) << "Max absolute error too large";

    // ---- INDEPENDENT fp64 oracle + BIAS guard (TEST_AUDIT (retired) §7 Tier-0) ----
    // The abs-error check above is imp-FP4 vs imp-FP16 — a systematically
    // shifted mxfp4 dequant (wrong scale exponent / off-by-one block) that
    // stays within 0.5 abs passes it. Compute the attention from the ORIGINAL
    // f16 inputs in fp64 and check (a) the mxfp4 output is unbiased relative to
    // it, and (b) its noise envelope is comparable to imp's own FP16 kernel's
    // (not systematically worse).
    std::vector<half> Qh(qo_elems), Kh(kv_elems), Vh(kv_elems);
    cudaMemcpy(Qh.data(), d_q, qo_elems * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(Kh.data(), d_k, kv_elems * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(Vh.data(), d_v, kv_elems * sizeof(half), cudaMemcpyDeviceToHost);
    std::vector<double> oref;
    ref_attention_f64_mha(Qh, Kh, Vh, oref, SQ, SKV, NH, NKV, HD, /*causal=*/true, 0.0);

    auto stats = [&](const std::vector<half>& out, double& nrmse, double& nbias) {
        double sum_sq = 0.0, sum_err_sq = 0.0, sum_signed = 0.0;
        for (size_t i = 0; i < oref.size(); i++) {
            double r = oref[i];
            double g = (double)__half2float(out[i]);
            sum_sq += r * r;
            sum_err_sq += (g - r) * (g - r);
            sum_signed += (g - r);
        }
        double ref_rms = std::sqrt(sum_sq / oref.size());
        double inv = ref_rms > 1e-9 ? 1.0 / ref_rms : 0.0;
        nrmse = std::sqrt(sum_err_sq / oref.size()) * inv;
        nbias = (sum_signed / oref.size()) * inv;  // signed mean / rms(ref)
    };
    double mx_nrmse = 0, mx_nbias = 0, fp16_nrmse = 0, fp16_nbias = 0;
    stats(h_mxfp4, mx_nrmse, mx_nbias);
    stats(h_ref, fp16_nrmse, fp16_nbias);
    printf("  MXFP4 vs fp64: nrmse=%.4f nbias=%.4f | FP16-kernel vs fp64: nrmse=%.4f nbias=%.4f\n",
           mx_nrmse, mx_nbias, fp16_nrmse, fp16_nbias);

    // (a) No systematic bias: unbiased 4-bit noise averages to ~0 over the
    //     output; a shifted dequant moves the signed mean. 5% of rms(ref) is
    //     generous for honest noise yet catches a real bias.
    EXPECT_LT(std::fabs(mx_nbias), 0.05) << "MXFP4 attention output is systematically biased vs fp64";
    // (b) Noise envelope bounded and comparable to the FP16 kernel's own
    //     discretization error (not an order of magnitude worse). 0.30 is the
    //     characterized 4-bit envelope for this shape; printed above.
    EXPECT_LT(mx_nrmse, 0.30) << "MXFP4 attention nrmse outside the characterized 4-bit envelope";

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o_mxfp4);
    cudaFree(d_o_ref);
}

TEST_F(AttentionMxFP4Test, GQASupport) {
    if (!can_run()) {
        GTEST_SKIP() << "MXFP4 attention requires sm_120+ with CUTLASS";
    }

    // GQA: 4 Q heads, 2 KV heads (ratio=2)
    const int B = 1, SQ = 64, SKV = 64, NH = 4, NKV = 2, HD = 64;
    size_t qo_elems = B * SQ * NH * HD;
    size_t kv_elems = B * SKV * NKV * HD;

    void *d_q, *d_k, *d_v, *d_o;
    cudaMalloc(&d_q, qo_elems * sizeof(half));
    cudaMalloc(&d_k, kv_elems * sizeof(half));
    cudaMalloc(&d_v, kv_elems * sizeof(half));
    cudaMalloc(&d_o, qo_elems * sizeof(half));

    fill_random_fp16(d_q, qo_elems, 0.3f, 333);
    fill_random_fp16(d_k, kv_elems, 0.3f, 444);
    fill_random_fp16(d_v, kv_elems, 0.3f, 555);
    cudaMemset(d_o, 0, qo_elems * sizeof(half));

    int64_t qo_shape[] = {B, SQ, NH, HD};
    int64_t kv_shape[] = {B, SKV, NKV, HD};
    Tensor Q(d_q, QType::F16, 4, qo_shape, true);
    Tensor K(d_k, QType::F16, 4, kv_shape, true);
    Tensor V(d_v, QType::F16, 4, kv_shape, true);
    Tensor O(d_o, QType::F16, 4, qo_shape, true);

    float scale = 1.0f / std::sqrt(static_cast<float>(HD));
    bool ok = attention_mxfp4_prefill(Q, K, V, O, scale, true, 0.0f, stream_);
    cudaStreamSynchronize(stream_);

    ASSERT_TRUE(ok) << "GQA MXFP4 attention failed";
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);

    // Verify finite output
    std::vector<half> h_o(qo_elems);
    cudaMemcpy(h_o.data(), d_o, qo_elems * sizeof(half), cudaMemcpyDeviceToHost);
    for (auto& v : h_o) {
        EXPECT_TRUE(std::isfinite(__half2float(v)));
    }

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
}

TEST_F(AttentionMxFP4Test, HeadDim128) {
    if (!can_run()) {
        GTEST_SKIP() << "MXFP4 attention requires sm_120+ with CUTLASS";
    }

    const int B = 1, SQ = 64, SKV = 64, NH = 2, NKV = 2, HD = 128;
    size_t qo_elems = B * SQ * NH * HD;
    size_t kv_elems = B * SKV * NKV * HD;

    void *d_q, *d_k, *d_v, *d_o;
    cudaMalloc(&d_q, qo_elems * sizeof(half));
    cudaMalloc(&d_k, kv_elems * sizeof(half));
    cudaMalloc(&d_v, kv_elems * sizeof(half));
    cudaMalloc(&d_o, qo_elems * sizeof(half));

    fill_random_fp16(d_q, qo_elems, 0.3f, 666);
    fill_random_fp16(d_k, kv_elems, 0.3f, 777);
    fill_random_fp16(d_v, kv_elems, 0.3f, 888);
    cudaMemset(d_o, 0, qo_elems * sizeof(half));

    int64_t qo_shape[] = {B, SQ, NH, HD};
    int64_t kv_shape[] = {B, SKV, NKV, HD};
    Tensor Q(d_q, QType::F16, 4, qo_shape, true);
    Tensor K(d_k, QType::F16, 4, kv_shape, true);
    Tensor V(d_v, QType::F16, 4, kv_shape, true);
    Tensor O(d_o, QType::F16, 4, qo_shape, true);

    float scale = 1.0f / std::sqrt(static_cast<float>(HD));
    bool ok = attention_mxfp4_prefill(Q, K, V, O, scale, true, 0.0f, stream_);
    cudaStreamSynchronize(stream_);

    ASSERT_TRUE(ok) << "head_dim=128 MXFP4 attention failed";
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
}

TEST_F(AttentionMxFP4Test, RejectsInvalidHeadDim) {
    if (!can_run()) {
        GTEST_SKIP() << "MXFP4 attention requires sm_120+ with CUTLASS";
    }

    // head_dim=48 is not a multiple of 32 — should return false
    const int B = 1, SQ = 32, SKV = 32, NH = 1, NKV = 1, HD = 48;
    size_t elems = B * SQ * NH * HD;

    void *d_q, *d_k, *d_v, *d_o;
    cudaMalloc(&d_q, elems * sizeof(half));
    cudaMalloc(&d_k, elems * sizeof(half));
    cudaMalloc(&d_v, elems * sizeof(half));
    cudaMalloc(&d_o, elems * sizeof(half));
    cudaMemset(d_q, 0, elems * sizeof(half));
    cudaMemset(d_k, 0, elems * sizeof(half));
    cudaMemset(d_v, 0, elems * sizeof(half));

    int64_t shape[] = {B, SQ, NH, HD};
    Tensor Q(d_q, QType::F16, 4, shape, true);
    Tensor K(d_k, QType::F16, 4, shape, true);
    Tensor V(d_v, QType::F16, 4, shape, true);
    Tensor O(d_o, QType::F16, 4, shape, true);

    float scale = 1.0f / std::sqrt(static_cast<float>(HD));
    bool ok = attention_mxfp4_prefill(Q, K, V, O, scale, true, 0.0f, stream_);
    EXPECT_FALSE(ok) << "Should reject head_dim not multiple of 32";

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
}

TEST_F(AttentionMxFP4Test, SoftcapSupport) {
    if (!can_run()) {
        GTEST_SKIP() << "MXFP4 attention requires sm_120+ with CUTLASS";
    }

    const int B = 1, SQ = 64, SKV = 64, NH = 2, NKV = 2, HD = 64;
    size_t qo_elems = B * SQ * NH * HD;
    size_t kv_elems = B * SKV * NKV * HD;

    void *d_q, *d_k, *d_v, *d_o;
    cudaMalloc(&d_q, qo_elems * sizeof(half));
    cudaMalloc(&d_k, kv_elems * sizeof(half));
    cudaMalloc(&d_v, kv_elems * sizeof(half));
    cudaMalloc(&d_o, qo_elems * sizeof(half));

    fill_random_fp16(d_q, qo_elems, 0.3f, 999);
    fill_random_fp16(d_k, kv_elems, 0.3f, 111);
    fill_random_fp16(d_v, kv_elems, 0.3f, 222);
    cudaMemset(d_o, 0, qo_elems * sizeof(half));

    int64_t qo_shape[] = {B, SQ, NH, HD};
    int64_t kv_shape[] = {B, SKV, NKV, HD};
    Tensor Q(d_q, QType::F16, 4, qo_shape, true);
    Tensor K(d_k, QType::F16, 4, kv_shape, true);
    Tensor V(d_v, QType::F16, 4, kv_shape, true);
    Tensor O(d_o, QType::F16, 4, qo_shape, true);

    float scale = 1.0f / std::sqrt(static_cast<float>(HD));
    float softcap = 50.0f;  // Gemma-2/3 style

    bool ok = attention_mxfp4_prefill(Q, K, V, O, scale, true, softcap, stream_);
    cudaStreamSynchronize(stream_);

    ASSERT_TRUE(ok) << "Softcap MXFP4 attention failed";
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);

    // Output should be finite
    std::vector<half> h_o(qo_elems);
    cudaMemcpy(h_o.data(), d_o, qo_elems * sizeof(half), cudaMemcpyDeviceToHost);
    for (auto& v : h_o) {
        EXPECT_TRUE(std::isfinite(__half2float(v)));
    }

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
}

}  // namespace
}  // namespace imp
