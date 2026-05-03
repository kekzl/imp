// Test for MXFP4 Flash Attention (sm_120): tiled FP4 E2M1 Q·K^T with online softmax.
//
// Compares the fused flash attention kernel output against the FP16 Blackwell
// reference kernel. FP4 quantization introduces ~1-2 bits of error.

#include <gtest/gtest.h>
#include "compute/attention_fmha_mxfp4_sm120.h"
#include "compute/attention.h"
#include "compute/attention_tc.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>

namespace imp {
namespace {

class FhmaMxFP4Test : public ::testing::Test {
protected:
    void SetUp() override {
        cudaStreamCreate(&stream_);
        int device = 0;
        cudaGetDevice(&device);
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
        sm_ = major * 10 + minor;
    }
    void TearDown() override { cudaStreamDestroy(stream_); }
    bool can_run() const { return sm_ >= 120; }

    cudaStream_t stream_ = nullptr;
    int sm_ = 0;
};

static void fill_random_fp16(void* d_ptr, size_t n, float amp, unsigned seed) {
    std::vector<half> h(n);
    for (size_t i = 0; i < n; i++) {
        seed = seed * 1103515245u + 12345u;
        float val = amp * (static_cast<float>((seed >> 16) & 0x7FFF) / 16384.0f - 1.0f);
        h[i] = __float2half(val);
    }
    cudaMemcpy(d_ptr, h.data(), n * sizeof(half), cudaMemcpyHostToDevice);
}

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

// Helper to run test with given dimensions
void run_compare_test(int B, int SQ, int SKV, int NH, int NKV, int HD, bool causal, int sliding_window,
                      float softcap, float max_err_limit, float mean_err_limit, cudaStream_t stream, int sm) {
    size_t qo_elems = (size_t)B * SQ * NH * HD;
    size_t kv_elems = (size_t)B * SKV * NKV * HD;

    void *d_q, *d_k, *d_v, *d_o_mxfp4, *d_o_ref;
    cudaMalloc(&d_q, qo_elems * sizeof(half));
    cudaMalloc(&d_k, kv_elems * sizeof(half));
    cudaMalloc(&d_v, kv_elems * sizeof(half));
    cudaMalloc(&d_o_mxfp4, qo_elems * sizeof(half));
    cudaMalloc(&d_o_ref, qo_elems * sizeof(half));

    fill_random_fp16(d_q, qo_elems, 0.3f, 42 + HD);
    fill_random_fp16(d_k, kv_elems, 0.3f, 123 + HD);
    fill_random_fp16(d_v, kv_elems, 0.3f, 456 + HD);

    int64_t qo_shape[] = {B, SQ, NH, HD};
    int64_t kv_shape[] = {B, SKV, NKV, HD};
    float scale = 1.0f / std::sqrt(static_cast<float>(HD));

    // MXFP4 Flash Attention
    {
        cudaMemset(d_o_mxfp4, 0, qo_elems * sizeof(half));
        Tensor Q(d_q, QType::F16, 4, qo_shape, true);
        Tensor K(d_k, QType::F16, 4, kv_shape, true);
        Tensor V(d_v, QType::F16, 4, kv_shape, true);
        Tensor O(d_o_mxfp4, QType::F16, 4, qo_shape, true);
        bool ok = fmha_sm120_mxfp4_prefill(Q, K, V, O, scale, causal, sliding_window, softcap, stream);
        ASSERT_TRUE(ok) << "MXFP4 FMHA returned false for HD=" << HD;
    }

    // FP16 reference
    {
        cudaMemset(d_o_ref, 0, qo_elems * sizeof(half));
        Tensor Q(d_q, QType::F16, 4, qo_shape, true);
        Tensor K(d_k, QType::F16, 4, kv_shape, true);
        Tensor V(d_v, QType::F16, 4, kv_shape, true);
        Tensor O(d_o_ref, QType::F16, 4, qo_shape, true);
        if (sm >= 120)
            flash_attention_blackwell(Q, K, V, O, scale, causal, sliding_window, softcap, stream);
        else
            flash_attention_prefill_tc(Q, K, V, O, scale, causal, sliding_window, softcap, stream);
    }

    cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    std::vector<half> h_mxfp4(qo_elems), h_ref(qo_elems);
    cudaMemcpy(h_mxfp4.data(), d_o_mxfp4, qo_elems * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref.data(), d_o_ref, qo_elems * sizeof(half), cudaMemcpyDeviceToHost);

    float max_err, mean_err;
    compute_errors(h_ref.data(), h_mxfp4.data(), qo_elems, max_err, mean_err);

    printf("  HD=%d SQ=%d SKV=%d causal=%d sw=%d softcap=%.1f: max_err=%.4f mean_err=%.6f\n", HD, SQ, SKV,
           causal, sliding_window, softcap, max_err, mean_err);

    EXPECT_LT(mean_err, mean_err_limit) << "Mean error too large";
    EXPECT_LT(max_err, max_err_limit) << "Max error too large";

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o_mxfp4);
    cudaFree(d_o_ref);
}

TEST_F(FhmaMxFP4Test, BasicHD64) {
    if (!can_run())
        GTEST_SKIP() << "Requires sm_120+";
    run_compare_test(1, 128, 128, 4, 2, 64, true, 0, 0.0f, 0.5f, 0.1f, stream_, sm_);
}

TEST_F(FhmaMxFP4Test, BasicHD128) {
    if (!can_run())
        GTEST_SKIP() << "Requires sm_120+";
    run_compare_test(1, 128, 128, 4, 2, 128, true, 0, 0.0f, 0.5f, 0.1f, stream_, sm_);
}

TEST_F(FhmaMxFP4Test, DISABLED_BasicHD256) {
    if (!can_run())
        GTEST_SKIP() << "Requires sm_120+";
    // HD=256 requires large shared memory; disabled pending smem optimization
    run_compare_test(1, 64, 64, 2, 2, 256, true, 0, 0.0f, 0.5f, 0.1f, stream_, sm_);
}

TEST_F(FhmaMxFP4Test, NonCausal) {
    if (!can_run())
        GTEST_SKIP() << "Requires sm_120+";
    run_compare_test(1, 128, 128, 4, 2, 128, false, 0, 0.0f, 0.5f, 0.1f, stream_, sm_);
}

TEST_F(FhmaMxFP4Test, SlidingWindow) {
    if (!can_run())
        GTEST_SKIP() << "Requires sm_120+";
    run_compare_test(1, 256, 256, 4, 2, 128, true, 64, 0.0f, 0.5f, 0.1f, stream_, sm_);
}

TEST_F(FhmaMxFP4Test, Softcap) {
    if (!can_run())
        GTEST_SKIP() << "Requires sm_120+";
    run_compare_test(1, 128, 128, 4, 2, 128, true, 0, 50.0f, 0.5f, 0.1f, stream_, sm_);
}

TEST_F(FhmaMxFP4Test, LongSequence) {
    if (!can_run())
        GTEST_SKIP() << "Requires sm_120+";
    // pp512 — multiple KV tiles
    run_compare_test(1, 512, 512, 4, 2, 128, true, 0, 0.0f, 0.5f, 0.1f, stream_, sm_);
}

TEST_F(FhmaMxFP4Test, AsymmetricSeqLens) {
    if (!can_run())
        GTEST_SKIP() << "Requires sm_120+";
    // Q shorter than KV (common in generation with KV cache)
    run_compare_test(1, 32, 256, 4, 2, 128, true, 0, 0.0f, 0.5f, 0.1f, stream_, sm_);
}

TEST_F(FhmaMxFP4Test, BasicHD96) {
    if (!can_run())
        GTEST_SKIP() << "Requires sm_120+";
    // HD=96 is a multiple of 32, should work with FP4 m16n8k32
    run_compare_test(1, 128, 128, 4, 2, 96, true, 0, 0.0f, 0.5f, 0.1f, stream_, sm_);
}

TEST_F(FhmaMxFP4Test, LongSequence2K) {
    if (!can_run())
        GTEST_SKIP() << "Requires sm_120+";
    // pp2048 — stress test with many KV tiles
    run_compare_test(1, 2048, 2048, 4, 2, 128, true, 0, 0.0f, 0.5f, 0.1f, stream_, sm_);
}

TEST_F(FhmaMxFP4Test, MHA_1to1) {
    if (!can_run())
        GTEST_SKIP() << "Requires sm_120+";
    // Multi-head attention (no GQA), 1:1 Q:KV ratio
    run_compare_test(1, 256, 256, 8, 8, 128, true, 0, 0.0f, 0.5f, 0.1f, stream_, sm_);
}

TEST_F(FhmaMxFP4Test, GQA_8x) {
    if (!can_run())
        GTEST_SKIP() << "Requires sm_120+";
    // High GQA ratio (8:1)
    run_compare_test(1, 256, 256, 32, 4, 128, true, 0, 0.0f, 0.5f, 0.1f, stream_, sm_);
}

TEST_F(FhmaMxFP4Test, HD64_LongSeq) {
    if (!can_run())
        GTEST_SKIP() << "Requires sm_120+";
    run_compare_test(1, 1024, 1024, 4, 2, 64, true, 0, 0.0f, 0.5f, 0.1f, stream_, sm_);
}

// Compare legacy vs blockscale paths. Both paths share the same Q/K quant
// and post-MMA scale application; only the MMA instruction differs.
// kind::mxf4nvf4.m16n8k64 internally sums 64 FP4 products per issue in FP32,
// whereas legacy f8f6f4.m16n8k32 sums 32 products per issue and then adds
// the partials. FP32 is not associative, so per-element differences at the
// ULP × sqrt(K) level are expected. After the softmax+PV pipeline, those
// tiny score deltas can propagate into a few-percent output outliers
// (max_err ~0.1), while mean_err stays at noise level (~1e-3).
static void run_blockscale_ab_test(int B, int SQ, int SKV, int NH, int NKV, int HD, bool causal,
                                   int sliding_window, float softcap, float max_err_limit,
                                   float mean_err_limit, cudaStream_t stream, int sm) {
    ASSERT_EQ(HD % 64, 0) << "Blockscale requires HD % 64 == 0";
    size_t qo_elems = (size_t)B * SQ * NH * HD;
    size_t kv_elems = (size_t)B * SKV * NKV * HD;

    void *d_q, *d_k, *d_v, *d_o_leg, *d_o_bs;
    cudaMalloc(&d_q, qo_elems * sizeof(half));
    cudaMalloc(&d_k, kv_elems * sizeof(half));
    cudaMalloc(&d_v, kv_elems * sizeof(half));
    cudaMalloc(&d_o_leg, qo_elems * sizeof(half));
    cudaMalloc(&d_o_bs, qo_elems * sizeof(half));

    fill_random_fp16(d_q, qo_elems, 0.3f, 42 + HD);
    fill_random_fp16(d_k, kv_elems, 0.3f, 123 + HD);
    fill_random_fp16(d_v, kv_elems, 0.3f, 456 + HD);

    int64_t qo_shape[] = {B, SQ, NH, HD};
    int64_t kv_shape[] = {B, SKV, NKV, HD};
    float scale = 1.0f / std::sqrt(static_cast<float>(HD));

    Tensor Q(d_q, QType::F16, 4, qo_shape, true);
    Tensor K(d_k, QType::F16, 4, kv_shape, true);
    Tensor V(d_v, QType::F16, 4, kv_shape, true);

    {
        cudaMemset(d_o_leg, 0, qo_elems * sizeof(half));
        Tensor O(d_o_leg, QType::F16, 4, qo_shape, true);
        bool ok = fmha_sm120_mxfp4_prefill(Q, K, V, O, scale, causal, sliding_window, softcap, stream,
                                           /*use_blockscale=*/false);
        ASSERT_TRUE(ok);
    }
    {
        cudaMemset(d_o_bs, 0, qo_elems * sizeof(half));
        Tensor O(d_o_bs, QType::F16, 4, qo_shape, true);
        bool ok = fmha_sm120_mxfp4_prefill(Q, K, V, O, scale, causal, sliding_window, softcap, stream,
                                           /*use_blockscale=*/true);
        ASSERT_TRUE(ok);
    }

    cudaStreamSynchronize(stream);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    std::vector<half> h_leg(qo_elems), h_bs(qo_elems);
    cudaMemcpy(h_leg.data(), d_o_leg, qo_elems * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bs.data(), d_o_bs, qo_elems * sizeof(half), cudaMemcpyDeviceToHost);

    float max_err, mean_err;
    compute_errors(h_leg.data(), h_bs.data(), qo_elems, max_err, mean_err);

    std::printf(
        "  [AB] HD=%d SQ=%d SKV=%d causal=%d sw=%d softcap=%.1f: "
        "max_err=%.4f mean_err=%.6f\n",
        HD, SQ, SKV, causal, sliding_window, softcap, max_err, mean_err);

    EXPECT_LT(mean_err, mean_err_limit);
    EXPECT_LT(max_err, max_err_limit);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o_leg);
    cudaFree(d_o_bs);
}

TEST_F(FhmaMxFP4Test, Blockscale_MatchesLegacy_HD64) {
    if (!can_run())
        GTEST_SKIP() << "Requires sm_120+";
    run_blockscale_ab_test(1, 256, 256, 4, 2, 64, true, 0, 0.0f, 0.25f, 0.01f, stream_, sm_);
}

TEST_F(FhmaMxFP4Test, Blockscale_MatchesLegacy_HD128) {
    if (!can_run())
        GTEST_SKIP() << "Requires sm_120+";
    run_blockscale_ab_test(1, 256, 256, 4, 2, 128, true, 0, 0.0f, 0.25f, 0.01f, stream_, sm_);
}

TEST_F(FhmaMxFP4Test, Blockscale_MatchesLegacy_HD128_LongSeq) {
    if (!can_run())
        GTEST_SKIP() << "Requires sm_120+";
    run_blockscale_ab_test(1, 1024, 1024, 4, 2, 128, true, 0, 0.0f, 0.25f, 0.01f, stream_, sm_);
}

TEST_F(FhmaMxFP4Test, Blockscale_MatchesLegacy_Softcap) {
    if (!can_run())
        GTEST_SKIP() << "Requires sm_120+";
    run_blockscale_ab_test(1, 256, 256, 4, 2, 128, true, 0, 50.0f, 0.25f, 0.01f, stream_, sm_);
}

TEST_F(FhmaMxFP4Test, Blockscale_MatchesLegacy_SlidingWindow) {
    if (!can_run())
        GTEST_SKIP() << "Requires sm_120+";
    run_blockscale_ab_test(1, 512, 512, 4, 2, 128, true, 64, 0.0f, 0.25f, 0.01f, stream_, sm_);
}

TEST_F(FhmaMxFP4Test, RejectsHD48) {
    if (!can_run())
        GTEST_SKIP() << "Requires sm_120+";
    // HD=48 is not a multiple of 32
    const int B = 1, SQ = 64, SKV = 64, NH = 2, NKV = 2, HD = 48;
    size_t elems = (size_t)B * SQ * NH * HD;

    void *d_q, *d_k, *d_v, *d_o;
    cudaMalloc(&d_q, elems * sizeof(half));
    cudaMalloc(&d_k, elems * sizeof(half));
    cudaMalloc(&d_v, elems * sizeof(half));
    cudaMalloc(&d_o, elems * sizeof(half));
    cudaMemset(d_q, 0, elems * sizeof(half));
    cudaMemset(d_k, 0, elems * sizeof(half));
    cudaMemset(d_v, 0, elems * sizeof(half));

    int64_t qo_shape[] = {B, SQ, NH, HD};
    int64_t kv_shape[] = {B, SKV, NKV, HD};
    Tensor Q(d_q, QType::F16, 4, qo_shape, true);
    Tensor K(d_k, QType::F16, 4, kv_shape, true);
    Tensor V(d_v, QType::F16, 4, kv_shape, true);
    Tensor O(d_o, QType::F16, 4, qo_shape, true);

    float scale = 1.0f / std::sqrt(static_cast<float>(HD));
    bool ok = fmha_sm120_mxfp4_prefill(Q, K, V, O, scale, true, 0, 0.0f, stream_);
    EXPECT_FALSE(ok) << "Should reject HD=48 (not multiple of 32)";

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
}

}  // namespace
}  // namespace imp
