// Standalone repro for the gemma-3-12b GGUF decode IMA (known issue): decode
// on the NVFP4 cache poisons the CUDA context asynchronously right after the
// first decode step (sticky 700 surfaces at sample_single_from_logits;
// bisected: NVFP4 decode cache ON + per-layer GEMVs — LM head exonerated via
// generation.lm_dequant_fp16, graphs exonerated via --no-cuda-graphs, and
// diagnostics.no_nvfp4_decode_cache makes the model fully coherent).
//
// Exercises every NVFP4 decode-GEMV entry the engine dispatches for a
// gemma-3-12b layer at its real dims (d_model=3840, q=4096, kv=2048,
// d_ff=15360, GeGLU): kpar per projection, the QKV fusion, the gate+up
// fusion, and the GeGLU-residual down projection (gemma-3 is the only GeGLU
// user — the least-exercised path). Checks NaN/Inf + a post-call device
// health probe; run under compute-sanitizer to pin an out-of-bounds access.

#include "quant/nvfp4_quant.h"
#include "quant/nvfp4_gemm.h"
#include "core/tensor.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <string>
#include <vector>

namespace imp {
namespace {

constexpr int kDModel = 3840;
constexpr int kQRows = 4096;
constexpr int kKVRows = 2048;
constexpr int kFfn = 15360;

class NvFP4GemvGemma3Dims : public ::testing::Test {
protected:
    void SetUp() override {
        if (cudaSetDevice(0) != cudaSuccess)
            GTEST_SKIP() << "No CUDA device";
        cudaStreamCreate(&stream_);
    }
    void TearDown() override {
        if (stream_)
            cudaStreamDestroy(stream_);
    }

    // Deterministic synthetic FP16 weight [N, K], quantized to NVFP4.
    void make_weight(int N, int K, half** d_w, NvFP4QuantResult& qr) {
        std::vector<half> h_w(static_cast<size_t>(N) * K);
        for (size_t i = 0; i < h_w.size(); ++i)
            h_w[i] = __float2half(((static_cast<int>(i * 17u) % 31) - 15) * 0.01f);
        ASSERT_EQ(cudaMalloc(d_w, h_w.size() * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(*d_w, h_w.data(), h_w.size() * sizeof(half), cudaMemcpyHostToDevice),
                  cudaSuccess);
        int64_t wshape[2] = {N, K};
        Tensor w_t(*d_w, QType::F16, 2, wshape, /*on_device=*/true);
        quantize_fp16_to_nvfp4(w_t, qr, stream_);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess) << "quantize failed for N=" << N
                                                               << " K=" << K;
    }

    half* make_vec(int n, float scale) {
        std::vector<half> h(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i)
            h[i] = __float2half(((i * 23) % 29 - 14) * scale);
        half* d = nullptr;
        EXPECT_EQ(cudaMalloc(&d, h.size() * sizeof(half)), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(d, h.data(), h.size() * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);
        return d;
    }

    // Sync + sticky-error check + NaN/Inf scan of a device half buffer.
    void check_output(const char* what, const half* d_y, int n) {
        cudaError_t sync = cudaStreamSynchronize(stream_);
        ASSERT_EQ(sync, cudaSuccess) << what << ": stream sync failed (" << cudaGetErrorString(sync)
                                     << ") — kernel poisoned the context";
        std::vector<half> h(static_cast<size_t>(n));
        ASSERT_EQ(cudaMemcpy(h.data(), d_y, h.size() * sizeof(half), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        int nan_cnt = 0, inf_cnt = 0;
        for (const half& v : h) {
            float f = __half2float(v);
            if (std::isnan(f))
                ++nan_cnt;
            else if (std::isinf(f))
                ++inf_cnt;
        }
        EXPECT_EQ(nan_cnt, 0) << what << " produced NaNs";
        EXPECT_EQ(inf_cnt, 0) << what << " produced Infs";
    }

    cudaStream_t stream_ = nullptr;
};

// Every per-projection kpar GEMV shape a gemma-3-12b decode step dispatches.
TEST_F(NvFP4GemvGemma3Dims, KparPerProjectionShapes) {
    struct Shape {
        const char* name;
        int N, K;
    };
    const Shape shapes[] = {
        {"wq [4096,3840]", kQRows, kDModel},   {"wk/wv [2048,3840]", kKVRows, kDModel},
        {"wo [3840,4096]", kDModel, kQRows},   {"w_gate/w_up [15360,3840]", kFfn, kDModel},
        {"w_down [3840,15360]", kDModel, kFfn},
    };
    for (const auto& s : shapes) {
        SCOPED_TRACE(s.name);
        half* d_w = nullptr;
        NvFP4QuantResult qr;
        make_weight(s.N, s.K, &d_w, qr);
        half* d_x = make_vec(s.K, 0.02f);
        half* d_y = nullptr;
        ASSERT_EQ(cudaMalloc(&d_y, static_cast<size_t>(s.N) * sizeof(half)), cudaSuccess);
        cudaMemsetAsync(d_y, 0, static_cast<size_t>(s.N) * sizeof(half), stream_);

        gemv_nvfp4_kpar(qr, d_x, d_y, s.N, s.K, stream_);
        check_output(s.name, d_y, s.N);

        free_nvfp4_result(qr);
        cudaFree(d_w);
        cudaFree(d_x);
        cudaFree(d_y);
    }
}

// The fused QKV projection at gemma-3 head geometry (hd=256: q=4096, kv=2048).
TEST_F(NvFP4GemvGemma3Dims, QkvFusedAtGemma3Heads) {
    half *d_wq = nullptr, *d_wk = nullptr, *d_wv = nullptr;
    NvFP4QuantResult qq, qk, qv;
    make_weight(kQRows, kDModel, &d_wq, qq);
    make_weight(kKVRows, kDModel, &d_wk, qk);
    make_weight(kKVRows, kDModel, &d_wv, qv);
    half* d_x = make_vec(kDModel, 0.02f);
    half *d_yq = nullptr, *d_yk = nullptr, *d_yv = nullptr;
    ASSERT_EQ(cudaMalloc(&d_yq, kQRows * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_yk, kKVRows * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_yv, kKVRows * sizeof(half)), cudaSuccess);

    gemv_nvfp4_qkv_fused(qq, qk, qv, d_x, d_yq, d_yk, d_yv, kQRows, kKVRows, kKVRows, kDModel, stream_);
    check_output("qkv_fused q", d_yq, kQRows);
    check_output("qkv_fused k", d_yk, kKVRows);
    check_output("qkv_fused v", d_yv, kKVRows);

    free_nvfp4_result(qq);
    free_nvfp4_result(qk);
    free_nvfp4_result(qv);
    for (auto* p : {d_wq, d_wk, d_wv, d_x, d_yq, d_yk, d_yv})
        cudaFree(p);
}

// Fused gate+up + the GeGLU-residual down projection — the gemma-3-only path.
TEST_F(NvFP4GemvGemma3Dims, GateUpFusedAndGegluResidualDown) {
    half *d_wg = nullptr, *d_wu = nullptr, *d_wd = nullptr;
    NvFP4QuantResult qg, qu, qd;
    make_weight(kFfn, kDModel, &d_wg, qg);
    make_weight(kFfn, kDModel, &d_wu, qu);
    make_weight(kDModel, kFfn, &d_wd, qd);
    half* d_x = make_vec(kDModel, 0.02f);
    half* d_res = make_vec(kDModel, 0.01f);
    half *d_yg = nullptr, *d_yu = nullptr, *d_y = nullptr;
    ASSERT_EQ(cudaMalloc(&d_yg, kFfn * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_yu, kFfn * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, kDModel * sizeof(half)), cudaSuccess);

    gemv_nvfp4_gate_up_fused(qg, qu, d_x, d_yg, d_yu, kFfn, kDModel, stream_);
    check_output("gate_up_fused gate", d_yg, kFfn);
    check_output("gate_up_fused up", d_yu, kFfn);

    gemv_nvfp4_geglu_residual(qd, d_yg, d_yu, d_y, d_res, kDModel, kFfn, stream_);
    check_output("geglu_residual down", d_y, kDModel);

    free_nvfp4_result(qg);
    free_nvfp4_result(qu);
    free_nvfp4_result(qd);
    for (auto* p : {d_wg, d_wu, d_wd, d_x, d_res, d_yg, d_yu, d_y})
        cudaFree(p);
}

}  // namespace
}  // namespace imp
