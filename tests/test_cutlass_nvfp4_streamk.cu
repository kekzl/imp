// Stream-K variant of the CUTLASS sm_120 NVFP4 prefill GEMM
// (gemm.nvfp4_cutlass_streamk): correctness against the data-parallel tile
// and an isolated bench on the Qwen3-14B pp512 shapes, where the 128x128
// grid quantises to 0.94 waves (N=5120: 160 CTAs on 170 SMs). The weight
// ring rotates 256 MB of copies to defeat the 96 MB L2, so the numbers read
// DRAM like the real prefill does. GPU required - skips without one.

#include "compute/gemm_cutlass_sm120.h"
#include "core/tensor.h"
#include "quant/nvfp4_quant.h"
#include "core/process_diag.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace imp;

namespace {

bool gpu_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

struct Shape {
    int M, N, K;
    const char* name;
};

// One weight + one activation on device, CUTLASS-ready.
struct Operands {
    NvFP4QuantResult qr{};
    CutlassNvFP4Weight cw{};
    void* act_data = nullptr;
    void* act_sf = nullptr;
    half* d_w = nullptr;
    half* d_x = nullptr;

    void build(int M, int N, int K, int seed, cudaStream_t stream) {
        std::vector<half> h_w((size_t)N * K), h_x((size_t)M * K);
        for (size_t i = 0; i < h_w.size(); ++i)
            h_w[i] = __float2half((float)((int)((i * 7 + seed) % 13) - 6) * 0.05f);
        for (size_t i = 0; i < h_x.size(); ++i)
            h_x[i] = __float2half((float)((int)((i * 11 + 3) % 17) - 8) * 0.05f);
        cudaMalloc(&d_w, h_w.size() * sizeof(half));
        cudaMalloc(&d_x, h_x.size() * sizeof(half));
        cudaMemcpy(d_w, h_w.data(), h_w.size() * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(half), cudaMemcpyHostToDevice);
        int64_t wshape[2] = {N, K};
        Tensor w_t(d_w, QType::F16, 2, wshape, true);
        quantize_fp16_to_nvfp4(w_t, qr, stream);
        convert_nvfp4_to_cutlass(qr, cw, stream);
        size_t act_sf_bytes = cutlass_nvfp4_sf_size(M, K);
        cudaMalloc(&act_data, (size_t)M * K / 2);
        cudaMalloc(&act_sf, act_sf_bytes);
        cudaMemsetAsync(act_sf, 0, act_sf_bytes, stream);
        quantize_fp16_to_nvfp4_cutlass(d_x, act_data, act_sf, M, K, stream);
        cudaStreamSynchronize(stream);
    }
    void release() {
        free_cutlass_nvfp4_weight(cw);
        free_nvfp4_result(qr);
        cudaFree(act_data);
        cudaFree(act_sf);
        cudaFree(d_w);
        cudaFree(d_x);
    }
};

class CutlassNvfp4StreamKTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!gpu_available())
            GTEST_SKIP() << "no GPU";
        cudaStreamCreate(&stream_);
        process_diag_set_nvfp4_cutlass_streamk(0);  // the dp arm goes through the default entry
    }
    void TearDown() override {
        process_diag_set_nvfp4_cutlass_streamk(1);
        if (stream_)
            cudaStreamDestroy(stream_);
    }
    cudaStream_t stream_ = nullptr;
};

float max_rel_diff(const std::vector<half>& a, const std::vector<half>& b) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
        max_abs = std::max(max_abs, std::fabs(__half2float(b[i])));
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
        worst = std::max(worst,
                         std::fabs(__half2float(a[i]) - __half2float(b[i])) / std::max(1e-3f, max_abs));
    return worst;
}

}  // namespace

// Forced stream-K must reproduce the data-parallel output: same MMAs, the
// K-split partials are reduced in f32 through the scheduler workspace.
TEST_F(CutlassNvfp4StreamKTest, MatchesDataParallel) {
    const Shape shapes[] = {{512, 5120, 5120, "q_proj"},
                            {512, 5120, 17408, "down"},
                            {200, 5120, 5120, "odd_m"},
                            {512, 17408, 5120, "gate_up"}};
    for (const auto& s : shapes) {
        Operands op;
        op.build(s.M, s.N, s.K, 1, stream_);
        size_t ws_dp = gemm_nvfp4_cutlass_sm120_workspace(s.M, s.N, s.K);
        size_t ws_sk = gemm_nvfp4_cutlass_sm120_streamk_workspace(s.M, s.N, s.K);
        size_t ws_bytes = std::max(ws_dp, ws_sk);
        void* ws = nullptr;
        cudaMalloc(&ws, ws_bytes > 0 ? ws_bytes : 1);
        half *y_dp = nullptr, *y_sk = nullptr;
        cudaMalloc(&y_dp, (size_t)s.M * s.N * sizeof(half));
        cudaMalloc(&y_sk, (size_t)s.M * s.N * sizeof(half));

        ASSERT_TRUE(gemm_nvfp4_cutlass_sm120(op.act_data, op.act_sf, op.cw, y_dp, s.M, s.N, s.K, ws, ws_bytes,
                                             stream_))
            << s.name;
        ASSERT_TRUE(gemm_nvfp4_cutlass_sm120_streamk(op.act_data, op.act_sf, op.cw, y_sk, s.M, s.N, s.K, ws,
                                                     ws_bytes, stream_, /*force=*/true))
            << s.name;
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        std::vector<half> h_dp((size_t)s.M * s.N), h_sk((size_t)s.M * s.N);
        cudaMemcpy(h_dp.data(), y_dp, h_dp.size() * sizeof(half), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_sk.data(), y_sk, h_sk.size() * sizeof(half), cudaMemcpyDeviceToHost);
        float rel = max_rel_diff(h_sk, h_dp);
        const int units_forced = gemm_nvfp4_cutlass_sm120_streamk_units(s.M, s.N, s.K, true);
        const int units_heur = gemm_nvfp4_cutlass_sm120_streamk_units(s.M, s.N, s.K, false);
        printf(
            "[streamk] %-8s M=%d N=%d K=%d  max_rel(sk vs dp)=%.2e  ws dp=%zu sk=%zu B  sk_units forced=%d "
            "heuristic=%d\n",
            s.name, s.M, s.N, s.K, rel, ws_dp, ws_sk, units_forced, units_heur);
        // A forced stream-K arm that launched zero stream-K units measured
        // the data-parallel kernel under another name.
        EXPECT_GT(units_forced, 0) << s.name;
        // The engine sizes ONE workspace at its max shape with the mode set;
        // it must cover what this shape's stream-K launch needs, or the GEMM
        // refuses and the whole projection falls back to dequant+cuBLAS.
        process_diag_set_nvfp4_cutlass_streamk(1);
        const size_t ws_engine = gemm_nvfp4_cutlass_sm120_workspace(4096, s.N, s.K);
        process_diag_set_nvfp4_cutlass_streamk(0);
        EXPECT_GE(ws_engine, ws_sk) << s.name << ": max-shape workspace does not cover the dispatched shape";
        // f16 output of an f32 reduction whose summation order differs.
        EXPECT_LT(rel, 2e-3f) << s.name;

        cudaFree(y_dp);
        cudaFree(y_sk);
        cudaFree(ws);
        op.release();
    }
}

// Isolated bench: data-parallel vs stream-K (heuristic and forced) on the
// pp512 shapes, weight ring >= 256 MB so every launch reads DRAM.
TEST_F(CutlassNvfp4StreamKTest, BenchPp512Shapes) {
    const Shape shapes[] = {{256, 5120, 5120, "m256"},     {384, 5120, 5120, "m384"},
                            {512, 5120, 5120, "q_proj"},   {640, 5120, 5120, "m640"},
                            {768, 5120, 5120, "m768"},     {1024, 5120, 5120, "m1024"},
                            {2048, 5120, 5120, "m2048"},   {512, 5120, 17408, "down"},
                            {1024, 5120, 17408, "down1k"}, {512, 17408, 5120, "gate_up"},
                            {1024, 17408, 5120, "gu1k"}};
    for (const auto& s : shapes) {
        const size_t w_bytes = (size_t)s.N * s.K / 2;
        // 2x the 96 MB L2: at 128 MB the ring was bistable (q_proj 27 vs 45
        // us between runs) because the one-CTA-per-SM K loop is latency
        // bound and a partially resident ring reads as L2 latency.
        const int ring = std::max<int>(2, (int)((256u << 20) / w_bytes) + 1);
        std::vector<Operands> ops(ring);
        for (int r = 0; r < ring; ++r)
            ops[r].build(s.M, s.N, s.K, r + 1, stream_);
        size_t ws_bytes = std::max(gemm_nvfp4_cutlass_sm120_workspace(s.M, s.N, s.K),
                                   gemm_nvfp4_cutlass_sm120_streamk_workspace(s.M, s.N, s.K));
        void* ws = nullptr;
        cudaMalloc(&ws, ws_bytes > 0 ? ws_bytes : 1);
        half* y = nullptr;
        cudaMalloc(&y, (size_t)s.M * s.N * sizeof(half));

        auto time_us = [&](auto&& launch) {
            for (int w = 0; w < 3; ++w)
                launch(w % ring);
            cudaStreamSynchronize(stream_);
            cudaEvent_t t0, t1;
            cudaEventCreate(&t0);
            cudaEventCreate(&t1);
            constexpr int kIters = 40;
            cudaEventRecord(t0, stream_);
            for (int i = 0; i < kIters; ++i)
                launch(i % ring);
            cudaEventRecord(t1, stream_);
            cudaEventSynchronize(t1);
            float ms = 0;
            cudaEventElapsedTime(&ms, t0, t1);
            cudaEventDestroy(t0);
            cudaEventDestroy(t1);
            return ms * 1000.0f / kIters;
        };
        {
            // Idle downclock: building the next shape's weight ring is seconds of
            // host work, so clocks drop again between shapes and a single warm-up
            // read the later shapes 1.6x apart between runs. Burn ~1 s per shape.
            cudaEvent_t w0, w1;
            cudaEventCreate(&w0);
            cudaEventCreate(&w1);
            cudaEventRecord(w0, stream_);
            float ms = 0.0f;
            while (ms < 1000.0f) {
                for (int i = 0; i < 50; ++i)
                    gemm_nvfp4_cutlass_sm120(ops[i % ring].act_data, ops[i % ring].act_sf, ops[i % ring].cw,
                                             y, s.M, s.N, s.K, ws, ws_bytes, stream_);
                cudaEventRecord(w1, stream_);
                cudaEventSynchronize(w1);
                cudaEventElapsedTime(&ms, w0, w1);
            }
            cudaEventDestroy(w0);
            cudaEventDestroy(w1);
        }
        float dp = time_us([&](int r) {
            gemm_nvfp4_cutlass_sm120(ops[r].act_data, ops[r].act_sf, ops[r].cw, y, s.M, s.N, s.K, ws,
                                     ws_bytes, stream_);
        });
        float sk_h = time_us([&](int r) {
            gemm_nvfp4_cutlass_sm120_streamk(ops[r].act_data, ops[r].act_sf, ops[r].cw, y, s.M, s.N, s.K, ws,
                                             ws_bytes, stream_, false);
        });
        float sk_f = time_us([&](int r) {
            gemm_nvfp4_cutlass_sm120_streamk(ops[r].act_data, ops[r].act_sf, ops[r].cw, y, s.M, s.N, s.K, ws,
                                             ws_bytes, stream_, true);
        });
        float pp = time_us([&](int r) {
            gemm_nvfp4_cutlass_sm120_smalln(ops[r].act_data, ops[r].act_sf, ops[r].cw, y, s.M, s.N, s.K, ws,
                                            ws_bytes, stream_);
        });
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        const double floor_us = (double)w_bytes / 1792e9 * 1e6;
        printf(
            "[streamk-bench] %-8s M=%d N=%d K=%d ring=%d  dp=%.1f us  sk-heur=%.1f us (units %d)  "
            "sk-forced=%.1f us (units %d)  pingpong128x64=%.1f us  (weight floor %.1f us)\n",
            s.name, s.M, s.N, s.K, ring, dp, sk_h,
            gemm_nvfp4_cutlass_sm120_streamk_units(s.M, s.N, s.K, false), sk_f,
            gemm_nvfp4_cutlass_sm120_streamk_units(s.M, s.N, s.K, true), pp, floor_us);

        cudaFree(y);
        cudaFree(ws);
        for (auto& o : ops)
            o.release();
    }
}
