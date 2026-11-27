// Timing cases of tests/test_nvfp4_smallm_ws.cu, built into test-bench (make test-bench), not into the
// default GPU run.
#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_quant.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace imp;

#include "test_nvfp4_smallm_ws_util.cuh"

// DRAM timing on the dense decode shapes: back-to-back calls over a ring of >= 4 weight copies (>= 400 MB,
// past the 96 MB L2), best of 5 x 200 calls after >= 1.2 s warm-up; the activation stays L2-resident as in
// decode. "v2" = what the dispatch ran before smallm ws: v2 sibling launch when the combined tiles allow one
// stripe, else v2 per tensor (stripes + reduce); "-" = v2 refuses (K % 256, N % 64: tile kernel in situ).
// floor = weight + scale bytes / 1792 GB/s.
// --gtest_also_run_disabled_tests --gtest_filter=*Nvfp4SmallMWsBench*
TEST(Nvfp4SmallMWsBench, DISABLED_VsSmallmV2) {
    cudaStream_t st;
    cudaStreamCreate(&st);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    struct Case {
        const char* name;
        int K;
        int N[3];
    };
    const Case cases[] = {
        {"gemma4 qkv sliding", 2816, {4096, 2048, 2048}}, {"gemma4 qk global", 2816, {8192, 1024, 0}},
        {"gemma4 o sliding", 4096, {2816, 0, 0}},         {"gemma4 o global", 8192, {2816, 0, 0}},
        {"gemma4 gate|up", 2816, {2112, 2112, 0}},        {"gemma4 down", 2112, {2816, 0, 0}},
        {"qwen3-8b qkv", 4096, {4096, 1024, 1024}},       {"qwen3-8b o", 4096, {4096, 0, 0}},
        {"qwen3-8b gate|up", 4096, {12288, 12288, 0}},    {"qwen3-8b down", 12288, {4096, 0, 0}},
        {"qwen3-14b qkv", 5120, {5120, 1024, 1024}},      {"qwen3-14b o", 5120, {5120, 0, 0}},
        {"qwen3-14b gate|up", 5120, {17408, 17408, 0}},   {"qwen3-14b down", 17408, {5120, 0, 0}},
        {"qwen3-30b qkv", 2048, {4096, 512, 512}},        {"qwen3-30b o", 4096, {2048, 0, 0}},
    };
    const char* only = std::getenv("IMP_WS_BENCH_ONLY");  // substring filter on the case name
    constexpr int kCfgN = 4;
    const int cfg[kCfgN][2] = {{16, 1}, {16, 2}, {32, 1}, {32, 2}};
    std::fprintf(stderr, "%-20s %3s %7s %7s %7s %7s %7s %7s %7s %6s\n", "shape", "M", "floor", "v2", "ws16u1",
                 "ws16u2", "ws32u1", "ws32u2", "default", "GB/s");
    for (const Case& c : cases) {
        if (only != nullptr && std::strstr(c.name, only) == nullptr)
            continue;
        const int K = c.K;
        int count = 0;
        double bytes = 0.0;
        Shape w[3];
        for (int i = 0; i < 3 && c.N[i] > 0; ++i, ++count) {
            w[i].init(c.N[i], K, false, i + 1);
            bytes += double(c.N[i]) * K * (0.5 + (1.0 / 16.0));
        }
        const int copies = std::max(4, static_cast<int>(400e6 / bytes) + 1);
        std::vector<Dev> ring(size_t(copies) * count);
        for (int r = 0; r < copies; ++r)
            for (int i = 0; i < count; ++i)
                ring[(size_t(r) * count) + i].up(w[i].wp, w[i].wsf);
        void* d_ws = nullptr;
        size_t ws_bytes = 0;
        bool v2_ok = (K % 256) == 0;
        int tiles = 0;
        for (int i = 0; i < count; ++i) {
            ws_bytes = std::max(ws_bytes, gemm_nvfp4_smallm_v2_workspace_bytes(c.N[i], K));
            v2_ok = v2_ok && (c.N[i] % 64) == 0;
            tiles += c.N[i] / 64;
        }
        cudaMalloc(&d_ws, std::max<size_t>(ws_bytes, 256));
        const NvFP4QuantResult xq = w[0].X();
        // One call on ring copy r: kind 0 = v2 dispatch, 1..kCfgN = ws tuned cfg, kCfgN + 1 = ws default.
        auto call = [&](int kind, int r, int M) {
            NvFP4QuantResult wq[3];
            SmallMV2Sibling sib[3];
            for (int i = 0; i < count; ++i) {
                wq[i] = ring[(size_t(r) * count) + i].q(c.N[i], K, kWTs);
                sib[i] = {&wq[i], w[i].d_y, c.N[i]};
            }
            if (kind == 0) {
                if (count > 1 && gemm_nvfp4_smallm_v2_stripes(tiles * 64, K) == 1) {
                    (void)gemm_nvfp4_smallm_v2_multi_a4(sib, count, xq, M, K, st);
                    return;
                }
                for (int i = 0; i < count; ++i)
                    (void)gemm_nvfp4_smallm_v2_a4(wq[i], xq, w[i].d_y, M, c.N[i], K, d_ws, st);
            } else if (kind == kCfgN + 1) {
                if (count > 1)
                    (void)gemm_nvfp4_smallm_ws_multi(sib, count, xq, M, K, st);
                else
                    (void)gemm_nvfp4_smallm_ws(wq[0], xq, w[0].d_y, M, c.N[0], K, 0.0f, st);
            } else if (count > 1) {
                (void)gemm_nvfp4_smallm_ws_multi_tuned(sib, count, xq, M, K, st, cfg[kind - 1][0],
                                                       cfg[kind - 1][1]);
            } else {
                (void)gemm_nvfp4_smallm_ws_tuned(wq[0], xq, w[0].d_y, M, c.N[0], K, 0.0f, st,
                                                 cfg[kind - 1][0], cfg[kind - 1][1]);
            }
        };
        // 200 calls captured into one graph (decode replays graphs; eager launches on WSL2 cost ~7 us each),
        // >= 1.2 s of warm replays, best of 15.
        auto time_us = [&](int kind, int M) {
            cudaGraph_t g = nullptr;
            cudaGraphExec_t ge = nullptr;
            call(kind, 0, M);  // eager first call: lazy cudaFuncSetAttribute fails inside a capture
            EXPECT_EQ(cudaStreamSynchronize(st), cudaSuccess);
            EXPECT_EQ(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal), cudaSuccess);
            for (int i = 0; i < 200; ++i)
                call(kind, i % copies, M);
            EXPECT_EQ(cudaStreamEndCapture(st, &g), cudaSuccess);
            EXPECT_EQ(cudaGraphInstantiate(&ge, g, 0), cudaSuccess);
            float ms = 0.0f;
            double warm = 0.0;
            while (warm < 1200.0) {
                cudaEventRecord(e0, st);
                cudaGraphLaunch(ge, st);
                cudaEventRecord(e1, st);
                cudaEventSynchronize(e1);
                cudaEventElapsedTime(&ms, e0, e1);
                warm += ms;
            }
            double best = 1e30;
            for (int rep = 0; rep < 15; ++rep) {
                cudaEventRecord(e0, st);
                cudaGraphLaunch(ge, st);
                cudaEventRecord(e1, st);
                cudaEventSynchronize(e1);
                cudaEventElapsedTime(&ms, e0, e1);
                best = std::min(best, double(ms));
            }
            cudaGraphExecDestroy(ge);
            cudaGraphDestroy(g);
            return best * 1000.0 / 200.0;
        };
        for (int M : {1, 4, 8, 16, 32}) {
            const double t_v2 = v2_ok ? time_us(0, M) : -1.0;
            double t_c[kCfgN];
            for (int k = 0; k < kCfgN; ++k) {
                bool ok = true;
                for (int i = 0; i < count; ++i)
                    ok = ok && c.N[i] % cfg[k][0] == 0;
                t_c[k] = ok ? time_us(k + 1, M) : -1.0;
            }
            const double t_def = time_us(kCfgN + 1, M);
            std::fprintf(stderr, "%-20s %3d %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %6.0f\n", c.name, M,
                         bytes / 1.792e6, t_v2, t_c[0], t_c[1], t_c[2], t_c[3], t_def, bytes / (t_def * 1e3));
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        cudaFree(d_ws);
        for (auto& d : ring)
            d.free_all();
        for (int i = 0; i < count; ++i)
            w[i].free_all();
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaStreamDestroy(st);
}
