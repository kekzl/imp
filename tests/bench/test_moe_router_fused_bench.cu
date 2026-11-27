// Timing cases of tests/test_moe_router_fused.cu, built into test-bench (make test-bench), not into the
// default GPU run.
#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "compute/moe_router_fused.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

using namespace imp;
#include "test_moe_router_fused_util.cuh"

namespace {

// Bench (not a gate): kernel time per row tile x split at the Qwen3-30B router (K 2048, 128 experts,
// top 8), L2-warm, 200 launches. --gtest_also_run_disabled_tests.
TEST(MoeRouterFusedBench, DISABLED_TileSplitSweep) {
    const int K = 2048, ne = 128, top_k = 8;
    const int max_m = 4096;
    DevBuf x(static_cast<size_t>(max_m) * K * 2), w(static_cast<size_t>(ne) * K * 2);
    DevBuf ids(static_cast<size_t>(max_m) * top_k * 4), wts(static_cast<size_t>(max_m) * top_k * 4);
    const size_t ws_bytes = moe_router_fused_workspace_bytes(max_m);
    DevBuf ws(ws_bytes);
    cudaMemset(ws.p, 0, ws_bytes);
    std::vector<__half> hx(static_cast<size_t>(max_m) * K), hw(static_cast<size_t>(ne) * K);
    std::mt19937 rng(7);
    std::normal_distribution<float> n1(0.0f, 1.0f), n2(0.0f, 0.03f);
    for (auto& v : hx)
        v = __float2half(n1(rng));
    for (auto& v : hw)
        v = __float2half(n2(rng));
    cudaMemcpy(x.p, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(w.p, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int M : {2, 16, 512, 1024, 2048, 3072, 4096})
        for (int bm : {0, 16, 32})
            for (int split : {0, 1, 2, 4, 8, 16}) {
                moe_router_fused_tune(bm, split);
                MoeRouterFusedArgs a;
                a.x = static_cast<const half*>(x.p);
                a.w = static_cast<const half*>(w.p);
                a.M = M;
                a.K = K;
                a.ne = ne;
                a.top_k = top_k;
                a.ids = static_cast<int32_t*>(ids.p);
                a.weights = static_cast<float*>(wts.p);
                if (!moe_router_fused(a, ws.p, ws_bytes, nullptr))
                    continue;
                // One graph of 200 launches: back-to-back stream launches are host-bound on WSL2.
                cudaStream_t st;
                cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
                cudaGraph_t g;
                cudaGraphExec_t ge;
                cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
                for (int i = 0; i < 200; ++i)
                    (void)moe_router_fused(a, ws.p, ws_bytes, st);
                cudaStreamEndCapture(st, &g);
                cudaGraphInstantiate(&ge, g, 0);
                cudaGraphLaunch(ge, st);  // warm
                cudaStreamSynchronize(st);
                cudaEventRecord(e0, st);
                cudaGraphLaunch(ge, st);
                cudaEventRecord(e1, st);
                cudaEventSynchronize(e1);
                cudaGraphExecDestroy(ge);
                cudaGraphDestroy(g);
                cudaStreamDestroy(st);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, e0, e1);
                printf("[router-bench] M=%d bm=%d split=%d: %.2f us\n", M, bm, split, ms * 1000.0f / 200.0f);
            }
    moe_router_fused_tune(0, 0);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
}

}  // namespace
