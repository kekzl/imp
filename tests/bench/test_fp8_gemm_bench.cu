// Timing cases of tests/test_fp8_gemm.cu, built into test-bench (make test-bench), not into the default GPU
// run.
#include <gtest/gtest.h>
#include "compute/gemm.h"
#include "quant/fp8_quant.h"
#include "quant/dequant_gpu.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

#include "test_fp8_gemm_util.cuh"

namespace imp {
namespace {

// Read-shape probe behind the streamed FP8 head (IMP_FP8_PROBE=1): read bandwidth of a [248320 x 5120]
// byte matrix when a warp instruction reads C contiguous bytes of 512 / C rows (64 B: 1397 GB/s, 128: 1665).
template <int C, int kU>
__global__ void probe_read_kernel(const uint4* __restrict__ W, int M, int K, uint32_t* out) {
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    constexpr int lpr = C / 16;    // lanes per row
    constexpr int rpi = 32 / lpr;  // rows per instruction
    const int row0 = warp * 16;
    if (row0 >= M)
        return;
    const int segs = K / C;
    const int n_inst = (16 / rpi) * segs;
    uint32_t acc = 0;
    for (int i0 = 0; i0 < n_inst; i0 += kU) {
        uint4 v[kU];
#pragma unroll
        for (int u = 0; u < kU; ++u) {
            const int i = i0 + u;
            const int row = row0 + ((i / segs) * rpi) + (lane / lpr);
            const int64_t off = (static_cast<int64_t>(row) * K) + ((i % segs) * C) + ((lane % lpr) * 16);
            v[u] = __ldcs(W + (off >> 4));
        }
#pragma unroll
        for (int u = 0; u < kU; ++u)
            acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
    if (acc == 0x12345678u)
        out[0] = acc;
}

TEST_F(FP8GemmTest, HeadReadShapeProbe) {
    if (getenv("IMP_FP8_PROBE") == nullptr)
        GTEST_SKIP() << "set IMP_FP8_PROBE=1 to run the read-shape probe";
    constexpr int M = 248320, K = 5120;
    void* d = nullptr;
    uint32_t* o = nullptr;
    ASSERT_EQ(cudaMalloc(&d, static_cast<size_t>(M) * K), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&o, 4), cudaSuccess);
    cudaMemset(d, 0x2a, static_cast<size_t>(M) * K);
    const auto* W = static_cast<const uint4*>(d);
    const int warps = M / 16;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    auto bench = [&](const char* name, auto launch) {
        for (int i = 0; i < 1500; ++i)
            launch();
        float ms = 0.0f;
        cudaEventRecord(t0, stream_);
        for (int i = 0; i < 300; ++i)
            launch();
        cudaEventRecord(t1, stream_);
        ASSERT_EQ(cudaEventSynchronize(t1), cudaSuccess);
        cudaEventElapsedTime(&ms, t0, t1);
        const double us = ms * 1000.0 / 300;
        printf("[ PROBE    ] %s: %.1f us, %.0f GB/s\n", name, us, static_cast<double>(M) * K / us / 1e3);
    };
#define PROBE(C, U, TPB)                                                                            \
    bench("C=" #C " kU=" #U " tpb=" #TPB, [&] {                                                     \
        probe_read_kernel<C, U><<<(warps * 32 + (TPB) - 1) / (TPB), TPB, 0, stream_>>>(W, M, K, o); \
    })
    PROBE(64, 8, 128);
    PROBE(128, 8, 128);
    PROBE(256, 8, 128);
    PROBE(512, 8, 128);
    PROBE(64, 4, 128);
    PROBE(512, 4, 128);
    PROBE(512, 2, 128);
    PROBE(512, 8, 256);
#undef PROBE
    cudaFree(d);
    cudaFree(o);
}

// FP8 head sweep, us/call per variant (nt 8 / 4 split-K, 0 default) vs 1628 GB/s. Env IMP_FP8_HEAD_M (151936
// = Qwen3-14B, 778 MB > L2), IMP_FP8_HEAD_ROWS (29), IMP_FP8_HEAD_NT (one variant).
TEST_F(FP8GemmTest, RowscaleFp32HeadRowTileSweep) {
    if (getenv("IMP_FP8_HEAD_SWEEP") == nullptr)
        GTEST_SKIP() << "set IMP_FP8_HEAD_SWEEP=1 to run the head sweep";
    const int M = getenv("IMP_FP8_HEAD_M") ? atoi(getenv("IMP_FP8_HEAD_M")) : 151936;
    constexpr int K = 5120;
    const int N = getenv("IMP_FP8_HEAD_ROWS") ? atoi(getenv("IMP_FP8_HEAD_ROWS")) : 29;
    void *d_w = nullptr, *d_q = nullptr, *d_x = nullptr;
    float *d_s = nullptr, *d_y = nullptr;
    ASSERT_EQ(cudaMalloc(&d_w, static_cast<size_t>(M) * K * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_q, static_cast<size_t>(M) * K), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, static_cast<size_t>(N) * K * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_s, M * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, static_cast<size_t>(N) * M * sizeof(float)), cudaSuccess);
    cudaMemset(d_w, 0x2a, static_cast<size_t>(M) * K * sizeof(half));
    cudaMemset(d_x, 0x2a, static_cast<size_t>(N) * K * sizeof(half));
    quantize_fp8_rows_async(d_w, d_q, M, K, d_s, stream_);
    cudaFree(d_w);
    const half* hx = static_cast<const half*>(d_x);
    const double floor_us = (static_cast<double>(M) * K + M * 4.0) / 1628e3;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    std::vector<int> nts = {8, 4, 0, 8};
    if (getenv("IMP_FP8_HEAD_NT") != nullptr)
        nts = {atoi(getenv("IMP_FP8_HEAD_NT"))};
    for (int nt : nts) {
        for (int i = 0; i < 2000; ++i)  // >1 s clock ramp
            ASSERT_TRUE(gemv_fp8_rowscale_fp32_tuned(d_q, d_s, hx, d_y, M, K, N, nt, stream_));
        double best = 1e30;
        for (int r = 0; r < 5; ++r) {
            float ms = 0.0f;
            cudaEventRecord(t0, stream_);
            for (int i = 0; i < 200; ++i)
                (void)gemv_fp8_rowscale_fp32_tuned(d_q, d_s, hx, d_y, M, K, N, nt, stream_);  // timing loop
            cudaEventRecord(t1, stream_);
            ASSERT_EQ(cudaEventSynchronize(t1), cudaSuccess);
            cudaEventElapsedTime(&ms, t0, t1);
            best = std::min(best, ms * 1000.0 / 200);
        }
        printf(
            "[ HEAD     ] M=%d K=%d rows=%d nt=%d (%d rows/block): %.1f us/call, %.0f GB/s, floor %.1f us\n",
            M, K, N, nt, nt * 8, best, (static_cast<double>(M) * K + M * 4.0) / best / 1e3, floor_us);
    }
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_q);
    cudaFree(d_x);
    cudaFree(d_s);
    cudaFree(d_y);
}

}  // namespace
}  // namespace imp
