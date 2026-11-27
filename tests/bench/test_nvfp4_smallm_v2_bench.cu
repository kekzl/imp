// Timing cases of tests/test_nvfp4_smallm_v2.cu, built into test-bench (make test-bench), not into the
// default GPU run.
#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_owned.h"
#include "quant/nvfp4_quant.h"
#include "core/tensor.h"

#include "test_nvfp4_smallm_v2_util.cuh"

namespace {

TEST_F(NvFP4SmallMV2Test, BandwidthAboveStarvationFloor) {
    const int M = 32, N = 5120, K = 5120;
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<__half> w_h((size_t)N * K), x_h((size_t)M * K);
    for (auto& v : w_h)
        v = __float2half(dist(rng));
    for (auto& v : x_h)
        v = __float2half(dist(rng));
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    // Weight bytes dominate: N*K/2 nibbles + N*K/16 scales = 14.75 MB.
    const double bytes = (double)N * K / 2 + (double)N * K / 16;
    const double floor_us = bytes / 1792e9 * 1e6;  // 8.2 us
    // The time is bimodal over minutes (10.3-14.9 vs 19-32 us, clocks 2917/13801 MHz in both; 0 of 132
    // slow later on the same box): one attempt is a lottery, best of up to 3 is the gate.
    std::vector<DeviceQuant> keep;  // attempts stay allocated: fresh buffers each time
    double best_pct = 0.0, best_us = 0.0;
    for (int attempt = 0; attempt < 3 && best_pct <= 40.0; ++attempt) {
        keep.emplace_back();
        keep.emplace_back();
        DeviceQuant& W = keep[keep.size() - 2];
        DeviceQuant& X = keep.back();
        W.quantize(w_h, N, K);
        X.quantize(x_h, M, K);

        void *d_y = nullptr, *d_ws = nullptr;
        ASSERT_EQ(cudaMalloc(&d_y, (size_t)M * N * sizeof(__half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_ws, imp::gemm_nvfp4_smallm_v2_workspace_bytes(N, K)), cudaSuccess);

        // Warmup >1s of busy time so clocks ramp (the box never throttles; it
        // DOES idle-downclock, and 20 ms of load measures the idle clocks).
        float ms = 0.0f;
        for (int w = 0; w < 100; ++w) {
            for (int i = 0; i < 1000; ++i)
                ASSERT_TRUE(
                    imp::gemm_nvfp4_smallm_v2_a4(W.q, X.q, static_cast<half*>(d_y), M, N, K, d_ws, nullptr));
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        }
        double best_ms = 1e30;
        for (int r = 0; r < 3; ++r) {
            cudaEventRecord(t0);
            for (int i = 0; i < 1000; ++i)
                ASSERT_TRUE(
                    imp::gemm_nvfp4_smallm_v2_a4(W.q, X.q, static_cast<half*>(d_y), M, N, K, d_ws, nullptr));
            cudaEventRecord(t1);
            ASSERT_EQ(cudaEventSynchronize(t1), cudaSuccess);
            cudaEventElapsedTime(&ms, t0, t1);
            best_ms = std::min(best_ms, (double)ms);
        }
        const double us = best_ms;  // 1000 calls -> ms == us/call
        const double pct = floor_us / us * 100.0;
        printf(
            "[ BENCH    ] smallm_v2 M=32 N=5120 K=5120 attempt %d: %.1f us/call, %.0f%% of weight floor "
            "(%.1f us)\n",
            attempt, us, pct, floor_us);
        if (pct > best_pct) {
            best_pct = pct;
            best_us = us;
        }
        cudaFree(d_y);
        cudaFree(d_ws);
    }
    EXPECT_GT(best_pct, 40.0) << best_us << " us — starvation regression (CUTLASS in-situ is 41.4 us)";

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
}

TEST_F(NvFP4SmallMV2Test, SweepTuning) {
    if (getenv("IMP_SMALLM_V2_SWEEP") == nullptr)
        GTEST_SKIP() << "set IMP_SMALLM_V2_SWEEP=1 to run the tuning sweep";
    const int M = 32, N = 5120, K = 5120;
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<__half> w_h((size_t)N * K), x_h((size_t)M * K);
    for (auto& v : w_h)
        v = __float2half(dist(rng));
    for (auto& v : x_h)
        v = __float2half(dist(rng));
    DeviceQuant W, X;
    W.quantize(w_h, N, K);
    X.quantize(x_h, M, K);
    void *d_y = nullptr, *d_ws = nullptr;
    const int kMaxStripes = 8;
    ASSERT_EQ(cudaMalloc(&d_y, (size_t)M * N * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_ws, (size_t)kMaxStripes * 32 * N * sizeof(float)), cudaSuccess);
    // clock ramp
    for (int i = 0; i < 20000; ++i)
        ASSERT_TRUE(imp::gemm_nvfp4_smallm_v2_a4(W.q, X.q, static_cast<half*>(d_y), M, N, K, d_ws, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    const int stages_v[] = {2, 3, 4, 6};
    const int stripes_v[] = {1, 2, 3, 4, 5, 8};
    for (int st : stages_v) {
        for (int sp : stripes_v) {
            float ms = 0.0f;
            double best = 1e30;
            if (!imp::gemm_nvfp4_smallm_v2_a4_tuned(W.q, X.q, static_cast<half*>(d_y), M, N, K, d_ws, nullptr,
                                                    false, st, sp))
                continue;
            for (int r = 0; r < 3; ++r) {
                cudaEventRecord(t0);
                for (int i = 0; i < 1000; ++i)
                    // timing loop: the launch status is not under test
                    (void)imp::gemm_nvfp4_smallm_v2_a4_tuned(W.q, X.q, static_cast<half*>(d_y), M, N, K, d_ws,
                                                             nullptr, false, st, sp);
                cudaEventRecord(t1);
                ASSERT_EQ(cudaEventSynchronize(t1), cudaSuccess);
                cudaEventElapsedTime(&ms, t0, t1);
                best = std::min(best, (double)ms);
            }
            printf("[ SWEEP    ] stages=%d stripes=%d ctas=%d: %.2f us/call\n", st, sp, (N / 64) * sp, best);
        }
    }
    cudaFree(d_y);
    cudaFree(d_ws);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
}

// Per-shape bandwidth on Qwen3-14B dense batched-decode shapes against an L2-defeating ring
// (>=4 copies, >=400 MB/shape) so the number is DRAM, not the 96 MB L2. Survey only, no
// assertion.
TEST_F(NvFP4SmallMV2Test, ShapeBandwidthSurvey) {
    if (getenv("IMP_SMALLM_V2_SHAPES") == nullptr)
        GTEST_SKIP() << "set IMP_SMALLM_V2_SHAPES=1 to run the shape survey";
    struct Shape {
        int N1, N2, K;  // N2 > 0: sibling pair launch
        const char* name;
    };
    const Shape shapes[] = {
        {5120, 0, 5120, "q/o 5120x5120"},        {1024, 0, 5120, "k/v 1024x5120"},
        {7168, 0, 5120, "qkv-as-one 7168x5120"}, {17408, 17408, 5120, "gate|up pair 2x17408x5120"},
        {5120, 0, 17408, "down 5120x17408"},     {151936, 0, 5120, "lm_head 151936x5120"},
    };
    const int M = 32;
    std::mt19937 rng(13);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    for (const Shape& s : shapes) {
        const int K = s.K;
        const int Nmax = std::max(s.N1, s.N2);
        std::vector<__half> w_h((size_t)Nmax * K), x_h((size_t)M * K);
        for (auto& v : w_h)
            v = __float2half(dist(rng));
        for (auto& v : x_h)
            v = __float2half(dist(rng));
        DeviceQuant W, X;
        W.quantize(w_h, Nmax, K);
        X.quantize(x_h, M, K);
        const size_t nib1 = (size_t)s.N1 * K / 2, sf1 = (size_t)s.N1 * K / 16;
        const size_t nib2 = (size_t)s.N2 * K / 2, sf2 = (size_t)s.N2 * K / 16;
        const double bytes = (double)(nib1 + sf1 + nib2 + sf2);
        const int copies = std::max(4, (int)(400e6 / bytes) + 1);
        // Ring: `copies` independent weight buffers with identical content.
        std::vector<imp::NvFP4QuantResult> ring1(copies), ring2(copies);
        for (int c = 0; c < copies; ++c) {
            void *p1 = nullptr, *s1 = nullptr;
            ASSERT_EQ(cudaMalloc(&p1, nib1), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&s1, sf1), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(p1, W.q.packed_data, nib1, cudaMemcpyDeviceToDevice), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(s1, W.q.micro_scales, sf1, cudaMemcpyDeviceToDevice), cudaSuccess);
            ring1[c] = W.q;
            ring1[c].packed_data = p1;
            ring1[c].micro_scales = s1;
            ring1[c].N = s.N1;
            if (s.N2 > 0) {
                void *p2 = nullptr, *s2 = nullptr;
                ASSERT_EQ(cudaMalloc(&p2, nib2), cudaSuccess);
                ASSERT_EQ(cudaMalloc(&s2, sf2), cudaSuccess);
                ASSERT_EQ(cudaMemcpy(p2, W.q.packed_data, nib2, cudaMemcpyDeviceToDevice), cudaSuccess);
                ASSERT_EQ(cudaMemcpy(s2, W.q.micro_scales, sf2, cudaMemcpyDeviceToDevice), cudaSuccess);
                ring2[c] = W.q;
                ring2[c].packed_data = p2;
                ring2[c].micro_scales = s2;
                ring2[c].N = s.N2;
            }
        }
        void *d_y1 = nullptr, *d_y2 = nullptr, *d_ws = nullptr;
        ASSERT_EQ(cudaMalloc(&d_y1, (size_t)M * s.N1 * sizeof(__half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_y2, (size_t)M * std::max(s.N2, 64) * sizeof(__half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_ws, imp::gemm_nvfp4_smallm_v2_workspace_bytes(s.N1, K)), cudaSuccess);
        auto call = [&](int i) {
            const int c = i % copies;
            if (s.N2 > 0)
                return imp::gemm_nvfp4_smallm_v2_pair_a4(ring1[c], ring2[c], X.q, static_cast<half*>(d_y1),
                                                         static_cast<half*>(d_y2), M, s.N1, s.N2, K, nullptr);
            return imp::gemm_nvfp4_smallm_v2_a4(ring1[c], X.q, static_cast<half*>(d_y1), M, s.N1, K, d_ws,
                                                nullptr);
        };
        ASSERT_TRUE(call(0)) << s.name;
        // Warm >= 1 s of busy time (idle downclock, STOP #3 of benchmark-cuda).
        float ms = 0.0f;
        double warm_ms = 0.0;
        for (int w = 0; warm_ms < 1200.0; ++w) {
            cudaEventRecord(t0);
            for (int i = 0; i < 200; ++i)
                call(i);
            cudaEventRecord(t1);
            ASSERT_EQ(cudaEventSynchronize(t1), cudaSuccess);
            cudaEventElapsedTime(&ms, t0, t1);
            warm_ms += ms;
        }
        double best = 1e30;
        for (int r = 0; r < 5; ++r) {
            cudaEventRecord(t0);
            for (int i = 0; i < 200; ++i)
                call(i);
            cudaEventRecord(t1);
            ASSERT_EQ(cudaEventSynchronize(t1), cudaSuccess);
            cudaEventElapsedTime(&ms, t0, t1);
            best = std::min(best, (double)ms);
        }
        const double us = best * 1000.0 / 200.0;
        const int ctas = (s.N1 + s.N2) / 64 * imp::gemm_nvfp4_smallm_v2_stripes(s.N1, K);
        printf("[ SHAPES   ] %-28s ctas=%4d stripes=%d copies=%d: %8.2f us  %7.1f GB/s  (%.1f MB)\n", s.name,
               ctas, imp::gemm_nvfp4_smallm_v2_stripes(s.N1, K), copies, us, bytes / us * 1e-3, bytes * 1e-6);
        for (int c = 0; c < copies; ++c) {
            cudaFree(ring1[c].packed_data);
            cudaFree(ring1[c].micro_scales);
            if (s.N2 > 0) {
                cudaFree(ring2[c].packed_data);
                cudaFree(ring2[c].micro_scales);
            }
        }
        cudaFree(d_y1);
        cudaFree(d_y2);
        cudaFree(d_ws);
    }
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
}

__global__ void ring_read_kernel(const uint4* __restrict__ src, size_t n, uint4* __restrict__ sink) {
    uint4 acc = make_uint4(0u, 0u, 0u, 0u);
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        const uint4 v = __ldcs(src + i);
        acc.x ^= v.x;
        acc.y ^= v.y;
        acc.z ^= v.z;
        acc.w ^= v.w;
    }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678u)
        *sink = acc;
}

// Pure-read ceiling per launch size: the same bytes as each GEMM, 8-copy ring, back-to-back launches.
TEST_F(NvFP4SmallMV2Test, RingReadCeiling) {
    if (getenv("IMP_SMALLM_V2_SWEEP") == nullptr)
        GTEST_SKIP() << "set IMP_SMALLM_V2_SWEEP=1 to run the ring sweep";
    constexpr int kRing = 8;
    const size_t sizes[] = {(size_t)5120 * 5120 * 9 / 16, (size_t)5120 * 17408 * 9 / 16,
                            (size_t)7168 * 5120 * 9 / 16, (size_t)34816 * 5120 * 9 / 16};
    uint4* sink = nullptr;
    ASSERT_EQ(cudaMalloc(&sink, sizeof(uint4)), cudaSuccess);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    for (size_t bytes : sizes) {
        std::vector<void*> ring(kRing);
        for (auto& p : ring) {
            ASSERT_EQ(cudaMalloc(&p, bytes), cudaSuccess);
            ASSERT_EQ(cudaMemset(p, 1, bytes), cudaSuccess);
        }
        const size_t n = bytes / 16;
        for (int blocks : {80, 170, 340, 680}) {
            auto call = [&](int i) {
                ring_read_kernel<<<blocks, 512>>>(static_cast<const uint4*>(ring[i % kRing]), n, sink);
            };
            for (int i = 0; i < 4000; ++i)
                call(i);
            double best = 1e30;
            for (int r = 0; r < 5; ++r) {
                float ms = 0.0f;
                cudaEventRecord(t0);
                for (int i = 0; i < 800; ++i)
                    call(i);
                cudaEventRecord(t1);
                ASSERT_EQ(cudaEventSynchronize(t1), cudaSuccess);
                cudaEventElapsedTime(&ms, t0, t1);
                best = std::min(best, ms * 1000.0 / 800);
            }
            printf("[ READ     ] %.1f MB blocks=%d: %.2f us/call, %.0f GB/s, floor %.2f us\n", bytes / 1e6,
                   blocks, best, bytes / best / 1e3, bytes / 1628e3);
        }
        for (auto p : ring)
            cudaFree(p);
    }
    cudaFree(sink);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
}

// DRAM-resident sweep over the Qwen3-14B decode shapes (#2414): 8 weight copies rotate so no
// launch reads L2-resident weights; reports us/call and GB/s against the 1628 GB/s floor.
TEST_F(NvFP4SmallMV2Test, RingSweepQwen3_14B) {
    if (getenv("IMP_SMALLM_V2_SWEEP") == nullptr)
        GTEST_SKIP() << "set IMP_SMALLM_V2_SWEEP=1 to run the ring sweep";
    const int M = getenv("IMP_SMALLM_V2_M") ? atoi(getenv("IMP_SMALLM_V2_M")) : 29;
    constexpr int kRing = 8;
    struct Shape {
        const char* name;
        int N, K;
    };
    const Shape shapes[] = {{"o", 5120, 5120},
                            {"down", 5120, 17408},
                            {"qkv", 7168, 5120},
                            {"gate|up", 34816, 5120}};
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    for (const Shape& sh : shapes) {
        std::mt19937 rng(23);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<__half> w_h((size_t)sh.N * sh.K), x_h((size_t)M * sh.K);
        for (auto& v : w_h)
            v = __float2half(dist(rng));
        for (auto& v : x_h)
            v = __float2half(dist(rng));
        DeviceQuant W, X;
        W.quantize(w_h, sh.N, sh.K);
        X.quantize(x_h, M, sh.K);
        const size_t pb = (size_t)sh.N * sh.K / 2, sb = (size_t)sh.N * sh.K / 16;
        imp::NvFP4QuantResult ring[kRing];
        for (auto& r : ring) {
            r = W.q;
            ASSERT_EQ(cudaMalloc(&r.packed_data, pb), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&r.micro_scales, sb), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(r.packed_data, W.q.packed_data, pb, cudaMemcpyDeviceToDevice), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(r.micro_scales, W.q.micro_scales, sb, cudaMemcpyDeviceToDevice),
                      cudaSuccess);
        }
        void *d_y = nullptr, *d_ws = nullptr;
        ASSERT_EQ(cudaMalloc(&d_y, (size_t)M * sh.N * sizeof(__half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_ws, (size_t)8 * 32 * sh.N * sizeof(float)), cudaSuccess);
        const double floor_us = (double)(pb + sb) / 1628e9 * 1e6;
        for (int sp : {1, 2, 4}) {
            if (sp > 1 && sh.N / 64 >= 160)
                continue;
            auto call = [&](int i) {
                return imp::gemm_nvfp4_smallm_v2_a4_tuned(ring[i % kRing], X.q, static_cast<half*>(d_y), M,
                                                          sh.N, sh.K, d_ws, nullptr, false, 6, sp);
            };
            for (int i = 0; i < 4000; ++i)  // >1 s clock ramp
                ASSERT_TRUE(call(i));
            double best = 1e30;
            for (int r = 0; r < 5; ++r) {
                float ms = 0.0f;
                cudaEventRecord(t0);
                for (int i = 0; i < 800; ++i)
                    (void)call(i);  // timing loop: launch status is not under test
                cudaEventRecord(t1);
                ASSERT_EQ(cudaEventSynchronize(t1), cudaSuccess);
                cudaEventElapsedTime(&ms, t0, t1);
                best = std::min(best, ms * 1000.0 / 800);
            }
            printf(
                "[ RING     ] %-7s M=%d N=%d K=%d stripes=%d ctas=%d: %.2f us/call, %.0f GB/s, floor %.2f "
                "us\n",
                sh.name, M, sh.N, sh.K, sp, sh.N / 64 * sp, best, (pb + sb) / best / 1e3, floor_us);
        }
        for (auto& r : ring) {
            cudaFree(r.packed_data);
            cudaFree(r.micro_scales);
        }
        cudaFree(d_y);
        cudaFree(d_ws);
    }
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
}

}  // namespace
