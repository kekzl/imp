// #2267: BM=128 Q8_0 IMMA pipeline kernel (cp.async ring + ldmatrix) against the pre-change kernel
// (tests/mmq_q8_imma_legacy.cuh), bit-exact. Same k order and scale expression per output: any
// differing half is a defect.

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "compute/mmq_q8_imma.h"
#include "compute/mmq_q8_imma_internal.cuh"
#include "mmq_q8_imma_legacy.cuh"
#include "scoped_engine_arena.h"

namespace imp {

IMP_TEST_ENGINE_ARENA(64ull << 20);

namespace {

void gen_q8_blocks(std::vector<uint8_t>& w, int N, int K, unsigned seed) {
    w.resize(static_cast<size_t>(N) * (K / 32) * 34);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> q(-128, 127);
    std::uniform_real_distribution<float> d(0.001f, 0.05f);
    for (size_t b = 0; b < w.size() / 34; ++b) {
        const __half h = __float2half(d(rng));
        std::memcpy(&w[b * 34], &h, 2);
        for (int i = 0; i < 32; ++i)
            w[b * 34 + 2 + i] = static_cast<uint8_t>(static_cast<int8_t>(q(rng)));
    }
}

// Device planes in gemm_common's layout. Weights: Q8_0 blocks split like q8_split_kernel (d, 0).
struct Planes {
    int8_t* xs8 = nullptr;
    __half* xscale = nullptr;
    float* xrowsum = nullptr;  // legacy signature only; WB=false never reads it
    int8_t* ws8 = nullptr;
    __half* wsc = nullptr;
    int M = 0, N = 0, K = 0;

    bool alloc(int m, int n, int k) {
        M = m, N = n, K = k;
        const size_t subs = static_cast<size_t>(k) / 32;
        return cudaMalloc(&xs8, static_cast<size_t>(m) * k) == cudaSuccess &&
               cudaMalloc(&xscale, static_cast<size_t>(m) * subs * sizeof(__half)) == cudaSuccess &&
               cudaMalloc(&xrowsum, static_cast<size_t>(m) * subs * 2 * sizeof(float)) == cudaSuccess &&
               cudaMemset(xrowsum, 0, static_cast<size_t>(m) * subs * 2 * sizeof(float)) == cudaSuccess &&
               cudaMalloc(&ws8, static_cast<size_t>(n) * k) == cudaSuccess &&
               cudaMalloc(&wsc, static_cast<size_t>(n) * subs * 2 * sizeof(__half)) == cudaSuccess;
    }
    void release() {
        cudaFree(xs8);
        cudaFree(xscale);
        cudaFree(xrowsum);
        cudaFree(ws8);
        cudaFree(wsc);
        *this = Planes{};
    }
};

bool fill_host(Planes& p, unsigned seed) {
    const int subs = p.K / 32;
    std::vector<uint8_t> blocks;
    gen_q8_blocks(blocks, p.N, p.K, seed);
    std::vector<int8_t> ws8(static_cast<size_t>(p.N) * p.K);
    std::vector<__half> wsc(static_cast<size_t>(p.N) * subs * 2);
    for (size_t b = 0; b < blocks.size() / 34; ++b) {
        std::memcpy(&wsc[b * 2], &blocks[b * 34], 2);
        wsc[b * 2 + 1] = __float2half(0.0f);
        std::memcpy(&ws8[b * 32], &blocks[b * 34 + 2], 32);
    }
    std::mt19937 rng(seed + 1);
    std::uniform_int_distribution<int> q(-128, 127);
    std::uniform_real_distribution<float> d(0.001f, 0.1f);
    std::vector<int8_t> xs8(static_cast<size_t>(p.M) * p.K);
    std::vector<__half> xscale(static_cast<size_t>(p.M) * subs);
    for (auto& v : xs8)
        v = static_cast<int8_t>(q(rng));
    for (auto& v : xscale)
        v = __float2half(d(rng));
    return cudaMemcpy(p.ws8, ws8.data(), ws8.size(), cudaMemcpyHostToDevice) == cudaSuccess &&
           cudaMemcpy(p.wsc, wsc.data(), wsc.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess &&
           cudaMemcpy(p.xs8, xs8.data(), xs8.size(), cudaMemcpyHostToDevice) == cudaSuccess &&
           cudaMemcpy(p.xscale, xscale.data(), xscale.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess;
}

// Pre-change BM=128 launch, as gemm_common issued it before #2267.
void launch_legacy(const int8_t* xs8, const __half* xscale, const float* xrowsum, const int8_t* ws8,
                   const __half* wsc, __half* out, int M, int N, int K, bool beta1, cudaStream_t s) {
    const dim3 grid((N + kBN - 1) / kBN, (M + 127) / 128, 1);
    if (beta1)
        legacy2267::mmq_imma_kernel<128, true, false>
            <<<grid, kThreads, 0, s>>>(xs8, xscale, xrowsum, ws8, wsc, out, M, N, K, nullptr, 0, 0);
    else
        legacy2267::mmq_imma_kernel<128, false, false>
            <<<grid, kThreads, 0, s>>>(xs8, xscale, xrowsum, ws8, wsc, out, M, N, K, nullptr, 0, 0);
}

size_t count_diff(const std::vector<__half>& a, const std::vector<__half>& b) {
    size_t n = 0;
    for (size_t i = 0; i < a.size(); ++i)
        n += std::memcmp(&a[i], &b[i], sizeof(__half)) != 0;
    return n;
}

std::vector<__half> random_halves(size_t n, unsigned seed) {
    std::vector<__half> v(n);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& h : v)
        h = __float2half(nd(rng));
    return v;
}

// N=6208 = 48.5 tiles (N-tail), N=4096 K=4096 = Qwen3-8B q_proj, K=64 = 1 K-step (< stages).
TEST(MmqQ8Imma, PipelineBitIdenticalToLegacy) {
    struct Shape {
        int N, K;
    };
    for (const Shape s : {Shape{6208, 1024}, Shape{4096, 4096}, Shape{384, 64}}) {
        const int max_m = 2048;
        Planes p;
        ASSERT_TRUE(p.alloc(max_m, s.N, s.K));
        ASSERT_TRUE(fill_host(p, static_cast<unsigned>(s.N + s.K)));
        __half* d_out = nullptr;
        ASSERT_EQ(cudaMalloc(&d_out, static_cast<size_t>(max_m) * s.N * sizeof(__half)), cudaSuccess);
        for (int M : {1, 33, 512, 2048}) {
            const std::vector<__half> base = random_halves(static_cast<size_t>(M) * s.N, 2268u + M);
            for (bool beta1 : {false, true}) {
                std::vector<__half> old_out(base.size()), new_out(base.size());
                ASSERT_EQ(cudaMemcpy(d_out, base.data(), base.size() * 2, cudaMemcpyHostToDevice), cudaSuccess);
                launch_legacy(p.xs8, p.xscale, p.xrowsum, p.ws8, p.wsc, d_out, M, s.N, s.K, beta1, nullptr);
                ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                ASSERT_EQ(cudaMemcpy(old_out.data(), d_out, base.size() * 2, cudaMemcpyDeviceToHost), cudaSuccess);
                ASSERT_EQ(cudaMemcpy(d_out, base.data(), base.size() * 2, cudaMemcpyHostToDevice), cudaSuccess);
                ASSERT_TRUE(mmq_q8_imma_plane128(p.xs8, p.xscale, p.ws8, p.wsc, d_out, M, s.N, s.K, beta1, nullptr));
                ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                ASSERT_EQ(cudaMemcpy(new_out.data(), d_out, base.size() * 2, cudaMemcpyDeviceToHost), cudaSuccess);
                // the kernel must have written, or equality proves nothing
                EXPECT_GT(count_diff(old_out, base), base.size() / 2)
                    << "N=" << s.N << " K=" << s.K << " M=" << M << " beta1=" << beta1;
                EXPECT_EQ(count_diff(old_out, new_out), 0u)
                    << "N=" << s.N << " K=" << s.K << " M=" << M << " beta1=" << beta1;
            }
        }
        cudaFree(d_out);
        p.release();
    }
}

// Production route: mmq_q8_imma_gemm (dense, BM=128 grid) vs the legacy kernel on the route's own planes.
// N=6208: 49 CTAs x 4 > 170 SMs keeps M=33 off the BM=32 small-grid rule.
TEST(MmqQ8Imma, RouteBitIdenticalToLegacy) {
    const int N = 6208, K = 1024, max_m = 2048;
    std::vector<uint8_t> W;
    gen_q8_blocks(W, N, K, 2267);
    uint8_t* d_w = nullptr;
    __half *d_x = nullptr, *d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_w, W.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, static_cast<size_t>(max_m) * K * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out, static_cast<size_t>(max_m) * N * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_w, W.data(), W.size(), cudaMemcpyHostToDevice), cudaSuccess);
    for (int M : {33, 512, 2048}) {
        const std::vector<__half> x = random_halves(static_cast<size_t>(M) * K, 2270u + M);
        const std::vector<__half> base = random_halves(static_cast<size_t>(M) * N, 2280u + M);
        ASSERT_EQ(cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice), cudaSuccess);
        for (float beta : {0.0f, 1.0f}) {
            std::vector<__half> route(base.size()), old_out(base.size());
            ASSERT_EQ(cudaMemcpy(d_out, base.data(), base.size() * 2, cudaMemcpyHostToDevice), cudaSuccess);
            ASSERT_TRUE(mmq_q8_imma_gemm(d_w, d_x, d_out, M, N, K, nullptr, beta, /*allow_splitk=*/false));
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(route.data(), d_out, base.size() * 2, cudaMemcpyDeviceToHost), cudaSuccess);
            const auto it = g_imma_weights.find(d_w);
            ASSERT_NE(it, g_imma_weights.end());
            ASSERT_EQ(cudaMemcpy(d_out, base.data(), base.size() * 2, cudaMemcpyHostToDevice), cudaSuccess);
            launch_legacy(g_imma_act.xs8, g_imma_act.xscale, g_imma_act.xrowsum, it->second.qs, it->second.sc,
                          d_out, M, N, K, beta == 1.0f, nullptr);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(old_out.data(), d_out, base.size() * 2, cudaMemcpyDeviceToHost), cudaSuccess);
            EXPECT_GT(count_diff(route, base), base.size() / 2) << "M=" << M << " beta=" << beta;
            EXPECT_EQ(count_diff(route, old_out), 0u) << "M=" << M << " beta=" << beta;
        }
    }
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_out);
    mmq_q8_imma_release_all();
}

__global__ void fill_planes_kernel(int8_t* s8, size_t n_s8, __half* sc, size_t n_sc, uint32_t seed) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n_s8 + n_sc;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
        h ^= h >> 15;
        h *= 2246822519u;
        h ^= h >> 13;
        if (i < n_s8)
            s8[i] = static_cast<int8_t>(h & 0xff);
        else
            sc[i - n_s8] = __float2half(0.001f + static_cast<float>(h & 0xffff) * (0.05f / 65536.0f));
    }
}

// Timing: legacy vs pipeline kernel per Qwen3 projection shape. Run with --gtest_also_run_disabled_tests.
// One line per (shape, M): `q8pipe ... old_ms= new_ms= new/old=`; scripts/accept_2267.sh reads q_o M=2048.
TEST(MmqQ8ImmaBench, DISABLED_PipelineVsLegacy) {
    struct Shape {
        const char* model;
        const char* name;
        int N, K;
    };
    const Shape shapes[] = {
        {"Qwen3-8B", "q_o", 4096, 4096},      {"Qwen3-8B", "k_v", 1024, 4096},
        {"Qwen3-8B", "gate_up", 12288, 4096}, {"Qwen3-8B", "down", 4096, 12288},
        {"Qwen3-4B", "q", 4096, 2560},        {"Qwen3-4B", "o", 2560, 4096},
        {"Qwen3-4B", "gate_up", 9728, 2560},  {"Qwen3-4B", "down", 2560, 9728},
    };
    const int rows[] = {256, 512, 1024, 2048, 4096, 8192};
    constexpr int kWarmup = 3, kIters = 10, kRepeats = 5;
    cudaEvent_t t0, t1;
    ASSERT_EQ(cudaEventCreate(&t0), cudaSuccess);
    ASSERT_EQ(cudaEventCreate(&t1), cudaSuccess);
    for (const Shape& s : shapes) {
        const int max_m = rows[sizeof(rows) / sizeof(rows[0]) - 1];
        Planes p;
        ASSERT_TRUE(p.alloc(max_m, s.N, s.K));
        const size_t subs = static_cast<size_t>(s.K) / 32;
        fill_planes_kernel<<<1024, 256>>>(p.xs8, static_cast<size_t>(max_m) * s.K, p.xscale, max_m * subs, 11u);
        fill_planes_kernel<<<1024, 256>>>(p.ws8, static_cast<size_t>(s.N) * s.K, p.wsc, s.N * subs * 2, 12u);
        __half* d_out = nullptr;
        ASSERT_EQ(cudaMalloc(&d_out, static_cast<size_t>(max_m) * s.N * sizeof(__half)), cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        for (int M : rows) {
            auto run = [&](int v) {
                if (v == 0)
                    launch_legacy(p.xs8, p.xscale, p.xrowsum, p.ws8, p.wsc, d_out, M, s.N, s.K, false, nullptr);
                else
                    mmq_q8_imma_plane128(p.xs8, p.xscale, p.ws8, p.wsc, d_out, M, s.N, s.K, false, nullptr);
            };
            for (int v = 0; v < 2; ++v)
                for (int i = 0; i < kWarmup; ++i)
                    run(v);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            std::vector<float> reps[2];
            for (int r = 0; r < kRepeats; ++r)
                for (int k = 0; k < 2; ++k) {
                    const int v = (r & 1) ? 1 - k : k;  // AB / BA per repeat
                    cudaEventRecord(t0, nullptr);
                    for (int i = 0; i < kIters; ++i)
                        run(v);
                    cudaEventRecord(t1, nullptr);
                    ASSERT_EQ(cudaEventSynchronize(t1), cudaSuccess);
                    float ms = 0.0f;
                    cudaEventElapsedTime(&ms, t0, t1);
                    reps[v].push_back(ms / kIters);
                }
            ASSERT_EQ(cudaGetLastError(), cudaSuccess);
            for (auto& r : reps)
                std::sort(r.begin(), r.end());
            const float o = reps[0][kRepeats / 2], n = reps[1][kRepeats / 2];
            printf("q8pipe model=%s shape=%s M=%d N=%d K=%d old_ms=%.4f new_ms=%.4f new/old=%.3f\n", s.model,
                   s.name, M, s.N, s.K, o, n, n / o);
        }
        cudaFree(d_out);
        p.release();
    }
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
}

}  // namespace
}  // namespace imp
