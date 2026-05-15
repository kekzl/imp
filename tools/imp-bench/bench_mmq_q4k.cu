// Microbench: tiled mmq_q4k vs ggml_mmvq_q4k on Qwen3-32B prefill shapes.
//
// Phase A acceptance gate: tiled kernel ≥ 1000 tok/s at M=512 N=K=5120.
// Reference points (memo q4k_mmvq_crossover_2026_05_15):
//   - mmvq saturates at ~250 tok/s for all M
//   - dequant+cuBLAS reaches ~1800 tok/s at M=512

#include "compute/ggml_mmvq.h"
#include "compute/mmq_q4k.h"
#include "compute/mmq_q4k_v2.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace imp {

namespace {

constexpr int kWarmup = 3;
constexpr int kTimed = 10;

struct Shape {
    int M, N, K;
    const char* label;
};

const Shape kShapes[] = {
    {32, 5120, 5120, "M=32  N=5120 K=5120 (mmvq territory)"},
    {64, 5120, 5120, "M=64  N=5120 K=5120"},
    {128, 5120, 5120, "M=128 N=5120 K=5120"},
    {256, 5120, 5120, "M=256 N=5120 K=5120"},
    {512, 5120, 5120, "M=512 N=5120 K=5120 (Phase A gate)"},
    {512, 27648, 5120, "M=512 N=27648 K=5120 (FFN up)"},
};

std::vector<uint8_t> make_random_q4k_host(int N, int K, unsigned seed) {
    const int blocks = K / 256;
    std::vector<uint8_t> h(static_cast<size_t>(N) * blocks * 144);
    std::srand(seed);
    for (int row = 0; row < N; ++row) {
        for (int b = 0; b < blocks; ++b) {
            uint8_t* bp = h.data() + (static_cast<size_t>(row) * blocks + b) * 144;
            half d = __float2half(0.01f + 0.005f * (std::rand() % 100) / 100.0f);
            half dmin = __float2half(0.003f + 0.001f * (std::rand() % 100) / 100.0f);
            std::memcpy(bp + 0, &d, 2);
            std::memcpy(bp + 2, &dmin, 2);
            for (int i = 0; i < 12; ++i) bp[4 + i] = static_cast<uint8_t>(std::rand() & 0x3F);
            for (int i = 0; i < 128; ++i) bp[16 + i] = static_cast<uint8_t>(std::rand() & 0xFF);
        }
    }
    return h;
}

void fill_random_fp16(std::vector<half>& v, unsigned seed) {
    std::srand(seed);
    for (auto& x : v)
        x = __float2half((std::rand() / static_cast<float>(RAND_MAX) - 0.5f) * 2.0f);
}

float time_iters(cudaEvent_t s, cudaEvent_t e,
                 void (*fn)(const void*, const half*, half*, int, int, int, void*, size_t,
                            cudaStream_t),
                 const void* W, const half* x, half* y, int M, int N, int K, void* scratch,
                 size_t scratch_size) {
    for (int i = 0; i < kWarmup; ++i)
        fn(W, x, y, M, N, K, scratch, scratch_size, nullptr);
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    for (int i = 0; i < kTimed; ++i)
        fn(W, x, y, M, N, K, scratch, scratch_size, nullptr);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0;
    cudaEventElapsedTime(&ms, s, e);
    return ms / kTimed;
}

}  // namespace

void bench_mmq_q4k() {
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        printf("bench_mmq_q4k: no CUDA device, skipping.\n");
        return;
    }

    printf("=== mmq_q4k Microbench (mmvq vs v1 tile sweep vs v2 HMMA) ===\n");
    printf("  v1 tiles: t0=<32,64,2,4>  t1=<16,32,1,1>  t2=<16,64,1,2>  "
           "t3=<64,128,4,4>  t4=<64,64,4,4>\n");
    printf("  v2:       HMMA <64,64,32> + WMMA m16n16k16 (FP32 acc)\n\n");

    const char* tile_labels[] = {
        "tile0 <32,64,2,4>",  "tile1 <16,32,1,1>", "tile2 <16,64,1,2>",
        "tile3 <64,128,4,4>", "tile4 <64,64,4,4>",
    };
    constexpr int kNumTiles = 5;

    for (const auto& sz : kShapes) {
        const int M = sz.M, N = sz.N, K = sz.K;
        const size_t bytes_W =
            static_cast<size_t>(N) * (K / 256) * 144;
        const size_t bytes_x = static_cast<size_t>(M) * K * sizeof(half);
        const size_t bytes_y = static_cast<size_t>(M) * N * sizeof(half);
        const size_t bytes_scratch = mmq_q4k_scratch_bytes(M, K);

        void* W_dev = nullptr;
        half* x_dev = nullptr;
        half* y_dev = nullptr;
        void* scratch = nullptr;
        uint8_t* eff_q4 = nullptr;
        half* eff_scale = nullptr;
        half* eff_min = nullptr;
        cudaMalloc(&W_dev, bytes_W);
        cudaMalloc(&x_dev, bytes_x);
        cudaMalloc(&y_dev, bytes_y);
        cudaMalloc(&scratch, bytes_scratch);
        cudaMalloc(&eff_q4, q4k_eff_q4_bytes(N, K));
        cudaMalloc(&eff_scale, q4k_eff_scale_bytes(N, K));
        cudaMalloc(&eff_min, q4k_eff_scale_bytes(N, K));

        auto h_W = make_random_q4k_host(N, K, 0xabcd);
        std::vector<half> h_x(static_cast<size_t>(M) * K);
        fill_random_fp16(h_x, 0xdcba);
        cudaMemcpy(W_dev, h_W.data(), bytes_W, cudaMemcpyHostToDevice);
        cudaMemcpy(x_dev, h_x.data(), bytes_x, cudaMemcpyHostToDevice);

        // One-shot v2 preprocessing — runs once at "model load" in production.
        q4k_precompute_eff_scales(W_dev, eff_scale, eff_min, N, K, nullptr);
        q4k_permute_to_v2_layout(W_dev, eff_q4, N, K, nullptr);
        cudaDeviceSynchronize();

        cudaEvent_t s, e;
        cudaEventCreate(&s);
        cudaEventCreate(&e);

        float mmvq_ms = time_iters(s, e, ggml_mmvq_q4k, W_dev, x_dev, y_dev, M, N, K, scratch,
                                   bytes_scratch);

        float tile_ms[kNumTiles] = {0};
        for (int t = 0; t < kNumTiles; ++t) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "%d", t);
            setenv("IMP_MMQ_Q4K_TILE", buf, 1);
            tile_ms[t] = time_iters(s, e, mmq_q4k, W_dev, x_dev, y_dev, M, N, K, scratch,
                                    bytes_scratch);
        }

        // v2 timing — no scratch (no Q8_1 quant), takes precomputed inputs.
        for (int i = 0; i < kWarmup; ++i)
            mmq_q4k_v2(x_dev, eff_q4, eff_scale, eff_min, y_dev, M, N, K, nullptr);
        cudaDeviceSynchronize();
        cudaEventRecord(s);
        for (int i = 0; i < kTimed; ++i)
            mmq_q4k_v2(x_dev, eff_q4, eff_scale, eff_min, y_dev, M, N, K, nullptr);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float v2_ms = 0;
        cudaEventElapsedTime(&v2_ms, s, e);
        v2_ms /= kTimed;

        cudaEventDestroy(s);
        cudaEventDestroy(e);

        int best = 0;
        for (int t = 1; t < kNumTiles; ++t)
            if (tile_ms[t] < tile_ms[best]) best = t;
        const float v1_best_ms = tile_ms[best];
        const double v2_toks = M / (v2_ms * 1e-3);
        const double v1_toks = M / (v1_best_ms * 1e-3);

        printf("  %-46s mmvq=%6.3fms", sz.label, mmvq_ms);
        for (int t = 0; t < kNumTiles; ++t) printf("  t%d=%6.3f", t, tile_ms[t]);
        printf("  v1_best=%s (%.2fx)  v2=%6.3fms (%.2fx mmvq, %.2fx v1_best)  "
               "v1=%.0f v2=%.0f tok/s\n",
               tile_labels[best], mmvq_ms / v1_best_ms, v2_ms, mmvq_ms / v2_ms,
               v1_best_ms / v2_ms, v1_toks, v2_toks);

        cudaFree(W_dev);
        cudaFree(x_dev);
        cudaFree(y_dev);
        cudaFree(scratch);
        cudaFree(eff_q4);
        cudaFree(eff_scale);
        cudaFree(eff_min);
    }
    printf("\n");
}

}  // namespace imp
