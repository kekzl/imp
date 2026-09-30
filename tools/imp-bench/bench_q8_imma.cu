// #2267: Q8_0 prefill GEMM per shape: dequant -> FP16-acc cuBLAS (the gemm.q8_imma_enabled=false
// route) against the IMMA kernel and the legacy IMMA kernel (#2267e A/B). Shapes: Qwen3-8B and Qwen3-4B.
// Output: one `q8imma` line per (shape, M) and one `layer` line per (model, M), times in ms.
#include "compute/gemm.h"
#include "compute/mmq_q8_imma.h"
#include "core/process_diag.h"
#include "core/tensor.h"
#include "memory/backend.h"
#include "memory/engine_arena.h"
#include "quant/dequant_gpu.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace imp {

namespace {

struct Q8Shape {
    const char* model;
    const char* name;
    int N, K;
    int per_layer;  // occurrences per decoder layer (k/v, gate/up = 2)
};

constexpr Q8Shape kShapes[] = {
    {"Qwen3-8B", "q_o", 4096, 4096, 2},      {"Qwen3-8B", "k_v", 1024, 4096, 2},
    {"Qwen3-8B", "gate_up", 12288, 4096, 2}, {"Qwen3-8B", "down", 4096, 12288, 1},
    {"Qwen3-4B", "q", 4096, 2560, 1},        {"Qwen3-4B", "k_v", 1024, 2560, 2},
    {"Qwen3-4B", "o", 2560, 4096, 1},        {"Qwen3-4B", "gate_up", 9728, 2560, 2},
    {"Qwen3-4B", "down", 2560, 9728, 1},
};
constexpr int kRows[] = {256, 512, 1024, 1536, 2048, 4096, 8192};
constexpr int kVariants = 3;  // 0 = off route, 1 = IMMA, 2 = legacy IMMA kernel (#2267e A/B)
constexpr int kWarmup = 3;
constexpr int kIters = 10;
constexpr int kRepeats = 5;

void fill_q8(std::vector<uint8_t>& w, int N, int K, unsigned seed) {
    w.resize(static_cast<size_t>(N) * (K / 32) * 34);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> q(-127, 127);
    std::uniform_real_distribution<float> d(0.001f, 0.05f);
    for (size_t b = 0; b < w.size() / 34; ++b) {
        const __half h = __float2half(d(rng));
        std::memcpy(&w[b * 34], &h, 2);
        for (int i = 0; i < 32; ++i)
            w[b * 34 + 2 + i] = static_cast<uint8_t>(static_cast<int8_t>(q(rng)));
    }
}

struct Bufs {
    void* w = nullptr;
    __half* x = nullptr;
    __half* out = nullptr;
    __half* deq = nullptr;
};

// One call of the variant: 0 = dequant + cuBLAS, 1 = IMMA, 2 = legacy IMMA kernel. False = IMMA declined.
bool run_variant(int v, const Bufs& b, int M, int N, int K) {
    if (v == 0) {
        dequant_gpu(b.w, b.deq, QType::Q8_0, N, K, nullptr);
        int64_t xs[2] = {M, K}, ws[2] = {N, K}, os[2] = {M, N};
        Tensor x(b.x, QType::F16, 2, xs, true), w(b.deq, QType::F16, 2, ws, true),
            o(b.out, QType::F16, 2, os, true);
        gemm(x, w, o, 1.0f, 0.0f, nullptr);
        return true;
    }
    return mmq_q8_imma_gemm(b.w, b.x, b.out, M, N, K, nullptr, 0.0f, /*allow_splitk=*/false);
}

// Median over kRepeats of the mean of kIters calls, ms; -1 = declined or CUDA error.
float time_variant(int v, const Bufs& b, int M, int N, int K) {
    for (int i = 0; i < kWarmup; ++i)
        if (!run_variant(v, b, M, N, K))
            return -1.0f;
    if (cudaDeviceSynchronize() != cudaSuccess)
        return -1.0f;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    std::vector<float> reps;
    for (int r = 0; r < kRepeats; ++r) {
        cudaEventRecord(t0, nullptr);
        for (int i = 0; i < kIters; ++i)
            run_variant(v, b, M, N, K);
        cudaEventRecord(t1, nullptr);
        cudaEventSynchronize(t1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t0, t1);
        reps.push_back(ms / kIters);
    }
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    std::sort(reps.begin(), reps.end());
    return cudaGetLastError() == cudaSuccess ? reps[kRepeats / 2] : -1.0f;
}

}  // namespace

bool bench_q8_imma() {
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        printf("bench_q8_imma: no CUDA device available, skipping.\n");
        return false;
    }
    process_diag_set_cublas_fp16_acc(true);  // Qwen3 resolves gemm.cublas_fp16_acc=auto to on
    const int max_m = kRows[sizeof(kRows) / sizeof(kRows[0]) - 1];
    int max_n = 0, max_k = 0;
    for (const auto& s : kShapes) {
        max_n = std::max(max_n, s.N);
        max_k = std::max(max_k, s.K);
    }
    // s8 + half scale + float rowsum = 19/16 B per element, plus the 8 MiB split-K slice
    const size_t act_bytes = static_cast<size_t>(max_m) * max_k * 2;
    if (!engine_arena().is_open() && engine_arena_open(cuda_malloc_backend(), act_bytes) != MemError::Ok) {
        printf("bench_q8_imma: engine arena (%zu B) did not open\n", act_bytes);
        return false;
    }
    mmq_q8_imma_preallocate(max_m, max_k);  // one activation take at the largest (M, K)
    Bufs b;
    std::vector<void*> weights;
    bool ok = cudaMalloc(&b.x, static_cast<size_t>(max_m) * max_k * 2) == cudaSuccess &&
              cudaMalloc(&b.out, static_cast<size_t>(max_m) * max_n * 2) == cudaSuccess &&
              cudaMalloc(&b.deq, static_cast<size_t>(max_n) * max_k * 2) == cudaSuccess;
    {
        std::vector<__half> hx(static_cast<size_t>(max_m) * max_k);
        std::mt19937 rng(2267);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto& v : hx)
            v = __float2half(nd(rng));
        ok = ok && cudaMemcpy(b.x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess;
    }
    printf(
        "=== Q8_0 prefill GEMM: dequant + FP16-acc cuBLAS (off) vs IMMA, median of %d x %d "
        "===\n",
        kRepeats, kIters);
    // layer_ms[model][m][variant]
    double layer_ms[2][sizeof(kRows) / sizeof(kRows[0])][kVariants] = {};
    bool all_measured = ok;
    // ncu filter (#2267e): IMP_Q8IMMA_MODEL, IMP_Q8IMMA_SHAPE, IMP_Q8IMMA_M; unset = every shape and M.
    const char* f_model = std::getenv("IMP_Q8IMMA_MODEL");
    const char* f_shape = std::getenv("IMP_Q8IMMA_SHAPE");
    const int f_m = std::getenv("IMP_Q8IMMA_M") != nullptr ? std::atoi(std::getenv("IMP_Q8IMMA_M")) : 0;
    for (const auto& s : kShapes) {
        if ((f_model != nullptr && std::strcmp(f_model, s.model) != 0) ||
            (f_shape != nullptr && std::strcmp(f_shape, s.name) != 0))
            continue;
        std::vector<uint8_t> hw;
        fill_q8(hw, s.N, s.K, static_cast<unsigned>(s.N * 31 + s.K));
        if (!ok || cudaMalloc(&b.w, hw.size()) != cudaSuccess ||
            cudaMemcpy(b.w, hw.data(), hw.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
            all_measured = false;
            break;
        }
        const int mi = std::strcmp(s.model, "Qwen3-8B") == 0 ? 0 : 1;
        for (size_t r = 0; r < sizeof(kRows) / sizeof(kRows[0]); ++r) {
            const int M = kRows[r];
            if (f_m > 0 && M != f_m) continue;
            float t[kVariants];
            for (int v = 0; v < kVariants; ++v) {
                mmq_q8_imma_set_legacy_kernel(v == 2);
                t[v] = time_variant(v, b, M, s.N, s.K);
                all_measured = all_measured && t[v] > 0.0f;
                layer_ms[mi][r][v] += static_cast<double>(t[v]) * s.per_layer;
            }
            mmq_q8_imma_set_legacy_kernel(false);
            printf("q8imma model=%s shape=%s M=%d N=%d K=%d off=%.4f imma=%.4f legacy=%.4f\n", s.model,
                   s.name, M, s.N, s.K, t[0], t[1], t[2]);
        }
        // Planes are keyed by weight pointer: keep every weight alive so no pointer is reused.
        weights.push_back(b.w);
        b.w = nullptr;
    }
    mmq_q8_imma_release_all();
    for (void* w : weights)
        cudaFree(w);
    for (int mi = 0; mi < 2; ++mi)
        for (size_t r = 0; r < sizeof(kRows) / sizeof(kRows[0]); ++r) {
            const double* l = layer_ms[mi][r];
            printf("layer model=%s M=%d off=%.4f imma=%.4f legacy=%.4f\n", mi == 0 ? "Qwen3-8B" : "Qwen3-4B",
                   kRows[r], l[0], l[1], l[2]);
        }
    cudaFree(b.x);
    cudaFree(b.out);
    cudaFree(b.deq);
    return all_measured;
}

}  // namespace imp
