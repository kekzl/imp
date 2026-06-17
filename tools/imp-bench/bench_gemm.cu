#include "compute/gemm.h"
#include "compute/gemm_cutlass_sm120.h"
#include "quant/quant_gemm.h"
#include "quant/nvfp4_quant.h"
#include "core/tensor.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cstdlib>

namespace imp {

static constexpr int kWarmupIters = 5;
static constexpr int kTimedIters = 20;

struct GemmSize {
    int64_t M;
    int64_t N;
    int64_t K;
    const char* label;
};

static const GemmSize kSizes[] = {
    {1, 4096, 4096, "M=1, N=4096, K=4096"},        // GEMV-like, single token
    {128, 4096, 4096, "M=128, N=4096, K=4096"},    // prefill batch
    {4096, 4096, 4096, "M=4096, N=4096, K=4096"},  // peak compute
    {32, 11008, 4096, "M=32, N=11008, K=4096"},    // Llama FFN gate/up
};
static constexpr int kNumSizes = sizeof(kSizes) / sizeof(kSizes[0]);

// Fill a host buffer with random FP16 values in [-1, 1].
static void fill_random_fp16(half* buf, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        float val = 2.0f * (static_cast<float>(rand()) / RAND_MAX) - 1.0f;
        buf[i] = __float2half(val);
    }
}

// Fill a host buffer with random bytes (for packed INT4 weights).
static void fill_random_bytes(uint8_t* buf, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        buf[i] = static_cast<uint8_t>(rand() & 0xFF);
    }
}

// Benchmark FP16 cuBLAS GEMM for a single size. Returns average latency in ms.
static float bench_fp16_gemm(const GemmSize& sz) {
    int64_t M = sz.M, N = sz.N, K = sz.K;

    // Allocate device memory for A [M, K], B [K, N], C [M, N]
    size_t bytes_A = static_cast<size_t>(M * K) * sizeof(half);
    size_t bytes_B = static_cast<size_t>(K * N) * sizeof(half);
    size_t bytes_C = static_cast<size_t>(M * N) * sizeof(half);

    void *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    if (cudaMalloc(&d_A, bytes_A) != cudaSuccess || cudaMalloc(&d_B, bytes_B) != cudaSuccess ||
        cudaMalloc(&d_C, bytes_C) != cudaSuccess) {
        if (d_A)
            cudaFree(d_A);
        if (d_B)
            cudaFree(d_B);
        if (d_C)
            cudaFree(d_C);
        fprintf(stderr, "bench_fp16_gemm: cudaMalloc failed for M=%ld N=%ld K=%ld\n", M, N, K);
        return -1.0f;
    }

    // Fill with random data
    std::vector<half> h_A(M * K);
    std::vector<half> h_B(K * N);
    fill_random_fp16(h_A.data(), M * K);
    fill_random_fp16(h_B.data(), K * N);
    cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, bytes_C);

    // Build tensor descriptors
    int64_t shape_A[] = {M, K};
    int64_t shape_B[] = {K, N};
    int64_t shape_C[] = {M, N};
    Tensor A(d_A, QType::F16, 2, shape_A, true);
    Tensor B(d_B, QType::F16, 2, shape_B, true);
    Tensor C(d_C, QType::F16, 2, shape_C, true);

    // Warmup
    for (int i = 0; i < kWarmupIters; ++i) {
        gemm(A, B, C, 1.0f, 0.0f, nullptr);
    }
    cudaDeviceSynchronize();

    // Timed iterations
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, nullptr);
    for (int i = 0; i < kTimedIters; ++i) {
        gemm(A, B, C, 1.0f, 0.0f, nullptr);
    }
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / static_cast<float>(kTimedIters);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return avg_ms;
}

// Benchmark INT4 quantized GEMM for a single size. Returns average latency in ms.
static float bench_int4_gemm(const GemmSize& sz) {
    int64_t M = sz.M, N = sz.N, K = sz.K;

    // A is FP16 [M, K]
    size_t bytes_A = static_cast<size_t>(M * K) * sizeof(half);
    void* d_A = nullptr;
    cudaMalloc(&d_A, bytes_A);

    std::vector<half> h_A(M * K);
    fill_random_fp16(h_A.data(), M * K);
    cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice);

    // B_quant is INT4-packed [N, K/2] (two 4-bit values per byte)
    int64_t K_packed = K / 2;
    size_t bytes_Bq = static_cast<size_t>(N * K_packed);
    void* d_Bq = nullptr;
    cudaMalloc(&d_Bq, bytes_Bq);

    std::vector<uint8_t> h_Bq(N * K_packed);
    fill_random_bytes(h_Bq.data(), N * K_packed);
    cudaMemcpy(d_Bq, h_Bq.data(), bytes_Bq, cudaMemcpyHostToDevice);

    // Scales are FP16 [N, K/32] (one scale per group of 32 elements)
    int64_t num_groups = K / 32;
    size_t bytes_scales = static_cast<size_t>(N * num_groups) * sizeof(half);
    void* d_scales = nullptr;
    cudaMalloc(&d_scales, bytes_scales);

    std::vector<half> h_scales(N * num_groups);
    fill_random_fp16(h_scales.data(), N * num_groups);
    cudaMemcpy(d_scales, h_scales.data(), bytes_scales, cudaMemcpyHostToDevice);

    // C is FP16 [M, N]
    size_t bytes_C = static_cast<size_t>(M * N) * sizeof(half);
    void* d_C = nullptr;
    cudaMalloc(&d_C, bytes_C);
    cudaMemset(d_C, 0, bytes_C);

    // Build tensor descriptors
    int64_t shape_A[] = {M, K};
    int64_t shape_Bq[] = {N, K_packed};
    int64_t shape_scales[] = {N, num_groups};
    int64_t shape_C[] = {M, N};

    Tensor A(d_A, QType::F16, 2, shape_A, true);
    Tensor B_quant(d_Bq, QType::INT4, 2, shape_Bq, true);
    Tensor scales(d_scales, QType::F16, 2, shape_scales, true);
    Tensor C(d_C, QType::F16, 2, shape_C, true);

    // Warmup
    for (int i = 0; i < kWarmupIters; ++i) {
        quant_gemm_int4(A, B_quant, scales, C, nullptr);
    }
    cudaDeviceSynchronize();

    // Timed iterations
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, nullptr);
    for (int i = 0; i < kTimedIters; ++i) {
        quant_gemm_int4(A, B_quant, scales, C, nullptr);
    }
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / static_cast<float>(kTimedIters);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_Bq);
    cudaFree(d_scales);
    cudaFree(d_C);

    return avg_ms;
}

// Benchmark the PRODUCTION CUTLASS sm_120 block-scaled NVFP4xNVFP4 dense GEMM
// (gemm_nvfp4_cutlass_sm120) — the kernel the from-scratch tools/standalone/
// gemm_nvfp4_sm120a.cu reference beats (48% vs ~41% of FP4 peak). Square shapes
// match the standalone's M=N=K cubed measurements so ncu profiles line up
// apples-to-apples (occupancy + lts__t_requests/sectors). Returns avg ms.
static float bench_nvfp4_cutlass_gemm(const GemmSize& sz) {
    int64_t M = sz.M, N = sz.N, K = sz.K;
    if (!cutlass_sm120_nvfp4_available()) {
        fprintf(stderr, "bench_nvfp4_cutlass_gemm: CUTLASS sm_120 NVFP4 GEMM not compiled\n");
        return -1.0f;
    }

    // --- Activation A [M,K] FP16 -> CUTLASS NVFP4 (packed FP4 + SfAtom scales) ---
    std::vector<half> h_A(static_cast<size_t>(M) * K);
    fill_random_fp16(h_A.data(), M * K);
    void* d_A_fp16 = nullptr;
    cudaMalloc(&d_A_fp16, h_A.size() * sizeof(half));
    cudaMemcpy(d_A_fp16, h_A.data(), h_A.size() * sizeof(half), cudaMemcpyHostToDevice);

    void* a_data = nullptr;  // [M, K/2] packed FP4
    void* a_sf = nullptr;    // SfAtom UE4M3 scales
    cudaMalloc(&a_data, static_cast<size_t>(M) * (K / 2));
    cudaMalloc(&a_sf, cutlass_nvfp4_sf_size(static_cast<int>(M), static_cast<int>(K)));
    quantize_fp16_to_nvfp4_cutlass(d_A_fp16, a_data, a_sf, static_cast<int>(M), static_cast<int>(K), nullptr);

    // --- Weight B [N,K] FP16 -> NvFP4QuantResult -> CUTLASS block-scaled ---
    std::vector<half> h_B(static_cast<size_t>(N) * K);
    fill_random_fp16(h_B.data(), N * K);
    void* d_B_fp16 = nullptr;
    cudaMalloc(&d_B_fp16, h_B.size() * sizeof(half));
    cudaMemcpy(d_B_fp16, h_B.data(), h_B.size() * sizeof(half), cudaMemcpyHostToDevice);
    int64_t bshape[2] = {N, K};
    Tensor b_t(d_B_fp16, QType::F16, 2, bshape, /*on_device=*/true);
    NvFP4QuantResult qr{};
    quantize_fp16_to_nvfp4(b_t, qr, nullptr);
    CutlassNvFP4Weight cw{};
    convert_nvfp4_to_cutlass(qr, cw, nullptr);

    // --- Output + workspace ---
    void* d_D = nullptr;
    cudaMalloc(&d_D, static_cast<size_t>(M) * N * sizeof(half));
    size_t ws_size = gemm_nvfp4_cutlass_sm120_workspace(static_cast<int>(M), static_cast<int>(N),
                                                        static_cast<int>(K));
    void* ws = nullptr;
    if (ws_size > 0)
        cudaMalloc(&ws, ws_size);
    cudaDeviceSynchronize();

    auto run = [&]() {
        return gemm_nvfp4_cutlass_sm120(a_data, a_sf, cw, d_D, static_cast<int>(M), static_cast<int>(N),
                                        static_cast<int>(K), ws, ws_size, nullptr);
    };

    // Warmup >1s to ramp clocks (idle-downclock is the dominant cold-start artifact).
    bool ok = true;
    for (int i = 0; i < 50; ++i)
        ok = run() && ok;
    cudaDeviceSynchronize();
    if (!ok)
        fprintf(stderr, "bench_nvfp4_cutlass_gemm: kernel returned false for M=%ld N=%ld K=%ld\n", M, N, K);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, nullptr);
    for (int i = 0; i < kTimedIters; ++i)
        run();
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / static_cast<float>(kTimedIters);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free_cutlass_nvfp4_weight(cw);
    free_nvfp4_result(qr);
    if (ws)
        cudaFree(ws);
    cudaFree(d_D);
    cudaFree(a_data);
    cudaFree(a_sf);
    cudaFree(d_A_fp16);
    cudaFree(d_B_fp16);
    return avg_ms;
}

// Square NVFP4 shapes matching tools/standalone/gemm_nvfp4_sm120a.cu (M=N=K cubed).
static const GemmSize kNvfp4Sizes[] = {
    {2048, 2048, 2048, "M=N=K=2048"},
    {4096, 4096, 4096, "M=N=K=4096"},
    {8192, 8192, 8192, "M=N=K=8192"},
    // Realistic prefill shapes (small M = chunk size, N/K = model dims) — the
    // grid-underfill regime where warp-spec's 1-block/SM could lose to a
    // higher-occupancy cp.async path. N=K=5120 ~ Qwen3-14B hidden.
    {128, 5120, 5120, "M=128 N=K=5120"},
    {256, 5120, 5120, "M=256 N=K=5120"},
    {512, 5120, 5120, "M=512 N=K=5120"},
    {1024, 5120, 5120, "M=1024 N=K=5120"},
    {2048, 5120, 5120, "M=2048 N=K=5120"},
};
static constexpr int kNumNvfp4Sizes = sizeof(kNvfp4Sizes) / sizeof(kNvfp4Sizes[0]);

void bench_gemm_nvfp4_cutlass() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        printf("bench_gemm_nvfp4_cutlass: no CUDA device available, skipping.\n");
        return;
    }
    printf("=== Production CUTLASS sm_120 NVFP4 dense GEMM ===\n");
    printf("(peak FP4 mma.sync ~2019 TOPS measured; standalone ref hits 48%%)\n\n");
    for (int i = 0; i < kNumNvfp4Sizes; ++i) {
        const GemmSize& sz = kNvfp4Sizes[i];
        float avg_ms = bench_nvfp4_cutlass_gemm(sz);
        if (avg_ms <= 0.0f)
            continue;
        double tops = 2.0 * sz.M * sz.N * sz.K / (avg_ms * 1e-3) / 1e12;
        double pct_peak = tops / 2019.0 * 100.0;
        printf("  [%-14s] %8.3f ms  %7.1f TOP/s  %5.1f%% of 2019 peak\n", sz.label, avg_ms, tops, pct_peak);
    }
    printf("\n");
}

void bench_gemm() {
    // Check for CUDA device availability
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        printf("bench_gemm: no CUDA device available, skipping.\n");
        return;
    }

    printf("=== GEMM Benchmark ===\n\n");

    // --- FP16 cuBLAS GEMM ---
    printf("FP16 cuBLAS GEMM:\n");
    for (int i = 0; i < kNumSizes; ++i) {
        const GemmSize& sz = kSizes[i];
        float avg_ms = bench_fp16_gemm(sz);
        double tflops = 2.0 * sz.M * sz.N * sz.K / (avg_ms * 1e-3) / 1e12;
        printf("  [%-30s] %8.3f ms  %7.2f TFLOPS\n", sz.label, avg_ms, tflops);
    }
    printf("\n");

    // --- INT4 Quantized GEMM ---
    printf("INT4 Quantized GEMM:\n");
    for (int i = 0; i < kNumSizes; ++i) {
        const GemmSize& sz = kSizes[i];
        float avg_ms = bench_int4_gemm(sz);
        double tflops = 2.0 * sz.M * sz.N * sz.K / (avg_ms * 1e-3) / 1e12;
        printf("  [%-30s] %8.3f ms  %7.2f TFLOPS\n", sz.label, avg_ms, tflops);
    }
    printf("\n");
}

}  // namespace imp
