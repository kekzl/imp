#include "compute/gemm.h"
#include "quant/quant_gemm.h"
#include "core/tensor.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cstdlib>

namespace imp {

static constexpr int kWarmupIters = 5;
static constexpr int kTimedIters  = 20;

struct GemmSize {
    int64_t M;
    int64_t N;
    int64_t K;
    const char* label;
};

static const GemmSize kSizes[] = {
    {    1, 4096,  4096, "M=1, N=4096, K=4096"      },  // GEMV-like, single token
    {  128, 4096,  4096, "M=128, N=4096, K=4096"     },  // prefill batch
    { 4096, 4096,  4096, "M=4096, N=4096, K=4096"    },  // peak compute
    {   32, 11008, 4096, "M=32, N=11008, K=4096"     },  // Llama FFN gate/up
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
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

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
    Tensor A(d_A, DType::FP16, 2, shape_A, true);
    Tensor B(d_B, DType::FP16, 2, shape_B, true);
    Tensor C(d_C, DType::FP16, 2, shape_C, true);

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
    int64_t shape_A[]      = {M, K};
    int64_t shape_Bq[]     = {N, K_packed};
    int64_t shape_scales[] = {N, num_groups};
    int64_t shape_C[]      = {M, N};

    Tensor A(d_A, DType::FP16, 2, shape_A, true);
    Tensor B_quant(d_Bq, DType::INT4, 2, shape_Bq, true);
    Tensor scales(d_scales, DType::FP16, 2, shape_scales, true);
    Tensor C(d_C, DType::FP16, 2, shape_C, true);

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

} // namespace imp
