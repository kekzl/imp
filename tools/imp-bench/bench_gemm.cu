#include "compute/gemm.h"
#include "compute/gemm_cutlass_sm120.h"
#include "quant/quant_gemm.h"
#include "quant/nvfp4_gemm_tile.h"
#include "quant/nvfp4_quant.h"
#include "core/logging.h"
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
            IMP_CUDA_CHECK_LOG(cudaFree(d_A));
        if (d_B)
            IMP_CUDA_CHECK_LOG(cudaFree(d_B));
        if (d_C)
            IMP_CUDA_CHECK_LOG(cudaFree(d_C));
        fprintf(stderr, "bench_fp16_gemm: cudaMalloc failed for M=%ld N=%ld K=%ld\n", M, N, K);
        return -1.0f;
    }

    // Fill with random data
    std::vector<half> h_A(M * K);
    std::vector<half> h_B(K * N);
    fill_random_fp16(h_A.data(), M * K);
    fill_random_fp16(h_B.data(), K * N);
    IMP_CUDA_CHECK_LOG(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    IMP_CUDA_CHECK_LOG(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));
    IMP_CUDA_CHECK_LOG(cudaMemset(d_C, 0, bytes_C));

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
    IMP_CUDA_CHECK_LOG(cudaDeviceSynchronize());

    // Timed iterations
    cudaEvent_t start, stop;
    IMP_CUDA_CHECK_LOG(cudaEventCreate(&start));
    IMP_CUDA_CHECK_LOG(cudaEventCreate(&stop));

    IMP_CUDA_CHECK_LOG(cudaEventRecord(start, nullptr));
    for (int i = 0; i < kTimedIters; ++i) {
        gemm(A, B, C, 1.0f, 0.0f, nullptr);
    }
    IMP_CUDA_CHECK_LOG(cudaEventRecord(stop, nullptr));
    IMP_CUDA_CHECK_LOG(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    IMP_CUDA_CHECK_LOG(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / static_cast<float>(kTimedIters);

    IMP_CUDA_CHECK_LOG(cudaEventDestroy(start));
    IMP_CUDA_CHECK_LOG(cudaEventDestroy(stop));
    IMP_CUDA_CHECK_LOG(cudaFree(d_A));
    IMP_CUDA_CHECK_LOG(cudaFree(d_B));
    IMP_CUDA_CHECK_LOG(cudaFree(d_C));

    return avg_ms;
}

// Benchmark INT4 quantized GEMM for a single size. Returns average latency in ms.
static float bench_int4_gemm(const GemmSize& sz) {
    int64_t M = sz.M, N = sz.N, K = sz.K;

    // A is FP16 [M, K]
    size_t bytes_A = static_cast<size_t>(M * K) * sizeof(half);
    void* d_A = nullptr;
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_A, bytes_A));

    std::vector<half> h_A(M * K);
    fill_random_fp16(h_A.data(), M * K);
    IMP_CUDA_CHECK_LOG(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));

    // B_quant is INT4-packed [N, K/2] (two 4-bit values per byte)
    int64_t K_packed = K / 2;
    size_t bytes_Bq = static_cast<size_t>(N * K_packed);
    void* d_Bq = nullptr;
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_Bq, bytes_Bq));

    std::vector<uint8_t> h_Bq(N * K_packed);
    fill_random_bytes(h_Bq.data(), N * K_packed);
    IMP_CUDA_CHECK_LOG(cudaMemcpy(d_Bq, h_Bq.data(), bytes_Bq, cudaMemcpyHostToDevice));

    // Scales are FP16 [N, K/32] (one scale per group of 32 elements)
    int64_t num_groups = K / 32;
    size_t bytes_scales = static_cast<size_t>(N * num_groups) * sizeof(half);
    void* d_scales = nullptr;
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_scales, bytes_scales));

    std::vector<half> h_scales(N * num_groups);
    fill_random_fp16(h_scales.data(), N * num_groups);
    IMP_CUDA_CHECK_LOG(cudaMemcpy(d_scales, h_scales.data(), bytes_scales, cudaMemcpyHostToDevice));

    // C is FP16 [M, N]
    size_t bytes_C = static_cast<size_t>(M * N) * sizeof(half);
    void* d_C = nullptr;
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_C, bytes_C));
    IMP_CUDA_CHECK_LOG(cudaMemset(d_C, 0, bytes_C));

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
    IMP_CUDA_CHECK_LOG(cudaDeviceSynchronize());

    // Timed iterations
    cudaEvent_t start, stop;
    IMP_CUDA_CHECK_LOG(cudaEventCreate(&start));
    IMP_CUDA_CHECK_LOG(cudaEventCreate(&stop));

    IMP_CUDA_CHECK_LOG(cudaEventRecord(start, nullptr));
    for (int i = 0; i < kTimedIters; ++i) {
        quant_gemm_int4(A, B_quant, scales, C, nullptr);
    }
    IMP_CUDA_CHECK_LOG(cudaEventRecord(stop, nullptr));
    IMP_CUDA_CHECK_LOG(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    IMP_CUDA_CHECK_LOG(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / static_cast<float>(kTimedIters);

    IMP_CUDA_CHECK_LOG(cudaEventDestroy(start));
    IMP_CUDA_CHECK_LOG(cudaEventDestroy(stop));
    IMP_CUDA_CHECK_LOG(cudaFree(d_A));
    IMP_CUDA_CHECK_LOG(cudaFree(d_Bq));
    IMP_CUDA_CHECK_LOG(cudaFree(d_scales));
    IMP_CUDA_CHECK_LOG(cudaFree(d_C));

    return avg_ms;
}

// Benchmarks the production NVFP4xNVFP4 dense prefill GEMM (gemm_nvfp4_tile, auto variant with the
// stream-K workspace). Square shapes match tools/standalone/gemm_nvfp4_sm120a.cu's M=N=K measurements
// so ncu profiles line up. Returns avg ms.
static float bench_nvfp4_tile_gemm(const GemmSize& sz) {
    int64_t M = sz.M, N = sz.N, K = sz.K;

    // --- Activation A [M,K] FP16 -> NVFP4 (packed FP4 + SfAtom scales) ---
    std::vector<half> h_A(static_cast<size_t>(M) * K);
    fill_random_fp16(h_A.data(), M * K);
    void* d_A_fp16 = nullptr;
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_A_fp16, h_A.size() * sizeof(half)));
    IMP_CUDA_CHECK_LOG(cudaMemcpy(d_A_fp16, h_A.data(), h_A.size() * sizeof(half), cudaMemcpyHostToDevice));

    void* a_data = nullptr;  // [M, K/2] packed FP4
    void* a_sf = nullptr;    // SfAtom UE4M3 scales
    IMP_CUDA_CHECK_LOG(cudaMalloc(&a_data, static_cast<size_t>(M) * (K / 2)));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&a_sf, cutlass_nvfp4_sf_size(static_cast<int>(M), static_cast<int>(K))));
    quantize_fp16_to_nvfp4_cutlass(d_A_fp16, a_data, a_sf, static_cast<int>(M), static_cast<int>(K), nullptr);

    // --- Weight B [N,K] FP16 -> NvFP4QuantResult -> SfAtom block-scaled ---
    std::vector<half> h_B(static_cast<size_t>(N) * K);
    fill_random_fp16(h_B.data(), N * K);
    void* d_B_fp16 = nullptr;
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_B_fp16, h_B.size() * sizeof(half)));
    IMP_CUDA_CHECK_LOG(cudaMemcpy(d_B_fp16, h_B.data(), h_B.size() * sizeof(half), cudaMemcpyHostToDevice));
    int64_t bshape[2] = {N, K};
    Tensor b_t(d_B_fp16, QType::F16, 2, bshape, /*on_device=*/true);
    NvFP4QuantResult qr{};
    quantize_fp16_to_nvfp4(b_t, qr, nullptr);
    CutlassNvFP4Weight cw{};
    convert_nvfp4_to_cutlass(qr, cw, nullptr);

    // --- Output + workspace ---
    void* d_D = nullptr;
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_D, static_cast<size_t>(M) * N * sizeof(half)));
    size_t ws_size = gemm_tile_streamk_workspace_bytes();
    void* ws = nullptr;
    if (ws_size > 0)
        IMP_CUDA_CHECK_LOG(cudaMalloc(&ws, ws_size));
    IMP_CUDA_CHECK_LOG(cudaDeviceSynchronize());

    auto run = [&]() {
        return gemm_nvfp4_tile(a_data, a_sf, cw.data, cw.scale_factors, cw.tensor_scale, d_D,
                               static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), nullptr,
                               /*variant=*/-1, ws, ws_size);
    };

    // Warmup >1s to ramp clocks (idle-downclock is the dominant cold-start artifact).
    bool ok = true;
    for (int i = 0; i < 50; ++i)
        ok = run() && ok;
    IMP_CUDA_CHECK_LOG(cudaDeviceSynchronize());
    if (!ok)
        fprintf(stderr, "bench_nvfp4_tile_gemm: kernel returned false for M=%ld N=%ld K=%ld\n", M, N, K);

    cudaEvent_t start, stop;
    IMP_CUDA_CHECK_LOG(cudaEventCreate(&start));
    IMP_CUDA_CHECK_LOG(cudaEventCreate(&stop));
    IMP_CUDA_CHECK_LOG(cudaEventRecord(start, nullptr));
    for (int i = 0; i < kTimedIters; ++i)
        run();
    IMP_CUDA_CHECK_LOG(cudaEventRecord(stop, nullptr));
    IMP_CUDA_CHECK_LOG(cudaEventSynchronize(stop));
    float total_ms = 0.0f;
    IMP_CUDA_CHECK_LOG(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / static_cast<float>(kTimedIters);

    IMP_CUDA_CHECK_LOG(cudaEventDestroy(start));
    IMP_CUDA_CHECK_LOG(cudaEventDestroy(stop));
    free_cutlass_nvfp4_weight(cw);
    free_nvfp4_result(qr);
    if (ws)
        IMP_CUDA_CHECK_LOG(cudaFree(ws));
    IMP_CUDA_CHECK_LOG(cudaFree(d_D));
    IMP_CUDA_CHECK_LOG(cudaFree(a_data));
    IMP_CUDA_CHECK_LOG(cudaFree(a_sf));
    IMP_CUDA_CHECK_LOG(cudaFree(d_A_fp16));
    IMP_CUDA_CHECK_LOG(cudaFree(d_B_fp16));
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

bool bench_gemm_nvfp4_cutlass() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        printf("bench_gemm_nvfp4_cutlass: no CUDA device available, skipping.\n");
        return false;  // measured nothing (#1584)
    }
    printf("=== Production NVFP4 dense GEMM (own tile, gemm_nvfp4_tile) ===\n");
    printf("(peak FP4 mma.sync ~2019 TOPS measured; standalone ref hits 48%%)\n\n");
    for (int i = 0; i < kNumNvfp4Sizes; ++i) {
        const GemmSize& sz = kNvfp4Sizes[i];
        float avg_ms = bench_nvfp4_tile_gemm(sz);
        if (avg_ms <= 0.0f)
            continue;
        double tops = 2.0 * sz.M * sz.N * sz.K / (avg_ms * 1e-3) / 1e12;
        double pct_peak = tops / 2019.0 * 100.0;
        printf("  [%-14s] %8.3f ms  %7.1f TOP/s  %5.1f%% of 2019 peak\n", sz.label, avg_ms, tops, pct_peak);
    }
    printf("\n");
    return true;
}

bool bench_gemm() {
    // Check for CUDA device availability
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        printf("bench_gemm: no CUDA device available, skipping.\n");
        return false;  // measured nothing (#1584)
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
    return true;
}

}  // namespace imp
