// =============================================================================
// mma_peak_saturated.cu -- Saturated tensor-core peak microbench (sm_120a)
// =============================================================================
//
// Measures the TRUE achievable mma.sync throughput per dtype: 8 warps/SM
// with 4 independent accumulator chains per warp (ILP hides the MMA
// pipeline latency). The older mxf4nvf4_mma_bench runs 1 warp/SM with a
// serial accumulator dependency — that measures issue LATENCY, not peak
// (it reads ~273 TOPS where saturation reaches ~2019).
//
// Calibration results (2026-06-07, RTX 5090 @ ~2.85 GHz boost under load):
//   FP4  mxf4nvf4 block_scale : ~2019 TOPS  (datasheet 3354 — mma.sync
//                               delivers HALF; 4096 ops/SM/clk, not 8192)
//   FP16 f16acc               : ~1956 TFLOPS (= datasheet 838 * 2.85/2.407 —
//                               methodology check, full rate confirmed)
//   FP16 f32acc               : ~253 TFLOPS  (1/4 rate on GeForce)
//   FP8  e4m3 f32acc          : ~496 TOPS    (1/4 rate on GeForce)
//
// These numbers feed tools/roofline/config.json flop_per_cycle (issues
// #595/#596): roofline %s against the 3354/838 datasheet values understate
// kernel quality by 2-4x for f32-accumulate / FP4 kernels.
// =============================================================================

#include "bench/mma_peak_saturated.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

namespace {

constexpr int kBlocks = 170;          // 1 block/SM on RTX 5090
constexpr int kWarpsPerBlock = 8;     // saturates the 4 TC schedulers
constexpr int kChains = 4;            // independent accumulators (ILP)
constexpr int kIterations = 1 << 14;  // per-chain MMA issues

__global__ void peak_mxf4nvf4_kernel(int iterations, float* sink) {
    uint32_t a0 = threadIdx.x * 37u + 1u, a1 = threadIdx.x * 41u + 2u;
    uint32_t a2 = threadIdx.x * 43u + 3u, a3 = threadIdx.x * 47u + 4u;
    uint32_t b0 = threadIdx.x * 53u + 5u, b1 = threadIdx.x * 59u + 6u;
    uint32_t sfa = 0x38383838u, sfb = 0x38383838u;  // UE4M3 ~1.0
    constexpr uint16_t zid = 0;

    float d0[kChains], d1[kChains], d2[kChains], d3[kChains];
#pragma unroll
    for (int c = 0; c < kChains; ++c) d0[c] = d1[c] = d2[c] = d3[c] = 0.0f;

#if __CUDA_ARCH__ >= 1200
#pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
#pragma unroll
        for (int c = 0; c < kChains; ++c) {
            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1."
                "e2m1.f32.ue4m3 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, "
                "{%14}, {%15, %16}, {%17}, {%18, %19};\n"
                : "=f"(d0[c]), "=f"(d1[c]), "=f"(d2[c]), "=f"(d3[c])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(d0[c]), "f"(d1[c]),
                  "f"(d2[c]), "f"(d3[c]), "r"(sfa), "h"(zid), "h"(zid), "r"(sfb), "h"(zid), "h"(zid));
        }
    }
#endif
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < kChains; ++c) s += d0[c] + d1[c] + d2[c] + d3[c];
        sink[0] = s;
    }
}

__global__ void peak_fp16_f16acc_kernel(int iterations, float* sink) {
    uint32_t a0 = threadIdx.x * 37u + 1u, a1 = threadIdx.x * 41u + 2u;
    uint32_t a2 = threadIdx.x * 43u + 3u, a3 = threadIdx.x * 47u + 4u;
    uint32_t b0 = threadIdx.x * 53u + 5u, b1 = threadIdx.x * 59u + 6u;

    uint32_t d0[kChains], d1[kChains];
#pragma unroll
    for (int c = 0; c < kChains; ++c) d0[c] = d1[c] = 0u;

#pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
#pragma unroll
        for (int c = 0; c < kChains; ++c) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
                : "=r"(d0[c]), "=r"(d1[c])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(d0[c]), "r"(d1[c]));
        }
    }
    if (threadIdx.x == 0 && blockIdx.x == 0)
        sink[0] = static_cast<float>(d0[0] + d1[0]);
}

__global__ void peak_fp16_f32acc_kernel(int iterations, float* sink) {
    uint32_t a0 = threadIdx.x * 37u + 1u, a1 = threadIdx.x * 41u + 2u;
    uint32_t a2 = threadIdx.x * 43u + 3u, a3 = threadIdx.x * 47u + 4u;
    uint32_t b0 = threadIdx.x * 53u + 5u, b1 = threadIdx.x * 59u + 6u;

    float d0[kChains], d1[kChains], d2[kChains], d3[kChains];
#pragma unroll
    for (int c = 0; c < kChains; ++c) d0[c] = d1[c] = d2[c] = d3[c] = 0.0f;

#pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
#pragma unroll
        for (int c = 0; c < kChains; ++c) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(d0[c]), "=f"(d1[c]), "=f"(d2[c]), "=f"(d3[c])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(d0[c]), "f"(d1[c]),
                  "f"(d2[c]), "f"(d3[c]));
        }
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < kChains; ++c) s += d0[c] + d1[c] + d2[c] + d3[c];
        sink[0] = s;
    }
}

__global__ void peak_fp8_f32acc_kernel(int iterations, float* sink) {
    uint32_t a0 = threadIdx.x * 37u + 1u, a1 = threadIdx.x * 41u + 2u;
    uint32_t a2 = threadIdx.x * 43u + 3u, a3 = threadIdx.x * 47u + 4u;
    uint32_t b0 = threadIdx.x * 53u + 5u, b1 = threadIdx.x * 59u + 6u;

    float d0[kChains], d1[kChains], d2[kChains], d3[kChains];
#pragma unroll
    for (int c = 0; c < kChains; ++c) d0[c] = d1[c] = d2[c] = d3[c] = 0.0f;

#if __CUDA_ARCH__ >= 890
#pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
#pragma unroll
        for (int c = 0; c < kChains; ++c) {
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(d0[c]), "=f"(d1[c]), "=f"(d2[c]), "=f"(d3[c])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(d0[c]), "f"(d1[c]),
                  "f"(d2[c]), "f"(d3[c]));
        }
    }
#endif
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < kChains; ++c) s += d0[c] + d1[c] + d2[c] + d3[c];
        sink[0] = s;
    }
}

// Best-of-3 reps of one kernel; returns effective ops/s across all warps.
double run_peak(void (*kernel)(int, float*), double ops_per_mma, cudaStream_t stream) {
    float* d_sink = nullptr;
    if (cudaMalloc(&d_sink, sizeof(float)) != cudaSuccess)
        return 0.0;

    dim3 block(32 * kWarpsPerBlock);
    kernel<<<kBlocks, block, 0, stream>>>(kIterations / 8, d_sink);  // warmup
    cudaStreamSynchronize(stream);

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    double best = 0.0;
    for (int r = 0; r < 3; ++r) {
        cudaEventRecord(s, stream);
        kernel<<<kBlocks, block, 0, stream>>>(kIterations, d_sink);
        cudaEventRecord(e, stream);
        cudaEventSynchronize(e);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, s, e);
        double total_mmas = static_cast<double>(kBlocks) * kWarpsPerBlock * kChains * kIterations;
        double tops = total_mmas * ops_per_mma / (ms * 1e-3) / 1e12;
        if (tops > best)
            best = tops;
    }
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    cudaFree(d_sink);
    return best;
}

}  // namespace

MmaPeakResult bench_mma_peak_saturated(cudaStream_t stream) {
    // Clock-ramp warmup >1s: idle downclock is the dominant cold-start
    // artifact on this box (see benchmark methodology).
    {
        float* d_sink = nullptr;
        cudaMalloc(&d_sink, sizeof(float));
        for (int i = 0; i < 10; ++i)
            peak_mxf4nvf4_kernel<<<kBlocks, 32 * kWarpsPerBlock, 0, stream>>>(kIterations, d_sink);
        cudaStreamSynchronize(stream);
        cudaFree(d_sink);
    }

    MmaPeakResult r{};
    r.fp4_blockscale_tops = run_peak(peak_mxf4nvf4_kernel, 16384.0, stream);   // 16*8*64*2
    r.fp16_f16acc_tflops = run_peak(peak_fp16_f16acc_kernel, 4096.0, stream);  // 16*8*16*2
    r.fp16_f32acc_tflops = run_peak(peak_fp16_f32acc_kernel, 4096.0, stream);
    r.fp8_f32acc_tops = run_peak(peak_fp8_f32acc_kernel, 8192.0, stream);      // 16*8*32*2
    return r;
}

}  // namespace imp
