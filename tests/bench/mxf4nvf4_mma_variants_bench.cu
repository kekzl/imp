// Benchmarks every block_scale/sparse PTX MMA CUTLASS sm120 headers expose (mma_sm120.hpp,
// mma_sm120_sparse.hpp), to find a variant beating the default mxf4nvf4.block_scale
// vec::4X.m16n8k64.ue4m3 on QK^T throughput. Sparse variants need 2:4 metadata on A.

#include "bench/mxf4nvf4_mma_variants_bench.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace imp {

// Stable per-thread input registers (defeat constant-folding) plus UE4M3=0x38 / UE8M0=0x7f
// (~1.0 scales). [[maybe_unused]] silences NVCC #550-D for regs a given variant skips.
#define BENCH_PREAMBLE                                                                   \
    [[maybe_unused]] uint32_t a0 = threadIdx.x * 37u + 1u;                               \
    [[maybe_unused]] uint32_t a1 = threadIdx.x * 41u + 2u;                               \
    [[maybe_unused]] uint32_t a2 = threadIdx.x * 43u + 3u;                               \
    [[maybe_unused]] uint32_t a3 = threadIdx.x * 47u + 4u;                               \
    [[maybe_unused]] uint32_t b0 = threadIdx.x * 53u + 5u;                               \
    [[maybe_unused]] uint32_t b1 = threadIdx.x * 59u + 6u;                               \
    [[maybe_unused]] uint32_t b2 = threadIdx.x * 61u + 7u;                               \
    [[maybe_unused]] uint32_t b3 = threadIdx.x * 67u + 8u;                               \
    [[maybe_unused]] uint32_t sfa_e4m3 = 0x38383838u; /* UE4M3 ≈ 1.0 */                  \
    [[maybe_unused]] uint32_t sfb_e4m3 = 0x38383838u;                                    \
    [[maybe_unused]] uint32_t sfa_e8m0 = 0x7f7f7f7fu; /* UE8M0 = 2^0 = 1.0 */            \
    [[maybe_unused]] uint32_t sfb_e8m0 = 0x7f7f7f7fu;                                    \
    [[maybe_unused]] uint32_t metadata = 0x44444444u; /* 2:4 sparse: positions 0,1 of 4 */ \
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;                                    \
    [[maybe_unused]] constexpr uint16_t bidA = 0, tidA = 0, bidB = 0, tidB = 0

#define BENCH_SINK_STORE                     \
    if (threadIdx.x == 0 && blockIdx.x == 0) \
    sink[0] = d0 + d1 + d2 + d3

// ---------------------------------------------------------------------------
// (1) DENSE_F8F6F4_K32 — legacy baseline (no scaling)
// ---------------------------------------------------------------------------
__global__ void bench_v_dense_f8f6f4_k32(int iterations, float* sink) {
    BENCH_PREAMBLE;
#if __CUDA_ARCH__ >= 1200
#pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile(
            "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
            : "r"(a0), "r"(a1), "r"(0), "r"(0), "r"(b0), "r"(0), "f"(d0), "f"(d1), "f"(d2), "f"(d3));
    }
#endif
    BENCH_SINK_STORE;
}

// ---------------------------------------------------------------------------
// (2) BS_NVFP4_4X_E4M3 — current Project B default
// ---------------------------------------------------------------------------
__global__ void bench_v_nvfp4_4x_e4m3(int iterations, float* sink) {
    BENCH_PREAMBLE;
#if __CUDA_ARCH__ >= 1200
#pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32."
            "ue4m3 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};\n"
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(d0), "f"(d1), "f"(d2), "f"(d3),
              "r"(sfa_e4m3), "h"(bidA), "h"(tidA), "r"(sfb_e4m3), "h"(bidB), "h"(tidB));
    }
#endif
    BENCH_SINK_STORE;
}

// ---------------------------------------------------------------------------
// (3) BS_MXFP4_4X_E8M0 — same K=64 NVFP4 layout, alternative UE8M0 scale type
// ---------------------------------------------------------------------------
__global__ void bench_v_nvfp4_4x_e8m0(int iterations, float* sink) {
    BENCH_PREAMBLE;
#if __CUDA_ARCH__ >= 1200
#pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32."
            "ue8m0 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};\n"
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(d0), "f"(d1), "f"(d2), "f"(d3),
              "r"(sfa_e8m0), "h"(bidA), "h"(tidA), "r"(sfb_e8m0), "h"(bidB), "h"(tidB));
    }
#endif
    BENCH_SINK_STORE;
}

// ---------------------------------------------------------------------------
// (4) BS_MXFP4_2X_E8M0 — per-32 scales (vs ::4X per-16). Half the SF traffic.
// ---------------------------------------------------------------------------
__global__ void bench_v_mxfp4_2x_e8m0(int iterations, float* sink) {
    BENCH_PREAMBLE;
#if __CUDA_ARCH__ >= 1200
#pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32."
            "ue8m0 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};\n"
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(d0), "f"(d1), "f"(d2), "f"(d3),
              "r"(sfa_e8m0), "h"(bidA), "h"(tidA), "r"(sfb_e8m0), "h"(bidB), "h"(tidB));
    }
#endif
    BENCH_SINK_STORE;
}

// (5) BS_MXF8F6F4_1X_E2M1: FP4xFP4 at K=32 with HW scale; same K as legacy so expect
// similar throughput.
__global__ void bench_v_mxf8f6f4_1x_e2m1(int iterations, float* sink) {
    BENCH_PREAMBLE;
#if __CUDA_ARCH__ >= 1200
#pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile(
            "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32."
            "ue8m0 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};\n"
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(d0), "f"(d1), "f"(d2), "f"(d3),
              "r"(sfa_e8m0), "h"(bidA), "h"(tidA), "r"(sfb_e8m0), "h"(bidB), "h"(tidB));
    }
#endif
    BENCH_SINK_STORE;
}

// ---------------------------------------------------------------------------
// (6) BS_MXF8F6F4_1X_E4M3 — FP8 × FP8 (Q FP8 quality, half the K of K=64)
// ---------------------------------------------------------------------------
__global__ void bench_v_mxf8f6f4_1x_e4m3(int iterations, float* sink) {
    BENCH_PREAMBLE;
#if __CUDA_ARCH__ >= 1200
#pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile(
            "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32."
            "ue8m0 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};\n"
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(d0), "f"(d1), "f"(d2), "f"(d3),
              "r"(sfa_e8m0), "h"(bidA), "h"(tidA), "r"(sfb_e8m0), "h"(bidB), "h"(tidB));
    }
#endif
    BENCH_SINK_STORE;
}

// ---------------------------------------------------------------------------
// (7) BS_MXF8F6F4_1X_E4M3xE2M1 — Q in FP8, K in FP4 (asymmetric quality)
// ---------------------------------------------------------------------------
__global__ void bench_v_mxf8f6f4_1x_e4m3_e2m1(int iterations, float* sink) {
    BENCH_PREAMBLE;
#if __CUDA_ARCH__ >= 1200
#pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile(
            "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32."
            "ue8m0 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};\n"
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(d0), "f"(d1), "f"(d2), "f"(d3),
              "r"(sfa_e8m0), "h"(bidA), "h"(tidA), "r"(sfb_e8m0), "h"(bidB), "h"(tidB));
    }
#endif
    BENCH_SINK_STORE;
}

// (8) BS_MXF8F6F4_2X_E2M1_K64: DISABLED. ptxas rejects m16n8k64 for kind::mxf8f6f4 on
// sm_120a (max is m16n8k32); k64 is valid only for kind::mxf4nvf4. Dead end, do not retry.
__global__ void bench_v_mxf8f6f4_2x_e2m1_k64(int iterations, float* sink) {
    BENCH_PREAMBLE;
    // Intentionally empty — ptxas rejects mxf8f6f4 m16n8k64 on sm_120a.
    BENCH_SINK_STORE;
}

// ---------------------------------------------------------------------------
// (9) SP_F8F6F4_K64 — sparse f8f6f4 at K=64 (no scaling; 2× K vs dense legacy)
// ---------------------------------------------------------------------------
__global__ void bench_v_sparse_f8f6f4_k64(int iterations, float* sink) {
    BENCH_PREAMBLE;
#if __CUDA_ARCH__ >= 1200
#pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile(
            "mma.sync.aligned.kind::f8f6f4.sp::ordered_metadata.m16n8k64.row.col.f32.e2m1.e2m1.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%12,%13,%14,%15},%16,0x0;\n"
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2), "r"(b3), "f"(d0), "f"(d1),
              "f"(d2), "f"(d3), "r"(metadata));
    }
#endif
    BENCH_SINK_STORE;
}

// (9) SP_NVFP4_4X_K128: DISABLED. ptxas rejects '.kind::mxf4nvf4 with .sp modifier' on
// sm_120f: CUTLASS exposes the PTX template but consumer Blackwell lacks the hardware.
// Dead end, do not retry.
__global__ void bench_v_sparse_nvfp4_k128(int iterations, float* sink) {
    BENCH_PREAMBLE;
    // Intentionally empty — see comment above.
    BENCH_SINK_STORE;
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
static float run_one(void (*kernel)(int, float*), int warps, int iterations, cudaStream_t stream) {
    float* d_sink = nullptr;
    if (cudaMalloc(&d_sink, sizeof(float)) != cudaSuccess)
        return -1.0f;

    kernel<<<warps, 32, 0, stream>>>(iterations / 10, d_sink);
    cudaStreamSynchronize(stream);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(d_sink);
        return -2.0f;  // launch failure
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    constexpr int NUM_REPS = 5;
    float total_ms = 0.0f;
    for (int rep = 0; rep < NUM_REPS; ++rep) {
        cudaEventRecord(start, stream);
        kernel<<<warps, 32, 0, stream>>>(iterations, d_sink);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_sink);
    return total_ms / NUM_REPS;
}

MmaVariantsBenchResult bench_mma_variants(int warps, int iterations, cudaStream_t stream) {
    MmaVariantsBenchResult r{};

    struct Entry {
        const char* label;
        void (*kernel)(int, float*);
        double ops_per_mma;  // FMAs * 2 = m * n * k * 2 for dense; sparse same since metadata selects
    };
    Entry entries[] = {
        {"dense_f8f6f4_k32", bench_v_dense_f8f6f4_k32, 16.0 * 8.0 * 32.0 * 2.0},
        {"bs_nvfp4_4x_e4m3_k64", bench_v_nvfp4_4x_e4m3, 16.0 * 8.0 * 64.0 * 2.0},
        {"bs_nvfp4_4x_e8m0_k64", bench_v_nvfp4_4x_e8m0, 16.0 * 8.0 * 64.0 * 2.0},
        {"bs_mxfp4_2x_e8m0_k64", bench_v_mxfp4_2x_e8m0, 16.0 * 8.0 * 64.0 * 2.0},
        {"bs_mxf8f6f4_1x_e2m1", bench_v_mxf8f6f4_1x_e2m1, 16.0 * 8.0 * 32.0 * 2.0},
        {"bs_mxf8f6f4_1x_e4m3", bench_v_mxf8f6f4_1x_e4m3, 16.0 * 8.0 * 32.0 * 2.0},
        {"bs_mxf8f6f4_1x_e4m3xe2m1", bench_v_mxf8f6f4_1x_e4m3_e2m1, 16.0 * 8.0 * 32.0 * 2.0},
        // bs_mxf8f6f4_2x_e2m1_k64: ptxas rejects mxf8f6f4 m16n8k64 on sm_120a. Stub kernel, not measured.
        {"sp_f8f6f4_k64", bench_v_sparse_f8f6f4_k64, 16.0 * 8.0 * 64.0 * 2.0},
        // sp_nvfp4_4x_k128: ptxas rejects on sm_120f. See comment above kernel.
    };

    int n = sizeof(entries) / sizeof(entries[0]);
    if (n > MmaVariantsBenchResult::kMaxEntries)
        n = MmaVariantsBenchResult::kMaxEntries;
    r.count = n;

    const double total_mmas = static_cast<double>(warps) * iterations;
    for (int i = 0; i < n; ++i) {
        float ms = run_one(entries[i].kernel, warps, iterations, stream);
        r.entries[i].label = entries[i].label;
        r.entries[i].ms = ms;
        if (ms > 0.0f) {
            r.entries[i].tops = (entries[i].ops_per_mma * total_mmas) / (ms * 1e-3) / 1e12;
        } else {
            r.entries[i].tops = -1.0;  // launch failed
        }
    }
    return r;
}

}  // namespace imp
