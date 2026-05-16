// =============================================================================
// mxf4nvf4_mma_variants_bench.cu -- Microbench all sm_120 FP4/FP8 MMA variants
// =============================================================================
//
// Compares throughput of every relevant block_scale + sparse PTX MMA the
// CUTLASS sm120 headers expose (mma_sm120.hpp + mma_sm120_sparse.hpp).
// Goal: identify any variant beyond `kind::mxf4nvf4.block_scale.scale_vec::
// 4X.m16n8k64.ue4m3` (Project B's current default) that would beat it on
// raw throughput for our QK^T workload.
//
// What we test (all dense; sparse requires 2:4 metadata, also tested):
//
//   Dense, no scaling:
//     [DENSE_F8F6F4_K32]   kind::f8f6f4.m16n8k32              (legacy, baseline)
//
//   Dense block_scale variants:
//     [BS_NVFP4_4X_E4M3]   mxf4nvf4 vec::4X k64 ue4m3  (current default)
//     [BS_MXFP4_4X_E8M0]   mxf4nvf4 vec::4X k64 ue8m0  (NVFP4 layout, MXFP4 scale)
//     [BS_MXFP4_2X_E8M0]   mxf4nvf4 vec::2X k64 ue8m0  (per-32 scales, half scale loads)
//     [BS_MXF8F6F4_1X_44]  mxf8f6f4 vec::1X k32 ue8m0 e4m3.e4m3  (FP8 × FP8 mixed)
//     [BS_MXF8F6F4_1X_41]  mxf8f6f4 vec::1X k32 ue8m0 e4m3.e2m1  (FP8 Q × FP4 K)
//     [BS_MXF8F6F4_1X_11]  mxf8f6f4 vec::1X k32 ue8m0 e2m1.e2m1  (FP4 × FP4 with HW scale)
//     [BS_MXF8F6F4_2X_11]  mxf8f6f4 vec::2X k64 e2m1.e2m1 — DISABLED (ptxas rejects k64 for mxf8f6f4)
//
//   Sparse (require 2:4 sparsity on A):
//     [SP_F8F6F4_K64]      f8f6f4.sp::ordered_metadata k64
//     [SP_NVFP4_4X_K128]   mxf4nvf4 vec::4X.sp k128 ue4m3
//
// Sparse ones double K per issue → 2× raw throughput vs dense if 2:4 sparsity
// is achievable. Mixed-precision ones don't change K but enable Q-precision
// trade-offs.
// =============================================================================

#include "bench/mxf4nvf4_mma_variants_bench.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace imp {

// Helper: stable input registers (different per thread to defeat any
// constant-folding) plus 0x38 UE4M3 / 0x7f UE8M0 ≈ 1.0 scales. Each kernel
// below references a subset of these; [[maybe_unused]] silences NVCC #550-D
// for the regs a given variant doesn't consume.
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

// ---------------------------------------------------------------------------
// (5) BS_MXF8F6F4_1X_E2M1 — FP4 × FP4 at K=32 with HW scale (alternative to
//     legacy + manual scale; same K so should be ~same throughput as legacy).
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// (8) BS_MXF8F6F4_2X_E2M1_K64 — DISABLED. ptxas rejects on sm_120a:
// "Incorrect instruction type specified for mma with shape '.m16n8k64'".
// kind::mxf8f6f4 is limited to m16n8k32 — k64 is only valid for kind::mxf4nvf4.
// The mxf8f6f4 kind covers FP8/FP6/FP4 formats (min element = 4 bits, max = 8
// bits), so k32 already exercises 256 elements per tile at 8 bpe. Combining
// scale_vec::2X with k64 would require PTX support that simply doesn't exist.
// Dead end — do not retry.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// (9) SP_NVFP4_4X_K128 — DISABLED. ptxas rejects the instruction on sm_120f:
// "Feature '.kind::mxf4nvf4 with .sp modifier' not supported on .target 'sm_120f'".
// CUTLASS exposes the PTX template in mma_sm120_sparse.hpp but the actual
// hardware doesn't have it on consumer Blackwell. Marking as a dead end.
// ---------------------------------------------------------------------------
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
