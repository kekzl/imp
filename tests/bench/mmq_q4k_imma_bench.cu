// =============================================================================
// mmq_q4k_imma_bench.cu — Phase 1 INT8 IMMA throughput microbench
// =============================================================================
//
// Companion to design memo docs/plans/q4k_imma_design_2026_05_17.md.
//
// Each kernel is a tight per-warp loop of one MMA opcode with all operands
// alive — same harness pattern as tests/bench/mxf4nvf4_mma_variants_bench.cu.
// We measure raw issue-rate of:
//
//   imma_s32_s8_s8_k32   : m16n8k32 .row.col.s32.s8.s8.s32      (Q4_K IMMA target)
//   imma_s32_u8_s8_k32   : m16n8k32 .row.col.s32.u8.s8.s32      (mixed-sign variant)
//   imma_s32_u8_u8_k32   : m16n8k32 .row.col.s32.u8.u8.s32      (unsigned variant)
//   hmma_f32_f16_f16_k16 : m16n8k16 .row.col.f32.f16.f16.f32    (FP16 HMMA baseline)
//
// Ops per MMA:  2 × M × N × K  (FMA counts as 2 ops).
//   IMMA m16n8k32 = 2 × 16 × 8 × 32 = 8192
//   HMMA m16n8k16 = 2 × 16 × 8 × 16 = 4096
//
// Theoretical sm_120a peaks:
//   INT8 IMMA: ~838 TOPS
//   FP16 HMMA: ~419 TFLOPS
//   ⇒ Raw-MMA IMMA / HMMA TOPS ratio ≈ 2.0×
//
// Decision gate (cf. design memo §7):
//   ratio ≥ 1.8×  ⇒  hardware ceiling is real ⇒ PROCEED to Phase 2 production
//                    kernel (cp.async pipelining + ldmatrix + Q4-symmetric s8
//                    reordering); multi-week port.
//   ratio < 1.5×  ⇒  hardware throttled to FP16-peak on consumer Blackwell
//                    (same fate as the SM100-only tcgen05 family) ⇒ DEFER
//                    indefinitely.
//
// Note: this bench measures *raw* MMA-pipe throughput in isolation. It tells us
// whether the hardware is *willing* to dispatch INT8 TC at the advertised
// peak. It does NOT tell us whether a realistic tiled kernel (memory-bound on
// weight reads, cp.async-latency-bound, ldmatrix-bound) can approach that
// ceiling — Phase 2 work measures that.

#include "bench/mmq_q4k_imma_bench.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace imp {

// Stable register inputs (different per thread) plus zero accumulators.
// [[maybe_unused]] silences NVCC warnings on variants that consume a subset.
#define IMMA_BENCH_PREAMBLE                                                                   \
    [[maybe_unused]] uint32_t a0 = threadIdx.x * 37u + 1u;                                    \
    [[maybe_unused]] uint32_t a1 = threadIdx.x * 41u + 2u;                                    \
    [[maybe_unused]] uint32_t a2 = threadIdx.x * 43u + 3u;                                    \
    [[maybe_unused]] uint32_t a3 = threadIdx.x * 47u + 4u;                                    \
    [[maybe_unused]] uint32_t b0 = threadIdx.x * 53u + 5u;                                    \
    [[maybe_unused]] uint32_t b1 = threadIdx.x * 59u + 6u;                                    \
    [[maybe_unused]] int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;                                  \
    [[maybe_unused]] float fd0 = 0.0f, fd1 = 0.0f, fd2 = 0.0f, fd3 = 0.0f

#define IMMA_BENCH_SINK_STORE(EXPR)         \
    if (threadIdx.x == 0 && blockIdx.x == 0) \
        sink[0] = static_cast<float>(EXPR)

// ---------------------------------------------------------------------------
// (1) IMMA s32.s8.s8.s32 m16n8k32 — Q4_K_M direct-GEMM candidate
// ---------------------------------------------------------------------------
__global__ void bench_imma_s32_s8_s8_k32(int iterations, float* sink) {
    IMMA_BENCH_PREAMBLE;
#if __CUDA_ARCH__ >= 750  // INT8 IMMA is sm_75+
#pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
#endif
    IMMA_BENCH_SINK_STORE(c0 + c1 + c2 + c3);
}

// ---------------------------------------------------------------------------
// (2) IMMA s32.u8.s8.s32 m16n8k32 — mixed-sign (A=u8, B=s8) variant
// ---------------------------------------------------------------------------
__global__ void bench_imma_s32_u8_s8_k32(int iterations, float* sink) {
    IMMA_BENCH_PREAMBLE;
#if __CUDA_ARCH__ >= 750
#pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
#endif
    IMMA_BENCH_SINK_STORE(c0 + c1 + c2 + c3);
}

// ---------------------------------------------------------------------------
// (3) IMMA s32.u8.u8.s32 m16n8k32 — fully unsigned variant
// ---------------------------------------------------------------------------
__global__ void bench_imma_s32_u8_u8_k32(int iterations, float* sink) {
    IMMA_BENCH_PREAMBLE;
#if __CUDA_ARCH__ >= 750
#pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
#endif
    IMMA_BENCH_SINK_STORE(c0 + c1 + c2 + c3);
}

// ---------------------------------------------------------------------------
// (4) HMMA f32.f16.f16.f32 m16n8k16 — FP16 baseline (sm_80+ universal)
// ---------------------------------------------------------------------------
__global__ void bench_hmma_f32_f16_f16_k16(int iterations, float* sink) {
    IMMA_BENCH_PREAMBLE;
#if __CUDA_ARCH__ >= 800
#pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(fd0), "+f"(fd1), "+f"(fd2), "+f"(fd3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
#endif
    IMMA_BENCH_SINK_STORE(fd0 + fd1 + fd2 + fd3);
}

// ---------------------------------------------------------------------------
// Host launcher (identical to mxf4nvf4_mma_variants_bench::run_one)
// ---------------------------------------------------------------------------
static float run_one(void (*kernel)(int, float*), int warps, int iterations, cudaStream_t stream) {
    float* d_sink = nullptr;
    if (cudaMalloc(&d_sink, sizeof(float)) != cudaSuccess) return -1.0f;

    // Warm-up + launch validity check.
    kernel<<<warps, 32, 0, stream>>>(iterations / 10, d_sink);
    cudaStreamSynchronize(stream);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(d_sink);
        return -2.0f;
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

ImmaBenchResult bench_mmq_q4k_imma(int warps, int iterations, cudaStream_t stream) {
    ImmaBenchResult r{};

    struct Entry {
        const char* label;
        void (*kernel)(int, float*);
        double ops_per_mma;
    };
    Entry entries[] = {
        {"imma_s32_s8_s8_k32",   bench_imma_s32_s8_s8_k32,   2.0 * 16.0 * 8.0 * 32.0},
        {"imma_s32_u8_s8_k32",   bench_imma_s32_u8_s8_k32,   2.0 * 16.0 * 8.0 * 32.0},
        {"imma_s32_u8_u8_k32",   bench_imma_s32_u8_u8_k32,   2.0 * 16.0 * 8.0 * 32.0},
        {"hmma_f32_f16_f16_k16", bench_hmma_f32_f16_f16_k16, 2.0 * 16.0 * 8.0 * 16.0},
    };

    int n = sizeof(entries) / sizeof(entries[0]);
    if (n > ImmaBenchResult::kMaxEntries) n = ImmaBenchResult::kMaxEntries;
    r.count = n;

    const double total_mmas = static_cast<double>(warps) * iterations;
    for (int i = 0; i < n; ++i) {
        float ms = run_one(entries[i].kernel, warps, iterations, stream);
        r.entries[i].label = entries[i].label;
        r.entries[i].ms = ms;
        r.entries[i].ops_per_mma = entries[i].ops_per_mma;
        if (ms > 0.0f) {
            r.entries[i].tops = (entries[i].ops_per_mma * total_mmas) / (ms * 1e-3) / 1e12;
        } else {
            r.entries[i].tops = -1.0;
        }
    }
    return r;
}

}  // namespace imp
