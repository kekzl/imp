// =============================================================================
// mxf4nvf4_mma_bench.cu -- Raw MMA instruction throughput microbench
// =============================================================================
//
// Compares two MMA variants head-to-head on sm_120f:
//
// (A) LEGACY — kind::f8f6f4.m16n8k32.f32.e2m1.e2m1.f32
//     imp's current MXFP4 FMHA path. 16*8*32*2 = 8192 FMA-equivalent ops.
//
// (B) BLOCKSCALE — kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64
//     SageAttention3's 5×-speedup path. 16*8*64*2 = 16384 ops. 2× raw
//     ops per instruction and HW handles per-16-elem scale.
//
// Method: each warp issues N iterations of the target MMA in a tight
// loop with dependency on the accumulator. Measures wall time, computes
// effective TOPS per warp. Scales to per-GPU estimate by 170 SMs.
//
// Expected relative outcome:
//   - Instruction rate (cycles/MMA): similar if both are ~12-cycle pipeline
//   - Ops per instruction: 2× for BLOCKSCALE (k=64 vs k=32)
//   - Effective TOPS: 2× for BLOCKSCALE
//
// This is NOT a full attention kernel benchmark — it isolates the MMA
// pipeline and answers "is this instruction swap worth the integration
// effort?" (Project B Stage 4 gate.)
// =============================================================================

#include "compute/mxf4nvf4_mma_bench.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace imp {

// ---------------------------------------------------------------------------
// Legacy kernel: loops kind::f8f6f4.m16n8k32
// ---------------------------------------------------------------------------
__global__ void bench_f8f6f4_m16n8k32_kernel(int iterations, float* sink) {
    uint32_t a0 = threadIdx.x * 37u + 1u;
    uint32_t a1 = threadIdx.x * 41u + 2u;
    uint32_t a2 = 0u;  // padding (FP4 uses 2 of 4 A regs)
    uint32_t a3 = 0u;
    uint32_t b0 = threadIdx.x * 43u + 3u;
    uint32_t b1 = 0u;  // padding

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

#if __CUDA_ARCH__ >= 1200
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile(
            "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
            "{%0, %1, %2, %3},"
            "{%4, %5, %6, %7},"
            "{%8, %9},"
            "{%10, %11, %12, %13};\n"
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "f"(d0), "f"(d1), "f"(d2), "f"(d3));
    }
#endif

    // Sink to prevent DCE. Only thread 0 writes.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        sink[0] = d0 + d1 + d2 + d3;
    }
}

// ---------------------------------------------------------------------------
// Blockscale kernel: loops kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64
// ---------------------------------------------------------------------------
__global__ void bench_mxf4nvf4_blockscale_m16n8k64_kernel(int iterations, float* sink) {
    uint32_t a0 = threadIdx.x * 37u + 1u;
    uint32_t a1 = threadIdx.x * 41u + 2u;
    uint32_t a2 = threadIdx.x * 43u + 3u;
    uint32_t a3 = threadIdx.x * 47u + 4u;
    uint32_t b0 = threadIdx.x * 53u + 5u;
    uint32_t b1 = threadIdx.x * 59u + 6u;
    uint32_t sfa = 0x38383838u;  // FP8 UE4M3 ~ 1.0
    uint32_t sfb = 0x38383838u;

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    constexpr uint16_t tidA = 0;
    constexpr uint16_t bidA = 0;
    constexpr uint16_t bidB = 0;
    constexpr uint16_t tidB0 = 0;

#if __CUDA_ARCH__ >= 1200
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0, %1, %2, %3},"
            "{%4, %5, %6, %7},"
            "{%8, %9},"
            "{%10, %11, %12, %13},"
            "{%14},"
            "{%15, %16},"
            "{%17},"
            "{%18, %19};\n"
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "f"(d0), "f"(d1), "f"(d2), "f"(d3),
              "r"(sfa), "h"(bidA), "h"(tidA),
              "r"(sfb), "h"(bidB), "h"(tidB0));
    }
#endif

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        sink[0] = d0 + d1 + d2 + d3;
    }
}

// ---------------------------------------------------------------------------
// Host-side measurement
// ---------------------------------------------------------------------------
static float run_bench(void(*kernel)(int, float*), int warps, int iterations,
                       cudaStream_t stream) {
    float* d_sink = nullptr;
    if (cudaMalloc(&d_sink, sizeof(float)) != cudaSuccess) return -1.0f;

    // Warmup
    kernel<<<warps, 32, 0, stream>>>(iterations / 10, d_sink);
    cudaStreamSynchronize(stream);

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

    return total_ms / NUM_REPS;  // avg ms per rep
}

MmaBenchResult bench_mma_comparison(int warps, int iterations, cudaStream_t stream) {
    MmaBenchResult r;

    float legacy_ms = run_bench(bench_f8f6f4_m16n8k32_kernel, warps, iterations, stream);
    float block_ms  = run_bench(bench_mxf4nvf4_blockscale_m16n8k64_kernel,
                                 warps, iterations, stream);

    r.legacy_ms = legacy_ms;
    r.blockscale_ms = block_ms;

    // Ops per MMA instruction (FMA counted as 2 ops):
    //   legacy  m16n8k32: 16*8*32*2 = 8192
    //   blockscale m16n8k64: 16*8*64*2 = 16384
    constexpr double kLegacyOps = 16.0 * 8.0 * 32.0 * 2.0;
    constexpr double kBlockOps  = 16.0 * 8.0 * 64.0 * 2.0;

    const double total_mmas = static_cast<double>(warps) * iterations;
    // TOPS = ops_per_mma * total_mmas / seconds / 1e12
    r.legacy_tops     = (kLegacyOps * total_mmas) / (legacy_ms * 1e-3) / 1e12;
    r.blockscale_tops = (kBlockOps  * total_mmas) / (block_ms  * 1e-3) / 1e12;
    r.speedup         = r.blockscale_tops / r.legacy_tops;

    return r;
}

} // namespace imp
