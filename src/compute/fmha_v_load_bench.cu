// =============================================================================
// fmha_v_load_bench.cu — cp.async vs TMA bulk for FMHA V-tile loads (SM120)
// =============================================================================
//
// Microbench. See header for context.
//
// Bench shape mirrors the FMHA-MXFP4 V-prefetch path in
// attention_fmha_mxfp4_sm120.cu lines 755-772:
//   - V tile geometry: Bkv rows × head_dim halves, contiguous gmem (row-major)
//   - 1 tile per kernel iteration
//   - SMEM destination is the KV_fp16 buffer (head_dim halves per row)
//
// Two variants:
//   A. cp.async (current): 128-thread CTA, each thread issues
//      `cp.async.ca.shared.global [%0],[%1],16;` for an 8-halves chunk.
//      Loop until tile is loaded. Single cp_async_commit + wait_group.
//   B. TMA bulk (proposed): single `cp.async.bulk.tensor.2d` issued by thread 0,
//      mbarrier coordinates completion across the CTA.
//
// Bandwidth is per-CTA. We launch on every SM (170 on RTX 5090) to saturate
// memory engines, the same way the FMHA kernel does in practice.
// =============================================================================

#include "compute/fmha_v_load_bench.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

using PFN_cuTensorMapEncodeTiled_t = CUresult (*)(
    CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
    const cuuint64_t*, const cuuint32_t*, const cuuint32_t*,
    CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion,
    CUtensorMapFloatOOBfill);

namespace imp {

static constexpr int kThreads = 128;

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ void bench_cp_async_ca_16(void* smem, const void* glob) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(glob));
}

__device__ __forceinline__ void bench_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void bench_cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n");
}

__device__ __forceinline__ void bench_mbarrier_init(uint64_t* bar, uint32_t count) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(s), "r"(count));
}

__device__ __forceinline__ void bench_mbarrier_inval(uint64_t* bar) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" ::"r"(s));
}

__device__ __forceinline__ void bench_mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(s), "r"(bytes));
}

__device__ __forceinline__ void bench_mbarrier_wait(uint64_t* bar, uint32_t phase) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAIT_V: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@p bra DONE_V;\n"
        "bra WAIT_V;\n"
        "DONE_V:\n"
        "}\n"
        :: "r"(s), "r"(phase));
}

__device__ __forceinline__ void bench_cp_async_bulk_tensor_2d(
    void* smem_dst, const void* desc, int x, int y, uint64_t* mbar) {
    uint32_t s_dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    uint32_t s_bar = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(s_dst), "l"(desc), "r"(x), "r"(y), "r"(s_bar)
        : "memory");
}

// ---------------------------------------------------------------------------
// Variant A: cp.async (current FMHA path)
//
// 1 tile per iter. 128 threads. Each thread loads an 8-halves chunk per
// position in the tile. CHUNK_HALVES=8 (16 bytes) matches FMHA-MXFP4.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kThreads) bench_cp_async_v_load(
    int iters, int Bkv, int head_dim,
    const __half* __restrict__ src,
    uint32_t* __restrict__ sink) {

    extern __shared__ __align__(128) uint8_t smem_raw[];
    __half* KV_fp16 = reinterpret_cast<__half*>(smem_raw);

    const int tid = threadIdx.x;
    constexpr int CHUNK_HALVES = 8;  // 16 bytes
    const int total_chunks = (Bkv * head_dim) / CHUNK_HALVES;

    uint32_t acc = 0u;

#pragma unroll 1
    for (int it = 0; it < iters; ++it) {
        // Load tile via cp.async loop (mirrors FMHA pattern).
        for (int c = tid; c < total_chunks; c += kThreads) {
            int elem = c * CHUNK_HALVES;
            int r = elem / head_dim;
            int d = elem % head_dim;
            bench_cp_async_ca_16(&KV_fp16[r * head_dim + d], &src[r * head_dim + d]);
        }
        bench_cp_async_commit();
        bench_cp_async_wait_all();
        __syncthreads();

        // Sink: prevent compiler from eliminating the SMEM writes.
        if (tid == 0) {
            acc ^= *reinterpret_cast<uint32_t*>(&KV_fp16[0]);
        }
    }

    if (tid == 0 && blockIdx.x == 0) *sink = acc;
}

// ---------------------------------------------------------------------------
// Variant B: TMA bulk
//
// Single thread issues cp.async.bulk.tensor.2d for the whole tile. mbarrier
// coordinates completion. Tile box: Bkv × head_dim halves (innermost = halves,
// outermost = rows).
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kThreads) bench_tma_v_load(
    int iters, int Bkv, int head_dim,
    const __grid_constant__ CUtensorMap desc,
    uint32_t* __restrict__ sink) {

    extern __shared__ __align__(128) uint8_t smem_raw[];
    __half* KV_fp16 = reinterpret_cast<__half*>(smem_raw);
    // Place mbar at end of tile, 128-byte aligned via SMEM alignment.
    const int tile_halves = Bkv * head_dim;
    const int tile_bytes  = tile_halves * sizeof(__half);
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem_raw + tile_bytes);

    const int tid = threadIdx.x;
    if (tid == 0) {
        bench_mbarrier_init(mbar, 1);
    }
    __syncthreads();

    uint32_t acc = 0u;
    uint32_t phase = 0u;

#pragma unroll 1
    for (int it = 0; it < iters; ++it) {
        if (tid == 0) {
            bench_mbarrier_arrive_expect_tx(mbar, tile_bytes);
            // TMA descriptor coords: (col_in_halves, row). Innermost dim first.
            bench_cp_async_bulk_tensor_2d(KV_fp16, &desc, 0, 0, mbar);
        }
        bench_mbarrier_wait(mbar, phase);
        phase ^= 1u;

        if (tid == 0) {
            acc ^= *reinterpret_cast<uint32_t*>(&KV_fp16[0]);
        }
    }

    if (tid == 0) {
        bench_mbarrier_inval(mbar);
        if (blockIdx.x == 0) *sink = acc;
    }
}

// ---------------------------------------------------------------------------
// Host-side TMA descriptor builder for FP16 V-tile.
// ---------------------------------------------------------------------------
static bool build_v_tma_desc(CUtensorMap* desc, void* gmem,
                              int Bkv, int head_dim) {
    static PFN_cuTensorMapEncodeTiled_t pfn = nullptr;
    if (pfn == nullptr) {
        cudaDriverEntryPointQueryResult q;
        void* p = nullptr;
        cudaError_t err = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",
                                                            &p, CUDA_VERSION,
                                                            cudaEnableDefault, &q);
        if (err != cudaSuccess || q != cudaDriverEntryPointSuccess || p == nullptr) {
            std::fprintf(stderr, "cuTensorMapEncodeTiled lookup failed\n");
            return false;
        }
        pfn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_t>(p);
    }

    // 2D tensor: innermost = head_dim halves (cols), outermost = Bkv (rows).
    // Element type FP16. Row stride = head_dim * 2 bytes (contiguous).
    cuuint64_t gmem_shape[2]  = { (cuuint64_t)head_dim, (cuuint64_t)Bkv };
    cuuint64_t gmem_stride[1] = { (cuuint64_t)(head_dim * sizeof(__half)) };
    cuuint32_t smem_box[2]    = { (cuuint32_t)head_dim, (cuuint32_t)Bkv };
    cuuint32_t smem_box_stride[2] = { 1u, 1u };

    CUresult r = pfn(
        desc,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        /*tensorRank=*/2,
        gmem, gmem_shape, gmem_stride,
        smem_box, smem_box_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) {
        std::fprintf(stderr, "cuTensorMapEncodeTiled failed: r=%d\n", (int)r);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
bool fmha_v_load_bench(int Bkv, int head_dim, FmhaVLoadBenchResult* out) {
    if (!out) return false;
    if (Bkv <= 0 || head_dim <= 0) return false;
    if ((Bkv * head_dim) % 8 != 0) {
        std::fprintf(stderr, "fmha_v_load_bench: Bkv*head_dim must be %% 8\n");
        return false;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int sms = prop.multiProcessorCount;
    const int tile_bytes = Bkv * head_dim * (int)sizeof(__half);
    // SMEM layout: tile + mbar (8B). Round up to 128B alignment.
    const int smem_bytes = ((tile_bytes + 16 + 127) / 128) * 128;

    // Allocate one V tile in gmem (shared across all CTAs).
    void* d_src = nullptr;
    cudaError_t err = cudaMalloc(&d_src, tile_bytes);
    if (err != cudaSuccess) return false;
    cudaMemset(d_src, 0x42, tile_bytes);

    uint32_t* d_sink = nullptr;
    cudaMalloc(&d_sink, sizeof(uint32_t));
    cudaMemset(d_sink, 0, sizeof(uint32_t));

    // Build TMA descriptor.
    CUtensorMap desc{};
    if (!build_v_tma_desc(&desc, d_src, Bkv, head_dim)) {
        cudaFree(d_src);
        cudaFree(d_sink);
        return false;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Set max dynamic smem for both kernels.
    cudaFuncSetAttribute(bench_cp_async_v_load,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes);
    cudaFuncSetAttribute(bench_tma_v_load,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes);

    constexpr int kIters = 4096;
    constexpr int kReps  = 7;

    // Warmup
    bench_cp_async_v_load<<<sms, kThreads, smem_bytes, stream>>>(
        kIters / 8, Bkv, head_dim, (const __half*)d_src, d_sink);
    bench_tma_v_load<<<sms, kThreads, smem_bytes, stream>>>(
        kIters / 8, Bkv, head_dim, desc, d_sink);
    cudaStreamSynchronize(stream);
    if (cudaGetLastError() != cudaSuccess) {
        std::fprintf(stderr, "fmha_v_load_bench: warmup launch failed\n");
        cudaFree(d_src); cudaFree(d_sink); cudaStreamDestroy(stream);
        return false;
    }

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    auto time_kernel = [&](auto launcher) -> double {
        float total_ms = 0.0f;
        for (int r = 0; r < kReps; ++r) {
            cudaEventRecord(ev0, stream);
            launcher();
            cudaEventRecord(ev1, stream);
            cudaEventSynchronize(ev1);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev0, ev1);
            total_ms += ms;
        }
        return static_cast<double>(total_ms) / kReps;
    };

    double cp_ms  = time_kernel([&]() {
        bench_cp_async_v_load<<<sms, kThreads, smem_bytes, stream>>>(
            kIters, Bkv, head_dim, (const __half*)d_src, d_sink);
    });
    double tma_ms = time_kernel([&]() {
        bench_tma_v_load<<<sms, kThreads, smem_bytes, stream>>>(
            kIters, Bkv, head_dim, desc, d_sink);
    });

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaStreamDestroy(stream);
    cudaFree(d_src);
    cudaFree(d_sink);

    // Bandwidth: each CTA loads tile_bytes per iter. sms CTAs × iters.
    double bytes_per_run = static_cast<double>(sms) * kIters * tile_bytes;
    out->cp_async_ms       = cp_ms;
    out->tma_bulk_ms       = tma_ms;
    out->speedup           = cp_ms / tma_ms;
    out->cp_async_gb_per_s = bytes_per_run / (cp_ms * 1e-3) / 1e9;
    out->tma_bulk_gb_per_s = bytes_per_run / (tma_ms * 1e-3) / 1e9;
    return true;
}

}  // namespace imp
