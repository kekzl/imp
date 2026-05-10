// =============================================================================
// tma_block_scale_bench.cu — Block-scale-aware TMA descriptor microbench (SM120)
// =============================================================================
//
// Validates spec assumption in docs/superpowers/specs/2026-05-10-nvfp4-smallM-kernel-design.md:
// "single TMA descriptor for data + scales is +10-20% faster than two separate
//  descriptors".
//
// Both variants use REAL Hopper/Blackwell TMA via `cp.async.bulk.tensor.2d`
// (driver API CUtensorMap descriptors built with cuTensorMapEncodeTiled, mirrors
// the pattern in src/compute/gemm_cutlass_grouped_3x.cu / cute's make_tma_copy).
//
// Variant SEPARATE  : two CUtensorMap descriptors (FP4 data tile + UE4M3 scale
//                     tile), two cp.async.bulk.tensor.2d issues per iter,
//                     single mbarrier covering both transactions.
// Variant FUSED     : one CUtensorMap descriptor over a packed contiguous gmem
//                     tile (FP4 data + UE4M3 scales packed as a single 2-D
//                     tile), one cp.async.bulk.tensor.2d issue per iter, single
//                     mbarrier.
//
// This is the apples-to-apples version of the spec question. CUTLASS itself
// always uses two TMA descriptors for NVFP4 (TMA_A + TMA_SFA in
// sm120_blockscaled_mma_tma.hpp:298-311) — the "fused" path is hypothetical and
// must be benched against real TMA, not against per-thread cp.async.cg.
//
// Hardware: RTX 5090 sm_120a, CUDA 13.2.1.
// =============================================================================

#include "compute/tma_block_scale_bench.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

// Resolve cuTensorMapEncodeTiled via cudaGetDriverEntryPoint (runtime API,
// no DT_NEEDED on libcuda.so.1) to keep test discovery working in CI builders
// without GPU drivers installed. Same trick CUTLASS uses
// (cutlass/cuda_host_adapter.hpp).
using PFN_cuTensorMapEncodeTiled_t = CUresult (*)(
    CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
    const cuuint64_t*, const cuuint32_t*, const cuuint32_t*,
    CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion,
    CUtensorMapFloatOOBfill);

namespace imp {

// ---------------------------------------------------------------------------
// Tile geometry. TMA on SM120 requires the innermost gmem stride to be a
// multiple of 16 bytes, so all tiles use 128-byte rows.
//
//   FP4 data tile:    128 rows × 128 cols = 16 KiB  (16384 B FP4 nibble-packed)
//   UE4M3 scale tile:  16 rows × 128 cols =  2 KiB  (2048 B UE4M3, realistic
//                     ratio 1 SF per 16 FP4 elements ≈ 1/16th of data bytes,
//                     rounded up to TMA-legal box geometry)
//   Combined tile:    144 rows × 128 cols = 18 KiB  (data + scales packed
//                     contiguously row-by-row; loaded by one TMA descriptor)
// ---------------------------------------------------------------------------
static constexpr int kRowBytes       = 128;     // bytes per row, all tiles
static constexpr int kDataRows       = 128;
static constexpr int kScaleRows      = 16;
static constexpr int kCombinedRows   = kDataRows + kScaleRows;  // 144

static constexpr int kDataBytes      = kDataRows     * kRowBytes;  // 16384
static constexpr int kScaleBytes     = kScaleRows    * kRowBytes;  //  2048
static constexpr int kCombinedBytes  = kCombinedRows * kRowBytes;  // 18432
static constexpr int kTotalBytes     = kDataBytes + kScaleBytes;   // 18432 (= combined)

// SMEM layout: 128-byte aligned, holds either two tiles or one combined tile
// plus an 8-byte mbarrier. We use the same union for both kernels so smem
// allocation matches.
struct __align__(128) SmemLayout {
    union {
        struct {
            uint8_t data[kDataBytes];
            uint8_t scales[kScaleBytes];
        } sep;
        uint8_t combined[kCombinedBytes];
    } u;
    __align__(8) uint64_t mbar;
};

// ---------------------------------------------------------------------------
// PTX wrappers
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t count) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(s), "r"(count));
}

__device__ __forceinline__ void mbarrier_invalidate(uint64_t* bar) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" ::"r"(s));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(s), "r"(bytes));
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAIT: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@p bra DONE;\n"
        "bra WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(s), "r"(phase));
}

// 2-D bulk-tensor load. Emits UTMALDG on SM120.
__device__ __forceinline__ void cp_async_bulk_tensor_2d(
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
// Separate variant: two TMA descriptors (FP4 data + UE4M3 scales).
// Each iteration issues two cp.async.bulk.tensor loads, a single mbarrier
// covers both transactions.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(128) bench_separate(
    int iters,
    const __grid_constant__ CUtensorMap desc_data,
    const __grid_constant__ CUtensorMap desc_scale,
    uint32_t* __restrict__ sink) {

    extern __shared__ __align__(128) uint8_t smem_raw[];
    SmemLayout* smem = reinterpret_cast<SmemLayout*>(smem_raw);
    const int tid = threadIdx.x;

    if (tid == 0) {
        mbarrier_init(&smem->mbar, 1);
    }
    __syncthreads();

    uint32_t acc = 0u;
    uint32_t phase = 0u;

#pragma unroll 1
    for (int it = 0; it < iters; ++it) {
        if (tid == 0) {
            // Single arrive covers both TMA transactions (sum of bytes).
            mbarrier_arrive_expect_tx(&smem->mbar, kTotalBytes);
            // Two distinct cp.async.bulk.tensor.2d issues.
            cp_async_bulk_tensor_2d(&smem->u.sep.data,   &desc_data,  0, 0, &smem->mbar);
            cp_async_bulk_tensor_2d(&smem->u.sep.scales, &desc_scale, 0, 0, &smem->mbar);
        }
        // All threads wait on completion (single barrier).
        mbarrier_wait(&smem->mbar, phase);
        phase ^= 1u;

        // Sink: prevent compiler from eliminating the SMEM writes.
        if (tid == 0) {
            acc ^= *reinterpret_cast<uint32_t*>(&smem->u.sep.data[0]);
            acc ^= *reinterpret_cast<uint32_t*>(&smem->u.sep.scales[0]);
        }
    }

    if (tid == 0) {
        mbarrier_invalidate(&smem->mbar);
        if (blockIdx.x == 0) *sink = acc;
    }
}

// ---------------------------------------------------------------------------
// Fused variant: ONE TMA descriptor over a packed combined gmem tile (FP4 +
// UE4M3 packed contiguously). Single cp.async.bulk.tensor issue per iter.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(128) bench_fused(
    int iters,
    const __grid_constant__ CUtensorMap desc_combined,
    uint32_t* __restrict__ sink) {

    extern __shared__ __align__(128) uint8_t smem_raw[];
    SmemLayout* smem = reinterpret_cast<SmemLayout*>(smem_raw);
    const int tid = threadIdx.x;

    if (tid == 0) {
        mbarrier_init(&smem->mbar, 1);
    }
    __syncthreads();

    uint32_t acc = 0u;
    uint32_t phase = 0u;

#pragma unroll 1
    for (int it = 0; it < iters; ++it) {
        if (tid == 0) {
            // Single arrive: combined transaction byte count.
            mbarrier_arrive_expect_tx(&smem->mbar, kCombinedBytes);
            // Single cp.async.bulk.tensor.2d issue.
            cp_async_bulk_tensor_2d(&smem->u.combined, &desc_combined, 0, 0, &smem->mbar);
        }
        mbarrier_wait(&smem->mbar, phase);
        phase ^= 1u;

        if (tid == 0) {
            // Read both regions from the single combined tile (data first, scales after).
            acc ^= *reinterpret_cast<uint32_t*>(&smem->u.combined[0]);
            acc ^= *reinterpret_cast<uint32_t*>(&smem->u.combined[kDataBytes]);
        }
    }

    if (tid == 0) {
        mbarrier_invalidate(&smem->mbar);
        if (blockIdx.x == 0) *sink = acc;
    }
}

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------
static bool make_tma_2d_u8(CUtensorMap* desc, void* gmem, int rows, int cols, int row_stride_bytes) {
    // Build a 2-D TMA descriptor over a uint8 buffer:
    //   gmem shape:     [cols, rows]    (innermost = cols)
    //   gmem stride[0]: implicit 1 byte
    //   gmem stride[1]: row_stride_bytes
    //   smem box:       [cols, rows]
    cuuint64_t gmem_shape[2]  = { static_cast<cuuint64_t>(cols),
                                   static_cast<cuuint64_t>(rows) };
    // cuTensorMapEncodeTiled stride array starts at element [1]; dim0 is implicit 1.
    cuuint64_t gmem_stride[1] = { static_cast<cuuint64_t>(row_stride_bytes) };
    cuuint32_t smem_box[2]    = { static_cast<cuuint32_t>(cols),
                                   static_cast<cuuint32_t>(rows) };
    cuuint32_t smem_box_stride[2] = { 1u, 1u };

    // Resolve cuTensorMapEncodeTiled lazily so we don't need libcuda.so.1 at
    // load time (test-discovery in builder containers has no GPU driver).
    static PFN_cuTensorMapEncodeTiled_t pfn = nullptr;
    if (pfn == nullptr) {
        cudaDriverEntryPointQueryResult q;
        void* p = nullptr;
        cudaError_t err = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",
                                                   &p, cudaEnableDefault, &q);
        if (err != cudaSuccess || q != cudaDriverEntryPointSuccess || p == nullptr) {
            std::fprintf(stderr, "cudaGetDriverEntryPoint(cuTensorMapEncodeTiled) failed (q=%d)\n", (int)q);
            return false;
        }
        pfn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_t>(p);
    }

    CUresult r = pfn(
        desc,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        /*tensorRank=*/2,
        gmem,
        gmem_shape,
        gmem_stride,
        smem_box,
        smem_box_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) {
        std::fprintf(stderr, "cuTensorMapEncodeTiled failed: CUresult=%d (rows=%d cols=%d stride=%d)\n",
                     (int)r, rows, cols, row_stride_bytes);
        return false;
    }
    return true;
}

static double run_separate(int sms, int iters,
                           const CUtensorMap& desc_data, const CUtensorMap& desc_scale,
                           uint32_t* d_sink, cudaStream_t stream) {
    // Set max dynamic smem (default cap is 48 KiB; we need ~17 KiB but bump to be safe).
    int dyn_smem = static_cast<int>(sizeof(SmemLayout));
    cudaFuncSetAttribute(bench_separate,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         dyn_smem);

    // Warmup
    bench_separate<<<sms, 128, dyn_smem, stream>>>(iters / 8, desc_data, desc_scale, d_sink);
    if (cudaStreamSynchronize(stream) != cudaSuccess) return -1.0;
    if (cudaGetLastError() != cudaSuccess) return -2.0;

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    constexpr int kReps = 7;
    float total_ms = 0.0f;
    for (int r = 0; r < kReps; ++r) {
        cudaEventRecord(ev0, stream);
        bench_separate<<<sms, 128, dyn_smem, stream>>>(iters, desc_data, desc_scale, d_sink);
        cudaEventRecord(ev1, stream);
        cudaEventSynchronize(ev1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev0, ev1);
        total_ms += ms;
    }
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    return static_cast<double>(total_ms) / kReps;
}

static double run_fused(int sms, int iters, const CUtensorMap& desc_combined,
                        uint32_t* d_sink, cudaStream_t stream) {
    int dyn_smem = static_cast<int>(sizeof(SmemLayout));
    cudaFuncSetAttribute(bench_fused,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         dyn_smem);

    bench_fused<<<sms, 128, dyn_smem, stream>>>(iters / 8, desc_combined, d_sink);
    if (cudaStreamSynchronize(stream) != cudaSuccess) return -1.0;
    if (cudaGetLastError() != cudaSuccess) return -2.0;

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    constexpr int kReps = 7;
    float total_ms = 0.0f;
    for (int r = 0; r < kReps; ++r) {
        cudaEventRecord(ev0, stream);
        bench_fused<<<sms, 128, dyn_smem, stream>>>(iters, desc_combined, d_sink);
        cudaEventRecord(ev1, stream);
        cudaEventSynchronize(ev1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev0, ev1);
        total_ms += ms;
    }
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    return static_cast<double>(total_ms) / kReps;
}

// ---------------------------------------------------------------------------
// Public entry
// ---------------------------------------------------------------------------
TmaBlockScaleResult bench_tma_block_scale(int iters) {
    TmaBlockScaleResult r{0.0, 0.0, 0.0};

    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);
    const int sms = prop.multiProcessorCount;

    // Allocate gmem buffers — separate variant uses two distinct buffers, fused
    // variant uses one packed buffer of equal total size.
    uint8_t*  d_data     = nullptr;
    uint8_t*  d_scales   = nullptr;
    uint8_t*  d_combined = nullptr;
    uint32_t* d_sink     = nullptr;
    if (cudaMalloc(&d_data,     kDataBytes)        != cudaSuccess) return r;
    if (cudaMalloc(&d_scales,   kScaleBytes)       != cudaSuccess) { cudaFree(d_data); return r; }
    if (cudaMalloc(&d_combined, kCombinedBytes)    != cudaSuccess) { cudaFree(d_data); cudaFree(d_scales); return r; }
    if (cudaMalloc(&d_sink,     sizeof(uint32_t))  != cudaSuccess) {
        cudaFree(d_data); cudaFree(d_scales); cudaFree(d_combined); return r;
    }
    cudaMemset(d_data,     0xAB, kDataBytes);
    cudaMemset(d_scales,   0x38, kScaleBytes);    // UE4M3 ≈ 1.0 (all entries)
    cudaMemset(d_combined, 0xCD, kCombinedBytes);

    // Build TMA descriptors. Each tile is innermost-128-byte (rows of 128 B)
    // to satisfy the 16-byte TMA stride alignment.
    CUtensorMap desc_data{}, desc_scale{}, desc_combined{};
    if (!make_tma_2d_u8(&desc_data,     d_data,     kDataRows,     kRowBytes, kRowBytes)) goto cleanup;
    if (!make_tma_2d_u8(&desc_scale,    d_scales,   kScaleRows,    kRowBytes, kRowBytes)) goto cleanup;
    if (!make_tma_2d_u8(&desc_combined, d_combined, kCombinedRows, kRowBytes, kRowBytes)) goto cleanup;

    {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        double ms_sep  = run_separate(sms, iters, desc_data, desc_scale, d_sink, stream);
        double ms_fuse = run_fused(   sms, iters, desc_combined,         d_sink, stream);
        cudaStreamDestroy(stream);

        r.ms_separate  = ms_sep;
        r.ms_fused     = ms_fuse;
        r.bytes_loaded = static_cast<double>(sms) * iters * kTotalBytes;
    }

cleanup:
    cudaFree(d_data);
    cudaFree(d_scales);
    cudaFree(d_combined);
    cudaFree(d_sink);
    return r;
}

}  // namespace imp
