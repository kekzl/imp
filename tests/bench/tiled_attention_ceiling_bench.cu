// =============================================================================
// tiled_attention_ceiling_bench.cu — Säule 3 implementation
// =============================================================================
//
// Decomposes the FA2 inner-loop pipeline into three independently-measurable
// stages, then computes the lower-bound for what a perfectly-overlapped tiled
// streaming attention kernel could achieve on sm_120a:
//
//   stage A: K+V tile cp.async load     (memory bandwidth bound)
//   stage B: QKᵀ + PV mma.sync.m16n8k16 (FP16 tensor core throughput)
//   stage C: online softmax              (warp reduce + exp + rescale)
//
// Ceiling tile_ns = max(A_ns, B_ns, C_ns)  if perfectly overlapped (FA4 ideal)
// Ceiling tile_ns = A_ns + B_ns + C_ns      if fully serial (lower bound)
//
// The realistic FA2 implementation lies between these. Compare against actual
// FMHA tile time (from Säule 2: total_ms / (n_heads × ceil(seq/Bq) × ceil(seq/Bkv)/2))
// to see how much headroom remains.
//
// Fixed geometry: Br=64, Bkv=64, HD=128 FP16. Per inner iter:
//   K+V load: 32 KB
//   QKᵀ:      256 × mma.sync.m16n8k16 (4 row-tiles × 8 col-tiles × 8 k-iters)
//   PV:       256 × mma.sync.m16n8k16 (4 row-tiles × 16 col-tiles × 4 k-iters)
//   Softmax:  4 row-tiles, each 16×64 → row-reduce + exp + rescale
// =============================================================================

#include "bench/tiled_attention_ceiling_bench.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdio>

namespace imp {

namespace {

constexpr int kBr = 64;
constexpr int kBkv = 64;
constexpr int kHD = 128;
constexpr int kThreads = 128;       // 4 warps
constexpr int kMmaPerIterQKt = 256;
constexpr int kMmaPerIterPV = 256;
constexpr int kFlopsPerMma = 16 * 8 * 16 * 2;  // m × n × k × (mul+add)
constexpr int kBytesPerTile = (kBkv * kHD + kBkv * kHD) * sizeof(__half);  // K + V

// -----------------------------------------------------------------------------
// Stage A: K+V cp.async tile load
//
// 4 warps cooperatively load a 64×128 K tile then a 64×128 V tile per iter.
// One CTA per SM (170 on RTX 5090) saturates the memory subsystem.
// -----------------------------------------------------------------------------

__device__ __forceinline__ void cp_async_16(void* smem, const void* glob) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(glob));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n");
}

__global__ void __launch_bounds__(kThreads) stage_a_kv_load_kernel(
    int iters, const __half* __restrict__ K_src,
    const __half* __restrict__ V_src, uint32_t* __restrict__ sink) {
    extern __shared__ __align__(128) uint8_t smem_raw[];
    __half* K_sm = reinterpret_cast<__half*>(smem_raw);
    __half* V_sm = K_sm + kBkv * kHD;

    const int tid = threadIdx.x;
    constexpr int kHalvesPerChunk = 8;  // 16 bytes
    constexpr int kKvChunks = (kBkv * kHD) / kHalvesPerChunk;

    uint32_t acc = 0u;
#pragma unroll 1
    for (int it = 0; it < iters; ++it) {
        for (int c = tid; c < kKvChunks; c += kThreads) {
            int elem = c * kHalvesPerChunk;
            int r = elem / kHD;
            int d = elem % kHD;
            cp_async_16(&K_sm[r * kHD + d], &K_src[r * kHD + d]);
        }
        for (int c = tid; c < kKvChunks; c += kThreads) {
            int elem = c * kHalvesPerChunk;
            int r = elem / kHD;
            int d = elem % kHD;
            cp_async_16(&V_sm[r * kHD + d], &V_src[r * kHD + d]);
        }
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();
        if (tid == 0) acc ^= *reinterpret_cast<uint32_t*>(&K_sm[0]);
    }
    if (tid == 0 && blockIdx.x == 0) *sink = acc;
}

// -----------------------------------------------------------------------------
// Stage B: FP16 m16n8k16 mma.sync peak throughput
//
// Each warp issues (kMmaPerIterQKt + kMmaPerIterPV) / 4 = 128 mmas per iter
// with accumulator-dependency chain. Per CTA: 512 mmas/iter × 4096 FLOPs each
// = 2.1 MFLOPs/iter/CTA. Across 170 SMs: ~356 MFLOPs/iter.
// -----------------------------------------------------------------------------

__global__ void __launch_bounds__(kThreads, 1) stage_b_mma_kernel(
    int iters, float* __restrict__ sink) {
    uint32_t a0 = threadIdx.x * 37u + 1u;
    uint32_t a1 = threadIdx.x * 41u + 2u;
    uint32_t a2 = threadIdx.x * 43u + 3u;
    uint32_t a3 = threadIdx.x * 47u + 4u;
    uint32_t b0 = threadIdx.x * 53u + 5u;
    uint32_t b1 = threadIdx.x * 59u + 6u;
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    constexpr int kMmaPerWarpPerIter =
        (kMmaPerIterQKt + kMmaPerIterPV) / 4;  // 4 warps

#if __CUDA_ARCH__ >= 800
#pragma unroll 1
    for (int it = 0; it < iters; ++it) {
#pragma unroll 1
        for (int m = 0; m < kMmaPerWarpPerIter; ++m) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3},"
                "{%4, %5, %6, %7},"
                "{%8, %9},"
                "{%10, %11, %12, %13};\n"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "f"(d0), "f"(d1), "f"(d2), "f"(d3));
        }
    }
#endif
    if (threadIdx.x == 0 && blockIdx.x == 0) sink[0] = d0 + d1 + d2 + d3;
}

// -----------------------------------------------------------------------------
// Stage C: online softmax per inner iter
//
// 4 row-tiles of 16 rows each. Per row: read 64 floats from regs (simulated
// as smem here), compute row-max via warp shuffle reduce, subtract+exp,
// row-sum reduce, rescale running O accumulator (head_dim=128 floats).
// -----------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_max(float x) {
    for (int off = 16; off > 0; off >>= 1)
        x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, off));
    return x;
}

__device__ __forceinline__ float warp_reduce_sum(float x) {
    for (int off = 16; off > 0; off >>= 1)
        x += __shfl_xor_sync(0xffffffffu, x, off);
    return x;
}

__global__ void __launch_bounds__(kThreads) stage_c_softmax_kernel(
    int iters, float* __restrict__ sink) {
    extern __shared__ __align__(128) uint8_t smem_raw[];
    float* S_sm = reinterpret_cast<float*>(smem_raw);   // [Br × Bkv]
    float* O_sm = S_sm + kBr * kBkv;                    // [Br × HD]
    float* m_sm = O_sm + kBr * kHD;                     // [Br]
    float* l_sm = m_sm + kBr;                            // [Br]

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid & 31;

    // One-time init
    if (tid < kBr) {
        m_sm[tid] = -INFINITY;
        l_sm[tid] = 0.0f;
    }
    for (int i = tid; i < kBr * kHD; i += kThreads) O_sm[i] = 0.0f;
    for (int i = tid; i < kBr * kBkv; i += kThreads) S_sm[i] = static_cast<float>(i & 31) * 0.01f;
    __syncthreads();

    float acc = 0.f;
#pragma unroll 1
    for (int it = 0; it < iters; ++it) {
        // Each warp owns 16 rows. Per warp: 16 rows × 64 cols.
        // 32 lanes × 2 cols each covers 64 cols, all 16 rows looped.
        for (int row_in_warp = 0; row_in_warp < 16; ++row_in_warp) {
            int row = warp_id * 16 + row_in_warp;
            // Two cols per lane.
            float s0 = S_sm[row * kBkv + lane * 2];
            float s1 = S_sm[row * kBkv + lane * 2 + 1];
            float r_max = fmaxf(s0, s1);
            r_max = warp_reduce_max(r_max);
            float prev_m = m_sm[row];
            float new_m = fmaxf(prev_m, r_max);
            float scale_prev = __expf(prev_m - new_m);
            float p0 = __expf(s0 - new_m);
            float p1 = __expf(s1 - new_m);
            float r_sum = warp_reduce_sum(p0 + p1);
            float new_l = scale_prev * l_sm[row] + r_sum;
            if (lane == 0) {
                m_sm[row] = new_m;
                l_sm[row] = new_l;
            }
            // Rescale O row: 128 cols, 32 lanes × 4 cols each.
            for (int c = 0; c < 4; ++c) {
                int col = lane * 4 + c;
                O_sm[row * kHD + col] *= scale_prev;
            }
            acc += new_l;
        }
        __syncthreads();
    }
    if (tid == 0 && blockIdx.x == 0) sink[0] = acc;
}

// -----------------------------------------------------------------------------
// Common timing helper
// -----------------------------------------------------------------------------

template <typename Launcher>
double time_kernel_ms(Launcher launcher, cudaStream_t stream) {
    constexpr int kReps = 5;
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    // Warmup
    launcher();
    cudaStreamSynchronize(stream);
    double total = 0.0;
    for (int r = 0; r < kReps; ++r) {
        cudaEventRecord(a, stream);
        launcher();
        cudaEventRecord(b, stream);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        total += ms;
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return total / kReps;
}

}  // namespace

bool tiled_attention_ceiling_bench(TiledAttnCeilingResult* out) {
    if (!out) return false;
    *out = {};

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int sms = prop.multiProcessorCount;

    // Source K/V tile in gmem (shared across all CTAs; one tile, reused).
    __half* d_K = nullptr;
    __half* d_V = nullptr;
    uint32_t* d_sink_a = nullptr;
    float* d_sink_b = nullptr;
    float* d_sink_c = nullptr;

    auto bail = [&](const char* why) -> bool {
        if (d_K) cudaFree(d_K);
        if (d_V) cudaFree(d_V);
        if (d_sink_a) cudaFree(d_sink_a);
        if (d_sink_b) cudaFree(d_sink_b);
        if (d_sink_c) cudaFree(d_sink_c);
        std::fprintf(stderr, "tiled_attention_ceiling_bench: %s\n", why);
        return false;
    };

    const size_t tile_bytes = kBkv * kHD * sizeof(__half);
    if (cudaMalloc(&d_K, tile_bytes) != cudaSuccess) return bail("alloc K");
    if (cudaMalloc(&d_V, tile_bytes) != cudaSuccess) return bail("alloc V");
    if (cudaMalloc(&d_sink_a, sizeof(uint32_t)) != cudaSuccess) return bail("alloc sink A");
    if (cudaMalloc(&d_sink_b, sizeof(float)) != cudaSuccess) return bail("alloc sink B");
    if (cudaMalloc(&d_sink_c, sizeof(float)) != cudaSuccess) return bail("alloc sink C");
    cudaMemset(d_K, 0x42, tile_bytes);
    cudaMemset(d_V, 0x33, tile_bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // ---- Stage A ----
    const int smem_a = static_cast<int>(2 * tile_bytes);  // K + V in smem
    cudaFuncSetAttribute(stage_a_kv_load_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_a);
    constexpr int kItersA = 8192;
    double ms_a = time_kernel_ms([&]() {
        stage_a_kv_load_kernel<<<sms, kThreads, smem_a, stream>>>(
            kItersA, d_K, d_V, d_sink_a);
    }, stream);
    if (cudaGetLastError() != cudaSuccess) return bail("stage A launch");

    // ---- Stage B ----
    // No smem needed; use 4 warps per CTA × 170 SMs.
    constexpr int kItersB = 4096;
    double ms_b = time_kernel_ms([&]() {
        stage_b_mma_kernel<<<sms, kThreads, 0, stream>>>(kItersB, d_sink_b);
    }, stream);
    if (cudaGetLastError() != cudaSuccess) return bail("stage B launch");

    // ---- Stage C ----
    const int smem_c =
        static_cast<int>((kBr * kBkv + kBr * kHD + 2 * kBr) * sizeof(float));
    cudaFuncSetAttribute(stage_c_softmax_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_c);
    constexpr int kItersC = 4096;
    double ms_c = time_kernel_ms([&]() {
        stage_c_softmax_kernel<<<sms, kThreads, smem_c, stream>>>(
            kItersC, d_sink_c);
    }, stream);
    if (cudaGetLastError() != cudaSuccess) return bail("stage C launch");

    cudaStreamDestroy(stream);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_sink_a);
    cudaFree(d_sink_b);
    cudaFree(d_sink_c);

    // ---- Derive per-tile times ----
    // sms CTAs × kIters tiles each. Per-tile time is per-CTA per-iter average.
    const double tile_ns_a = (ms_a * 1.0e6) / kItersA;            // per CTA per iter
    const double tile_ns_b = (ms_b * 1.0e6) / kItersB;
    const double tile_ns_c = (ms_c * 1.0e6) / kItersC;

    // Pipeline ceiling: serial sum (lower bound, no overlap)
    const double tile_ns_serial = tile_ns_a + tile_ns_b + tile_ns_c;
    // Pipeline ceiling: max (upper bound, perfect overlap)
    const double tile_ns_overlap = fmax(tile_ns_a, fmax(tile_ns_b, tile_ns_c));

    // Effective TFLOPS based on MMA stage alone (best case).
    const double flops_per_tile =
        static_cast<double>(kMmaPerIterQKt + kMmaPerIterPV) * kFlopsPerMma;
    out->tile_ns = tile_ns_overlap;
    out->effective_tflops = (flops_per_tile / tile_ns_overlap);  // FLOPs/ns = TFLOPS
    out->kv_bandwidth_gb_per_s = (kBytesPerTile / tile_ns_a);    // bytes/ns = GB/s

    std::printf(
        "TILED_CEILING_BENCH | Br=%d Bkv=%d HD=%d (FP16)\n"
        "  Stage A cp.async K+V :  %7.2f ns/tile  (%.0f GB/s, %d bytes/tile)\n"
        "  Stage B mma.sync 512 :  %7.2f ns/tile  (%.0f TFLOPS)\n"
        "  Stage C softmax+resc :  %7.2f ns/tile\n"
        "  Pipeline (serial sum):  %7.2f ns/tile  ← lower-bound ceiling\n"
        "  Pipeline (max overlap): %7.2f ns/tile  ← upper-bound ceiling\n",
        kBr, kBkv, kHD,
        tile_ns_a, kBytesPerTile / tile_ns_a, kBytesPerTile,
        tile_ns_b, flops_per_tile / tile_ns_b,
        tile_ns_c,
        tile_ns_serial, tile_ns_overlap);
    std::fflush(stdout);
    return true;
}

}  // namespace imp
