#include "quant/mxfp4_gemm.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace imp {

// ---------------------------------------------------------------------------
// MXFP4 K-parallel GEMV kernels (optimized)
//
// Same prmt register LUT as NVFP4 GEMV. Key difference: 32 elements per
// scale group (2 micro-blocks × 16 nibbles each share 1 UE8M0 scale).
// uint4 loads for full 16-byte group in one 128-bit transaction.
// ---------------------------------------------------------------------------

static constexpr int kMxGroupSize = 32;
static constexpr int kMxGroupBytes = 16;  // 32 nibbles = 16 bytes
static constexpr int kKparWarps = 4;
static constexpr int kKparThreads = kKparWarps * 32;  // 128
static constexpr int kMRWarps = 8;
static constexpr int kMRThreads = kMRWarps * 32;  // 256

// UE8M0 → float: pure exponent, no mantissa. Value = 2^(byte - 127).
__device__ __forceinline__ float ue8m0_to_float(uint8_t byte) {
    return __uint_as_float(static_cast<uint32_t>(byte) << 23);
}

// Cached SM count for multi-row dispatch.
static int mxfp4_n_sms() {
    static int n_sms = 0;
    if (__builtin_expect(n_sms == 0, 0)) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        n_sms = prop.multiProcessorCount;
    }
    return n_sms;
}

static bool use_multirow(int n_groups, int mr_blocks) {
    return n_groups <= 256 && mr_blocks >= 6 * mxfp4_n_sms();
}

// Process 8 packed bytes (16 nibbles = 16 FP4 values) via prmt register LUT.
// Returns UNSCALED dot product: sum(dequant(nibble) * activation).
// Identical to NVFP4 dot_micro_block — same E2M1 encoding, same prmt LUT.
__device__ __forceinline__ float dot_micro_block(const uint8_t* __restrict__ pb, const half* __restrict__ x,
                                                 int elem_base) {
    constexpr uint32_t kLutLo = 0x3E3C3800u;
    constexpr uint32_t kLutHi = 0x46444240u;

    float acc = 0.0f;
#pragma unroll
    for (int b = 0; b < 8; b++) {
        uint32_t byte_val = pb[b];
        const half2 xh = *reinterpret_cast<const half2*>(x + elem_base + b * 2);
        const float2 xf = __half22float2(xh);

        uint32_t lo_mag = byte_val & 0x07u;
        uint32_t lo_hi_byte;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lo_hi_byte) : "r"(kLutLo), "r"(kLutHi), "r"(lo_mag));
        uint32_t lo_fp16 = (lo_hi_byte & 0xFFu) << 8;
        lo_fp16 |= ((byte_val & 0x08u) << 12);
        float lo_val = __half2float(*reinterpret_cast<const half*>(&lo_fp16));
        acc = __fmaf_rn(lo_val, xf.x, acc);

        uint32_t hi_mag = (byte_val >> 4) & 0x07u;
        uint32_t hi_hi_byte;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hi_hi_byte) : "r"(kLutLo), "r"(kLutHi), "r"(hi_mag));
        uint32_t hi_fp16 = (hi_hi_byte & 0xFFu) << 8;
        hi_fp16 |= ((byte_val & 0x80u) << 8);
        float hi_val = __half2float(*reinterpret_cast<const half*>(&hi_fp16));
        acc = __fmaf_rn(hi_val, xf.y, acc);
    }
    return acc;
}

// Core row accumulation: iterate over MXFP4 scale groups.
// Each group = 16 packed bytes (32 FP4) + 1 UE8M0 scale.
// Uses uint4 (128-bit) vectorized data loads + 2 micro-blocks per group.
__device__ __forceinline__ float gemv_mxfp4_row(const uint8_t* __restrict__ row_packed,
                                                const uint8_t* __restrict__ row_scales,
                                                const half* __restrict__ x, int n_groups, int tid) {
    float acc = 0.0f;
    for (int gi = tid; gi < n_groups; gi += kKparThreads) {
        // Load 16 bytes as uint4 (128-bit coalesced)
        uint4 packed4 = *reinterpret_cast<const uint4*>(row_packed + gi * kMxGroupBytes);
        const uint8_t* pb = reinterpret_cast<const uint8_t*>(&packed4);
        float scale = ue8m0_to_float(row_scales[gi]);
        // Two micro-blocks of 8 bytes (16 nibbles) each, shared scale
        float dot0 = dot_micro_block(pb, x, gi * kMxGroupSize);
        float dot1 = dot_micro_block(pb + 8, x, gi * kMxGroupSize + 16);
        acc = __fmaf_rn(dot0 + dot1, scale, acc);
    }
    return acc;
}

// Warp-level row accumulation for multi-row kernels.
template <typename DotFn>
__device__ __forceinline__ float warp_k_loop(const uint8_t* __restrict__ row_packed,
                                             const uint8_t* __restrict__ row_scales, int n_groups, int lane,
                                             DotFn dot_fn) {
    float acc = 0.0f;
    for (int gi = lane; gi < n_groups; gi += 32) {
        uint4 packed4 = *reinterpret_cast<const uint4*>(row_packed + gi * kMxGroupBytes);
        const uint8_t* pb = reinterpret_cast<const uint8_t*>(&packed4);
        float scale = ue8m0_to_float(row_scales[gi]);
        acc = __fmaf_rn(dot_fn(pb, gi * kMxGroupSize), scale, acc);
    }
    return acc;
}

// Warp-level reduction: shuffle-down within 32 threads.
__device__ __forceinline__ float warp_reduce(float acc) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    return acc;
}

// K-parallel reduction: warp shuffle + cross-warp shared memory.
__device__ __forceinline__ float reduce_kpar(float acc, int tid, float* warp_sums) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    int warp_id = tid / 32;
    if ((tid & 31) == 0)
        warp_sums[warp_id] = acc;
    __syncthreads();
    if (tid == 0) {
        float total = warp_sums[0];
#pragma unroll
        for (int w = 1; w < kKparWarps; w++)
            total += warp_sums[w];
        return total;
    }
    return 0.0f;
}

// TU-local: same layout as the identically-named struct in
// nvfp4_gemm_internal.cuh; kept internal-linkage to avoid an ODR clash in the
// single core lib (both are namespace-scope in namespace imp).
namespace {
struct SmemKpar {
    float warp_sums[kKparWarps];
};
}  // namespace

// ---------------------------------------------------------------------------
// Basic GEMV: y[row] = W_mxfp4[row,:] @ x
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kKparThreads, 12) gemv_mxfp4_kpar_kernel(
    const uint8_t* __restrict__ packed_data, const uint8_t* __restrict__ linear_scales,
    const half* __restrict__ x, half* __restrict__ y, int N, int K) {
    const int row = blockIdx.x;
    if (row >= N)
        return;
    const int tid = threadIdx.x;
    const int n_groups = K / kMxGroupSize;
    __shared__ SmemKpar smem;
    float acc = gemv_mxfp4_row(packed_data + (int64_t)row * (K / 2), linear_scales + (int64_t)row * n_groups,
                               x, n_groups, tid);
    float total = reduce_kpar(acc, tid, smem.warp_sums);
    if (tid == 0)
        y[row] = __float2half(total);
}

__global__ void __launch_bounds__(kKparThreads, 12) gemv_mxfp4_kpar_fp32_kernel(
    const uint8_t* __restrict__ packed_data, const uint8_t* __restrict__ linear_scales,
    const half* __restrict__ x, float* __restrict__ y, int N, int K) {
    const int row = blockIdx.x;
    if (row >= N)
        return;
    const int tid = threadIdx.x;
    const int n_groups = K / kMxGroupSize;
    __shared__ SmemKpar smem;
    float acc = gemv_mxfp4_row(packed_data + (int64_t)row * (K / 2), linear_scales + (int64_t)row * n_groups,
                               x, n_groups, tid);
    float total = reduce_kpar(acc, tid, smem.warp_sums);
    if (tid == 0)
        y[row] = total;
}

// ---------------------------------------------------------------------------
// Multi-row GEMV: NR rows per block, 256 threads (8 warps).
// ---------------------------------------------------------------------------
template <int NR>
__global__ void __launch_bounds__(kMRThreads) gemv_mxfp4_multirow_kernel(
    const uint8_t* __restrict__ packed_data, const uint8_t* __restrict__ linear_scales,
    const half* __restrict__ x, half* __restrict__ y, int N, int K) {
    const int block_row_base = blockIdx.x * NR;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int K_half = K / 2;
    const int n_groups = K / kMxGroupSize;
    const int row = block_row_base + warp_id;
    if (row >= N || warp_id >= NR)
        return;

    float acc = warp_k_loop(packed_data + (int64_t)row * K_half, linear_scales + (int64_t)row * n_groups,
                            n_groups, lane, [&] __device__(const uint8_t* pb, int elem_base) {
                                return dot_micro_block(pb, x, elem_base) +
                                       dot_micro_block(pb + 8, x, elem_base + 16);
                            });
    acc = warp_reduce(acc);
    if (lane == 0)
        y[row] = __float2half(acc);
}

template <int NR>
__global__ void __launch_bounds__(kMRThreads) gemv_mxfp4_multirow_fp32_kernel(
    const uint8_t* __restrict__ packed_data, const uint8_t* __restrict__ linear_scales,
    const half* __restrict__ x, float* __restrict__ y, int N, int K) {
    const int block_row_base = blockIdx.x * NR;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int K_half = K / 2;
    const int n_groups = K / kMxGroupSize;
    const int row = block_row_base + warp_id;
    if (row >= N || warp_id >= NR)
        return;

    float acc = warp_k_loop(packed_data + (int64_t)row * K_half, linear_scales + (int64_t)row * n_groups,
                            n_groups, lane, [&] __device__(const uint8_t* pb, int elem_base) {
                                return dot_micro_block(pb, x, elem_base) +
                                       dot_micro_block(pb + 8, x, elem_base + 16);
                            });
    acc = warp_reduce(acc);
    if (lane == 0)
        y[row] = acc;
}

// ---------------------------------------------------------------------------
// GEMV with residual: y[row] = W[row,:] @ x + residual[row]
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kKparThreads, 12) gemv_mxfp4_residual_kernel(
    const uint8_t* __restrict__ packed_data, const uint8_t* __restrict__ linear_scales,
    const half* __restrict__ x, half* __restrict__ y, const half* __restrict__ residual, int N, int K) {
    const int row = blockIdx.x;
    if (row >= N)
        return;
    const int tid = threadIdx.x;
    const int n_groups = K / kMxGroupSize;
    __shared__ SmemKpar smem;
    float acc = gemv_mxfp4_row(packed_data + (int64_t)row * (K / 2), linear_scales + (int64_t)row * n_groups,
                               x, n_groups, tid);
    float total = reduce_kpar(acc, tid, smem.warp_sums);
    if (tid == 0)
        y[row] = __float2half(total + __half2float(residual[row]));
}

// ---------------------------------------------------------------------------
// Fused QKV: 3 weight matrices, shared x, separate outputs
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kKparThreads, 12) gemv_mxfp4_qkv_kernel(
    const uint8_t* __restrict__ q_packed, const uint8_t* __restrict__ q_scales,
    const uint8_t* __restrict__ k_packed, const uint8_t* __restrict__ k_scales,
    const uint8_t* __restrict__ v_packed, const uint8_t* __restrict__ v_scales, const half* __restrict__ x,
    half* __restrict__ yq, half* __restrict__ yk, half* __restrict__ yv, int q_rows, int k_rows, int v_rows,
    int K) {
    const int global_row = blockIdx.x;
    const int tid = threadIdx.x;
    const int K_half = K / 2;
    const int n_groups = K / kMxGroupSize;
    __shared__ SmemKpar smem;

    const uint8_t* packed;
    const uint8_t* scales;
    half* y_out;
    int row;
    if (global_row < q_rows) {
        row = global_row;
        packed = q_packed + (int64_t)row * K_half;
        scales = q_scales + (int64_t)row * n_groups;
        y_out = yq;
    } else if (global_row < q_rows + k_rows) {
        row = global_row - q_rows;
        packed = k_packed + (int64_t)row * K_half;
        scales = k_scales + (int64_t)row * n_groups;
        y_out = yk;
    } else if (global_row < q_rows + k_rows + v_rows) {
        row = global_row - q_rows - k_rows;
        packed = v_packed + (int64_t)row * K_half;
        scales = v_scales + (int64_t)row * n_groups;
        y_out = yv;
    } else {
        return;
    }

    float acc = gemv_mxfp4_row(packed, scales, x, n_groups, tid);
    float total = reduce_kpar(acc, tid, smem.warp_sums);
    if (tid == 0)
        y_out[row] = __float2half(total);
}

// ---------------------------------------------------------------------------
// Fused Gate+Up: 2 weight matrices, shared x, separate outputs
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kKparThreads, 12) gemv_mxfp4_gate_up_kernel(
    const uint8_t* __restrict__ g_packed, const uint8_t* __restrict__ g_scales,
    const uint8_t* __restrict__ u_packed, const uint8_t* __restrict__ u_scales, const half* __restrict__ x,
    half* __restrict__ yg, half* __restrict__ yu, int rows, int K) {
    const int global_row = blockIdx.x;
    const int tid = threadIdx.x;
    const int K_half = K / 2;
    const int n_groups = K / kMxGroupSize;
    __shared__ SmemKpar smem;

    const uint8_t* packed;
    const uint8_t* scales;
    half* y_out;
    int row;
    if (global_row < rows) {
        row = global_row;
        packed = g_packed + (int64_t)row * K_half;
        scales = g_scales + (int64_t)row * n_groups;
        y_out = yg;
    } else if (global_row < rows * 2) {
        row = global_row - rows;
        packed = u_packed + (int64_t)row * K_half;
        scales = u_scales + (int64_t)row * n_groups;
        y_out = yu;
    } else {
        return;
    }

    float acc = gemv_mxfp4_row(packed, scales, x, n_groups, tid);
    float total = reduce_kpar(acc, tid, smem.warp_sums);
    if (tid == 0)
        y_out[row] = __float2half(total);
}

// ---------------------------------------------------------------------------
// Fused SwiGLU + GEMV + residual
// ---------------------------------------------------------------------------
__device__ __forceinline__ float dot_micro_block_swiglu(const uint8_t* __restrict__ pb,
                                                        const half* __restrict__ gate,
                                                        const half* __restrict__ up, int elem_base,
                                                        const float* s_lut) {
    float acc = 0.0f;
#pragma unroll
    for (int b = 0; b < 8; b++) {
        const half2 gh = *reinterpret_cast<const half2*>(gate + elem_base + b * 2);
        const half2 uh = *reinterpret_cast<const half2*>(up + elem_base + b * 2);
        const float2 gf = __half22float2(gh);
        const float2 uf = __half22float2(uh);
        float s0 = gf.x / (1.0f + expf(-gf.x)) * uf.x;
        float s1 = gf.y / (1.0f + expf(-gf.y)) * uf.y;
        acc = __fmaf_rn(s_lut[pb[b] & 0x0F], s0, acc);
        acc = __fmaf_rn(s_lut[pb[b] >> 4], s1, acc);
    }
    return acc;
}

__device__ __forceinline__ void init_lut(float* s_lut, int tid) {
    if (tid < 16) {
        constexpr float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        s_lut[tid] = (tid < 8) ? kMag[tid] : -kMag[tid & 7];
    }
}

__global__ void __launch_bounds__(kKparThreads, 12) gemv_mxfp4_swiglu_residual_kernel(
    const uint8_t* __restrict__ packed_data, const uint8_t* __restrict__ linear_scales,
    const half* __restrict__ gate, const half* __restrict__ up, half* __restrict__ y,
    const half* __restrict__ residual, int N, int K) {
    const int row = blockIdx.x;
    if (row >= N)
        return;
    const int tid = threadIdx.x;
    const int K_half = K / 2;
    const int n_groups = K / kMxGroupSize;

    __shared__ float s_lut[16];
    __shared__ float warp_sums[kKparWarps];
    init_lut(s_lut, tid);
    __syncthreads();

    const uint8_t* row_packed = packed_data + (int64_t)row * K_half;
    const uint8_t* row_scales = linear_scales + (int64_t)row * n_groups;
    float acc = 0.0f;
    for (int gi = tid; gi < n_groups; gi += kKparThreads) {
        uint4 packed4 = *reinterpret_cast<const uint4*>(row_packed + gi * kMxGroupBytes);
        const uint8_t* pb = reinterpret_cast<const uint8_t*>(&packed4);
        float scale = ue8m0_to_float(row_scales[gi]);
        float dot0 = dot_micro_block_swiglu(pb, gate, up, gi * kMxGroupSize, s_lut);
        float dot1 = dot_micro_block_swiglu(pb + 8, gate, up, gi * kMxGroupSize + 16, s_lut);
        acc = __fmaf_rn(dot0 + dot1, scale, acc);
    }
    float total = reduce_kpar(acc, tid, warp_sums);
    if (tid == 0)
        y[row] = __float2half(total + __half2float(residual[row]));
}

// ---------------------------------------------------------------------------
// Fused GeGLU + GEMV + residual
// ---------------------------------------------------------------------------
__device__ __forceinline__ float dot_micro_block_geglu(const uint8_t* __restrict__ pb,
                                                       const half* __restrict__ gate,
                                                       const half* __restrict__ up, int elem_base,
                                                       const float* s_lut) {
    constexpr float SQRT_2_PI = 0.7978845608028654f;
    constexpr float COEFF = 0.044715f;
    float acc = 0.0f;
#pragma unroll
    for (int b = 0; b < 8; b++) {
        const half2 gh = *reinterpret_cast<const half2*>(gate + elem_base + b * 2);
        const half2 uh = *reinterpret_cast<const half2*>(up + elem_base + b * 2);
        const float2 gf = __half22float2(gh);
        const float2 uf = __half22float2(uh);
        float g0 = gf.x * 0.5f * (1.0f + tanhf(SQRT_2_PI * (gf.x + COEFF * gf.x * gf.x * gf.x)));
        float g1 = gf.y * 0.5f * (1.0f + tanhf(SQRT_2_PI * (gf.y + COEFF * gf.y * gf.y * gf.y)));
        acc = __fmaf_rn(s_lut[pb[b] & 0x0F], g0 * uf.x, acc);
        acc = __fmaf_rn(s_lut[pb[b] >> 4], g1 * uf.y, acc);
    }
    return acc;
}

__global__ void __launch_bounds__(kKparThreads, 12) gemv_mxfp4_geglu_residual_kernel(
    const uint8_t* __restrict__ packed_data, const uint8_t* __restrict__ linear_scales,
    const half* __restrict__ gate, const half* __restrict__ up, half* __restrict__ y,
    const half* __restrict__ residual, int N, int K) {
    const int row = blockIdx.x;
    if (row >= N)
        return;
    const int tid = threadIdx.x;
    const int K_half = K / 2;
    const int n_groups = K / kMxGroupSize;

    __shared__ float s_lut[16];
    __shared__ float warp_sums[kKparWarps];
    init_lut(s_lut, tid);
    __syncthreads();

    const uint8_t* row_packed = packed_data + (int64_t)row * K_half;
    const uint8_t* row_scales = linear_scales + (int64_t)row * n_groups;
    float acc = 0.0f;
    for (int gi = tid; gi < n_groups; gi += kKparThreads) {
        uint4 packed4 = *reinterpret_cast<const uint4*>(row_packed + gi * kMxGroupBytes);
        const uint8_t* pb = reinterpret_cast<const uint8_t*>(&packed4);
        float scale = ue8m0_to_float(row_scales[gi]);
        float dot0 = dot_micro_block_geglu(pb, gate, up, gi * kMxGroupSize, s_lut);
        float dot1 = dot_micro_block_geglu(pb + 8, gate, up, gi * kMxGroupSize + 16, s_lut);
        acc = __fmaf_rn(dot0 + dot1, scale, acc);
    }
    float total = reduce_kpar(acc, tid, warp_sums);
    if (tid == 0)
        y[row] = __float2half(total + __half2float(residual[row]));
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

void gemv_mxfp4_kpar(const CutlassMxFP4Weight& W, const half* x, half* y, int N, int K, cudaStream_t stream) {
    static int kpar_dbg = 0;
    if (++kpar_dbg <= 200 && (K != 5120 || kpar_dbg <= 3))
        fprintf(stderr, "[GEMV_DBG] N=%d K=%d W.N=%lld W.K=%lld data=%p scales=%p x=%p y=%p\n", N, K,
                (long long)W.N, (long long)W.K, W.data, W.linear_scales, x, y);
    int n_groups = K / kMxGroupSize;
    int mr_blocks = (N + kMRWarps - 1) / kMRWarps;
    if (use_multirow(n_groups, mr_blocks)) {
        gemv_mxfp4_multirow_kernel<kMRWarps>
            <<<mr_blocks, kMRThreads, 0, stream>>>(static_cast<const uint8_t*>(W.data),
                                                   static_cast<const uint8_t*>(W.linear_scales), x, y, N, K);
    } else {
        gemv_mxfp4_kpar_kernel<<<N, kKparThreads, 0, stream>>>(static_cast<const uint8_t*>(W.data),
                                                               static_cast<const uint8_t*>(W.linear_scales),
                                                               x, y, N, K);
    }
}

void gemv_mxfp4_kpar_fp32(const CutlassMxFP4Weight& W, const half* x, float* y, int N, int K,
                          cudaStream_t stream) {
    int n_groups = K / kMxGroupSize;
    int mr_blocks = (N + kMRWarps - 1) / kMRWarps;
    if (use_multirow(n_groups, mr_blocks)) {
        gemv_mxfp4_multirow_fp32_kernel<kMRWarps>
            <<<mr_blocks, kMRThreads, 0, stream>>>(static_cast<const uint8_t*>(W.data),
                                                   static_cast<const uint8_t*>(W.linear_scales), x, y, N, K);
    } else {
        gemv_mxfp4_kpar_fp32_kernel<<<N, kKparThreads, 0, stream>>>(
            static_cast<const uint8_t*>(W.data), static_cast<const uint8_t*>(W.linear_scales), x, y, N, K);
    }
}

void gemv_mxfp4_qkv_fused(const CutlassMxFP4Weight& wq, const CutlassMxFP4Weight& wk,
                          const CutlassMxFP4Weight& wv, const half* x, half* yq, half* yk, half* yv,
                          int q_rows, int k_rows, int v_rows, int K, cudaStream_t stream) {
    int total_rows = q_rows + k_rows + v_rows;
    gemv_mxfp4_qkv_kernel<<<total_rows, kKparThreads, 0, stream>>>(
        static_cast<const uint8_t*>(wq.data), static_cast<const uint8_t*>(wq.linear_scales),
        static_cast<const uint8_t*>(wk.data), static_cast<const uint8_t*>(wk.linear_scales),
        static_cast<const uint8_t*>(wv.data), static_cast<const uint8_t*>(wv.linear_scales), x, yq, yk, yv,
        q_rows, k_rows, v_rows, K);
}

void gemv_mxfp4_gate_up_fused(const CutlassMxFP4Weight& wg, const CutlassMxFP4Weight& wu, const half* x,
                              half* yg, half* yu, int rows, int K, cudaStream_t stream) {
    gemv_mxfp4_gate_up_kernel<<<rows * 2, kKparThreads, 0, stream>>>(
        static_cast<const uint8_t*>(wg.data), static_cast<const uint8_t*>(wg.linear_scales),
        static_cast<const uint8_t*>(wu.data), static_cast<const uint8_t*>(wu.linear_scales), x, yg, yu, rows,
        K);
}

void gemv_mxfp4_residual(const CutlassMxFP4Weight& W, const half* x, half* y, const half* residual, int N,
                         int K, cudaStream_t stream) {
    gemv_mxfp4_residual_kernel<<<N, kKparThreads, 0, stream>>>(static_cast<const uint8_t*>(W.data),
                                                               static_cast<const uint8_t*>(W.linear_scales),
                                                               x, y, residual, N, K);
}

void gemv_mxfp4_swiglu_residual(const CutlassMxFP4Weight& W, const half* gate, const half* up, half* y,
                                const half* residual, int N, int K, cudaStream_t stream) {
    gemv_mxfp4_swiglu_residual_kernel<<<N, kKparThreads, 0, stream>>>(static_cast<const uint8_t*>(W.data),
                                                                      static_cast<const uint8_t*>(
                                                                          W.linear_scales),
                                                                      gate, up, y, residual, N, K);
}

void gemv_mxfp4_geglu_residual(const CutlassMxFP4Weight& W, const half* gate, const half* up, half* y,
                               const half* residual, int N, int K, cudaStream_t stream) {
    gemv_mxfp4_geglu_residual_kernel<<<N, kKparThreads, 0, stream>>>(static_cast<const uint8_t*>(W.data),
                                                                     static_cast<const uint8_t*>(
                                                                         W.linear_scales),
                                                                     gate, up, y, residual, N, K);
}

// One-time L1 cache carveout for MXFP4 GEMV kernels (bandwidth-bound, no SMEM).
void mxfp4_gemv_set_l1_carveout() {
#define MX_L1(kern) \
    cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1)
    MX_L1(gemv_mxfp4_kpar_kernel);
    MX_L1(gemv_mxfp4_kpar_fp32_kernel);
    MX_L1(gemv_mxfp4_multirow_kernel<kMRWarps>);
    MX_L1(gemv_mxfp4_multirow_fp32_kernel<kMRWarps>);
    MX_L1(gemv_mxfp4_qkv_kernel);
    MX_L1(gemv_mxfp4_gate_up_kernel);
    MX_L1(gemv_mxfp4_residual_kernel);
    MX_L1(gemv_mxfp4_swiglu_residual_kernel);
    MX_L1(gemv_mxfp4_geglu_residual_kernel);
#undef MX_L1
}

}  // namespace imp
