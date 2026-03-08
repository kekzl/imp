#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_quant.h"
#include "compute/gemm.h"
#include "core/tensor.h"
#include "core/logging.h"
#include "runtime/pdl.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include <cstdint>
#include <cassert>

namespace imp {

// ---------------------------------------------------------------------------
// Optimized K-parallel NVFP4 GEMV kernels (v2)
//
// Architecture: 128 threads (4 warps), 1 row per block, M blocks.
// Each iteration processes one micro-block (uint2 = 8 packed bytes = 16 FP4).
//
// Key optimizations vs v1:
//   1. Signed 16-entry LUT in shared memory (branchless, no cmem serial)
//   2. uint2 loads = 1 micro-block boundary (1 scale per load, no waste)
//   3. Deferred scale: accumulate LUT*activation, scale once per block
//   4. half2 vectorized activation loads (halves load instruction count)
//   5. Bitwise FP8->FP32 decode (no exp2f SFU call)
// ---------------------------------------------------------------------------

static constexpr int kMicroBlockSize = 16;
static constexpr int kKparWarps = 4;
static constexpr int kKparThreads = kKparWarps * 32;  // 128

// Multi-row kernel: 8 warps, NR rows per block for small-K models.
// Reduces launch overhead and improves SM occupancy.
static constexpr int kMRWarps = 8;
static constexpr int kMRThreads = kMRWarps * 32;  // 256

// Fast FP8 E4M3 -> FP32 via bit manipulation (no exp2f).
__device__ __forceinline__ float fp8_e4m3_to_float_fast(uint8_t bits)
{
    uint32_t sign = (bits >> 7) & 1;
    uint32_t exp  = (bits >> 3) & 0x0F;
    uint32_t man  = bits & 0x07;

    if (exp == 0) {
        // Denorm: value = man * 2^(-9)
        float val = (float)man * (1.0f / 512.0f);
        return sign ? -val : val;
    }
    // Normal: FP8 bias=7, FP32 bias=127 -> exp offset=120.
    uint32_t fp32 = (sign << 31) | ((exp + 120u) << 23) | (man << 20);
    return __uint_as_float(fp32);
}

// Process one micro-block (8 packed bytes = 16 FP4 values).
// Returns unscaled dot product: sum(LUT[nibble] * activation).
// Caller multiplies by combined_scale once.
__device__ __forceinline__ float dot_micro_block(
    const uint8_t* __restrict__ pb,
    const half*    __restrict__ x,
    int            elem_base,
    const float*   s_lut)
{
    float acc = 0.0f;
    #pragma unroll
    for (int b = 0; b < 8; b++) {
        const half2 xh = *reinterpret_cast<const half2*>(x + elem_base + b * 2);
        const float2 xf = __half22float2(xh);
        acc = __fmaf_rn(s_lut[pb[b] & 0x0F], xf.x, acc);
        acc = __fmaf_rn(s_lut[pb[b] >> 4],   xf.y, acc);
    }
    return acc;
}

// Core GEMV loop: accumulates dot(row, x) for one row.
__device__ __forceinline__ float gemv_nvfp4_row(
    const uint8_t* __restrict__ row_packed,
    const uint8_t* __restrict__ row_ms,
    float          tensor_scale,
    const half*    __restrict__ x,
    int            n_mb,
    int            tid,
    const float*   s_lut)
{
    float acc = 0.0f;
    for (int mi = tid; mi < n_mb; mi += kKparThreads) {
        int byte_off = mi * 8;
        uint2 packed2 = *reinterpret_cast<const uint2*>(row_packed + byte_off);
        const uint8_t* pb = reinterpret_cast<const uint8_t*>(&packed2);
        float cs = tensor_scale * fp8_e4m3_to_float_fast(row_ms[mi]);
        float local_dot = dot_micro_block(pb, x, byte_off * 2, s_lut);
        acc = __fmaf_rn(local_dot, cs, acc);
    }
    return acc;
}

// K-parallel reduction: warp shuffle + cross-warp shared memory.
__device__ __forceinline__ float reduce_kpar(float acc, int tid, float* warp_sums)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);

    int warp_id = tid / 32;
    if ((tid & 31) == 0) warp_sums[warp_id] = acc;
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

// Shared memory layout: 16 floats for signed FP4 LUT + 4 floats for warp sums.
// Total: 80 bytes per block.
struct SmemKpar {
    float lut[16];
    float warp_sums[kKparWarps];
};

__device__ __forceinline__ void init_lut(float* s_lut, int tid)
{
    if (tid < 16) {
        constexpr float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        s_lut[tid] = (tid < 8) ? kMag[tid] : -kMag[tid & 7];
    }
}

// ---------------------------------------------------------------------------
// Basic GEMV: y[row] = A_nvfp4[row,:] @ x
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kKparThreads, 12)
gemv_nvfp4_kpar_kernel(
    const uint8_t* __restrict__ packed_data,
    const uint8_t* __restrict__ micro_scales,
    float                       tensor_scale,
    const half*    __restrict__ x,
    half*          __restrict__ y,
    int M, int K)
{
    const int row = blockIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;
    const int K_half = K / 2;
    const int n_mb = K / kMicroBlockSize;

    __shared__ SmemKpar smem;
    init_lut(smem.lut, tid);
    __syncthreads();

    float acc = gemv_nvfp4_row(
        packed_data + (int64_t)row * K_half,
        micro_scales + (int64_t)row * n_mb,
        tensor_scale, x, n_mb, tid, smem.lut);

    float total = reduce_kpar(acc, tid, smem.warp_sums);
    if (tid == 0) y[row] = __float2half(total);
}

// ---------------------------------------------------------------------------
// Multi-row GEMV: NR rows per block, 256 threads (8 warps).
// Each warp handles one row, multiple warps process multiple rows in parallel.
// Amortizes block launch overhead and improves occupancy for small K.
// ---------------------------------------------------------------------------
template<int NR>
__global__ void __launch_bounds__(kMRThreads, 6)
gemv_nvfp4_multirow_kernel(
    const uint8_t* __restrict__ packed_data,
    const uint8_t* __restrict__ micro_scales,
    float                       tensor_scale,
    const half*    __restrict__ x,
    half*          __restrict__ y,
    int M, int K)
{
    const int block_row_base = blockIdx.x * NR;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int K_half = K / 2;
    const int n_mb = K / kMicroBlockSize;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) {
        constexpr float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        s_lut[threadIdx.x] = (threadIdx.x < 8) ? kMag[threadIdx.x] : -kMag[threadIdx.x & 7];
    }
    __syncthreads();

    // Each warp handles one row within the NR-row tile
    const int row = block_row_base + warp_id;
    if (row >= M || warp_id >= NR) return;

    const uint8_t* row_packed = packed_data + (int64_t)row * K_half;
    const uint8_t* row_ms = micro_scales + (int64_t)row * n_mb;

    // K-parallel within warp (32 threads)
    float acc = 0.0f;
    for (int mi = lane; mi < n_mb; mi += 32) {
        int byte_off = mi * 8;
        uint2 packed2 = *reinterpret_cast<const uint2*>(row_packed + byte_off);
        const uint8_t* pb = reinterpret_cast<const uint8_t*>(&packed2);
        float cs = tensor_scale * fp8_e4m3_to_float_fast(row_ms[mi]);
        float local_dot = dot_micro_block(pb, x, byte_off * 2, s_lut);
        acc = __fmaf_rn(local_dot, cs, acc);
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);

    if (lane == 0) y[row] = __float2half(acc);
}

// ---------------------------------------------------------------------------
// GEMV with residual: y[row] = A_nvfp4[row,:] @ x + residual[row]
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kKparThreads, 12)
gemv_nvfp4_residual_kernel(
    const uint8_t* __restrict__ packed_data,
    const uint8_t* __restrict__ micro_scales,
    float                       tensor_scale,
    const half*    __restrict__ x,
    half*          __restrict__ y,
    const half*    __restrict__ residual,
    int M, int K)
{
    const int row = blockIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;
    const int K_half = K / 2;
    const int n_mb = K / kMicroBlockSize;

    __shared__ SmemKpar smem;
    init_lut(smem.lut, tid);
    __syncthreads();

    float acc = gemv_nvfp4_row(
        packed_data + (int64_t)row * K_half,
        micro_scales + (int64_t)row * n_mb,
        tensor_scale, x, n_mb, tid, smem.lut);

    float total = reduce_kpar(acc, tid, smem.warp_sums);
    if (tid == 0) y[row] = __float2half(total + __half2float(residual[row]));
}

// ---------------------------------------------------------------------------
// Fused SwiGLU + GEMV + residual:
//   y[row] = A_nvfp4[row,:] @ swiglu(gate, up) + residual[row]
// Eliminates the separate SwiGLU kernel launch.
// ---------------------------------------------------------------------------

// SwiGLU micro-block: reads gate[16] and up[16], computes silu(gate)*up,
// then dots with weight nibbles.
__device__ __forceinline__ float dot_micro_block_swiglu(
    const uint8_t* __restrict__ pb,
    const half*    __restrict__ gate,
    const half*    __restrict__ up,
    int            elem_base,
    const float*   s_lut)
{
    float acc = 0.0f;
    #pragma unroll
    for (int b = 0; b < 8; b++) {
        const half2 gh = *reinterpret_cast<const half2*>(gate + elem_base + b * 2);
        const half2 uh = *reinterpret_cast<const half2*>(up   + elem_base + b * 2);
        const float2 gf = __half22float2(gh);
        const float2 uf = __half22float2(uh);
        // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        float s0 = gf.x / (1.0f + expf(-gf.x)) * uf.x;
        float s1 = gf.y / (1.0f + expf(-gf.y)) * uf.y;
        acc = __fmaf_rn(s_lut[pb[b] & 0x0F], s0, acc);
        acc = __fmaf_rn(s_lut[pb[b] >> 4],   s1, acc);
    }
    return acc;
}

// GeGLU micro-block: reads gate[16] and up[16], computes gelu_tanh(gate)*up.
__device__ __forceinline__ float dot_micro_block_geglu(
    const uint8_t* __restrict__ pb,
    const half*    __restrict__ gate,
    const half*    __restrict__ up,
    int            elem_base,
    const float*   s_lut)
{
    constexpr float SQRT_2_PI = 0.7978845608028654f;
    constexpr float COEFF = 0.044715f;
    float acc = 0.0f;
    #pragma unroll
    for (int b = 0; b < 8; b++) {
        const half2 gh = *reinterpret_cast<const half2*>(gate + elem_base + b * 2);
        const half2 uh = *reinterpret_cast<const half2*>(up   + elem_base + b * 2);
        const float2 gf = __half22float2(gh);
        const float2 uf = __half22float2(uh);
        float g0 = gf.x * 0.5f * (1.0f + tanhf(SQRT_2_PI * (gf.x + COEFF * gf.x * gf.x * gf.x)));
        float g1 = gf.y * 0.5f * (1.0f + tanhf(SQRT_2_PI * (gf.y + COEFF * gf.y * gf.y * gf.y)));
        acc = __fmaf_rn(s_lut[pb[b] & 0x0F], g0 * uf.x, acc);
        acc = __fmaf_rn(s_lut[pb[b] >> 4],   g1 * uf.y, acc);
    }
    return acc;
}

__device__ __forceinline__ float gemv_nvfp4_row_swiglu(
    const uint8_t* __restrict__ row_packed,
    const uint8_t* __restrict__ row_ms,
    float          tensor_scale,
    const half*    __restrict__ gate,
    const half*    __restrict__ up,
    int            n_mb,
    int            tid,
    const float*   s_lut)
{
    float acc = 0.0f;
    for (int mi = tid; mi < n_mb; mi += kKparThreads) {
        int byte_off = mi * 8;
        uint2 packed2 = *reinterpret_cast<const uint2*>(row_packed + byte_off);
        const uint8_t* pb = reinterpret_cast<const uint8_t*>(&packed2);
        float cs = tensor_scale * fp8_e4m3_to_float_fast(row_ms[mi]);
        float local_dot = dot_micro_block_swiglu(pb, gate, up, byte_off * 2, s_lut);
        acc = __fmaf_rn(local_dot, cs, acc);
    }
    return acc;
}

__global__ void __launch_bounds__(kKparThreads, 12)
gemv_nvfp4_swiglu_residual_kernel(
    const uint8_t* __restrict__ packed_data,
    const uint8_t* __restrict__ micro_scales,
    float                       tensor_scale,
    const half*    __restrict__ gate,
    const half*    __restrict__ up,
    half*          __restrict__ y,
    const half*    __restrict__ residual,
    int M, int K)
{
    const int row = blockIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;
    const int K_half = K / 2;
    const int n_mb = K / kMicroBlockSize;

    __shared__ SmemKpar smem;
    init_lut(smem.lut, tid);
    __syncthreads();

    float acc = gemv_nvfp4_row_swiglu(
        packed_data + (int64_t)row * K_half,
        micro_scales + (int64_t)row * n_mb,
        tensor_scale, gate, up, n_mb, tid, smem.lut);

    float total = reduce_kpar(acc, tid, smem.warp_sums);
    if (tid == 0) y[row] = __float2half(total + __half2float(residual[row]));
}

// ---------------------------------------------------------------------------
// Fused GeGLU + GEMV + residual (for Gemma-3 and similar)
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kKparThreads, 12)
gemv_nvfp4_geglu_residual_kernel(
    const uint8_t* __restrict__ packed_data,
    const uint8_t* __restrict__ micro_scales,
    float                       tensor_scale,
    const half*    __restrict__ gate,
    const half*    __restrict__ up,
    half*          __restrict__ y,
    const half*    __restrict__ residual,
    int M, int K)
{
    const int row = blockIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;
    const int K_half = K / 2;
    const int n_mb = K / kMicroBlockSize;

    __shared__ SmemKpar smem;
    init_lut(smem.lut, tid);
    __syncthreads();

    const uint8_t* row_packed = packed_data + (int64_t)row * K_half;
    const uint8_t* row_ms = micro_scales + (int64_t)row * n_mb;

    float acc = 0.0f;
    for (int mi = tid; mi < n_mb; mi += kKparThreads) {
        int byte_off = mi * 8;
        uint2 packed2 = *reinterpret_cast<const uint2*>(row_packed + byte_off);
        const uint8_t* pb = reinterpret_cast<const uint8_t*>(&packed2);
        float cs = tensor_scale * fp8_e4m3_to_float_fast(row_ms[mi]);
        float local_dot = dot_micro_block_geglu(pb, gate, up, byte_off * 2, smem.lut);
        acc = __fmaf_rn(local_dot, cs, acc);
    }

    float total = reduce_kpar(acc, tid, smem.warp_sums);
    if (tid == 0) y[row] = __float2half(total + __half2float(residual[row]));
}

// ---------------------------------------------------------------------------
// Fused QKV: 3 weight matrices, shared input, separate outputs
// Grid: (q_rows + k_rows + v_rows) blocks.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kKparThreads, 12)
gemv_nvfp4_qkv_fused_kernel(
    const uint8_t* __restrict__ packed_q,
    const uint8_t* __restrict__ ms_q,
    float ts_q,
    const uint8_t* __restrict__ packed_k,
    const uint8_t* __restrict__ ms_k,
    float ts_k,
    const uint8_t* __restrict__ packed_v,
    const uint8_t* __restrict__ ms_v,
    float ts_v,
    const half*    __restrict__ x,
    half*          __restrict__ yq,
    half*          __restrict__ yk,
    half*          __restrict__ yv,
    int q_rows, int k_rows, int v_rows, int K)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int K_half = K / 2;
    const int n_mb = K / kMicroBlockSize;

    __shared__ SmemKpar smem;
    init_lut(smem.lut, tid);
    __syncthreads();

    const uint8_t* row_packed;
    const uint8_t* row_ms;
    float ts;
    half* out;
    int local_row;

    if (bid < q_rows) {
        local_row = bid;
        row_packed = packed_q + (int64_t)local_row * K_half;
        row_ms     = ms_q + (int64_t)local_row * n_mb;
        ts = ts_q;
        out = yq;
    } else if (bid < q_rows + k_rows) {
        local_row = bid - q_rows;
        row_packed = packed_k + (int64_t)local_row * K_half;
        row_ms     = ms_k + (int64_t)local_row * n_mb;
        ts = ts_k;
        out = yk;
    } else {
        local_row = bid - q_rows - k_rows;
        row_packed = packed_v + (int64_t)local_row * K_half;
        row_ms     = ms_v + (int64_t)local_row * n_mb;
        ts = ts_v;
        out = yv;
    }

    float acc = gemv_nvfp4_row(row_packed, row_ms, ts, x, n_mb, tid, smem.lut);
    float total = reduce_kpar(acc, tid, smem.warp_sums);
    if (tid == 0) out[local_row] = __float2half(total);
}

// ---------------------------------------------------------------------------
// Fused Gate+Up: 2 weight matrices, shared input, separate outputs
// Grid: 2 * rows blocks. First half = gate, second half = up.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kKparThreads, 12)
gemv_nvfp4_gate_up_fused_kernel(
    const uint8_t* __restrict__ packed_g,
    const uint8_t* __restrict__ ms_g,
    float ts_g,
    const uint8_t* __restrict__ packed_u,
    const uint8_t* __restrict__ ms_u,
    float ts_u,
    const half*    __restrict__ x,
    half*          __restrict__ yg,
    half*          __restrict__ yu,
    int rows, int K)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int K_half = K / 2;
    const int n_mb = K / kMicroBlockSize;

    __shared__ SmemKpar smem;
    init_lut(smem.lut, tid);
    __syncthreads();

    const uint8_t* row_packed;
    const uint8_t* row_ms;
    float ts;
    half* out;
    int local_row;

    if (bid < rows) {
        local_row = bid;
        row_packed = packed_g + (int64_t)local_row * K_half;
        row_ms     = ms_g + (int64_t)local_row * n_mb;
        ts = ts_g;
        out = yg;
    } else {
        local_row = bid - rows;
        row_packed = packed_u + (int64_t)local_row * K_half;
        row_ms     = ms_u + (int64_t)local_row * n_mb;
        ts = ts_u;
        out = yu;
    }

    float acc = gemv_nvfp4_row(row_packed, row_ms, ts, x, n_mb, tid, smem.lut);
    float total = reduce_kpar(acc, tid, smem.warp_sums);
    if (tid == 0) out[local_row] = __float2half(total);
}

// ---------------------------------------------------------------------------
// Multi-row variants of fused kernels (NR rows/block, 256 threads, 8 warps).
// Used when K is small (n_mb ≤ 512) to reduce block count and improve
// per-thread work (32 threads/row vs 128 threads/row).
// ---------------------------------------------------------------------------

// Multi-row QKV fused: each warp determines its matrix and row independently.
template<int NR>
__global__ void __launch_bounds__(kMRThreads, 6)
gemv_nvfp4_qkv_fused_mr_kernel(
    const uint8_t* __restrict__ packed_q, const uint8_t* __restrict__ ms_q, float ts_q,
    const uint8_t* __restrict__ packed_k, const uint8_t* __restrict__ ms_k, float ts_k,
    const uint8_t* __restrict__ packed_v, const uint8_t* __restrict__ ms_v, float ts_v,
    const half* __restrict__ x,
    half* __restrict__ yq, half* __restrict__ yk, half* __restrict__ yv,
    int q_rows, int k_rows, int v_rows, int K)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int global_row = blockIdx.x * NR + warp_id;
    const int total_rows = q_rows + k_rows + v_rows;
    if (global_row >= total_rows || warp_id >= NR) return;

    const int K_half = K / 2;
    const int n_mb = K / kMicroBlockSize;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) {
        constexpr float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        s_lut[threadIdx.x] = (threadIdx.x < 8) ? kMag[threadIdx.x] : -kMag[threadIdx.x & 7];
    }
    __syncthreads();

    const uint8_t* row_packed;
    const uint8_t* row_ms;
    float ts;
    half* out;
    int local_row;

    if (global_row < q_rows) {
        local_row = global_row;
        row_packed = packed_q + (int64_t)local_row * K_half;
        row_ms = ms_q + (int64_t)local_row * n_mb;
        ts = ts_q; out = yq;
    } else if (global_row < q_rows + k_rows) {
        local_row = global_row - q_rows;
        row_packed = packed_k + (int64_t)local_row * K_half;
        row_ms = ms_k + (int64_t)local_row * n_mb;
        ts = ts_k; out = yk;
    } else {
        local_row = global_row - q_rows - k_rows;
        row_packed = packed_v + (int64_t)local_row * K_half;
        row_ms = ms_v + (int64_t)local_row * n_mb;
        ts = ts_v; out = yv;
    }

    float acc = 0.0f;
    for (int mi = lane; mi < n_mb; mi += 32) {
        int byte_off = mi * 8;
        uint2 packed2 = *reinterpret_cast<const uint2*>(row_packed + byte_off);
        const uint8_t* pb = reinterpret_cast<const uint8_t*>(&packed2);
        float cs = ts * fp8_e4m3_to_float_fast(row_ms[mi]);
        float local_dot = dot_micro_block(pb, x, byte_off * 2, s_lut);
        acc = __fmaf_rn(local_dot, cs, acc);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    if (lane == 0) out[local_row] = __float2half(acc);
}

// Multi-row gate+up fused.
template<int NR>
__global__ void __launch_bounds__(kMRThreads, 6)
gemv_nvfp4_gate_up_fused_mr_kernel(
    const uint8_t* __restrict__ packed_g, const uint8_t* __restrict__ ms_g, float ts_g,
    const uint8_t* __restrict__ packed_u, const uint8_t* __restrict__ ms_u, float ts_u,
    const half* __restrict__ x,
    half* __restrict__ yg, half* __restrict__ yu,
    int rows, int K)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int global_row = blockIdx.x * NR + warp_id;
    const int total_rows = 2 * rows;
    if (global_row >= total_rows || warp_id >= NR) return;

    const int K_half = K / 2;
    const int n_mb = K / kMicroBlockSize;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) {
        constexpr float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        s_lut[threadIdx.x] = (threadIdx.x < 8) ? kMag[threadIdx.x] : -kMag[threadIdx.x & 7];
    }
    __syncthreads();

    const uint8_t* row_packed;
    const uint8_t* row_ms;
    float ts;
    half* out;
    int local_row;

    if (global_row < rows) {
        local_row = global_row;
        row_packed = packed_g + (int64_t)local_row * K_half;
        row_ms = ms_g + (int64_t)local_row * n_mb;
        ts = ts_g; out = yg;
    } else {
        local_row = global_row - rows;
        row_packed = packed_u + (int64_t)local_row * K_half;
        row_ms = ms_u + (int64_t)local_row * n_mb;
        ts = ts_u; out = yu;
    }

    float acc = 0.0f;
    for (int mi = lane; mi < n_mb; mi += 32) {
        int byte_off = mi * 8;
        uint2 packed2 = *reinterpret_cast<const uint2*>(row_packed + byte_off);
        const uint8_t* pb = reinterpret_cast<const uint8_t*>(&packed2);
        float cs = ts * fp8_e4m3_to_float_fast(row_ms[mi]);
        float local_dot = dot_micro_block(pb, x, byte_off * 2, s_lut);
        acc = __fmaf_rn(local_dot, cs, acc);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    if (lane == 0) out[local_row] = __float2half(acc);
}

// Multi-row residual.
template<int NR>
__global__ void __launch_bounds__(kMRThreads, 6)
gemv_nvfp4_residual_mr_kernel(
    const uint8_t* __restrict__ packed_data, const uint8_t* __restrict__ micro_scales,
    float tensor_scale, const half* __restrict__ x,
    half* __restrict__ y, const half* __restrict__ residual,
    int M, int K)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * NR + warp_id;
    if (row >= M || warp_id >= NR) return;

    const int K_half = K / 2;
    const int n_mb = K / kMicroBlockSize;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) {
        constexpr float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        s_lut[threadIdx.x] = (threadIdx.x < 8) ? kMag[threadIdx.x] : -kMag[threadIdx.x & 7];
    }
    __syncthreads();

    const uint8_t* row_packed = packed_data + (int64_t)row * K_half;
    const uint8_t* row_ms = micro_scales + (int64_t)row * n_mb;

    float acc = 0.0f;
    for (int mi = lane; mi < n_mb; mi += 32) {
        int byte_off = mi * 8;
        uint2 packed2 = *reinterpret_cast<const uint2*>(row_packed + byte_off);
        const uint8_t* pb = reinterpret_cast<const uint8_t*>(&packed2);
        float cs = tensor_scale * fp8_e4m3_to_float_fast(row_ms[mi]);
        float local_dot = dot_micro_block(pb, x, byte_off * 2, s_lut);
        acc = __fmaf_rn(local_dot, cs, acc);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    if (lane == 0) y[row] = __float2half(acc + __half2float(residual[row]));
}

// Multi-row SwiGLU + residual.
template<int NR>
__global__ void __launch_bounds__(kMRThreads, 6)
gemv_nvfp4_swiglu_residual_mr_kernel(
    const uint8_t* __restrict__ packed_data, const uint8_t* __restrict__ micro_scales,
    float tensor_scale,
    const half* __restrict__ gate, const half* __restrict__ up,
    half* __restrict__ y, const half* __restrict__ residual,
    int M, int K)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * NR + warp_id;
    if (row >= M || warp_id >= NR) return;

    const int K_half = K / 2;
    const int n_mb = K / kMicroBlockSize;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) {
        constexpr float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        s_lut[threadIdx.x] = (threadIdx.x < 8) ? kMag[threadIdx.x] : -kMag[threadIdx.x & 7];
    }
    __syncthreads();

    const uint8_t* row_packed = packed_data + (int64_t)row * K_half;
    const uint8_t* row_ms = micro_scales + (int64_t)row * n_mb;

    float acc = 0.0f;
    for (int mi = lane; mi < n_mb; mi += 32) {
        int byte_off = mi * 8;
        uint2 packed2 = *reinterpret_cast<const uint2*>(row_packed + byte_off);
        const uint8_t* pb = reinterpret_cast<const uint8_t*>(&packed2);
        float cs = tensor_scale * fp8_e4m3_to_float_fast(row_ms[mi]);
        float local_dot = dot_micro_block_swiglu(pb, gate, up, byte_off * 2, s_lut);
        acc = __fmaf_rn(local_dot, cs, acc);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    if (lane == 0) y[row] = __float2half(acc + __half2float(residual[row]));
}

// Multi-row GeGLU + residual.
template<int NR>
__global__ void __launch_bounds__(kMRThreads, 6)
gemv_nvfp4_geglu_residual_mr_kernel(
    const uint8_t* __restrict__ packed_data, const uint8_t* __restrict__ micro_scales,
    float tensor_scale,
    const half* __restrict__ gate, const half* __restrict__ up,
    half* __restrict__ y, const half* __restrict__ residual,
    int M, int K)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * NR + warp_id;
    if (row >= M || warp_id >= NR) return;

    const int K_half = K / 2;
    const int n_mb = K / kMicroBlockSize;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) {
        constexpr float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        s_lut[threadIdx.x] = (threadIdx.x < 8) ? kMag[threadIdx.x] : -kMag[threadIdx.x & 7];
    }
    __syncthreads();

    const uint8_t* row_packed = packed_data + (int64_t)row * K_half;
    const uint8_t* row_ms = micro_scales + (int64_t)row * n_mb;

    float acc = 0.0f;
    for (int mi = lane; mi < n_mb; mi += 32) {
        int byte_off = mi * 8;
        uint2 packed2 = *reinterpret_cast<const uint2*>(row_packed + byte_off);
        const uint8_t* pb = reinterpret_cast<const uint8_t*>(&packed2);
        float cs = tensor_scale * fp8_e4m3_to_float_fast(row_ms[mi]);
        float local_dot = dot_micro_block_geglu(pb, gate, up, byte_off * 2, s_lut);
        acc = __fmaf_rn(local_dot, cs, acc);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    if (lane == 0) y[row] = __float2half(acc + __half2float(residual[row]));
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

void gemv_nvfp4_kpar(const NvFP4QuantResult& A, const half* x, half* y,
                     int M, int K, cudaStream_t stream)
{
    const int n_mb = K / kMicroBlockSize;
    // Use multi-row kernel when K is small relative to thread count.
    // With 128 threads doing K-parallel, each thread only gets n_mb/128 iterations.
    // When that's < 4, the kernel is launch-overhead-dominated → switch to multi-row.
    if (n_mb <= 512) {
        // NR=8: each warp (32 threads) handles one row, 8 rows per block.
        // Each thread gets n_mb/32 iterations — much better work per thread.
        constexpr int NR = 8;
        int grid = (M + NR - 1) / NR;
        pdl::launch(gemv_nvfp4_multirow_kernel<NR>,
            dim3(grid), dim3(kMRThreads), size_t(0), stream,
            reinterpret_cast<const uint8_t*>(A.packed_data),
            reinterpret_cast<const uint8_t*>(A.micro_scales),
            A.tensor_scale, x, y, M, K);
    } else {
        pdl::launch(gemv_nvfp4_kpar_kernel,
            dim3(M), dim3(kKparThreads), size_t(0), stream,
            reinterpret_cast<const uint8_t*>(A.packed_data),
            reinterpret_cast<const uint8_t*>(A.micro_scales),
            A.tensor_scale, x, y, M, K);
    }
}

void gemv_nvfp4_qkv_fused(const NvFP4QuantResult& wq, const NvFP4QuantResult& wk,
                           const NvFP4QuantResult& wv, const half* x,
                           half* yq, half* yk, half* yv,
                           int q_rows, int k_rows, int v_rows, int K,
                           cudaStream_t stream)
{
    int total_rows = q_rows + k_rows + v_rows;
    const int n_mb = K / kMicroBlockSize;
    if (n_mb <= 512) {
        constexpr int NR = 8;
        int grid = (total_rows + NR - 1) / NR;
        pdl::launch(gemv_nvfp4_qkv_fused_mr_kernel<NR>,
            dim3(grid), dim3(kMRThreads), size_t(0), stream,
            reinterpret_cast<const uint8_t*>(wq.packed_data),
            reinterpret_cast<const uint8_t*>(wq.micro_scales), wq.tensor_scale,
            reinterpret_cast<const uint8_t*>(wk.packed_data),
            reinterpret_cast<const uint8_t*>(wk.micro_scales), wk.tensor_scale,
            reinterpret_cast<const uint8_t*>(wv.packed_data),
            reinterpret_cast<const uint8_t*>(wv.micro_scales), wv.tensor_scale,
            x, yq, yk, yv, q_rows, k_rows, v_rows, K);
    } else {
        pdl::launch(gemv_nvfp4_qkv_fused_kernel,
            dim3(total_rows), dim3(kKparThreads), size_t(0), stream,
            reinterpret_cast<const uint8_t*>(wq.packed_data),
            reinterpret_cast<const uint8_t*>(wq.micro_scales), wq.tensor_scale,
            reinterpret_cast<const uint8_t*>(wk.packed_data),
            reinterpret_cast<const uint8_t*>(wk.micro_scales), wk.tensor_scale,
            reinterpret_cast<const uint8_t*>(wv.packed_data),
            reinterpret_cast<const uint8_t*>(wv.micro_scales), wv.tensor_scale,
            x, yq, yk, yv, q_rows, k_rows, v_rows, K);
    }
}

void gemv_nvfp4_gate_up_fused(const NvFP4QuantResult& wg, const NvFP4QuantResult& wu,
                               const half* x, half* yg, half* yu,
                               int rows, int K, cudaStream_t stream)
{
    int total_rows = 2 * rows;
    const int n_mb = K / kMicroBlockSize;
    if (n_mb <= 512) {
        constexpr int NR = 8;
        int grid = (total_rows + NR - 1) / NR;
        pdl::launch(gemv_nvfp4_gate_up_fused_mr_kernel<NR>,
            dim3(grid), dim3(kMRThreads), size_t(0), stream,
            reinterpret_cast<const uint8_t*>(wg.packed_data),
            reinterpret_cast<const uint8_t*>(wg.micro_scales), wg.tensor_scale,
            reinterpret_cast<const uint8_t*>(wu.packed_data),
            reinterpret_cast<const uint8_t*>(wu.micro_scales), wu.tensor_scale,
            x, yg, yu, rows, K);
    } else {
        pdl::launch(gemv_nvfp4_gate_up_fused_kernel,
            dim3(total_rows), dim3(kKparThreads), size_t(0), stream,
            reinterpret_cast<const uint8_t*>(wg.packed_data),
            reinterpret_cast<const uint8_t*>(wg.micro_scales), wg.tensor_scale,
            reinterpret_cast<const uint8_t*>(wu.packed_data),
            reinterpret_cast<const uint8_t*>(wu.micro_scales), wu.tensor_scale,
            x, yg, yu, rows, K);
    }
}

void gemv_nvfp4_residual(const NvFP4QuantResult& A, const half* x, half* y,
                          const half* residual, int M, int K, cudaStream_t stream)
{
    const int n_mb = K / kMicroBlockSize;
    if (n_mb <= 512) {
        constexpr int NR = 8;
        int grid = (M + NR - 1) / NR;
        pdl::launch(gemv_nvfp4_residual_mr_kernel<NR>,
            dim3(grid), dim3(kMRThreads), size_t(0), stream,
            reinterpret_cast<const uint8_t*>(A.packed_data),
            reinterpret_cast<const uint8_t*>(A.micro_scales),
            A.tensor_scale, x, y, residual, M, K);
    } else {
        pdl::launch(gemv_nvfp4_residual_kernel,
            dim3(M), dim3(kKparThreads), size_t(0), stream,
            reinterpret_cast<const uint8_t*>(A.packed_data),
            reinterpret_cast<const uint8_t*>(A.micro_scales),
            A.tensor_scale, x, y, residual, M, K);
    }
}

void gemv_nvfp4_swiglu_residual(const NvFP4QuantResult& A,
                                 const half* gate, const half* up,
                                 half* y, const half* residual,
                                 int M, int K, cudaStream_t stream)
{
    const int n_mb = K / kMicroBlockSize;
    if (n_mb <= 512) {
        constexpr int NR = 8;
        int grid = (M + NR - 1) / NR;
        pdl::launch(gemv_nvfp4_swiglu_residual_mr_kernel<NR>,
            dim3(grid), dim3(kMRThreads), size_t(0), stream,
            reinterpret_cast<const uint8_t*>(A.packed_data),
            reinterpret_cast<const uint8_t*>(A.micro_scales),
            A.tensor_scale, gate, up, y, residual, M, K);
    } else {
        pdl::launch(gemv_nvfp4_swiglu_residual_kernel,
            dim3(M), dim3(kKparThreads), size_t(0), stream,
            reinterpret_cast<const uint8_t*>(A.packed_data),
            reinterpret_cast<const uint8_t*>(A.micro_scales),
            A.tensor_scale, gate, up, y, residual, M, K);
    }
}

void gemv_nvfp4_geglu_residual(const NvFP4QuantResult& A,
                                const half* gate, const half* up,
                                half* y, const half* residual,
                                int M, int K, cudaStream_t stream)
{
    const int n_mb = K / kMicroBlockSize;
    if (n_mb <= 512) {
        constexpr int NR = 8;
        int grid = (M + NR - 1) / NR;
        pdl::launch(gemv_nvfp4_geglu_residual_mr_kernel<NR>,
            dim3(grid), dim3(kMRThreads), size_t(0), stream,
            reinterpret_cast<const uint8_t*>(A.packed_data),
            reinterpret_cast<const uint8_t*>(A.micro_scales),
            A.tensor_scale, gate, up, y, residual, M, K);
    } else {
        pdl::launch(gemv_nvfp4_geglu_residual_kernel,
            dim3(M), dim3(kKparThreads), size_t(0), stream,
            reinterpret_cast<const uint8_t*>(A.packed_data),
            reinterpret_cast<const uint8_t*>(A.micro_scales),
            A.tensor_scale, gate, up, y, residual, M, K);
    }
}

// ---------------------------------------------------------------------------
// MoE NVFP4 GEMV: per-expert decode projections.
// Grid: top_k * rows blocks, 128 threads.  Each block computes one row of
// one expert's output.  FP16 input (no Q8_1 pre-quantization).
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kKparThreads, 12)
gemv_nvfp4_moe_decode_kernel(
    const uint8_t* __restrict__ packed_data,
    const uint8_t* __restrict__ micro_scales,
    const float*   __restrict__ tensor_scales,
    const int32_t* __restrict__ expert_indices,
    const half*    __restrict__ x,
    half*          __restrict__ y,
    int rows, int K,
    size_t expert_stride_packed,
    size_t expert_stride_ms,
    int x_stride,
    int blocks_per_expert)
{
    const int expert_slot = blockIdx.x / blocks_per_expert;
    const int row = blockIdx.x % blocks_per_expert;
    if (row >= rows) return;

    const int tid = threadIdx.x;
    const int n_mb = K / kMicroBlockSize;
    const int expert_id = expert_indices[expert_slot];

    const uint8_t* W = packed_data + (size_t)expert_id * expert_stride_packed;
    const uint8_t* MS = micro_scales + (size_t)expert_id * expert_stride_ms;
    float ts = tensor_scales[expert_id];
    const half* xi = x + (size_t)expert_slot * x_stride;

    __shared__ SmemKpar smem;
    init_lut(smem.lut, tid);
    __syncthreads();

    float acc = gemv_nvfp4_row(
        W + (size_t)row * (K / 2),
        MS + (size_t)row * n_mb,
        ts, xi, n_mb, tid, smem.lut);

    float total = reduce_kpar(acc, tid, smem.warp_sums);
    if (tid == 0) y[(size_t)expert_slot * rows + row] = __float2half(total);
}

// ---------------------------------------------------------------------------
// Fused gate+up MoE NVFP4 GEMV.
// Grid: dim3(top_k * rows, 2).  blockIdx.y=0 → gate, blockIdx.y=1 → up.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kKparThreads, 12)
gemv_nvfp4_moe_gate_up_fused_kernel(
    const uint8_t* __restrict__ gate_packed,
    const uint8_t* __restrict__ gate_ms,
    const float*   __restrict__ gate_ts,
    const uint8_t* __restrict__ up_packed,
    const uint8_t* __restrict__ up_ms,
    const float*   __restrict__ up_ts,
    const int32_t* __restrict__ expert_indices,
    const half*    __restrict__ x,
    half*          __restrict__ y_gate,
    half*          __restrict__ y_up,
    int rows, int K,
    size_t expert_stride_packed,
    size_t expert_stride_ms,
    int blocks_per_expert)
{
    const int expert_slot = blockIdx.x / blocks_per_expert;
    const int row = blockIdx.x % blocks_per_expert;
    if (row >= rows) return;

    const int tid = threadIdx.x;
    const int n_mb = K / kMicroBlockSize;
    const int expert_id = expert_indices[expert_slot];
    const bool is_up = (blockIdx.y == 1);

    const uint8_t* W = is_up
        ? (up_packed + (size_t)expert_id * expert_stride_packed)
        : (gate_packed + (size_t)expert_id * expert_stride_packed);
    const uint8_t* MS = is_up
        ? (up_ms + (size_t)expert_id * expert_stride_ms)
        : (gate_ms + (size_t)expert_id * expert_stride_ms);
    float ts = is_up ? up_ts[expert_id] : gate_ts[expert_id];
    half* out = is_up ? y_up : y_gate;

    // x_stride = 0 (shared input for gate+up)
    const half* xi = x;

    __shared__ SmemKpar smem;
    init_lut(smem.lut, tid);
    __syncthreads();

    float acc = gemv_nvfp4_row(
        W + (size_t)row * (K / 2),
        MS + (size_t)row * n_mb,
        ts, xi, n_mb, tid, smem.lut);

    float total = reduce_kpar(acc, tid, smem.warp_sums);
    if (tid == 0) out[(size_t)expert_slot * rows + row] = __float2half(total);
}

// ---------------------------------------------------------------------------
// MoE NVFP4 host launchers
// ---------------------------------------------------------------------------

void gemv_nvfp4_moe_decode(const NvFP4MoEQuantResult& w,
                            const int32_t* expert_indices, const half* x, half* y,
                            int rows, int K, int x_stride, int top_k,
                            cudaStream_t stream)
{
    int total_blocks = top_k * rows;
    pdl::launch(gemv_nvfp4_moe_decode_kernel,
        dim3(total_blocks), dim3(kKparThreads), size_t(0), stream,
        reinterpret_cast<const uint8_t*>(w.packed_data),
        reinterpret_cast<const uint8_t*>(w.micro_scales),
        w.tensor_scales,
        expert_indices, x, y,
        rows, K,
        w.expert_stride_packed,
        w.expert_stride_ms,
        x_stride, rows);
}

void gemv_nvfp4_moe_gate_up_fused(const NvFP4MoEQuantResult& gate,
                                    const NvFP4MoEQuantResult& up,
                                    const int32_t* expert_indices, const half* x,
                                    half* y_gate, half* y_up,
                                    int rows, int K, int top_k,
                                    cudaStream_t stream)
{
    dim3 grid(top_k * rows, 2);
    pdl::launch(gemv_nvfp4_moe_gate_up_fused_kernel,
        grid, dim3(kKparThreads), size_t(0), stream,
        reinterpret_cast<const uint8_t*>(gate.packed_data),
        reinterpret_cast<const uint8_t*>(gate.micro_scales),
        gate.tensor_scales,
        reinterpret_cast<const uint8_t*>(up.packed_data),
        reinterpret_cast<const uint8_t*>(up.micro_scales),
        up.tensor_scales,
        expert_indices, x,
        y_gate, y_up,
        rows, K,
        gate.expert_stride_packed,
        gate.expert_stride_ms,
        rows);
}

// ---------------------------------------------------------------------------
// Fused SwiGLU + MoE NVFP4 GEMV for down projection.
// Eliminates separate swiglu() kernel launch.
// Grid: top_k * rows blocks, 128 threads.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kKparThreads, 12)
gemv_nvfp4_moe_swiglu_decode_kernel(
    const uint8_t* __restrict__ packed_data,
    const uint8_t* __restrict__ micro_scales,
    const float*   __restrict__ tensor_scales,
    const int32_t* __restrict__ expert_indices,
    const half*    __restrict__ gate,
    const half*    __restrict__ up,
    half*          __restrict__ y,
    int rows, int K,
    size_t expert_stride_packed,
    size_t expert_stride_ms,
    int x_stride,
    int blocks_per_expert)
{
    const int expert_slot = blockIdx.x / blocks_per_expert;
    const int row = blockIdx.x % blocks_per_expert;
    if (row >= rows) return;

    const int tid = threadIdx.x;
    const int n_mb = K / kMicroBlockSize;
    const int expert_id = expert_indices[expert_slot];

    const uint8_t* W = packed_data + (size_t)expert_id * expert_stride_packed;
    const uint8_t* MS = micro_scales + (size_t)expert_id * expert_stride_ms;
    float ts = tensor_scales[expert_id];
    const half* gi = gate + (size_t)expert_slot * x_stride;
    const half* ui = up   + (size_t)expert_slot * x_stride;

    __shared__ SmemKpar smem;
    init_lut(smem.lut, tid);
    __syncthreads();

    float acc = gemv_nvfp4_row_swiglu(
        W + (size_t)row * (K / 2),
        MS + (size_t)row * n_mb,
        ts, gi, ui, n_mb, tid, smem.lut);

    float total = reduce_kpar(acc, tid, smem.warp_sums);
    if (tid == 0) y[(size_t)expert_slot * rows + row] = __float2half(total);
}

void gemv_nvfp4_moe_swiglu_decode(const NvFP4MoEQuantResult& w,
                                    const int32_t* expert_indices,
                                    const half* gate, const half* up, half* y,
                                    int rows, int K, int x_stride, int top_k,
                                    cudaStream_t stream)
{
    int total_blocks = top_k * rows;
    pdl::launch(gemv_nvfp4_moe_swiglu_decode_kernel,
        dim3(total_blocks), dim3(kKparThreads), size_t(0), stream,
        reinterpret_cast<const uint8_t*>(w.packed_data),
        reinterpret_cast<const uint8_t*>(w.micro_scales),
        w.tensor_scales,
        expert_indices, gate, up, y,
        rows, K,
        w.expert_stride_packed,
        w.expert_stride_ms,
        x_stride, rows);
}

// ---------------------------------------------------------------------------
// Tensor-based launcher (existing API, delegates to K-parallel kernel)
// ---------------------------------------------------------------------------
void gemv_nvfp4(const NvFP4QuantResult& A, const Tensor& x, Tensor& y,
                cudaStream_t stream)
{
    assert(A.packed_data != nullptr && "A must be quantized");
    assert(x.on_device && "x must be on device");
    assert(y.on_device && "y must be on device");
    assert(x.dtype == DType::FP16 && "x must be FP16");
    assert(y.dtype == DType::FP16 && "y must be FP16");

    int M = static_cast<int>(A.N);
    int K = static_cast<int>(A.K);

    gemv_nvfp4_kpar(A,
                    reinterpret_cast<const half*>(x.data),
                    reinterpret_cast<half*>(y.data),
                    M, K, stream);
}

// ---------------------------------------------------------------------------
// GEMM for NVFP4 weights:  C = input @ A^T
//
//   A (NvFP4QuantResult): weight matrix [N, K] in NVFP4 packed format
//   input (Tensor):       activation     [M, K] in FP16
//   C (Tensor):           output         [M, N] in FP16
//
// Strategy: dequantize A to FP16, then call cuBLAS gemm.
// ---------------------------------------------------------------------------

static void* s_nvfp4_dequant_buf = nullptr;
static size_t s_nvfp4_dequant_buf_size = 0;

static void* ensure_dequant_buffer(size_t needed) {
    if (needed <= s_nvfp4_dequant_buf_size) return s_nvfp4_dequant_buf;
    if (s_nvfp4_dequant_buf) cudaFree(s_nvfp4_dequant_buf);
    s_nvfp4_dequant_buf = nullptr;
    s_nvfp4_dequant_buf_size = 0;
    cudaError_t err = cudaMalloc(&s_nvfp4_dequant_buf, needed);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("gemm_nvfp4: failed to allocate %zu bytes for dequant buffer: %s",
                      needed, cudaGetErrorString(err));
        return nullptr;
    }
    s_nvfp4_dequant_buf_size = needed;
    return s_nvfp4_dequant_buf;
}

void gemm_nvfp4(const NvFP4QuantResult& A, const Tensor& B, Tensor& C,
                cudaStream_t stream)
{
    assert(A.packed_data != nullptr && "A must be quantized");
    assert(B.on_device && "B (input) must be on device");
    assert(C.on_device && "C (output) must be on device");
    assert(B.ndim == 2 && "B (input) must be 2D [M, K]");
    assert(C.ndim == 2 && "C (output) must be 2D [M, N]");

    const int64_t N = A.N;
    const int64_t K = A.K;
    const int64_t M = B.shape[0];

    assert(B.shape[1] == K && "input columns must match weight in_features");
    assert(C.shape[0] == M && C.shape[1] == N && "output shape must be [M, N]");

    if (M == 1) {
        gemv_nvfp4(A, B, C, stream);
        return;
    }

    size_t A_fp16_bytes = (size_t)(N * K) * sizeof(half);
    void* dequant_buf = ensure_dequant_buffer(A_fp16_bytes);
    if (!dequant_buf) return;

    dequantize_nvfp4_to_fp16(A, dequant_buf, stream);

    int64_t A_shape[2] = {N, K};
    Tensor A_fp16(dequant_buf, DType::FP16, 2, A_shape, /*on_device=*/true);

    gemm(B, A_fp16, C, 1.0f, 0.0f, stream);
}

// ---------------------------------------------------------------------------
// PDL registration for all NVFP4 GEMV kernels.
// Called from GraphExecutor::init() when PDL is enabled.
// ---------------------------------------------------------------------------
void nvfp4_gemv_pdl_register() {
    constexpr int NR = 8;
    // Dense GEMV kernels
    pdl::enable_kernel(gemv_nvfp4_kpar_kernel);
    pdl::enable_kernel(gemv_nvfp4_multirow_kernel<NR>);
    pdl::enable_kernel(gemv_nvfp4_residual_kernel);
    pdl::enable_kernel(gemv_nvfp4_residual_mr_kernel<NR>);
    pdl::enable_kernel(gemv_nvfp4_qkv_fused_kernel);
    pdl::enable_kernel(gemv_nvfp4_qkv_fused_mr_kernel<NR>);
    pdl::enable_kernel(gemv_nvfp4_gate_up_fused_kernel);
    pdl::enable_kernel(gemv_nvfp4_gate_up_fused_mr_kernel<NR>);
    pdl::enable_kernel(gemv_nvfp4_swiglu_residual_kernel);
    pdl::enable_kernel(gemv_nvfp4_swiglu_residual_mr_kernel<NR>);
    pdl::enable_kernel(gemv_nvfp4_geglu_residual_kernel);
    pdl::enable_kernel(gemv_nvfp4_geglu_residual_mr_kernel<NR>);
    // MoE GEMV kernels
    pdl::enable_kernel(gemv_nvfp4_moe_decode_kernel);
    pdl::enable_kernel(gemv_nvfp4_moe_gate_up_fused_kernel);
    pdl::enable_kernel(gemv_nvfp4_moe_swiglu_decode_kernel);
}

} // namespace imp
