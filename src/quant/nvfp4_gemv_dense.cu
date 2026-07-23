// Dense single-input NVFP4 GEMV kernels + host launchers.
// Split out of nvfp4_gemm.cu (kernel .cu size gate). All kernels and launchers
// MOVED VERBATIM — hot-path numeric code, must stay bit-identical. Shared device
// helpers + tuning constants live in nvfp4_gemm_internal.cuh.

#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_gemm_internal.cuh"
#include "quant/nvfp4_quant.h"
#include "runtime/pdl.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace imp {

// ---------------------------------------------------------------------------
// Basic GEMV: y[row] = A_nvfp4[row,:] @ x
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kKparThreads, 12) gemv_nvfp4_kpar_kernel(
    const uint8_t* __restrict__ packed_data, const uint8_t* __restrict__ micro_scales, float tensor_scale,
    const half* __restrict__ x, half* __restrict__ y, int M, int K) {
    const int row = blockIdx.x;
    if (row >= M)
        return;

    const int tid = threadIdx.x;
    const int K_half = K / 2;
    const int n_mb = K / kMicroBlockSize;

    __shared__ SmemKpar smem;

    float acc = gemv_nvfp4_row(packed_data + (int64_t)row * K_half, micro_scales + (int64_t)row * n_mb,
                               tensor_scale, x, n_mb, tid);

    float total = reduce_kpar(acc, tid, smem.warp_sums);
    if (tid == 0)
        y[row] = __float2half(total);
}

// FP32 output variant for LM head projection (sampling needs float logits).
__global__ void __launch_bounds__(kKparThreads, 12) gemv_nvfp4_kpar_fp32_kernel(
    const uint8_t* __restrict__ packed_data, const uint8_t* __restrict__ micro_scales, float tensor_scale,
    const half* __restrict__ x, float* __restrict__ y, int M, int K) {
    const int row = blockIdx.x;
    if (row >= M)
        return;

    const int tid = threadIdx.x;
    const int n_mb = K / kMicroBlockSize;

    __shared__ SmemKpar smem;

    float acc = gemv_nvfp4_row(packed_data + (int64_t)row * (K / 2), micro_scales + (int64_t)row * n_mb,
                               tensor_scale, x, n_mb, tid);

    float total = reduce_kpar(acc, tid, smem.warp_sums);
    if (tid == 0)
        y[row] = total;
}

// ---------------------------------------------------------------------------
// Batched-M K-parallel GEMV (FP32 out): y[m, n] = A_nvfp4[n,:] @ x[m,:] for the
// MR activation rows m in this launch. One block per output (vocab) row n; the
// weight row is loaded ONCE and reused across all MR activation rows (x[m]
// streams from L2). This removes the per-sequence weight re-read of the batched
// decode LM head — a single M=1 GEMV per sequence re-read the whole ~389 MiB
// LM-head matrix from HBM (it does not fit in L2), making it the #2 decode GPU
// consumer at batch>1. x is [n_act, K] row-major, y is [n_act, N_out] row-major;
// the caller offsets x/y to this launch's first row.
// ---------------------------------------------------------------------------
template <int MR>
__global__ void __launch_bounds__(kKparThreads) gemv_nvfp4_kpar_mb_fp32_kernel(
    const uint8_t* __restrict__ packed_data, const uint8_t* __restrict__ micro_scales, float tensor_scale,
    const half* __restrict__ x, float* __restrict__ y, int N_out, int K) {
    const int n = blockIdx.x;
    if (n >= N_out)
        return;
    const int tid = threadIdx.x;
    const int n_mb = K / kMicroBlockSize;
    const uint8_t* row_packed = packed_data + (int64_t)n * (K / 2);
    const uint8_t* row_ms = micro_scales + (int64_t)n * n_mb;

    float acc[MR];
#pragma unroll
    for (int m = 0; m < MR; ++m)
        acc[m] = 0.0f;

    for (int mi = tid; mi < n_mb; mi += kKparThreads) {
        int byte_off = mi * 8;
        uint2 packed2 = *reinterpret_cast<const uint2*>(row_packed + byte_off);  // weight read once
        const uint8_t* pb = reinterpret_cast<const uint8_t*>(&packed2);
        // Decode the 16 FP4 weights ONCE and reuse across all MR activation rows
        // (the old per-row path re-decoded the weight byte per row — 16x cvt).
        float wf[16];
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            uint32_t w_fp16x2;
            asm("{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }"
                : "=r"(w_fp16x2)
                : "r"(static_cast<uint32_t>(pb[b])));
            float2 wf2 = __half22float2(*reinterpret_cast<const half2*>(&w_fp16x2));
            wf[b * 2] = wf2.x;
            wf[b * 2 + 1] = wf2.y;
        }
        float cs = tensor_scale * fp8_e4m3_to_float_fast(row_ms[mi]);
        const int elem_base = byte_off * 2;
#pragma unroll
        for (int m = 0; m < MR; ++m) {
            const half* xm = x + (int64_t)m * K + elem_base;
            const uint4 xv0 = *reinterpret_cast<const uint4*>(xm);
            const uint4 xv1 = *reinterpret_cast<const uint4*>(xm + 8);
            half2 xh[8];
            *reinterpret_cast<uint4*>(&xh[0]) = xv0;
            *reinterpret_cast<uint4*>(&xh[4]) = xv1;
            float d = 0.0f;
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                float2 xf = __half22float2(xh[b]);
                d = __fmaf_rn(wf[b * 2], xf.x, d);
                d = __fmaf_rn(wf[b * 2 + 1], xf.y, d);
            }
            acc[m] = __fmaf_rn(d, cs, acc[m]);
        }
    }

    __shared__ float warp_sums[kKparWarps];
#pragma unroll
    for (int m = 0; m < MR; ++m) {
        float total = reduce_kpar(acc[m], tid, warp_sums);
        if (tid == 0)
            y[(int64_t)m * N_out + n] = total;
        __syncthreads();  // reuse warp_sums for the next activation row
    }
}

// ---------------------------------------------------------------------------
// Multi-row GEMV: NR rows per block, 256 threads (8 warps).
// Each warp handles one row, multiple warps process multiple rows in parallel.
// Amortizes block launch overhead and improves occupancy for small K.
// ---------------------------------------------------------------------------
template <int NR>
__global__ void __launch_bounds__(kMRThreads) gemv_nvfp4_multirow_kernel(
    const uint8_t* __restrict__ packed_data, const uint8_t* __restrict__ micro_scales, float tensor_scale,
    const half* __restrict__ x, half* __restrict__ y, int M, int K) {
    const int block_row_base = blockIdx.x * NR;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int K_half = K / 2;
    const int n_mb = K / kMicroBlockSize;

    // Each warp handles one row within the NR-row tile
    const int row = block_row_base + warp_id;
    if (row >= M || warp_id >= NR)
        return;

    const uint8_t* row_packed = packed_data + (int64_t)row * K_half;
    const uint8_t* row_ms = micro_scales + (int64_t)row * n_mb;

    // K-parallel within warp (32 threads), prmt register LUT
    float acc = warp_k_loop(row_packed, row_ms, tensor_scale, n_mb, lane,
                            [&]
                            __device__(const uint8_t* pb, int off) { return dot_micro_block(pb, x, off); });

    acc = warp_reduce(acc);
    if (lane == 0)
        y[row] = __float2half(acc);
}

// FP32 output multi-row variant for LM head projection.
template <int NR>
__global__ void __launch_bounds__(kMRThreads) gemv_nvfp4_multirow_fp32_kernel(
    const uint8_t* __restrict__ packed_data, const uint8_t* __restrict__ micro_scales, float tensor_scale,
    const half* __restrict__ x, float* __restrict__ y, int M, int K) {
    const int block_row_base = blockIdx.x * NR;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int K_half = K / 2;
    const int n_mb = K / kMicroBlockSize;

    const int row = block_row_base + warp_id;
    if (row >= M || warp_id >= NR)
        return;

    const uint8_t* row_packed = packed_data + (int64_t)row * K_half;
    const uint8_t* row_ms = micro_scales + (int64_t)row * n_mb;

    float acc = warp_k_loop(row_packed, row_ms, tensor_scale, n_mb, lane,
                            [&]
                            __device__(const uint8_t* pb, int off) { return dot_micro_block(pb, x, off); });

    acc = warp_reduce(acc);
    if (lane == 0)
        y[row] = acc;
}

// ---------------------------------------------------------------------------
// GEMV with residual: y[row] = A_nvfp4[row,:] @ x + residual[row]
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kKparThreads, 12) gemv_nvfp4_residual_kernel(
    const uint8_t* __restrict__ packed_data, const uint8_t* __restrict__ micro_scales, float tensor_scale,
    const half* __restrict__ x, half* __restrict__ y, const half* __restrict__ residual, int M, int K) {
    const int row = blockIdx.x;
    if (row >= M)
        return;

    const int tid = threadIdx.x;
    const int K_half = K / 2;
    const int n_mb = K / kMicroBlockSize;

    __shared__ SmemKpar smem;

    float acc = gemv_nvfp4_row(packed_data + (int64_t)row * K_half, micro_scales + (int64_t)row * n_mb,
                               tensor_scale, x, n_mb, tid);

    float total = reduce_kpar(acc, tid, smem.warp_sums);
    if (tid == 0)
        y[row] = __float2half(total + __half2float(residual[row]));
}

// ---------------------------------------------------------------------------
// Fused SwiGLU + GEMV + residual:
//   y[row] = A_nvfp4[row,:] @ swiglu(gate, up) + residual[row]
// Eliminates the separate SwiGLU kernel launch.
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(kKparThreads, 12) gemv_nvfp4_swiglu_residual_kernel(
    const uint8_t* __restrict__ packed_data, const uint8_t* __restrict__ micro_scales, float tensor_scale,
    const half* __restrict__ gate, const half* __restrict__ up, half* __restrict__ y,
    const half* __restrict__ residual, int M, int K) {
    const int row = blockIdx.x;
    if (row >= M)
        return;

    const int tid = threadIdx.x;
    const int K_half = K / 2;
    const int n_mb = K / kMicroBlockSize;

    __shared__ float s_lut[16];
    __shared__ float warp_sums[kKparWarps];
    init_lut(s_lut, tid);
    __syncthreads();

    float acc = gemv_nvfp4_row_swiglu(packed_data + (int64_t)row * K_half, micro_scales + (int64_t)row * n_mb,
                                      tensor_scale, gate, up, n_mb, tid, s_lut);

    float total = reduce_kpar(acc, tid, warp_sums);
    if (tid == 0)
        y[row] = __float2half(total + __half2float(residual[row]));
}

// ---------------------------------------------------------------------------
// Fused GeGLU + GEMV + residual (for Gemma-3 and similar)
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kKparThreads, 12) gemv_nvfp4_geglu_residual_kernel(
    const uint8_t* __restrict__ packed_data, const uint8_t* __restrict__ micro_scales, float tensor_scale,
    const half* __restrict__ gate, const half* __restrict__ up, half* __restrict__ y,
    const half* __restrict__ residual, int M, int K) {
    const int row = blockIdx.x;
    if (row >= M)
        return;

    const int tid = threadIdx.x;
    const int K_half = K / 2;
    const int n_mb = K / kMicroBlockSize;

    __shared__ float s_lut[16];
    __shared__ float warp_sums[kKparWarps];
    init_lut(s_lut, tid);
    __syncthreads();

    const uint8_t* row_packed = packed_data + (int64_t)row * K_half;
    const uint8_t* row_ms = micro_scales + (int64_t)row * n_mb;

    float acc = 0.0f;
    for (int mi = tid; mi < n_mb; mi += kKparThreads) {
        int byte_off = mi * 8;
        uint2 packed2 = *reinterpret_cast<const uint2*>(row_packed + byte_off);
        const uint8_t* pb = reinterpret_cast<const uint8_t*>(&packed2);
        float cs = tensor_scale * fp8_e4m3_to_float_fast(row_ms[mi]);
        float local_dot = dot_micro_block_geglu(pb, gate, up, byte_off * 2, s_lut);
        acc = __fmaf_rn(local_dot, cs, acc);
    }

    float total = reduce_kpar(acc, tid, warp_sums);
    if (tid == 0)
        y[row] = __float2half(total + __half2float(residual[row]));
}

// Multi-row residual.
template <int NR>
__global__ void __launch_bounds__(kMRThreads) gemv_nvfp4_residual_mr_kernel(
    const uint8_t* __restrict__ packed_data, const uint8_t* __restrict__ micro_scales, float tensor_scale,
    const half* __restrict__ x, half* __restrict__ y, const half* __restrict__ residual, int M, int K) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * NR + warp_id;
    if (row >= M || warp_id >= NR)
        return;

    const int K_half = K / 2;
    const int n_mb = K / kMicroBlockSize;

    const uint8_t* row_packed = packed_data + (int64_t)row * K_half;
    const uint8_t* row_ms = micro_scales + (int64_t)row * n_mb;

    float acc = warp_k_loop(row_packed, row_ms, tensor_scale, n_mb, lane,
                            [&]
                            __device__(const uint8_t* pb, int off) { return dot_micro_block(pb, x, off); });

    acc = warp_reduce(acc);
    if (lane == 0)
        y[row] = __float2half(acc + __half2float(residual[row]));
}

// Multi-row SwiGLU + residual.
template <int NR>
__global__ void __launch_bounds__(kMRThreads) gemv_nvfp4_swiglu_residual_mr_kernel(
    const uint8_t* __restrict__ packed_data, const uint8_t* __restrict__ micro_scales, float tensor_scale,
    const half* __restrict__ gate, const half* __restrict__ up, half* __restrict__ y,
    const half* __restrict__ residual, int M, int K) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * NR + warp_id;
    if (row >= M || warp_id >= NR)
        return;

    const int K_half = K / 2;
    const int n_mb = K / kMicroBlockSize;

    __shared__ float s_lut[16];
    init_lut(s_lut, threadIdx.x);
    __syncthreads();

    const uint8_t* row_packed = packed_data + (int64_t)row * K_half;
    const uint8_t* row_ms = micro_scales + (int64_t)row * n_mb;

    float acc = warp_k_loop(row_packed, row_ms, tensor_scale, n_mb, lane,
                            [&] __device__(const uint8_t* pb, int off) {
                                return dot_micro_block_swiglu(pb, gate, up, off, s_lut);
                            });

    acc = warp_reduce(acc);
    if (lane == 0)
        y[row] = __float2half(acc + __half2float(residual[row]));
}

// Multi-row GeGLU + residual.
template <int NR>
__global__ void __launch_bounds__(kMRThreads) gemv_nvfp4_geglu_residual_mr_kernel(
    const uint8_t* __restrict__ packed_data, const uint8_t* __restrict__ micro_scales, float tensor_scale,
    const half* __restrict__ gate, const half* __restrict__ up, half* __restrict__ y,
    const half* __restrict__ residual, int M, int K) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * NR + warp_id;
    if (row >= M || warp_id >= NR)
        return;

    const int K_half = K / 2;
    const int n_mb = K / kMicroBlockSize;

    __shared__ float s_lut[16];
    init_lut(s_lut, threadIdx.x);
    __syncthreads();

    const uint8_t* row_packed = packed_data + (int64_t)row * K_half;
    const uint8_t* row_ms = micro_scales + (int64_t)row * n_mb;

    float acc = warp_k_loop(row_packed, row_ms, tensor_scale, n_mb, lane,
                            [&] __device__(const uint8_t* pb, int off) {
                                return dot_micro_block_geglu(pb, gate, up, off, s_lut);
                            });

    acc = warp_reduce(acc);
    if (lane == 0)
        y[row] = __float2half(acc + __half2float(residual[row]));
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

void gemv_nvfp4_kpar(const NvFP4QuantResult& A, const half* x, half* y, int M, int K, cudaStream_t stream) {
    const int n_mb = K / kMicroBlockSize;
    constexpr int NR = 8;
    int mr_blocks = (M + NR - 1) / NR;
    if (use_multirow(n_mb, mr_blocks)) {
        pdl::launch(gemv_nvfp4_multirow_kernel<NR>, dim3(mr_blocks), dim3(kMRThreads), size_t(0), stream,
                    reinterpret_cast<const uint8_t*>(A.packed_data),
                    reinterpret_cast<const uint8_t*>(A.micro_scales), A.tensor_scale, x, y, M, K);
    } else {
        pdl::launch(gemv_nvfp4_kpar_kernel, dim3(M), dim3(kKparThreads), size_t(0), stream,
                    reinterpret_cast<const uint8_t*>(A.packed_data),
                    reinterpret_cast<const uint8_t*>(A.micro_scales), A.tensor_scale, x, y, M, K);
    }
}

void gemv_nvfp4_kpar_fp32(const NvFP4QuantResult& A, const half* x, float* y, int M, int K,
                          cudaStream_t stream) {
    const int n_mb = K / kMicroBlockSize;
    constexpr int NR = 8;
    int mr_blocks = (M + NR - 1) / NR;
    if (use_multirow(n_mb, mr_blocks)) {
        pdl::launch(gemv_nvfp4_multirow_fp32_kernel<NR>, dim3(mr_blocks), dim3(kMRThreads), size_t(0), stream,
                    reinterpret_cast<const uint8_t*>(A.packed_data),
                    reinterpret_cast<const uint8_t*>(A.micro_scales), A.tensor_scale, x, y, M, K);
    } else {
        pdl::launch(gemv_nvfp4_kpar_fp32_kernel, dim3(M), dim3(kKparThreads), size_t(0), stream,
                    reinterpret_cast<const uint8_t*>(A.packed_data),
                    reinterpret_cast<const uint8_t*>(A.micro_scales), A.tensor_scale, x, y, M, K);
    }
}

// Batched-M FP32 GEMV for the LM head at batch>1: y[n_act, N_out] = x[n_act,K] @ A^T.
// Reads the weight matrix ONCE per launch (vs once per activation row in the old
// per-row M=1 loop). n_act is processed in power-of-two MR chunks so each launch
// reuses the weight across MR rows; typical decode batches (<=16) need one launch.
void gemv_nvfp4_kpar_batched_fp32(const NvFP4QuantResult& A, const half* x, float* y, int N_out, int K,
                                  int n_act, cudaStream_t stream) {
    const auto* pd = reinterpret_cast<const uint8_t*>(A.packed_data);
    const auto* ms = reinterpret_cast<const uint8_t*>(A.micro_scales);
    const float ts = A.tensor_scale;
    // MR is capped at 4: each accumulator row costs registers, and beyond MR=4
    // the kernel spills (measured: 91 us/row at MR=4 vs 118 us/row at MR=16). For
    // M>4 the weight is re-read per MR=4 tile, but the tiled cost still beats the
    // larger-MR spill (M=16: 4x MR=4 = 1.45 ms < 1x MR=16 = 1.89 ms) and the old
    // per-row loop (16x M=1 = 4.2 ms).
    int done = 0;
    while (done < n_act) {
        const int rem = n_act - done;
        const half* xm = x + (int64_t)done * K;
        float* ym = y + (int64_t)done * N_out;
        int used;
        if (rem >= 4) {
            pdl::launch(gemv_nvfp4_kpar_mb_fp32_kernel<4>, dim3(N_out), dim3(kKparThreads), size_t(0), stream,
                        pd, ms, ts, xm, ym, N_out, K);
            used = 4;
        } else if (rem == 3) {
            // Bucket-3 verify lm_head: one sweep instead of <2>+<1> (#1055).
            pdl::launch(gemv_nvfp4_kpar_mb_fp32_kernel<3>, dim3(N_out), dim3(kKparThreads), size_t(0), stream,
                        pd, ms, ts, xm, ym, N_out, K);
            used = 3;
        } else if (rem >= 2) {
            pdl::launch(gemv_nvfp4_kpar_mb_fp32_kernel<2>, dim3(N_out), dim3(kKparThreads), size_t(0), stream,
                        pd, ms, ts, xm, ym, N_out, K);
            used = 2;
        } else {
            pdl::launch(gemv_nvfp4_kpar_mb_fp32_kernel<1>, dim3(N_out), dim3(kKparThreads), size_t(0), stream,
                        pd, ms, ts, xm, ym, N_out, K);
            used = 1;
        }
        done += used;
    }
}

// FP16-output twin of gemv_nvfp4_kpar_mb_fp32_kernel for spec-verify chunk
// GEMMs (#998): one block per weight row n, the row decoded once and reused
// across the MR activation rows of this launch. kAcc adds into the existing
// output (cuBLAS beta=1 semantics) for the o/down residual-add GEMMs (#1055).
template <int MR, bool kAcc = false>
__global__ void __launch_bounds__(kKparThreads) gemv_nvfp4_kpar_mb_fp16_kernel(
    const uint8_t* __restrict__ packed_data, const uint8_t* __restrict__ micro_scales, float tensor_scale,
    const half* __restrict__ x, half* __restrict__ y, int N_out, int K) {
    const int n = blockIdx.x;
    if (n >= N_out)
        return;
    const int tid = threadIdx.x;
    const int n_mb = K / kMicroBlockSize;
    const uint8_t* row_packed = packed_data + (int64_t)n * (K / 2);
    const uint8_t* row_ms = micro_scales + (int64_t)n * n_mb;

    float acc[MR];
#pragma unroll
    for (int m = 0; m < MR; ++m)
        acc[m] = 0.0f;

    for (int mi = tid; mi < n_mb; mi += kKparThreads) {
        int byte_off = mi * 8;
        uint2 packed2 = *reinterpret_cast<const uint2*>(row_packed + byte_off);  // weight read once
        const uint8_t* pb = reinterpret_cast<const uint8_t*>(&packed2);
        float wf[16];
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            uint32_t w_fp16x2;
            asm("{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }"
                : "=r"(w_fp16x2)
                : "r"(static_cast<uint32_t>(pb[b])));
            float2 wf2 = __half22float2(*reinterpret_cast<const half2*>(&w_fp16x2));
            wf[b * 2] = wf2.x;
            wf[b * 2 + 1] = wf2.y;
        }
        float cs = tensor_scale * fp8_e4m3_to_float_fast(row_ms[mi]);
        const int elem_base = byte_off * 2;
#pragma unroll
        for (int m = 0; m < MR; ++m) {
            const half* xm = x + (int64_t)m * K + elem_base;
            const uint4 xv0 = *reinterpret_cast<const uint4*>(xm);
            const uint4 xv1 = *reinterpret_cast<const uint4*>(xm + 8);
            half2 xh[8];
            *reinterpret_cast<uint4*>(&xh[0]) = xv0;
            *reinterpret_cast<uint4*>(&xh[4]) = xv1;
            float d = 0.0f;
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                float2 xf = __half22float2(xh[b]);
                d = __fmaf_rn(wf[b * 2], xf.x, d);
                d = __fmaf_rn(wf[b * 2 + 1], xf.y, d);
            }
            acc[m] = __fmaf_rn(d, cs, acc[m]);
        }
    }

    __shared__ float warp_sums[kKparWarps];
#pragma unroll
    for (int m = 0; m < MR; ++m) {
        float total = reduce_kpar(acc[m], tid, warp_sums);
        if (tid == 0) {
            half* yp = y + (int64_t)m * N_out + n;
            if (kAcc)
                total += __half2float(*yp);
            *yp = __float2half(total);
        }
        __syncthreads();  // reuse warp_sums for the next activation row
    }
}

// Batched-M FP16 GEMM for spec-verify chunks (#998). MR cap of 4 mirrors the
// FP32 LM-head launcher: larger MR spills registers and loses to tiling.
void gemm_nvfp4_batched(const NvFP4QuantResult& A, const half* x, half* y, int N_out, int K,
                        int n_act, cudaStream_t stream) {
    const auto* pd = reinterpret_cast<const uint8_t*>(A.packed_data);
    const auto* ms = reinterpret_cast<const uint8_t*>(A.micro_scales);
    const float ts = A.tensor_scale;
    int done = 0;
    while (done < n_act) {
        const int rem = n_act - done;
        const half* xm = x + (int64_t)done * K;
        half* ym = y + (int64_t)done * N_out;
        int used;
        if (rem >= 4) {
            pdl::launch(gemv_nvfp4_kpar_mb_fp16_kernel<4>, dim3(N_out), dim3(kKparThreads), size_t(0), stream,
                        pd, ms, ts, xm, ym, N_out, K);
            used = 4;
        } else if (rem == 3) {
            // Dedicated 3-row tile: capture bucket 3 (the k=2 verify chunk)
            // otherwise pays TWO weight sweeps (<2> + <1>) — #1055.
            pdl::launch(gemv_nvfp4_kpar_mb_fp16_kernel<3>, dim3(N_out), dim3(kKparThreads), size_t(0), stream,
                        pd, ms, ts, xm, ym, N_out, K);
            used = 3;
        } else if (rem >= 2) {
            pdl::launch(gemv_nvfp4_kpar_mb_fp16_kernel<2>, dim3(N_out), dim3(kKparThreads), size_t(0), stream,
                        pd, ms, ts, xm, ym, N_out, K);
            used = 2;
        } else {
            pdl::launch(gemv_nvfp4_kpar_mb_fp16_kernel<1>, dim3(N_out), dim3(kKparThreads), size_t(0), stream,
                        pd, ms, ts, xm, ym, N_out, K);
            used = 1;
        }
        done += used;
    }
}

// beta=1 twin: y[m,n] += A[n,:] @ x[m,:] (o/down residual-add verify GEMMs).
void gemm_nvfp4_batched_acc(const NvFP4QuantResult& A, const half* x, half* y, int N_out, int K,
                            int n_act, cudaStream_t stream) {
    const auto* pd = reinterpret_cast<const uint8_t*>(A.packed_data);
    const auto* ms = reinterpret_cast<const uint8_t*>(A.micro_scales);
    const float ts = A.tensor_scale;
    int done = 0;
    while (done < n_act) {
        const int rem = n_act - done;
        const half* xm = x + (int64_t)done * K;
        half* ym = y + (int64_t)done * N_out;
        int used;
        if (rem >= 4) {
            pdl::launch(gemv_nvfp4_kpar_mb_fp16_kernel<4, true>, dim3(N_out), dim3(kKparThreads),
                        size_t(0), stream, pd, ms, ts, xm, ym, N_out, K);
            used = 4;
        } else if (rem == 3) {
            pdl::launch(gemv_nvfp4_kpar_mb_fp16_kernel<3, true>, dim3(N_out), dim3(kKparThreads),
                        size_t(0), stream, pd, ms, ts, xm, ym, N_out, K);
            used = 3;
        } else if (rem >= 2) {
            pdl::launch(gemv_nvfp4_kpar_mb_fp16_kernel<2, true>, dim3(N_out), dim3(kKparThreads),
                        size_t(0), stream, pd, ms, ts, xm, ym, N_out, K);
            used = 2;
        } else {
            pdl::launch(gemv_nvfp4_kpar_mb_fp16_kernel<1, true>, dim3(N_out), dim3(kKparThreads),
                        size_t(0), stream, pd, ms, ts, xm, ym, N_out, K);
            used = 1;
        }
        done += used;
    }
}

void gemv_nvfp4_residual(const NvFP4QuantResult& A, const half* x, half* y, const half* residual, int M,
                         int K, cudaStream_t stream) {
    const int n_mb = K / kMicroBlockSize;
    constexpr int NR = 8;
    int mr_blocks = (M + NR - 1) / NR;
    if (use_multirow(n_mb, mr_blocks)) {
        pdl::launch(gemv_nvfp4_residual_mr_kernel<NR>, dim3(mr_blocks), dim3(kMRThreads), size_t(0), stream,
                    reinterpret_cast<const uint8_t*>(A.packed_data),
                    reinterpret_cast<const uint8_t*>(A.micro_scales), A.tensor_scale, x, y, residual, M, K);
    } else {
        pdl::launch(gemv_nvfp4_residual_kernel, dim3(M), dim3(kKparThreads), size_t(0), stream,
                    reinterpret_cast<const uint8_t*>(A.packed_data),
                    reinterpret_cast<const uint8_t*>(A.micro_scales), A.tensor_scale, x, y, residual, M, K);
    }
}

void gemv_nvfp4_swiglu_residual(const NvFP4QuantResult& A, const half* gate, const half* up, half* y,
                                const half* residual, int M, int K, cudaStream_t stream) {
    const int n_mb = K / kMicroBlockSize;
    constexpr int NR = 8;
    int mr_blocks = (M + NR - 1) / NR;
    if (use_multirow(n_mb, mr_blocks)) {
        pdl::launch(gemv_nvfp4_swiglu_residual_mr_kernel<NR>, dim3(mr_blocks), dim3(kMRThreads), size_t(0),
                    stream, reinterpret_cast<const uint8_t*>(A.packed_data),
                    reinterpret_cast<const uint8_t*>(A.micro_scales), A.tensor_scale, gate, up, y, residual,
                    M, K);
    } else {
        pdl::launch(gemv_nvfp4_swiglu_residual_kernel, dim3(M), dim3(kKparThreads), size_t(0), stream,
                    reinterpret_cast<const uint8_t*>(A.packed_data),
                    reinterpret_cast<const uint8_t*>(A.micro_scales), A.tensor_scale, gate, up, y, residual,
                    M, K);
    }
}

void gemv_nvfp4_geglu_residual(const NvFP4QuantResult& A, const half* gate, const half* up, half* y,
                               const half* residual, int M, int K, cudaStream_t stream) {
    const int n_mb = K / kMicroBlockSize;
    constexpr int NR = 8;
    int mr_blocks = (M + NR - 1) / NR;
    if (use_multirow(n_mb, mr_blocks)) {
        pdl::launch(gemv_nvfp4_geglu_residual_mr_kernel<NR>, dim3(mr_blocks), dim3(kMRThreads), size_t(0),
                    stream, reinterpret_cast<const uint8_t*>(A.packed_data),
                    reinterpret_cast<const uint8_t*>(A.micro_scales), A.tensor_scale, gate, up, y, residual,
                    M, K);
    } else {
        pdl::launch(gemv_nvfp4_geglu_residual_kernel, dim3(M), dim3(kKparThreads), size_t(0), stream,
                    reinterpret_cast<const uint8_t*>(A.packed_data),
                    reinterpret_cast<const uint8_t*>(A.micro_scales), A.tensor_scale, gate, up, y, residual,
                    M, K);
    }
}

}  // namespace imp
