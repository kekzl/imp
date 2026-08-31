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
#include "compute/pdl_device.cuh"

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
    pdl_wait();

    const int tid = threadIdx.x;
    const int K_half = K / 2;
    const int n_mb = K / kMicroBlockSize;

    __shared__ SmemKpar smem;

    float acc = gemv_nvfp4_row(packed_data + (int64_t)row * K_half, micro_scales + (int64_t)row * n_mb,
                               tensor_scale, x, n_mb, tid);

    pdl_trigger();
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
    pdl_wait();

    const int tid = threadIdx.x;
    const int n_mb = K / kMicroBlockSize;

    __shared__ SmemKpar smem;

    float acc = gemv_nvfp4_row(packed_data + (int64_t)row * (K / 2), micro_scales + (int64_t)row * n_mb,
                               tensor_scale, x, n_mb, tid);

    pdl_trigger();
    float total = reduce_kpar(acc, tid, smem.warp_sums);
    if (tid == 0)
        y[row] = total;
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
    pdl_wait();

    const uint8_t* row_packed = packed_data + (int64_t)row * K_half;
    const uint8_t* row_ms = micro_scales + (int64_t)row * n_mb;

    // K-parallel within warp (32 threads), prmt register LUT
    float acc = warp_k_loop(row_packed, row_ms, tensor_scale, n_mb, lane,
                            [&]
                            __device__(const uint8_t* pb, int off) { return dot_micro_block(pb, x, off); });

    pdl_trigger();
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
    pdl_wait();

    const uint8_t* row_packed = packed_data + (int64_t)row * K_half;
    const uint8_t* row_ms = micro_scales + (int64_t)row * n_mb;

    float acc = warp_k_loop(row_packed, row_ms, tensor_scale, n_mb, lane,
                            [&]
                            __device__(const uint8_t* pb, int off) { return dot_micro_block(pb, x, off); });

    pdl_trigger();
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
    pdl_wait();

    const int tid = threadIdx.x;
    const int K_half = K / 2;
    const int n_mb = K / kMicroBlockSize;

    __shared__ SmemKpar smem;

    float acc = gemv_nvfp4_row(packed_data + (int64_t)row * K_half, micro_scales + (int64_t)row * n_mb,
                               tensor_scale, x, n_mb, tid);

    pdl_trigger();
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
    pdl_wait();

    const int tid = threadIdx.x;
    const int K_half = K / 2;
    const int n_mb = K / kMicroBlockSize;

    __shared__ float s_lut[16];
    __shared__ float warp_sums[kKparWarps];
    init_lut(s_lut, tid);
    __syncthreads();

    float acc = gemv_nvfp4_row_swiglu(packed_data + (int64_t)row * K_half, micro_scales + (int64_t)row * n_mb,
                                      tensor_scale, gate, up, n_mb, tid, s_lut);

    pdl_trigger();
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
    pdl_wait();

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

    pdl_trigger();
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
    pdl_wait();

    const int K_half = K / 2;
    const int n_mb = K / kMicroBlockSize;

    const uint8_t* row_packed = packed_data + (int64_t)row * K_half;
    const uint8_t* row_ms = micro_scales + (int64_t)row * n_mb;

    float acc = warp_k_loop(row_packed, row_ms, tensor_scale, n_mb, lane,
                            [&]
                            __device__(const uint8_t* pb, int off) { return dot_micro_block(pb, x, off); });

    pdl_trigger();
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
    pdl_wait();

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

    pdl_trigger();
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
    pdl_wait();

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

    pdl_trigger();
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
