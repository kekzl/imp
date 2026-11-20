// M=1 GEMVs for mixed NVFP4 + FP8 checkpoints (modelopt MIXED_PRECISION: FP8 GDN projections beside
// NVFP4 MLPs) that take part in the RMSNorm fold of quant/nvfp4_gemm.h: the GDN input projections
// (FP8 in | FP8 gate | FP16 alpha | FP16 beta) on a folded norm row in one launch, and an FP8 weight
// with the residual add and the fold emit in its epilogue.
// The FP8 dot loop is gemv_fp8_e4m3_kernel's (compute/gemm_gemv_dtype.cu): same bits before the epilogue.
#include "quant/fp8_gemv_fold.h"
#include "quant/nvfp4_gemm_internal.cuh"
#include "core/pdl_device.cuh"
#include "core/pdl_launch.cuh"

#include <cuda_fp8.h>
#include <cstring>

namespace imp {

namespace {

constexpr int kFoldGemvThreads = 256;  // 8 warps, one output row each
constexpr int kFoldGemvWarps = kFoldGemvThreads / 32;
int fold_gemv_blocks(int M) { return (M + kFoldGemvWarps - 1) / kFoldGemvWarps; }

// One lane's share of dot(A_row * scale, x); the caller warp-reduces.
__device__ __forceinline__ float fp8_row_dot(const uint8_t* __restrict__ A_row, const half* __restrict__ x,
                                             int K, float scale, int lane) {
    float sum = 0.0f;
    const int K_vec = K / 16;
    const float4* A_row_v = reinterpret_cast<const float4*>(A_row);
    const float4* x_v = reinterpret_cast<const float4*>(x);
    for (int i = lane; i < K_vec; i += 32) {
        float4 a_raw = A_row_v[i];
        float4 x_raw0 = x_v[static_cast<int64_t>(2) * i];
        float4 x_raw1 = x_v[(2 * i) + 1];
        const uint8_t* a_bytes = reinterpret_cast<const uint8_t*>(&a_raw);
        const half* x_lo = reinterpret_cast<const half*>(&x_raw0);
        const half* x_hi = reinterpret_cast<const half*>(&x_raw1);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            __nv_fp8_e4m3 v;
            memcpy(&v, &a_bytes[j], 1);
            sum += (static_cast<float>(v) * scale) * __half2float(x_lo[j]);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            __nv_fp8_e4m3 v;
            memcpy(&v, &a_bytes[8 + j], 1);
            sum += (static_cast<float>(v) * scale) * __half2float(x_hi[j]);
        }
    }
    for (int i = (K_vec * 16) + lane; i < K; i += 32) {
        __nv_fp8_e4m3 v;
        memcpy(&v, &A_row[i], 1);
        sum += (static_cast<float>(v) * scale) * __half2float(x[i]);
    }
    return sum;
}

// FP16 weight row, fp32 products, 16 halves per lane step.
__device__ __forceinline__ float f16_row_dot(const half* __restrict__ A_row, const half* __restrict__ x,
                                             int K, int lane) {
    float sum = 0.0f;
    const int K_vec = K / 16;
    const float4* A_row_v = reinterpret_cast<const float4*>(A_row);
    const float4* x_v = reinterpret_cast<const float4*>(x);
    for (int i = lane; i < K_vec; i += 32) {
        float4 a0 = A_row_v[static_cast<int64_t>(2) * i];
        float4 a1 = A_row_v[(2 * i) + 1];
        float4 x0 = x_v[static_cast<int64_t>(2) * i];
        float4 x1 = x_v[(2 * i) + 1];
        const half* a_lo = reinterpret_cast<const half*>(&a0);
        const half* a_hi = reinterpret_cast<const half*>(&a1);
        const half* x_lo = reinterpret_cast<const half*>(&x0);
        const half* x_hi = reinterpret_cast<const half*>(&x1);
#pragma unroll
        for (int j = 0; j < 8; ++j)
            sum += (__half2float(a_lo[j]) * __half2float(x_lo[j])) +
                   (__half2float(a_hi[j]) * __half2float(x_hi[j]));
    }
    for (int i = (K_vec * 16) + lane; i < K; i += 32)
        sum += __half2float(A_row[i]) * __half2float(x[i]);
    return sum;
}

// h[row] = dot + h[row], then the fold emit for the norm that follows.
__global__ void gemv_fp8_residual_kernel(const uint8_t* __restrict__ A, const half* __restrict__ x, half* h,
                                         int M, int K, float scale, const float* __restrict__ row_scales,
                                         NvFP4NormFoldOut fout) {
    const int lane = threadIdx.x % 32;
    const int row = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32);
    if (row >= M)
        return;
    // Weights: read ahead of the predecessor.
    if (row_scales != nullptr)
        scale = row_scales[row];
    const float gain = norm_fold_gain(fout, row);
    pdl_wait();
    const float sum = warp_reduce(fp8_row_dot(A + (static_cast<int64_t>(row) * K), x, K, scale, lane));
    if (lane != 0)
        return;
    // One warp owns the row: it reads h[row] before it writes it. One ssq atomic per row: a per-block
    // sum measured +0.08 % (83.58 -> 83.65 tok/s, DaisyRose-27B), not worth the smem reduce.
    const half hv = __float2half(sum + __half2float(h[row]));
    h[row] = hv;
    if (fout.ssq)
        norm_fold_emit(fout, row, hv, gain);
}

// Rows [0, n_in) FP8 in, [n_in, n_in + n_gate) FP8 gate, then n_ab FP16 alpha and n_ab FP16 beta rows.
// Every row: y = dot(A_row, x) * norm_fold_scale(fold).
__global__ void gemv_fp8_gdn_input_kernel(Fp8GdnInputArgs a) {
    const int lane = threadIdx.x % 32;
    int row = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32);
    if (row >= a.n_in + a.n_gate + (2 * a.n_ab))
        return;
    const uint8_t* w8 = nullptr;
    const half* w16 = nullptr;
    half* y = nullptr;
    float scale = 1.0f;
    if (row < a.n_in) {
        w8 = a.w_in + (static_cast<int64_t>(row) * a.K);
        scale = a.rs_in != nullptr ? a.rs_in[row] : a.s_in;
        y = a.y_in + row;
    } else if (row < a.n_in + a.n_gate) {
        row -= a.n_in;
        w8 = a.w_gate + (static_cast<int64_t>(row) * a.K);
        scale = a.rs_gate != nullptr ? a.rs_gate[row] : a.s_gate;
        y = a.y_gate + row;
    } else if (row < a.n_in + a.n_gate + a.n_ab) {
        row -= a.n_in + a.n_gate;
        w16 = a.w_alpha + (static_cast<int64_t>(row) * a.K);
        y = a.y_alpha + row;
    } else {
        row -= a.n_in + a.n_gate + a.n_ab;
        w16 = a.w_beta + (static_cast<int64_t>(row) * a.K);
        y = a.y_beta + row;
    }
    pdl_wait();
    const float part = w8 != nullptr ? fp8_row_dot(w8, a.x, a.K, scale, lane)
                                     : f16_row_dot(w16, a.x, a.K, lane);
    const float sum = warp_reduce(part);
    if (lane == 0)
        *y = __float2half(sum * norm_fold_scale(a.fold));
}

}  // namespace

void gemv_fp8_gdn_input(const Fp8GdnInputArgs& a, cudaStream_t stream) {
    const int rows = a.n_in + a.n_gate + (2 * a.n_ab);
    pdl::enable_kernel(gemv_fp8_gdn_input_kernel);
    pdl::launch(gemv_fp8_gdn_input_kernel, dim3(fold_gemv_blocks(rows)), dim3(kFoldGemvThreads), size_t(0),
                stream, a);
    IMP_CUDA_CHECK_LAUNCH();
}

void gemv_fp8_residual(const void* A, float scale, const float* d_row_scales, const half* x, half* h, int M,
                       int K, const NvFP4NormFoldOut& fold, cudaStream_t stream) {
    pdl::enable_kernel(gemv_fp8_residual_kernel);
    pdl::launch(gemv_fp8_residual_kernel, dim3(fold_gemv_blocks(M)), dim3(kFoldGemvThreads), size_t(0),
                stream, static_cast<const uint8_t*>(A), x, h, M, K, scale, d_row_scales, fold);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
