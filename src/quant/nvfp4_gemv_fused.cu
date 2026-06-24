// Fused multi-matrix NVFP4 GEMV kernels (QKV-fused, gate+up-fused) + launchers.
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
// Fused QKV: 3 weight matrices, shared input, separate outputs
// Grid: (q_rows + k_rows + v_rows) blocks.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kKparThreads, 12) gemv_nvfp4_qkv_fused_kernel(
    const uint8_t* __restrict__ packed_q, const uint8_t* __restrict__ ms_q, float ts_q,
    const uint8_t* __restrict__ packed_k, const uint8_t* __restrict__ ms_k, float ts_k,
    const uint8_t* __restrict__ packed_v, const uint8_t* __restrict__ ms_v, float ts_v,
    const half* __restrict__ x, half* __restrict__ yq, half* __restrict__ yk, half* __restrict__ yv,
    int q_rows, int k_rows, int v_rows, int K) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int K_half = K / 2;
    const int n_mb = K / kMicroBlockSize;

    __shared__ SmemKpar smem;

    const uint8_t* row_packed;
    const uint8_t* row_ms;
    float ts;
    half* out;
    int local_row;

    if (bid < q_rows) {
        local_row = bid;
        row_packed = packed_q + (int64_t)local_row * K_half;
        row_ms = ms_q + (int64_t)local_row * n_mb;
        ts = ts_q;
        out = yq;
    } else if (bid < q_rows + k_rows) {
        local_row = bid - q_rows;
        row_packed = packed_k + (int64_t)local_row * K_half;
        row_ms = ms_k + (int64_t)local_row * n_mb;
        ts = ts_k;
        out = yk;
    } else {
        local_row = bid - q_rows - k_rows;
        row_packed = packed_v + (int64_t)local_row * K_half;
        row_ms = ms_v + (int64_t)local_row * n_mb;
        ts = ts_v;
        out = yv;
    }

    float acc = gemv_nvfp4_row(row_packed, row_ms, ts, x, n_mb, tid);
    float total = reduce_kpar(acc, tid, smem.warp_sums);
    if (tid == 0)
        out[local_row] = __float2half(total);
}

// ---------------------------------------------------------------------------
// Fused Gate+Up: 2 weight matrices, shared input, separate outputs
// Grid: 2 * rows blocks. First half = gate, second half = up.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kKparThreads, 12) gemv_nvfp4_gate_up_fused_kernel(
    const uint8_t* __restrict__ packed_g, const uint8_t* __restrict__ ms_g, float ts_g,
    const uint8_t* __restrict__ packed_u, const uint8_t* __restrict__ ms_u, float ts_u,
    const half* __restrict__ x, half* __restrict__ yg, half* __restrict__ yu, int rows, int K) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int K_half = K / 2;
    const int n_mb = K / kMicroBlockSize;

    __shared__ SmemKpar smem;

    const uint8_t* row_packed;
    const uint8_t* row_ms;
    float ts;
    half* out;
    int local_row;

    if (bid < rows) {
        local_row = bid;
        row_packed = packed_g + (int64_t)local_row * K_half;
        row_ms = ms_g + (int64_t)local_row * n_mb;
        ts = ts_g;
        out = yg;
    } else {
        local_row = bid - rows;
        row_packed = packed_u + (int64_t)local_row * K_half;
        row_ms = ms_u + (int64_t)local_row * n_mb;
        ts = ts_u;
        out = yu;
    }

    float acc = gemv_nvfp4_row(row_packed, row_ms, ts, x, n_mb, tid);
    float total = reduce_kpar(acc, tid, smem.warp_sums);
    if (tid == 0)
        out[local_row] = __float2half(total);
}

// ---------------------------------------------------------------------------
// Multi-row variants of fused kernels (NR rows/block, 256 threads, 8 warps).
// Used when K is small (n_mb ≤ 512) to reduce block count and improve
// per-thread work (32 threads/row vs 128 threads/row).
// ---------------------------------------------------------------------------

// Multi-row QKV fused: each warp determines its matrix and row independently.
template <int NR>
__global__ void __launch_bounds__(kMRThreads) gemv_nvfp4_qkv_fused_mr_kernel(
    const uint8_t* __restrict__ packed_q, const uint8_t* __restrict__ ms_q, float ts_q,
    const uint8_t* __restrict__ packed_k, const uint8_t* __restrict__ ms_k, float ts_k,
    const uint8_t* __restrict__ packed_v, const uint8_t* __restrict__ ms_v, float ts_v,
    const half* __restrict__ x, half* __restrict__ yq, half* __restrict__ yk, half* __restrict__ yv,
    int q_rows, int k_rows, int v_rows, int K) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int global_row = blockIdx.x * NR + warp_id;
    const int total_rows = q_rows + k_rows + v_rows;
    if (global_row >= total_rows || warp_id >= NR)
        return;

    const int K_half = K / 2;
    const int n_mb = K / kMicroBlockSize;

    const uint8_t* row_packed;
    const uint8_t* row_ms;
    float ts;
    half* out;
    int local_row;

    if (global_row < q_rows) {
        local_row = global_row;
        row_packed = packed_q + (int64_t)local_row * K_half;
        row_ms = ms_q + (int64_t)local_row * n_mb;
        ts = ts_q;
        out = yq;
    } else if (global_row < q_rows + k_rows) {
        local_row = global_row - q_rows;
        row_packed = packed_k + (int64_t)local_row * K_half;
        row_ms = ms_k + (int64_t)local_row * n_mb;
        ts = ts_k;
        out = yk;
    } else {
        local_row = global_row - q_rows - k_rows;
        row_packed = packed_v + (int64_t)local_row * K_half;
        row_ms = ms_v + (int64_t)local_row * n_mb;
        ts = ts_v;
        out = yv;
    }

    float acc = warp_k_loop(row_packed, row_ms, ts, n_mb, lane, [&] __device__(const uint8_t* pb, int off) {
        return dot_micro_block(pb, x, off);
    });

    acc = warp_reduce(acc);
    if (lane == 0)
        out[local_row] = __float2half(acc);
}

// Multi-row gate+up fused.
template <int NR>
__global__ void __launch_bounds__(kMRThreads) gemv_nvfp4_gate_up_fused_mr_kernel(
    const uint8_t* __restrict__ packed_g, const uint8_t* __restrict__ ms_g, float ts_g,
    const uint8_t* __restrict__ packed_u, const uint8_t* __restrict__ ms_u, float ts_u,
    const half* __restrict__ x, half* __restrict__ yg, half* __restrict__ yu, int rows, int K) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int global_row = blockIdx.x * NR + warp_id;
    const int total_rows = 2 * rows;
    if (global_row >= total_rows || warp_id >= NR)
        return;

    const int K_half = K / 2;
    const int n_mb = K / kMicroBlockSize;

    const uint8_t* row_packed;
    const uint8_t* row_ms;
    float ts;
    half* out;
    int local_row;

    if (global_row < rows) {
        local_row = global_row;
        row_packed = packed_g + (int64_t)local_row * K_half;
        row_ms = ms_g + (int64_t)local_row * n_mb;
        ts = ts_g;
        out = yg;
    } else {
        local_row = global_row - rows;
        row_packed = packed_u + (int64_t)local_row * K_half;
        row_ms = ms_u + (int64_t)local_row * n_mb;
        ts = ts_u;
        out = yu;
    }

    float acc = warp_k_loop(row_packed, row_ms, ts, n_mb, lane, [&] __device__(const uint8_t* pb, int off) {
        return dot_micro_block(pb, x, off);
    });

    acc = warp_reduce(acc);
    if (lane == 0)
        out[local_row] = __float2half(acc);
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

void gemv_nvfp4_qkv_fused(const NvFP4QuantResult& wq, const NvFP4QuantResult& wk, const NvFP4QuantResult& wv,
                          const half* x, half* yq, half* yk, half* yv, int q_rows, int k_rows, int v_rows,
                          int K, cudaStream_t stream) {
    int total_rows = q_rows + k_rows + v_rows;
    const int n_mb = K / kMicroBlockSize;
    constexpr int NR = 8;
    int mr_blocks = (total_rows + NR - 1) / NR;
    if (use_multirow(n_mb, mr_blocks)) {
        pdl::launch(gemv_nvfp4_qkv_fused_mr_kernel<NR>, dim3(mr_blocks), dim3(kMRThreads), size_t(0), stream,
                    reinterpret_cast<const uint8_t*>(wq.packed_data),
                    reinterpret_cast<const uint8_t*>(wq.micro_scales), wq.tensor_scale,
                    reinterpret_cast<const uint8_t*>(wk.packed_data),
                    reinterpret_cast<const uint8_t*>(wk.micro_scales), wk.tensor_scale,
                    reinterpret_cast<const uint8_t*>(wv.packed_data),
                    reinterpret_cast<const uint8_t*>(wv.micro_scales), wv.tensor_scale, x, yq, yk, yv, q_rows,
                    k_rows, v_rows, K);
    } else {
        pdl::launch(gemv_nvfp4_qkv_fused_kernel, dim3(total_rows), dim3(kKparThreads), size_t(0), stream,
                    reinterpret_cast<const uint8_t*>(wq.packed_data),
                    reinterpret_cast<const uint8_t*>(wq.micro_scales), wq.tensor_scale,
                    reinterpret_cast<const uint8_t*>(wk.packed_data),
                    reinterpret_cast<const uint8_t*>(wk.micro_scales), wk.tensor_scale,
                    reinterpret_cast<const uint8_t*>(wv.packed_data),
                    reinterpret_cast<const uint8_t*>(wv.micro_scales), wv.tensor_scale, x, yq, yk, yv, q_rows,
                    k_rows, v_rows, K);
    }
}

void gemv_nvfp4_gate_up_fused(const NvFP4QuantResult& wg, const NvFP4QuantResult& wu, const half* x, half* yg,
                              half* yu, int rows, int K, cudaStream_t stream) {
    int total_rows = 2 * rows;
    const int n_mb = K / kMicroBlockSize;
    constexpr int NR = 8;
    int mr_blocks = (total_rows + NR - 1) / NR;
    if (use_multirow(n_mb, mr_blocks)) {
        pdl::launch(gemv_nvfp4_gate_up_fused_mr_kernel<NR>, dim3(mr_blocks), dim3(kMRThreads), size_t(0),
                    stream, reinterpret_cast<const uint8_t*>(wg.packed_data),
                    reinterpret_cast<const uint8_t*>(wg.micro_scales), wg.tensor_scale,
                    reinterpret_cast<const uint8_t*>(wu.packed_data),
                    reinterpret_cast<const uint8_t*>(wu.micro_scales), wu.tensor_scale, x, yg, yu, rows, K);
    } else {
        pdl::launch(gemv_nvfp4_gate_up_fused_kernel, dim3(total_rows), dim3(kKparThreads), size_t(0), stream,
                    reinterpret_cast<const uint8_t*>(wg.packed_data),
                    reinterpret_cast<const uint8_t*>(wg.micro_scales), wg.tensor_scale,
                    reinterpret_cast<const uint8_t*>(wu.packed_data),
                    reinterpret_cast<const uint8_t*>(wu.micro_scales), wu.tensor_scale, x, yg, yu, rows, K);
    }
}

}  // namespace imp
