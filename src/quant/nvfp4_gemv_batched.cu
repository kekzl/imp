// NVFP4 GEMV/GEMM kernels serving MANY activation rows (spec-verify M=1..4, batched LM
// head). Split from nvfp4_gemv_dense.cu (2026-08-21): every kernel there computes ONE
// activation row against many weight rows, every kernel here computes MANY rows against the
// same weight read - also the boundary numerical-parity work runs along
// (gemv_nvfp4_multirow_mb_kernel reduces K exactly like the one-row side). Templated over
// MR, so compile-time isolation stops every touch from re-ptxasing four instantiations and
// dragging the single-row decode kernels through it.

#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_gemm_internal.cuh"
#include "core/pdl.h"

#include <cuda_fp16.h>
#include "core/pdl_device.cuh"

namespace imp {
namespace {

// Batched-M K-parallel GEMV (FP32 out): y[m,n]=A_nvfp4[n,:]@x[m,:] for the MR activation
// rows in this launch. One block per output row n; the weight row loads ONCE and is reused
// across all MR rows (x[m] streams from L2). Removes the per-sequence weight re-read of the
// batched decode LM head (a single M=1 GEMV per sequence re-read the whole ~389 MiB
// LM-head matrix from HBM, the #2 decode GPU consumer at batch>1).
template <int MR>
__global__ void __launch_bounds__(kKparThreads) gemv_nvfp4_kpar_mb_fp32_kernel(
    const uint8_t* __restrict__ packed_data, const uint8_t* __restrict__ micro_scales, float tensor_scale,
    const half* __restrict__ x, float* __restrict__ y, int N_out, int K) {
    const int n = blockIdx.x;
    if (n >= N_out)
        return;
    pdl_wait();
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

    pdl_trigger();
    __shared__ float warp_sums[kKparWarps];
#pragma unroll
    for (int m = 0; m < MR; ++m) {
        float total = reduce_kpar(acc[m], tid, warp_sums);
        if (tid == 0)
            y[(int64_t)m * N_out + n] = total;
        __syncthreads();  // reuse warp_sums for the next activation row
    }
}

// Batched twin of gemv_nvfp4_multirow_kernel, existing for NUMERICAL PARITY not speed: the
// spec-verify chunk and M=1 decode compute the same projections with instruction-identical
// inner loops that differ in ONE thing, K-reduction width. Decode's multirow kernel gives
// each WARP an output row (32 partial sums, mi=lane;+=32); the verify chunk's kpar kernel
// gave the whole 128-thread BLOCK an output row (128 partial sums). Same math, different
// float rounding - reached the stop decision on Qwen3.8-27B-NVFP4 mtp_k=1, truncating 2/6
// answers after ~40 tokens (docs/LIMITATIONS.md). This kernel keeps the 32-lane warp
// partition so each row reproduces decode bit-for-bit, while still reading each weight
// micro-block once for all MR rows. Only fires for shapes where decode takes the multirow
// branch (use_multirow(): true for 10240x5120/12288x5120 q/k/v, false for
// 5120x5120/5120x17408 o/down, where the existing batched kernel already agrees).
template <int NR, int MR, bool kAcc = false>
__global__ void __launch_bounds__(kMRThreads) gemv_nvfp4_multirow_mb_kernel(
    const uint8_t* __restrict__ packed_data, const uint8_t* __restrict__ micro_scales, float tensor_scale,
    const half* __restrict__ x, half* __restrict__ y, int N_out, int K) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int K_half = K / 2;
    const int n_mb = K / kMicroBlockSize;

    const int row = blockIdx.x * NR + warp_id;
    if (row >= N_out || warp_id >= NR)
        return;
    pdl_wait();

    const uint8_t* row_packed = packed_data + (int64_t)row * K_half;
    const uint8_t* row_ms = micro_scales + (int64_t)row * n_mb;

    float acc[MR];
#pragma unroll
    for (int m = 0; m < MR; ++m)
        acc[m] = 0.0f;

    // Same stride/order/helper as warp_k_loop+dot_micro_block in gemv_nvfp4_multirow_kernel.
    // Do not "optimise" the accumulation without re-running tools/analysis/mtp_truncation_check.sh:
    // the parity IS the feature.
    for (int mi = lane; mi < n_mb; mi += 32) {
        const int byte_off = mi * 8;
        uint2 packed2 = *reinterpret_cast<const uint2*>(row_packed + byte_off);
        const uint8_t* pb = reinterpret_cast<const uint8_t*>(&packed2);
        const float cs = tensor_scale * fp8_e4m3_to_float_fast(row_ms[mi]);
#pragma unroll
        for (int m = 0; m < MR; ++m)
            acc[m] = __fmaf_rn(dot_micro_block(pb, x + (int64_t)m * K, byte_off * 2), cs, acc[m]);
    }

    pdl_trigger();
#pragma unroll
    for (int m = 0; m < MR; ++m) {
        const float total = warp_reduce(acc[m]);
        if (lane == 0) {
            half* yp = y + (int64_t)m * N_out + row;
            *yp = __float2half(kAcc ? total + __half2float(*yp) : total);
        }
    }
}

// FP16-output twin of gemv_nvfp4_kpar_mb_fp32_kernel for spec-verify chunk GEMMs (#998):
// one block per weight row n, decoded once and reused across the MR rows of this launch.
// kAcc adds into the existing output (cuBLAS beta=1 semantics) for the o/down residual-add
// GEMMs (#1055).
template <int MR, bool kAcc = false>
__global__ void __launch_bounds__(kKparThreads) gemv_nvfp4_kpar_mb_fp16_kernel(
    const uint8_t* __restrict__ packed_data, const uint8_t* __restrict__ micro_scales, float tensor_scale,
    const half* __restrict__ x, half* __restrict__ y, int N_out, int K) {
    const int n = blockIdx.x;
    if (n >= N_out)
        return;
    pdl_wait();
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

    pdl_trigger();
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

}  // namespace

// PDL registration + MaxL1 carveout for the batched-M kernels production reaches:
// gemv_nvfp4_kpar_mb_fp32_kernel<1..4> behind gemv_nvfp4_kpar_batched_fp32 (batched decode
// LM head). Kernel waits before its first global read, triggers after its last (core/pdl.h
// contract); until AUDIT_arch_2026 A2-2 it was instrumented but never registered, so
// pdl::launch fell through to <<<>>> with a default graph edge. multirow_mb and
// kpar_mb_fp16 kernels (test oracles only) stay unregistered.
void nvfp4_gemv_batched_pdl_register() {
#define NVFP4_MB_REGISTER(kern)                                                    \
    do {                                                                           \
        pdl::enable_kernel(kern);                                                  \
        cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, \
                             cudaSharedmemCarveoutMaxL1);                          \
    } while (0)
    NVFP4_MB_REGISTER(gemv_nvfp4_kpar_mb_fp32_kernel<1>);
    NVFP4_MB_REGISTER(gemv_nvfp4_kpar_mb_fp32_kernel<2>);
    NVFP4_MB_REGISTER(gemv_nvfp4_kpar_mb_fp32_kernel<3>);
    NVFP4_MB_REGISTER(gemv_nvfp4_kpar_mb_fp32_kernel<4>);
#undef NVFP4_MB_REGISTER
}

// Batched-M FP32 GEMV for the LM head at batch>1: y[n_act,N_out]=x[n_act,K]@A^T. Reads the
// weight matrix ONCE per launch (vs once per row in the old per-row M=1 loop); n_act is
// processed in power-of-two MR chunks so each launch reuses the weight across MR rows,
// typical decode batches (<=16) need one launch.
void gemv_nvfp4_kpar_batched_fp32(const NvFP4QuantResult& A, const half* x, float* y, int N_out, int K,
                                  int n_act, cudaStream_t stream) {
    const auto* pd = reinterpret_cast<const uint8_t*>(A.packed_data);
    const auto* ms = reinterpret_cast<const uint8_t*>(A.micro_scales);
    const float ts = A.tensor_scale;
    // MR is capped at 4: each accumulator row costs registers, beyond MR=4 the kernel spills
    // (measured 91 us/row at MR=4 vs 118 us/row at MR=16). For M>4 the weight re-reads per
    // MR=4 tile, but tiled still beats a larger-MR spill (M=16: 4x MR=4=1.45ms < 1x MR=16=1.89ms)
    // and the old per-row loop (16x M=1=4.2ms).
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

// Batched-M FP16 GEMM for spec-verify chunks (#998). MR cap of 4 mirrors the
// FP32 LM-head launcher: larger MR spills registers and loses to tiling.
void gemm_nvfp4_batched(const NvFP4QuantResult& A, const half* x, half* y, int N_out, int K, int n_act,
                        cudaStream_t stream) {
    const auto* pd = reinterpret_cast<const uint8_t*>(A.packed_data);
    // Numerical parity with the M=1 decode path, when decode would take the 32-lane multirow
    // branch for this shape (see gemv_nvfp4_multirow_mb_kernel): the 128-wide kpar reduction
    // below is a different float grouping of the same products, and on a speculative arm that
    // difference reaches the stop decision.
    if (nvfp4_verify_row_parity()) {
        constexpr int NR = 8;
        const int n_mb_p = K / kMicroBlockSize;
        const int mr_blocks_p = (N_out + NR - 1) / NR;
        if (use_multirow(n_mb_p, mr_blocks_p)) {
            const auto* msp = reinterpret_cast<const uint8_t*>(A.micro_scales);
            const float tsp = A.tensor_scale;
            int d = 0;
            while (d < n_act) {
                const int rem = n_act - d;
                const half* xm = x + (int64_t)d * K;
                half* ym = y + (int64_t)d * N_out;
                int used;
                if (rem >= 4) {
                    pdl::launch(gemv_nvfp4_multirow_mb_kernel<NR, 4, false>, dim3(mr_blocks_p),
                                dim3(kMRThreads), size_t(0), stream, pd, msp, tsp, xm, ym, N_out, K);
                    used = 4;
                } else if (rem == 3) {
                    pdl::launch(gemv_nvfp4_multirow_mb_kernel<NR, 3, false>, dim3(mr_blocks_p),
                                dim3(kMRThreads), size_t(0), stream, pd, msp, tsp, xm, ym, N_out, K);
                    used = 3;
                } else if (rem >= 2) {
                    pdl::launch(gemv_nvfp4_multirow_mb_kernel<NR, 2, false>, dim3(mr_blocks_p),
                                dim3(kMRThreads), size_t(0), stream, pd, msp, tsp, xm, ym, N_out, K);
                    used = 2;
                } else {
                    pdl::launch(gemv_nvfp4_multirow_mb_kernel<NR, 1, false>, dim3(mr_blocks_p),
                                dim3(kMRThreads), size_t(0), stream, pd, msp, tsp, xm, ym, N_out, K);
                    used = 1;
                }
                d += used;
            }
            return;
        }
    }
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
void gemm_nvfp4_batched_acc(const NvFP4QuantResult& A, const half* x, half* y, int N_out, int K, int n_act,
                            cudaStream_t stream) {
    const auto* pd = reinterpret_cast<const uint8_t*>(A.packed_data);
    // Numerical parity with the M=1 decode path, when decode would take the 32-lane multirow
    // branch for this shape (see gemv_nvfp4_multirow_mb_kernel): the 128-wide kpar reduction
    // below is a different float grouping of the same products, and on a speculative arm that
    // difference reaches the stop decision.
    if (nvfp4_verify_row_parity()) {
        constexpr int NR = 8;
        const int n_mb_p = K / kMicroBlockSize;
        const int mr_blocks_p = (N_out + NR - 1) / NR;
        if (use_multirow(n_mb_p, mr_blocks_p)) {
            const auto* msp = reinterpret_cast<const uint8_t*>(A.micro_scales);
            const float tsp = A.tensor_scale;
            int d = 0;
            while (d < n_act) {
                const int rem = n_act - d;
                const half* xm = x + (int64_t)d * K;
                half* ym = y + (int64_t)d * N_out;
                int used;
                if (rem >= 4) {
                    pdl::launch(gemv_nvfp4_multirow_mb_kernel<NR, 4, true>, dim3(mr_blocks_p),
                                dim3(kMRThreads), size_t(0), stream, pd, msp, tsp, xm, ym, N_out, K);
                    used = 4;
                } else if (rem == 3) {
                    pdl::launch(gemv_nvfp4_multirow_mb_kernel<NR, 3, true>, dim3(mr_blocks_p),
                                dim3(kMRThreads), size_t(0), stream, pd, msp, tsp, xm, ym, N_out, K);
                    used = 3;
                } else if (rem >= 2) {
                    pdl::launch(gemv_nvfp4_multirow_mb_kernel<NR, 2, true>, dim3(mr_blocks_p),
                                dim3(kMRThreads), size_t(0), stream, pd, msp, tsp, xm, ym, N_out, K);
                    used = 2;
                } else {
                    pdl::launch(gemv_nvfp4_multirow_mb_kernel<NR, 1, true>, dim3(mr_blocks_p),
                                dim3(kMRThreads), size_t(0), stream, pd, msp, tsp, xm, ym, N_out, K);
                    used = 1;
                }
                d += used;
            }
            return;
        }
    }
    const auto* ms = reinterpret_cast<const uint8_t*>(A.micro_scales);
    const float ts = A.tensor_scale;
    int done = 0;
    while (done < n_act) {
        const int rem = n_act - done;
        const half* xm = x + (int64_t)done * K;
        half* ym = y + (int64_t)done * N_out;
        int used;
        if (rem >= 4) {
            pdl::launch(gemv_nvfp4_kpar_mb_fp16_kernel<4, true>, dim3(N_out), dim3(kKparThreads), size_t(0),
                        stream, pd, ms, ts, xm, ym, N_out, K);
            used = 4;
        } else if (rem == 3) {
            pdl::launch(gemv_nvfp4_kpar_mb_fp16_kernel<3, true>, dim3(N_out), dim3(kKparThreads), size_t(0),
                        stream, pd, ms, ts, xm, ym, N_out, K);
            used = 3;
        } else if (rem >= 2) {
            pdl::launch(gemv_nvfp4_kpar_mb_fp16_kernel<2, true>, dim3(N_out), dim3(kKparThreads), size_t(0),
                        stream, pd, ms, ts, xm, ym, N_out, K);
            used = 2;
        } else {
            pdl::launch(gemv_nvfp4_kpar_mb_fp16_kernel<1, true>, dim3(N_out), dim3(kKparThreads), size_t(0),
                        stream, pd, ms, ts, xm, ym, N_out, K);
            used = 1;
        }
        done += used;
    }
}

}  // namespace imp
