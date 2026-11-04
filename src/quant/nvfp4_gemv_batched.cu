// NVFP4 GEMV/GEMM kernels serving MANY activation rows (spec-verify M=1..4, batched LM
// head). Split from nvfp4_gemv_dense.cu (2026-08-21): every kernel there computes ONE
// activation row against many weight rows, every kernel here computes MANY rows against the
// same weight read - also the boundary numerical-parity work runs along
// (gemv_nvfp4_multirow_mb_kernel reduces K exactly like the one-row side). Templated over
// MR, so compile-time isolation stops every touch from re-ptxasing four instantiations and
// dragging the single-row decode kernels through it.

#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_gemm_internal.cuh"
#include "core/pdl_launch.cuh"

#include <cuda_fp16.h>
#include "core/pdl_device.cuh"

namespace imp {
namespace {

// f16 x f16 + f32 in one rounding (SASS FHFMA), equal to __fmaf_rn on the converted floats: an f16
// product is exact in f32, so the HADD2.F32 per operand goes.
__device__ __forceinline__ float fma_f16_f32(uint32_t a, uint32_t b, float c) {
    float d;
    asm("fma.rn.f32.f16 %0, %1, %2, %3;"
        : "=f"(d)
        : "h"(static_cast<uint16_t>(a)), "h"(static_cast<uint16_t>(b)), "f"(c));
    return d;
}

// Weight rows per block / micro-blocks a thread loads ahead (Qwen3.8 head [248320 x 5120], 3-4 rows:
// one row per block 600 / 660 us -> 430 / 450 us; the x fragments serve all kKparNR weight rows).
constexpr int kKparNR = 4;
constexpr int kKparAhead = 3;

// Batched-M K-parallel GEMV (FP32 out): y[m,n]=A_nvfp4[n,:]@x[m,:] for the MR activation rows in this
// launch, kKparNR weight rows n per block, each weight read once for all MR rows. Per (n, m) the float
// op order is the 128-lane one of reduce_kpar (lane t: micro-blocks t, t+128, ...): every row's bits
// are independent of MR and kKparNR (tests/test_lm_head_draft_rows.cu pins them against a host replica).
template <int MR>
__global__ void __launch_bounds__(kKparThreads) gemv_nvfp4_kpar_mb_fp32_kernel(
    const uint8_t* __restrict__ packed_data, const uint8_t* __restrict__ micro_scales, float tensor_scale,
    const half* __restrict__ x, float* __restrict__ y, int N_out, int K) {
    const int n0 = blockIdx.x * kKparNR;
    pdl_wait();
    const int tid = threadIdx.x;
    const int n_mb = K / kMicroBlockSize;

    float acc[kKparNR][MR];
#pragma unroll
    for (int r = 0; r < kKparNR; ++r)
#pragma unroll
        for (int m = 0; m < MR; ++m)
            acc[r][m] = 0.0f;

    for (int base = tid; base < n_mb; base += kKparAhead * kKparThreads) {
        uint2 w[kKparNR][kKparAhead];
        uint32_t sb[kKparNR][kKparAhead];
#pragma unroll
        for (int r = 0; r < kKparNR; ++r) {
            const int n = min(n0 + r, N_out - 1);  // clamped rows are never stored
            const uint8_t* row_packed = packed_data + ((int64_t)n * (K / 2));
            const uint8_t* row_ms = micro_scales + ((int64_t)n * n_mb);
#pragma unroll
            for (int j = 0; j < kKparAhead; ++j) {
                const int mi = base + (j * kKparThreads);
                if (mi < n_mb) {
                    const int byte_off = mi * 8;
                    w[r][j] = *reinterpret_cast<const uint2*>(row_packed + byte_off);
                    sb[r][j] = row_ms[mi];
                }
            }
        }
#pragma unroll
        for (int j = 0; j < kKparAhead; ++j) {
            const int mi = base + (j * kKparThreads);
            if (mi < n_mb) {
                const int elem_base = mi * kMicroBlockSize;
                uint4 xv[MR][2];
#pragma unroll
                for (int m = 0; m < MR; ++m) {
                    const uint4* xm = reinterpret_cast<const uint4*>(x + ((int64_t)m * K) + elem_base);
                    xv[m][0] = xm[0];
                    xv[m][1] = xm[1];
                }
#pragma unroll
                for (int r = 0; r < kKparNR; ++r) {
                    // 16 FP4 weights decoded once as f16x2, reused across the MR activation rows.
                    const uint8_t* pb = reinterpret_cast<const uint8_t*>(&w[r][j]);
                    uint32_t wh[8];
#pragma unroll
                    for (int b = 0; b < 8; ++b)
                        asm("{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }"
                            : "=r"(wh[b])
                            : "r"(static_cast<uint32_t>(pb[b])));
                    const float cs = tensor_scale * fp8_e4m3_to_float_fast(static_cast<uint8_t>(sb[r][j]));
#pragma unroll
                    for (int m = 0; m < MR; ++m) {
                        const uint32_t xw[8] = {xv[m][0].x, xv[m][0].y, xv[m][0].z, xv[m][0].w,
                                                xv[m][1].x, xv[m][1].y, xv[m][1].z, xv[m][1].w};
                        float d = 0.0f;
#pragma unroll
                        for (int b = 0; b < 8; ++b) {
                            d = fma_f16_f32(wh[b], xw[b], d);
                            d = fma_f16_f32(wh[b] >> 16, xw[b] >> 16, d);
                        }
                        acc[r][m] = __fmaf_rn(d, cs, acc[r][m]);
                    }
                }
            }
        }
    }

    pdl_trigger();
    // Same warp tree and warp order as reduce_kpar, all kKparNR x MR sums behind one barrier.
    __shared__ float warp_sums[kKparNR * MR][kKparWarps];
#pragma unroll
    for (int r = 0; r < kKparNR; ++r)
#pragma unroll
        for (int m = 0; m < MR; ++m) {
            float v = acc[r][m];
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                v += __shfl_down_sync(0xFFFFFFFF, v, offset);
            if ((tid & 31) == 0)
                warp_sums[(r * MR) + m][tid / 32] = v;
        }
    __syncthreads();
    if (tid < kKparNR * MR) {
        const int r = tid / MR;
        const int m = tid % MR;
        float total = warp_sums[tid][0];
#pragma unroll
        for (int w = 1; w < kKparWarps; w++)
            total += warp_sums[tid][w];
        if (n0 + r < N_out)
            y[((int64_t)m * N_out) + n0 + r] = total;
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

    const int row = (blockIdx.x * NR) + warp_id;
    if (row >= N_out || warp_id >= NR)
        return;
    pdl_wait();

    const uint8_t* row_packed = packed_data + ((int64_t)row * K_half);
    const uint8_t* row_ms = micro_scales + ((int64_t)row * n_mb);

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
            acc[m] = __fmaf_rn(dot_micro_block(pb, x + ((int64_t)m * K), byte_off * 2), cs, acc[m]);
    }

    pdl_trigger();
#pragma unroll
    for (int m = 0; m < MR; ++m) {
        const float total = warp_reduce(acc[m]);
        if (lane == 0) {
            half* yp = y + ((int64_t)m * N_out) + row;
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
    const uint8_t* row_packed = packed_data + ((int64_t)n * (K / 2));
    const uint8_t* row_ms = micro_scales + ((int64_t)n * n_mb);

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
        // #2218 bounded: b * 2 <= 14 (loop b < 8 literal, float wf[16] extent)
        for (int b = 0; b < 8; ++b) {
            uint32_t w_fp16x2;
            asm("{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }"
                : "=r"(w_fp16x2)
                : "r"(static_cast<uint32_t>(pb[b])));
            float2 wf2 = __half22float2(*reinterpret_cast<const half2*>(&w_fp16x2));
            wf[static_cast<ptrdiff_t>(b * 2)] = wf2.x;
            wf[(b * 2) + 1] = wf2.y;
        }
        float cs = tensor_scale * fp8_e4m3_to_float_fast(row_ms[mi]);
        const int elem_base = byte_off * 2;
#pragma unroll
        for (int m = 0; m < MR; ++m) {
            const half* xm = x + ((int64_t)m * K) + elem_base;
            const uint4 xv0 = *reinterpret_cast<const uint4*>(xm);
            const uint4 xv1 = *reinterpret_cast<const uint4*>(xm + 8);
            half2 xh[8];
            *reinterpret_cast<uint4*>(&xh[0]) = xv0;
            *reinterpret_cast<uint4*>(&xh[4]) = xv1;
            float d = 0.0f;
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                float2 xf = __half22float2(xh[b]);
                d = __fmaf_rn(wf[static_cast<ptrdiff_t>(b * 2)], xf.x, d);
                d = __fmaf_rn(wf[(b * 2) + 1], xf.y, d);
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
            half* yp = y + ((int64_t)m * N_out) + n;
            if (kAcc)
                total += __half2float(*yp);
            *yp = __float2half(total);
        }
        __syncthreads();  // reuse warp_sums for the next activation row
    }
}

}  // namespace

namespace {

template <bool kAcc>
void register_verify_mb() {
    constexpr int NR = 8;  // = gemm_nvfp4_batched(_acc)
    pdl::enable_kernel(gemv_nvfp4_multirow_mb_kernel<NR, 1, kAcc>);
    pdl::enable_kernel(gemv_nvfp4_multirow_mb_kernel<NR, 2, kAcc>);
    pdl::enable_kernel(gemv_nvfp4_multirow_mb_kernel<NR, 3, kAcc>);
    pdl::enable_kernel(gemv_nvfp4_multirow_mb_kernel<NR, 4, kAcc>);
    pdl::enable_kernel(gemv_nvfp4_kpar_mb_fp16_kernel<1, kAcc>);
    pdl::enable_kernel(gemv_nvfp4_kpar_mb_fp16_kernel<2, kAcc>);
    pdl::enable_kernel(gemv_nvfp4_kpar_mb_fp16_kernel<3, kAcc>);
    pdl::enable_kernel(gemv_nvfp4_kpar_mb_fp16_kernel<4, kAcc>);
}

}  // namespace

// PDL for every batched-M kernel: kpar_mb_fp32 (batched decode LM head) and
// the multirow_mb / kpar_mb_fp16 spec-verify chunk kernels (gemm_nvfp4_batched(_acc)). Each
// waits before its first global read (core/pdl_device.cuh contract).
void nvfp4_gemv_batched_pdl_register() {
    pdl::enable_kernel(gemv_nvfp4_kpar_mb_fp32_kernel<1>);
    pdl::enable_kernel(gemv_nvfp4_kpar_mb_fp32_kernel<2>);
    pdl::enable_kernel(gemv_nvfp4_kpar_mb_fp32_kernel<3>);
    pdl::enable_kernel(gemv_nvfp4_kpar_mb_fp32_kernel<4>);
    register_verify_mb<false>();
    register_verify_mb<true>();
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
    const dim3 grid((N_out + kKparNR - 1) / kKparNR);
    int done = 0;
    while (done < n_act) {
        const int rem = n_act - done;
        const half* xm = x + ((int64_t)done * K);
        float* ym = y + ((int64_t)done * N_out);
        int used;
        if (rem >= 4) {
            pdl::launch(gemv_nvfp4_kpar_mb_fp32_kernel<4>, grid, dim3(kKparThreads), size_t(0), stream, pd,
                        ms, ts, xm, ym, N_out, K);
            used = 4;
        } else if (rem == 3) {
            // Bucket-3 verify lm_head: one sweep instead of <2>+<1> (#1055).
            pdl::launch(gemv_nvfp4_kpar_mb_fp32_kernel<3>, grid, dim3(kKparThreads), size_t(0), stream, pd,
                        ms, ts, xm, ym, N_out, K);
            used = 3;
        } else if (rem >= 2) {
            pdl::launch(gemv_nvfp4_kpar_mb_fp32_kernel<2>, grid, dim3(kKparThreads), size_t(0), stream, pd,
                        ms, ts, xm, ym, N_out, K);
            used = 2;
        } else {
            pdl::launch(gemv_nvfp4_kpar_mb_fp32_kernel<1>, grid, dim3(kKparThreads), size_t(0), stream, pd,
                        ms, ts, xm, ym, N_out, K);
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
                const half* xm = x + ((int64_t)d * K);
                half* ym = y + ((int64_t)d * N_out);
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
        const half* xm = x + ((int64_t)done * K);
        half* ym = y + ((int64_t)done * N_out);
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
                const half* xm = x + ((int64_t)d * K);
                half* ym = y + ((int64_t)d * N_out);
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
        const half* xm = x + ((int64_t)done * K);
        half* ym = y + ((int64_t)done * N_out);
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
