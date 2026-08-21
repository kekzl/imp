#ifndef IMP_QUANT_NVFP4_GEMM_INTERNAL_CUH
#define IMP_QUANT_NVFP4_GEMM_INTERNAL_CUH

// Internal shared device helpers + tuning constants for the NVFP4 GEMV kernels.
// Split out of nvfp4_gemm.cu so the per-family kernel TUs (dense / fused / moe)
// stay under the kernel .cu size gate. MOVED VERBATIM — do not rewrite, reorder,
// or change any numeric behavior; the kernels are hot-path numeric code and must
// stay bit-identical.

#include "quant/fp8_utils.cuh"
#include "runtime/process_diag.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace imp {

static constexpr int kMicroBlockSize = 16;
static constexpr int kKparWarps = 4;
static constexpr int kKparThreads = kKparWarps * 32;  // 128

// Multi-row kernel: 8 warps, NR rows per block for small-K models.
// Reduces launch overhead and improves SM occupancy.
static constexpr int kMRWarps = 8;
static constexpr int kMRThreads = kMRWarps * 32;  // 256

// Minimum blocks/SM for multi-row to be efficient. Below this, kpar is better
// because more blocks (M instead of M/NR) give higher warp occupancy.
static constexpr int kMinBlocksPerSM = 6;

// Cached SM count for dispatch decisions.
// speculative.verify_row_parity. Kernel TUs do not include runtime/config.h;
// process_diag is the seam. NOT cached in a function-local static like
// nvfp4_n_sms(): this is read once per GEMM launch on the host, so the read
// costs nothing, and caching it would make the knob impossible to toggle in a
// test - which is how the first version of this was written.
static bool nvfp4_verify_row_parity() { return imp::process_diag_verify_row_parity(); }

static int nvfp4_n_sms() {
    static int n_sms = 0;
    if (__builtin_expect(n_sms == 0, 0)) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        n_sms = prop.multiProcessorCount;
    }
    return n_sms;
}

// Returns true if multi-row should be used: K is small enough AND there are
// enough blocks for good SM occupancy (>= kMinBlocksPerSM blocks/SM).
static bool use_multirow(int n_mb, int mr_blocks) {
    return n_mb <= 512 && mr_blocks >= kMinBlocksPerSM * nvfp4_n_sms();
}

// fp8_e4m3_to_float_fast moved to quant/fp8_utils.cuh (shared with MXFP4 +
// CUTLASS block-scale kernels). Single correct implementation, denorm-safe.

// Process one micro-block (8 packed bytes = 16 FP4 values).
// Returns unscaled dot product: sum(dequant(nibble) * activation).
//
// HW FP4 decode via `cvt.rn.f16x2.e2m1x2` — single PTX instruction converts
// one byte (2 packed E2M1 nibbles) to a packed f16x2. Replaces the prior
// 8-op-per-byte prmt+shift+OR cascade. Verified supported on sm_120f /
// CUDA 13.2 by tools/analysis/ptx_cvt_survey.sh.
//
// Layout: low nibble (bits 0-3) decodes to f16x2.x (lower half),
//         high nibble (bits 4-7) decodes to f16x2.y (upper half).
__device__ __forceinline__ float dot_micro_block(const uint8_t* __restrict__ pb, const half* __restrict__ x,
                                                 int elem_base) {
    // x slice for one micro-block: 16 halfs = 32 B, loaded as two 16-B
    // vectors. The previous per-pair half2 loads issued 8 narrow 4-B
    // requests at a 32-B lane stride — each request touched 32 distinct
    // sectors and the 8 unrolled iterations re-touched the same sectors,
    // an 8x L1TEX sector-access amplification on x that saturated the
    // L1TEX data path (~90%) while DRAM idled at ~70% (#667 lever-3
    // profile: x re-reads were ~77% of all GEMV L1TEX sector accesses).
    // elem_base is a multiple of 16 halfs, so both loads are 16-B aligned.
    const uint4 xv0 = *reinterpret_cast<const uint4*>(x + elem_base);
    const uint4 xv1 = *reinterpret_cast<const uint4*>(x + elem_base + 8);
    half2 xh[8];
    *reinterpret_cast<uint4*>(&xh[0]) = xv0;
    *reinterpret_cast<uint4*>(&xh[4]) = xv1;

    float acc = 0.0f;
#pragma unroll
    for (int b = 0; b < 8; b++) {
        uint32_t byte_val = pb[b];
        uint32_t w_fp16x2;
        asm("{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }" : "=r"(w_fp16x2) : "r"(byte_val));

        const half2 wh = *reinterpret_cast<const half2*>(&w_fp16x2);
        const float2 xf = __half22float2(xh[b]);
        const float2 wf = __half22float2(wh);
        acc = __fmaf_rn(wf.x, xf.x, acc);
        acc = __fmaf_rn(wf.y, xf.y, acc);
    }
    return acc;
}

// Core GEMV loop: accumulates dot(row, x) for one row.
// Uses prmt register-based LUT (no shared memory needed).
__device__ __forceinline__ float gemv_nvfp4_row(const uint8_t* __restrict__ row_packed,
                                                const uint8_t* __restrict__ row_ms, float tensor_scale,
                                                const half* __restrict__ x, int n_mb, int tid) {
    float acc = 0.0f;
    for (int mi = tid; mi < n_mb; mi += kKparThreads) {
        int byte_off = mi * 8;
        uint2 packed2 = *reinterpret_cast<const uint2*>(row_packed + byte_off);
        const uint8_t* pb = reinterpret_cast<const uint8_t*>(&packed2);
        float cs = tensor_scale * fp8_e4m3_to_float_fast(row_ms[mi]);
        float local_dot = dot_micro_block(pb, x, byte_off * 2);
        acc = __fmaf_rn(local_dot, cs, acc);
    }
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

// Shared memory for K-parallel reduction (warp sums only).
// Non-SwiGLU/GeGLU kernels use prmt register LUT — no smem LUT needed.
struct SmemKpar {
    float warp_sums[kKparWarps];
};

// Shared memory LUT initializer for SwiGLU/GeGLU kernels only.
__device__ __forceinline__ void init_lut(float* s_lut, int tid) {
    if (tid < 16) {
        constexpr float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        s_lut[tid] = (tid < 8) ? kMag[tid] : -kMag[tid & 7];
    }
}

// Warp-level K-parallel micro-block loop: accumulates dot(row, x) for one row.
// DotFn signature: float(const uint8_t* pb, int elem_offset)
// Used by multi-row kernels where each warp (32 threads) handles one row.
template <typename DotFn>
__device__ __forceinline__ float warp_k_loop(const uint8_t* __restrict__ row_packed,
                                             const uint8_t* __restrict__ row_ms, float tensor_scale, int n_mb,
                                             int lane, DotFn dot_fn) {
    // (A two-micro-blocks-per-lane variant — 16-B weight vector + 2-B scale
    // pair for deeper memory-level parallelism — measured exactly neutral
    // here: the kernels already sit at the practical GDDR7 ceiling once the
    // x sector amplification is gone. Keep the simple form; it also keeps
    // the accumulation order bit-identical to the pre-vectorization kernel.)
    float acc = 0.0f;
    for (int mi = lane; mi < n_mb; mi += 32) {
        int byte_off = mi * 8;
        uint2 packed2 = *reinterpret_cast<const uint2*>(row_packed + byte_off);
        const uint8_t* pb = reinterpret_cast<const uint8_t*>(&packed2);
        float cs = tensor_scale * fp8_e4m3_to_float_fast(row_ms[mi]);
        acc = __fmaf_rn(dot_fn(pb, byte_off * 2), cs, acc);
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

// SwiGLU micro-block: reads gate[16] and up[16], computes silu(gate)*up,
// then dots with weight nibbles.
__device__ __forceinline__ float dot_micro_block_swiglu(const uint8_t* __restrict__ pb,
                                                        const half* __restrict__ gate,
                                                        const half* __restrict__ up, int elem_base,
                                                        const float* s_lut) {
    // 16-B vector loads for the gate/up slices (see dot_micro_block — the
    // narrow per-pair half2 loads were the GEMV L1TEX sector-amplification).
    half2 ghv[8], uhv[8];
    *reinterpret_cast<uint4*>(&ghv[0]) = *reinterpret_cast<const uint4*>(gate + elem_base);
    *reinterpret_cast<uint4*>(&ghv[4]) = *reinterpret_cast<const uint4*>(gate + elem_base + 8);
    *reinterpret_cast<uint4*>(&uhv[0]) = *reinterpret_cast<const uint4*>(up + elem_base);
    *reinterpret_cast<uint4*>(&uhv[4]) = *reinterpret_cast<const uint4*>(up + elem_base + 8);

    float acc = 0.0f;
#pragma unroll
    for (int b = 0; b < 8; b++) {
        const float2 gf = __half22float2(ghv[b]);
        const float2 uf = __half22float2(uhv[b]);
        // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        float s0 = gf.x / (1.0f + expf(-gf.x)) * uf.x;
        float s1 = gf.y / (1.0f + expf(-gf.y)) * uf.y;
        acc = __fmaf_rn(s_lut[pb[b] & 0x0F], s0, acc);
        acc = __fmaf_rn(s_lut[pb[b] >> 4], s1, acc);
    }
    return acc;
}

// GeGLU micro-block: reads gate[16] and up[16], computes gelu_tanh(gate)*up.
__device__ __forceinline__ float dot_micro_block_geglu(const uint8_t* __restrict__ pb,
                                                       const half* __restrict__ gate,
                                                       const half* __restrict__ up, int elem_base,
                                                       const float* s_lut) {
    constexpr float SQRT_2_PI = 0.7978845608028654f;
    constexpr float COEFF = 0.044715f;
    half2 ghv[8], uhv[8];
    *reinterpret_cast<uint4*>(&ghv[0]) = *reinterpret_cast<const uint4*>(gate + elem_base);
    *reinterpret_cast<uint4*>(&ghv[4]) = *reinterpret_cast<const uint4*>(gate + elem_base + 8);
    *reinterpret_cast<uint4*>(&uhv[0]) = *reinterpret_cast<const uint4*>(up + elem_base);
    *reinterpret_cast<uint4*>(&uhv[4]) = *reinterpret_cast<const uint4*>(up + elem_base + 8);

    float acc = 0.0f;
#pragma unroll
    for (int b = 0; b < 8; b++) {
        const float2 gf = __half22float2(ghv[b]);
        const float2 uf = __half22float2(uhv[b]);
        float g0 = gf.x * 0.5f * (1.0f + tanhf(SQRT_2_PI * (gf.x + COEFF * gf.x * gf.x * gf.x)));
        float g1 = gf.y * 0.5f * (1.0f + tanhf(SQRT_2_PI * (gf.y + COEFF * gf.y * gf.y * gf.y)));
        acc = __fmaf_rn(s_lut[pb[b] & 0x0F], g0 * uf.x, acc);
        acc = __fmaf_rn(s_lut[pb[b] >> 4], g1 * uf.y, acc);
    }
    return acc;
}

__device__ __forceinline__ float gemv_nvfp4_row_swiglu(const uint8_t* __restrict__ row_packed,
                                                       const uint8_t* __restrict__ row_ms, float tensor_scale,
                                                       const half* __restrict__ gate,
                                                       const half* __restrict__ up, int n_mb, int tid,
                                                       const float* s_lut) {
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

}  // namespace imp

#endif  // IMP_QUANT_NVFP4_GEMM_INTERNAL_CUH
