// =============================================================================
// quantize_fp16_nvfp4_moe_native.cu
//
// Per-expert FP16 -> NVFP4 quantization with native row-major UE4M3 scale
// layout.  Produces the layout consumed by gemm_grouped_nvfp4_smallM (the
// smallM prefill GEMM for MoE activations).
//
// Algorithm matches quantize_fp16_to_nvfp4 (nvfp4_quant.cu) exactly so that
// bit-exact equivalence holds for a single-expert problem:
//   - Two-level scaling: tensor_scale (per expert) + micro_scale (per 16 elems)
//   - FP8 UE4M3 micro-scales, HW E2M1 FP4 conversion (cvt.rn.satfinite.e2m1x2)
//   - Output layout: [M_e, K/2] packed FP4 + [M_e, K/16] linear UE4M3 scales
// =============================================================================

#include "compute/quantize_fp16_nvfp4_moe_native.h"
#include "quant/fp8_utils.cuh"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cfloat>

namespace imp {

// ---------------------------------------------------------------------------
// Constants — must match nvfp4_quant.cu so the two-level scaling is identical.
// ---------------------------------------------------------------------------
static constexpr int kNativeMicroBlockSize = 16;  // elements per micro-block
static constexpr float kNativeFP4E2M1Max = 6.0f;  // max representable in E2M1

// ---------------------------------------------------------------------------
// HW FP4 pair conversion (identical to pack_fp4_pair_hw in gemm_cutlass_sm120.cu
// and nvfp4_pack_pair_hw in nvfp4_quant.cu).  Low nibble = v0, high nibble = v1.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint8_t native_pack_fp4_pair(float v0, float v1) {
#if __CUDA_ARCH__ >= 1200
    uint32_t out;
    asm volatile(
        "{ .reg .b8 b;\n"
        "  cvt.rn.satfinite.e2m1x2.f32 b, %2, %1;\n"
        "  cvt.u32.u8 %0, b; }\n"
        : "=r"(out)
        : "f"(v0), "f"(v1));
    return static_cast<uint8_t>(out);
#else
    // Software fallback for non-sm120 builds (used by test-unit CPU filter etc.)
    auto abs_to_code = [](float v) -> uint8_t {
        float a = fabsf(v);
        uint8_t c = (a >= 0.25f) + (a >= 0.75f) + (a >= 1.25f) + (a >= 1.75f) +
                    (a >= 2.5f)  + (a >= 3.5f)  + (a >= 5.0f);
        return c;
    };
    uint8_t c0 = (v0 < 0.0f ? 0x8u : 0u) | abs_to_code(v0);
    uint8_t c1 = (v1 < 0.0f ? 0x8u : 0u) | abs_to_code(v1);
    return (c1 << 4) | c0;
#endif
}

// ---------------------------------------------------------------------------
// Kernel 1: per-expert absmax reduction.
//
// Grid:   blockIdx.x = expert index
// Block:  256 threads
// Each CTA reduces all rows [offsets[e], offsets[e+1]) over all K columns and
// writes a single float via atomicMax to d_absmax[e].
//
// Uses uint32 atomicMax on IEEE754 bit pattern of non-negative floats —
// identical trick to absmax_kernel in nvfp4_quant.cu.
// ---------------------------------------------------------------------------
__global__ void nvfp4_moe_native_absmax_kernel(
    const __half* __restrict__ src,      // [expanded, K]
    const int* __restrict__ offsets,     // [ne+1]
    int K,
    float* __restrict__ d_absmax)        // [ne], pre-zeroed
{
    int e = blockIdx.x;
    int M0 = offsets[e];
    int M1 = offsets[e + 1];
    int M_e = M1 - M0;
    if (M_e <= 0)
        return;

    int64_t n_elem = (int64_t)M_e * K;

    float local_max = 0.0f;
    for (int64_t i = threadIdx.x; i < n_elem; i += blockDim.x) {
        int64_t src_idx = (int64_t)M0 * K + i;
        float v = fabsf(__half2float(src[src_idx]));
        if (v > local_max)
            local_max = v;
    }

    // Shared-memory block reduction.
    __shared__ float smem[256];
    smem[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s && smem[threadIdx.x + s] > smem[threadIdx.x])
            smem[threadIdx.x] = smem[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        unsigned int* ptr = reinterpret_cast<unsigned int*>(d_absmax + e);
        atomicMax(ptr, __float_as_uint(smem[0]));
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: per-expert FP16 -> NVFP4 quantize (native row-major output).
//
// One thread per micro-block of 16 elements.
// Grid:   blockIdx.x = expert, blockIdx.y = slice-of-work
// Block:  256 threads
//
// Scale convention (matches quantize_micro_block_nvfp4 in nvfp4_quant.cu):
//   tensor_scale  = absmax / 6.0  (if absmax==0 → 1.0 to avoid /0)
//   micro_scale_f = local_absmax / (tensor_scale * 6.0)
//   micro_scale_f = clamp(micro_scale_f, 1/512, 448)  → FP8 UE4M3 byte
//   actual_scale  = fp8_e4m3_to_float(ue4m3_byte)     (0 → 1/512 fallback)
//   fp4_val       = input_val / (tensor_scale * actual_scale)
//
// Output layout (native, row-major dense):
//   packed_e[m * (K/2)  + kb * 8 + i/2]  = packed nibble byte
//   sf_e   [m * (K/16) + kb]             = UE4M3 byte
// ---------------------------------------------------------------------------
__global__ void nvfp4_moe_native_quant_kernel(
    const __half* __restrict__ src,          // [expanded, K]
    void* const* __restrict__ d_packed,      // [ne] per-expert packed FP4
    void* const* __restrict__ d_sf,          // [ne] per-expert UE4M3
    const int* __restrict__ offsets,         // [ne+1]
    const float* __restrict__ d_absmax,      // [ne]
    int K)
{
    int e = blockIdx.x;
    int M0 = offsets[e];
    int M1 = offsets[e + 1];
    int M_e = M1 - M0;
    if (M_e <= 0)
        return;

    auto* packed_e = static_cast<uint8_t*>(d_packed[e]);
    auto* sf_e     = static_cast<uint8_t*>(d_sf[e]);

    float absmax_e = d_absmax[e];
    float tensor_scale = (absmax_e == 0.0f) ? 1.0f : (absmax_e / kNativeFP4E2M1Max);

    int K_blocks = K / kNativeMicroBlockSize;  // number of micro-blocks per row
    int total_mb = M_e * K_blocks;             // total micro-blocks for this expert

    for (int t = (int)(blockIdx.y * blockDim.x) + threadIdx.x; t < total_mb;
         t += (int)(gridDim.y * blockDim.x)) {
        int m  = t / K_blocks;
        int kb = t % K_blocks;

        // Load 16 FP16 values via vectorized half2 loads.
        const half2* src_h2 = reinterpret_cast<const half2*>(
            src + (int64_t)(M0 + m) * K + kb * kNativeMicroBlockSize);

        float vals[kNativeMicroBlockSize];
        float local_absmax = 0.0f;
#pragma unroll
        for (int i = 0; i < kNativeMicroBlockSize / 2; i++) {
            half2 h2 = src_h2[i];
            vals[i * 2]     = __half2float(h2.x);
            vals[i * 2 + 1] = __half2float(h2.y);
            local_absmax = fmaxf(local_absmax,
                                 fmaxf(fabsf(vals[i * 2]), fabsf(vals[i * 2 + 1])));
        }

        // Micro-scale — same formula and clamping as quantize_micro_block_nvfp4.
        constexpr float kFP8E4M3Max = 448.0f;
        float micro_scale_f = local_absmax / (tensor_scale * kNativeFP4E2M1Max);
        if (micro_scale_f < (1.0f / 512.0f))
            micro_scale_f = 1.0f / 512.0f;
        if (micro_scale_f > kFP8E4M3Max)
            micro_scale_f = kFP8E4M3Max;

        uint8_t ue4m3 = float_to_fp8_e4m3(micro_scale_f);

        // Reconstruct actual_scale from UE4M3 byte for consistent quantization.
        float actual_scale = fp8_e4m3_to_float(ue4m3);
        if (actual_scale == 0.0f)
            actual_scale = 1.0f / 512.0f;

        // Write UE4M3 scale — native row-major: sf[m * K_blocks + kb]
        sf_e[m * K_blocks + kb] = ue4m3;

        // Quantize 16 FP16 -> 8 packed bytes — native row-major:
        // packed[m * (K/2) + kb * 8 + byte_idx]
        float inv_combined = 1.0f / (tensor_scale * actual_scale);
        uint8_t* packed_at = packed_e + (int64_t)m * (K / 2) + kb * (kNativeMicroBlockSize / 2);
#pragma unroll
        for (int i = 0; i < kNativeMicroBlockSize; i += 2) {
            float s0 = vals[i]     * inv_combined;
            float s1 = vals[i + 1] * inv_combined;
            packed_at[i / 2] = native_pack_fp4_pair(s0, s1);
        }
    }
}

// ---------------------------------------------------------------------------
// Host entry point
// ---------------------------------------------------------------------------
void quantize_fp16_to_nvfp4_moe_native(
    const __half* src_fp16,
    void* const* d_packed_ptrs,
    void* const* d_sf_ptrs,
    const int* d_expert_offsets,
    int expanded,
    int K,
    int n_experts,
    cudaStream_t stream)
{
    if (n_experts <= 0 || K <= 0 || expanded < 0)
        return;
    if ((K % kNativeMicroBlockSize) != 0) {
        IMP_LOG_ERROR("quantize_fp16_to_nvfp4_moe_native: K=%d not divisible by 16", K);
        return;
    }
    if (expanded == 0)
        return;

    // Copy per-expert pointer arrays to device (host arrays passed in).
    void** d_packed_dev = nullptr;
    void** d_sf_dev     = nullptr;
    float* d_absmax     = nullptr;
    IMP_CUDA_CHECK_LOG(cudaMallocAsync(&d_packed_dev, sizeof(void*) * n_experts, stream));
    IMP_CUDA_CHECK_LOG(cudaMallocAsync(&d_sf_dev,     sizeof(void*) * n_experts, stream));
    IMP_CUDA_CHECK_LOG(cudaMallocAsync(&d_absmax,     sizeof(float) * n_experts, stream));
    IMP_CUDA_CHECK_LOG(cudaMemsetAsync(d_absmax, 0,   sizeof(float) * n_experts, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_packed_dev, d_packed_ptrs, sizeof(void*) * n_experts,
                                       cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_sf_dev,     d_sf_ptrs,     sizeof(void*) * n_experts,
                                       cudaMemcpyHostToDevice, stream));

    // Pass 1: per-expert absmax (one CTA per expert, 256 threads).
    {
        dim3 grid(n_experts);
        dim3 block(256);
        nvfp4_moe_native_absmax_kernel<<<grid, block, 0, stream>>>(
            src_fp16, d_expert_offsets, K, d_absmax);
    }

    // Pass 2: quantize (one CTA-x per expert, CTA-y slices for parallelism).
    {
        dim3 block(256);
        dim3 grid(n_experts, 16);
        nvfp4_moe_native_quant_kernel<<<grid, block, 0, stream>>>(
            src_fp16,
            const_cast<void* const*>(d_packed_dev),
            const_cast<void* const*>(d_sf_dev),
            d_expert_offsets,
            d_absmax,
            K);
    }

    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_packed_dev, stream));
    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_sf_dev,     stream));
    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_absmax,     stream));
}

}  // namespace imp
