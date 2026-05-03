// =============================================================================
// nvfp4_quant_ref.cu -- Reference NVFP4 quantization (round-trip validation)
// =============================================================================
//
// Port of SageAttention3's scaled_fp4_quant_kernel, simplified to a linear
// storage layout. Used to validate that the FP16→NVFP4 quant math
// produces sensible values before Project B's Stage-3 integration
// into the actual FMHA kernel (which needs the HW scale-interleaving).
//
// Key PTX instruction: cvt.rn.satfinite.e2m1x2.f32
//   (sm_120 + CUDA 13.2 — verified by probe commit b9ec21a)
// =============================================================================

#include "compute/nvfp4_quant_ref.h"
#include <cuda_fp8.h>
#include <cstdint>

namespace imp {

// ---------------------------------------------------------------------------
// FP32 → 2× E2M1 nibbles (packed in one byte). Hardware instruction.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t fp32x8_to_e2m1x8(const float2* vals /* [4] */) {
    // Produces 8 E2M1 nibbles = 4 bytes = 1 uint32.
    uint32_t val;
    asm volatile(
        "{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"  // pair 0: (x0, y0) low-high
        "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
        "mov.b32 %0, {b0, b1, b2, b3};\n"
        "}"
        : "=r"(val)
        : "f"(vals[0].x), "f"(vals[0].y), "f"(vals[1].x), "f"(vals[1].y), "f"(vals[2].x), "f"(vals[2].y),
          "f"(vals[3].x), "f"(vals[3].y));
    return val;
}

// ---------------------------------------------------------------------------
// E2M1 (4-bit) → FP32 LUT. E2M1 values: {±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}.
// Code layout (SageAttention3 & NVFP4 standard):
//   bit 3 = sign, bits 2..0 = magnitude code 0..7
//   magnitude: 0→0, 1→0.5, 2→1.0, 3→1.5, 4→2.0, 5→3.0, 6→4.0, 7→6.0
// ---------------------------------------------------------------------------
__device__ __forceinline__ float e2m1_nibble_to_fp32(uint8_t nib) {
    static const float mags[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float m = mags[nib & 0x7];
    return (nib & 0x8) ? -m : m;
}

// ---------------------------------------------------------------------------
// Quant kernel: each thread handles 16 FP16 elements.
//   blockDim.x = threads; gridDim.x covers ceil(n_elements / (16 * blockDim.x))
// ---------------------------------------------------------------------------
__global__ void nvfp4_quant_linear_kernel(const half* __restrict__ input,
                                          uint8_t* __restrict__ nvfp4_out,  // [n_elements / 2]
                                          uint8_t* __restrict__ sf_out,     // [n_elements / 16]
                                          int n_elements) {
    const int group_id = blockIdx.x * blockDim.x + threadIdx.x;  // one group = 16 FP16 elems
    const int start = group_id * 16;
    if (start >= n_elements)
        return;

    // 1. Load 16 FP16 elements into 8 half2 (via vectorized cast).
    half2 h2[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int idx = start + i * 2;
        h2[i] = __halves2half2((idx < n_elements) ? input[idx] : __float2half(0.0f),
                               (idx + 1 < n_elements) ? input[idx + 1] : __float2half(0.0f));
    }

    // 2. Per-group absmax (over 16 elements).
    half2 maxabs = __habs2(h2[0]);
#pragma unroll
    for (int i = 1; i < 8; ++i) {
        maxabs = __hmax2(maxabs, __habs2(h2[i]));
    }
    float vec_max = float(__hmax(maxabs.x, maxabs.y));

    // 3. Scale (FP8 UE4M3). E2M1 representable max = 6.0 → sc = max / 6.
    //    Round-trip through FP8 so quantized scale matches storage.
    float sc = vec_max / 6.0f;
    uint8_t sc_fp8;
    reinterpret_cast<__nv_fp8_e4m3&>(sc_fp8) = __nv_fp8_e4m3(sc);
    sc = float(reinterpret_cast<__nv_fp8_e4m3&>(sc_fp8));
    float inv_sc = (sc == 0.0f) ? 0.0f : 1.0f / sc;

    // 4. Apply inverse scale and convert to float2 pairs.
    float2 fp2[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        fp2[i] = __half22float2(h2[i]);
        fp2[i].x = fp2[i].x * inv_sc;
        fp2[i].y = fp2[i].y * inv_sc;
    }

    // 5. Pack: 4 float2 → 1 uint32 (8 nibbles). Two uint32 per group of 16.
    uint32_t lo = fp32x8_to_e2m1x8(&fp2[0]);
    uint32_t hi = fp32x8_to_e2m1x8(&fp2[4]);

    // 6. Store. 16 nibbles = 8 bytes per group.
    uint32_t* dst = reinterpret_cast<uint32_t*>(nvfp4_out + start / 2);
    dst[0] = lo;
    dst[1] = hi;

    // 7. Store scale.
    sf_out[group_id] = sc_fp8;
}

// ---------------------------------------------------------------------------
// Dequant kernel: inverse, for round-trip testing.
// ---------------------------------------------------------------------------
__global__ void nvfp4_dequant_linear_kernel(const uint8_t* __restrict__ nvfp4_in,
                                            const uint8_t* __restrict__ sf_in, half* __restrict__ output,
                                            int n_elements) {
    const int group_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int start = group_id * 16;
    if (start >= n_elements)
        return;

    // Load scale (FP8 UE4M3 → FP32).
    uint8_t sc_fp8 = sf_in[group_id];
    float sc = float(reinterpret_cast<const __nv_fp8_e4m3&>(sc_fp8));

// Load 8 packed bytes (16 NVFP4 nibbles).
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        int idx = start + i;
        if (idx >= n_elements)
            break;
        int byte_idx = idx / 2;
        uint8_t byte = nvfp4_in[byte_idx];
        uint8_t nib = (idx & 1) ? (byte >> 4) : (byte & 0xF);
        float val = e2m1_nibble_to_fp32(nib) * sc;
        output[idx] = __float2half(val);
    }
}

// ---------------------------------------------------------------------------
// Host entry points
// ---------------------------------------------------------------------------
void nvfp4_quant_linear_fp16(const half* d_input, uint8_t* d_nvfp4, uint8_t* d_sf, int n_elements,
                             cudaStream_t stream) {
    const int threads = 256;
    const int groups = (n_elements + 15) / 16;
    const int blocks = (groups + threads - 1) / threads;
    nvfp4_quant_linear_kernel<<<blocks, threads, 0, stream>>>(d_input, d_nvfp4, d_sf, n_elements);
}

void nvfp4_dequant_linear_fp16(const uint8_t* d_nvfp4, const uint8_t* d_sf, half* d_output, int n_elements,
                               cudaStream_t stream) {
    const int threads = 256;
    const int groups = (n_elements + 15) / 16;
    const int blocks = (groups + threads - 1) / threads;
    nvfp4_dequant_linear_kernel<<<blocks, threads, 0, stream>>>(d_nvfp4, d_sf, d_output, n_elements);
}

}  // namespace imp
