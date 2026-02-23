#include "quant/fp8_utils.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

namespace imp {

// ---------------------------------------------------------------------------
// FP16 <-> FP8 E4M3 conversion utilities.
//
// FP8 E4M3 layout (8 bits): 1 sign | 4 exponent | 3 mantissa
//   bias = 7
//   normal value   : (-1)^s * 2^(e-7) * (1 + m/8)    e in [1,14]
//   subnormal value: (-1)^s * 2^(-6)  * (m/8)         e == 0, m != 0
//   zero           : s=0|1, e=0, m=0
//   NaN            : e=15, m != 0    (no infinity in E4M3)
//   max normal     : +/- 448  (e=14, m=7 -> 2^7 * 1.875)
//
// When CUDA 12 native FP8 types are available (__CUDA_FP8_TYPES_EXIST__ or
// sm_89+) we delegate to the hardware-backed __nv_fp8_e4m3.  Otherwise a
// portable software path is used.
// ---------------------------------------------------------------------------

// ---- Software fallback helpers (always compiled for host-side unit tests) --

// FP16 -> FP8 E4M3: software conversion with saturation (no Inf in E4M3).
__device__ __forceinline__ uint8_t fp16_bits_to_fp8_e4m3(uint16_t h)
{
    const uint16_t sign = (h >> 15) & 1;
    int            exp  = (int)((h >> 10) & 0x1F);  // biased exponent (bias 15)
    uint16_t       man  = h & 0x03FF;                // 10-bit mantissa

    // --- Handle special FP16 values -----------------------------------------
    // FP16 Inf/NaN -> FP8 NaN (0x7F with sign)
    if (exp == 31) {
        return (uint8_t)((sign << 7) | 0x7F);  // e=15, m=7 => NaN
    }

    // FP16 zero / subnormal that rounds to zero in E4M3 range.
    if (exp == 0 && man == 0) {
        return (uint8_t)(sign << 7);  // +/- 0
    }

    // --- Re-bias exponent: FP16 bias = 15, FP8 E4M3 bias = 7 ---------------
    // Effective unbiased exponent:
    //   FP16 normal : e_unbiased = exp - 15
    //   FP16 subnorm: needs normalisation first

    float val = __half2float(*reinterpret_cast<const half*>(&h));
    float abs_val = fabsf(val);

    // Clamp to E4M3 max representable magnitude: 448.0
    if (abs_val > 448.0f) abs_val = 448.0f;

    // Clamp to smallest E4M3 subnormal: 2^-9 = 1/512
    if (abs_val < (1.0f / 512.0f) && abs_val != 0.0f) {
        return (uint8_t)(sign << 7);  // flush to zero
    }

    // Reconstruct from float for simplicity and correctness.
    // Extract float fields.
    uint32_t fbits;
    memcpy(&fbits, &abs_val, sizeof(float));
    int f_exp = (int)((fbits >> 23) & 0xFF) - 127; // unbiased
    uint32_t f_man = fbits & 0x7FFFFF;              // 23-bit mantissa

    int e4 = f_exp + 7;  // bias for E4M3

    uint8_t result;
    if (e4 <= 0) {
        // Subnormal in E4M3 range.
        // Value = 2^(-6) * (m/8), so m = round(abs_val / 2^(-6) * 8)
        int shift = 1 - e4;  // how many positions to shift mantissa right
        // Implicit 1 + fractional mantissa, total 24 bits with implicit leading 1.
        uint32_t full_man = (1u << 23) | f_man;
        // We need 3-bit mantissa; shift right by (23 - 3 + shift) = 20 + shift
        int right_shift = 20 + shift;
        uint8_t m3;
        if (right_shift >= 32) {
            m3 = 0;
        } else {
            // Round to nearest even.
            uint32_t shifted = full_man >> right_shift;
            uint32_t remainder = full_man & ((1u << right_shift) - 1);
            uint32_t half_point = 1u << (right_shift - 1);
            if (remainder > half_point ||
                (remainder == half_point && (shifted & 1))) {
                shifted += 1;
            }
            m3 = (uint8_t)(shifted & 0x07);
            if (shifted > 7) {
                // Rounded up into normal range.
                m3 = 0;
                e4 = 1;
                result = (uint8_t)((sign << 7) | (1 << 3) | m3);
                return result;
            }
        }
        result = (uint8_t)((sign << 7) | m3);
    } else if (e4 >= 15) {
        // Overflow -> NaN (E4M3 has no infinity; 0x7E = max normal, 0x7F = NaN).
        result = (uint8_t)((sign << 7) | 0x7E);  // saturate to max normal
    } else {
        // Normal value.
        // Round 23-bit float mantissa to 3-bit.
        uint32_t m3 = (f_man + (1u << 19)) >> 20;  // round to nearest (bit 19 is half)
        if (m3 > 7) {
            m3 = 0;
            e4 += 1;
            if (e4 >= 15) {
                result = (uint8_t)((sign << 7) | 0x7E);  // saturate
                return result;
            }
        }
        result = (uint8_t)((sign << 7) | (e4 << 3) | (m3 & 0x07));
    }
    return result;
}

// FP8 E4M3 -> FP16: software conversion.
__device__ __forceinline__ uint16_t fp8_e4m3_to_fp16_bits(uint8_t x)
{
    const uint16_t sign = (uint16_t)((x >> 7) & 1);
    int            exp  = (int)((x >> 3) & 0x0F);  // 4-bit biased exponent (bias 7)
    uint16_t       man  = x & 0x07;                 // 3-bit mantissa

    // Zero.
    if (exp == 0 && man == 0) {
        return (uint16_t)(sign << 15);
    }

    // NaN: e=15, m!=0.
    if (exp == 15 && man != 0) {
        // Map to FP16 NaN.
        return (uint16_t)((sign << 15) | 0x7E00);  // FP16 NaN (e=31, m=512)
    }

    // e=15, m=0 is treated as normal in E4M3 (no Inf):
    //   value = (-1)^s * 2^(15-7) * (1 + 0/8) = (-1)^s * 256
    // We'll handle it in the normal path below.

    float val;
    if (exp == 0) {
        // Subnormal: value = (-1)^s * 2^(-6) * (m / 8)
        val = ldexpf((float)man / 8.0f, -6);
    } else {
        // Normal: value = (-1)^s * 2^(exp-7) * (1 + m/8)
        val = ldexpf(1.0f + (float)man / 8.0f, exp - 7);
    }
    if (sign) val = -val;

    half h = __float2half(val);
    uint16_t bits;
    memcpy(&bits, &h, sizeof(uint16_t));
    return bits;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

static constexpr int kBlockSize = 256;
static constexpr int kElemsPerThread = 4;

__global__ void cast_fp16_to_fp8_kernel(
    const half*    __restrict__ input,
    uint8_t*       __restrict__ output,
    int n)
{
    const int base = (blockIdx.x * blockDim.x + threadIdx.x) * kElemsPerThread;
    if (base >= n) return;

#if defined(__CUDA_FP8_TYPES_EXIST__)
    // ---- Native FP8 path (CUDA 12+ with fp8 header) -----------------------
    for (int i = 0; i < kElemsPerThread && base + i < n; ++i) {
        __nv_fp8_e4m3 fp8_val = __nv_fp8_e4m3(__half2float(input[base + i]));
        memcpy(&output[base + i], &fp8_val, 1);
    }
#else
    // ---- Software fallback -------------------------------------------------
    for (int i = 0; i < kElemsPerThread && base + i < n; ++i) {
        uint16_t hbits;
        half hval = input[base + i];
        memcpy(&hbits, &hval, sizeof(uint16_t));
        output[base + i] = fp16_bits_to_fp8_e4m3(hbits);
    }
#endif
}

__global__ void cast_fp8_to_fp16_kernel(
    const uint8_t* __restrict__ input,
    half*          __restrict__ output,
    int n)
{
    const int base = (blockIdx.x * blockDim.x + threadIdx.x) * kElemsPerThread;
    if (base >= n) return;

#if defined(__CUDA_FP8_TYPES_EXIST__)
    // ---- Native FP8 path ---------------------------------------------------
    for (int i = 0; i < kElemsPerThread && base + i < n; ++i) {
        __nv_fp8_e4m3 fp8_val;
        memcpy(&fp8_val, &input[base + i], 1);
        output[base + i] = __float2half((float)fp8_val);
    }
#else
    // ---- Software fallback -------------------------------------------------
    for (int i = 0; i < kElemsPerThread && base + i < n; ++i) {
        uint16_t hbits = fp8_e4m3_to_fp16_bits(input[base + i]);
        memcpy(&output[base + i], &hbits, sizeof(uint16_t));
    }
#endif
}

// ---------------------------------------------------------------------------
// Host-side launch wrappers
// ---------------------------------------------------------------------------

void cast_fp16_to_fp8(const void* input, void* output, int n,
                      cudaStream_t stream)
{
    if (n <= 0) return;

    const int threads_needed = (n + kElemsPerThread - 1) / kElemsPerThread;
    const int grid = (threads_needed + kBlockSize - 1) / kBlockSize;

    cast_fp16_to_fp8_kernel<<<grid, kBlockSize, 0, stream>>>(
        static_cast<const half*>(input),
        static_cast<uint8_t*>(output),
        n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "cast_fp16_to_fp8 launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

void cast_fp8_to_fp16(const void* input, void* output, int n,
                      cudaStream_t stream)
{
    if (n <= 0) return;

    const int threads_needed = (n + kElemsPerThread - 1) / kElemsPerThread;
    const int grid = (threads_needed + kBlockSize - 1) / kBlockSize;

    cast_fp8_to_fp16_kernel<<<grid, kBlockSize, 0, stream>>>(
        static_cast<const uint8_t*>(input),
        static_cast<half*>(output),
        n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "cast_fp8_to_fp16 launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

} // namespace imp
