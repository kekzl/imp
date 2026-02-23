#include "quant/dequant_int8.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

namespace imp {

// ---------------------------------------------------------------------------
// INT8 -> FP16 dequantization kernel with per-element scales.
//
// The caller is responsible for pre-expanding (broadcasting) scales so that
// scales[i] is the correct scale factor for element i.  This keeps the kernel
// flexible: it works for per-tensor, per-channel, and per-group quantization
// as long as the scales buffer has been prepared accordingly.
//
//   output[i] = (float)input[i] * __half2float(scales[i])
//
// For bandwidth efficiency each thread processes 4 consecutive elements using
// a 32-bit load for the INT8 data and two 32-bit loads for two packed half2
// values, then writes two half2 values.
// ---------------------------------------------------------------------------

static constexpr int kBlockSize = 256;
static constexpr int kElemsPerThread = 4;

__global__ void dequant_int8_fp16_kernel(
    const int8_t* __restrict__ input,
    half*         __restrict__ output,
    const half*   __restrict__ scales,
    int n)
{
    const int base = (blockIdx.x * blockDim.x + threadIdx.x) * kElemsPerThread;
    if (base >= n) return;

    // --- Vectorised path: all 4 elements are in-bounds ----------------------
    if (base + kElemsPerThread <= n) {
        // Load 4 x int8 in one 32-bit transaction.
        int32_t packed;
        memcpy(&packed, input + base, sizeof(int32_t));

        const int8_t v0 = static_cast<int8_t>(packed & 0xFF);
        const int8_t v1 = static_cast<int8_t>((packed >> 8) & 0xFF);
        const int8_t v2 = static_cast<int8_t>((packed >> 16) & 0xFF);
        const int8_t v3 = static_cast<int8_t>((packed >> 24) & 0xFF);

        // Load 4 x half scales (2 x half2).
        const half2 s01 = *reinterpret_cast<const half2*>(scales + base);
        const half2 s23 = *reinterpret_cast<const half2*>(scales + base + 2);

        const float s0 = __half2float(__low2half(s01));
        const float s1 = __half2float(__high2half(s01));
        const float s2 = __half2float(__low2half(s23));
        const float s3 = __half2float(__high2half(s23));

        half2 out01;
        out01 = __halves2half2(__float2half((float)v0 * s0),
                               __float2half((float)v1 * s1));
        half2 out23;
        out23 = __halves2half2(__float2half((float)v2 * s2),
                               __float2half((float)v3 * s3));

        *reinterpret_cast<half2*>(output + base)     = out01;
        *reinterpret_cast<half2*>(output + base + 2) = out23;
    } else {
        // --- Scalar tail: handle remaining elements -------------------------
        for (int i = 0; i < kElemsPerThread && base + i < n; ++i) {
            const float val   = static_cast<float>(input[base + i]);
            const float scale = __half2float(scales[base + i]);
            output[base + i]  = __float2half(val * scale);
        }
    }
}

void dequant_int8_fp16(const void* input, void* output,
                       const void* scales, int n,
                       cudaStream_t stream)
{
    if (n <= 0) return;

    const int threads_needed = (n + kElemsPerThread - 1) / kElemsPerThread;
    const int grid = (threads_needed + kBlockSize - 1) / kBlockSize;

    dequant_int8_fp16_kernel<<<grid, kBlockSize, 0, stream>>>(
        static_cast<const int8_t*>(input),
        static_cast<half*>(output),
        static_cast<const half*>(scales),
        n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "dequant_int8_fp16 launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

} // namespace imp
