#include "quant/dequant_fp16.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

namespace imp {

// ---------------------------------------------------------------------------
// INT4 -> FP16 dequantization kernel (GGML Q4_0 compatible packing)
//
// Packing format: each byte holds 2 x 4-bit unsigned values.
//   low  nibble (bits 0-3)  = first element  (even index)
//   high nibble (bits 4-7)  = second element (odd index)
// Unsigned range [0, 15] is centered at 8:
//   dequant_value = (nibble - 8) * scale
//
// Scales are FP16, one per group of `group_size` elements.
// ---------------------------------------------------------------------------

static constexpr int kBlockSize = 256;

__global__ void dequant_int4_fp16_kernel(
    const uint8_t* __restrict__ input,
    half*          __restrict__ output,
    const half*    __restrict__ scales,
    int n,
    int group_size)
{
    // Each thread processes one byte = 2 elements.
    const int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int elem_idx = byte_idx * 2;  // first of the two elements

    if (elem_idx >= n) return;

    const uint8_t packed = input[byte_idx];
    const int lo = packed & 0x0F;        // low nibble  -> element at elem_idx
    const int hi = (packed >> 4) & 0x0F; // high nibble -> element at elem_idx+1

    // Determine the group for each element and fetch the corresponding scale.
    const float scale_lo = __half2float(scales[elem_idx / group_size]);

    output[elem_idx] = __float2half((float)(lo - 8) * scale_lo);

    // Guard: the last byte may encode only one valid element when n is odd.
    if (elem_idx + 1 < n) {
        const float scale_hi = __half2float(scales[(elem_idx + 1) / group_size]);
        output[elem_idx + 1] = __float2half((float)(hi - 8) * scale_hi);
    }
}

void dequant_int4_fp16(const void* input, void* output,
                       const void* scales, int n, int group_size,
                       cudaStream_t stream)
{
    if (n <= 0) return;

    // Number of packed bytes = ceil(n / 2)
    const int n_bytes = (n + 1) / 2;
    const int grid = (n_bytes + kBlockSize - 1) / kBlockSize;

    dequant_int4_fp16_kernel<<<grid, kBlockSize, 0, stream>>>(
        static_cast<const uint8_t*>(input),
        static_cast<half*>(output),
        static_cast<const half*>(scales),
        n,
        group_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "dequant_int4_fp16 launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

} // namespace imp
