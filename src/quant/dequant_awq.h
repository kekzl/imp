#pragma once

// AWQ 4-bit GEMM-layout dequant to FP16, one element rule for the device kernel and the host
// reference: qweight [K, N/8] and qzeros [K/g, N/8] INT32 packed along N, scales [K/g, N] FP16.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <string>

#include "quant/dequant_gptq.h"

#if defined(__CUDACC__)
#define IMP_AWQ_HD __host__ __device__
#else
#define IMP_AWQ_HD
#endif

namespace imp::awq {

// AutoAWQ awq/utils/packing_utils.py: AWQ_ORDER [0,2,4,6,1,3,5,7], nibble s holds column AWQ_ORDER[s].
// Column n reads nibble AWQ_REVERSE_ORDER[n%8] = [0,4,1,5,2,6,3,7] = ((n%8)>>1) | ((n%8)&1)<<2.
IMP_AWQ_HD inline int nibble_shift(int n) {
    const int i = n & 7;
    return ((i >> 1) | ((i & 1) << 2)) * 4;
}

// w[n,k] = (q - z) * s, exact integer times FP16 scale in FP32, one RN rounding: bit-equal to FP16 math.
IMP_AWQ_HD inline __half dequant_elem(const int32_t* qweight, const int32_t* qzeros, const __half* scales,
                                      int N, int group_size, int k, int n) {
    const int64_t packed_n = N / 8;
    const int64_t g = k / group_size;
    const int sh = nibble_shift(n);
    const int q = (qweight[k * packed_n + n / 8] >> sh) & 0xF;
    const int z = (qzeros[g * packed_n + n / 8] >> sh) & 0xF;
    return __float2half_rn(static_cast<float>(q - z) * __half2float(scales[g * N + n]));
}

// Checks one projection's AWQ tensor shapes; derives K, N. False with `err` set on any mismatch.
[[nodiscard]] bool check_shapes(const int64_t* qweight_shape, const int64_t* qzeros_shape, const int64_t* scales_shape,
                  int group_size, int* K, int* N, std::string* err);

// Host reference: out [N, K] FP16 (row n = output feature), same element rule as the kernel.
void dequant4_host(__half* out, const int32_t* qweight, const int32_t* qzeros, const __half* scales, int N,
                   int K, int group_size);

}  // namespace imp::awq

namespace imp {

// Device: out [N, K] FP16 from AWQ GEMM-layout qweight/qzeros/scales (see imp::awq above).
void dequant_awq4(half* out, const int32_t* qweight, const int32_t* qzeros, const half* scales, int N, int K,
                  int group_size, cudaStream_t stream = nullptr);

// Shared GPTQ/AWQ upload path: dequant_awq4 when awq_gemm, else dequant_gptq4 (g_idx, gptq_fmt GPTQ only).
void dequant_packed4(bool awq_gemm, half* out, const int32_t* qweight, const int32_t* qzeros, const half* scales,
                     const int32_t* g_idx, int N, int K, int group_size, gptq::ZeroFormat gptq_fmt,
                     cudaStream_t stream);

}  // namespace imp
