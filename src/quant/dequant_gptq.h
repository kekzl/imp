#pragma once

// GPTQ 4-bit dequant to FP16, one element rule for the device kernel and the host reference.
// AutoGPTQ layout: qweight [K/8, N] packed along K, qzeros [G, N/8] packed along N, scales [G, N],
// g_idx [K] optional; G = ceil(K / group_size), group_size -1 = one group.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <string>

#if defined(__CUDACC__)
#define IMP_GPTQ_HD __host__ __device__
#else
#define IMP_GPTQ_HD
#endif

namespace imp::gptq {

// Added to the stored zero nibble, then masked to 4 bits. v1 (checkpoint_format gptq) stores z - 1
// (AutoGPTQ qlinear_cuda_old.py:166 packs zeros -= 1, :301 unpacks zeros + 1); gptq_v2 stores z.
enum class ZeroFormat : int { V2 = 0, V1 = 1 };

// "" / "gptq" -> V1, "gptq_v2" -> V2 (case-insensitive). False for any other format (marlin, ...).
[[nodiscard]] bool parse_zero_format(const std::string& checkpoint_format, ZeroFormat* out);

// w[n,k] = (q - z) * s: exact integer times FP16 scale in FP32, one RN rounding, bit-equal to FP16 math.
IMP_GPTQ_HD inline __half dequant_elem(const int32_t* qweight, const int32_t* qzeros, const __half* scales,
                                       const int32_t* g_idx, int N, int group_size, ZeroFormat fmt, int k,
                                       int n) {
    const int64_t g = g_idx ? g_idx[k] : k / group_size;
    const int q = (qweight[static_cast<int64_t>(k / 8) * N + n] >> ((k & 7) * 4)) & 0xF;
    const int zs = (qzeros[g * (N / 8) + n / 8] >> ((n & 7) * 4)) & 0xF;
    const int z = (zs + static_cast<int>(fmt)) & 0xF;
    return __float2half_rn(static_cast<float>(q - z) * __half2float(scales[g * N + n]));
}

struct Dims {
    int K = 0, N = 0, groups = 0;
    int group_size = 0;  // resolved: -1 (per-channel) becomes K
};

// Checks one projection's tensor shapes and derives Dims. False with `err` set on any mismatch.
[[nodiscard]] bool check_shapes(const int64_t* qweight_shape, const int64_t* qzeros_shape, const int64_t* scales_shape,
                  int group_size, Dims* d, std::string* err);

// g_idx must hold K entries, each in [0, groups).
[[nodiscard]] bool check_g_idx(const int32_t* g_idx, int64_t len, const Dims& d, std::string* err);

// Host reference: out [N, K] FP16 (row n = output feature), same element rule as the kernel.
void dequant4_host(__half* out, const int32_t* qweight, const int32_t* qzeros, const __half* scales,
                   const int32_t* g_idx, int N, int K, int group_size, ZeroFormat fmt);

}  // namespace imp::gptq

namespace imp {

// Device: out [N, K] FP16. group_size is the resolved one (Dims::group_size); g_idx may be nullptr.
void dequant_gptq4(half* out, const int32_t* qweight, const int32_t* qzeros, const half* scales,
                   const int32_t* g_idx, int N, int K, int group_size, gptq::ZeroFormat fmt,
                   cudaStream_t stream = nullptr);

}  // namespace imp
