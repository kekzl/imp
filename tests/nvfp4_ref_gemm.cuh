#pragma once

// Library-free reference for the NVFP4 GEMMs: D[m,n] = alpha * sum_k e2m1(A[m,k]) ue4m3(SA) e2m1(W[n,k])
// ue4m3(SW), fp32 per 16-element block, one thread per output. Operands in the SfAtom layout of
// cutlass_nvfp4_sf_size (low nibble = even k). Sums in another order than the MMA: compare with a tolerance.

#include "quant/fp8_utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace imp::test {

__device__ __forceinline__ float ref_e2m1(uint32_t c) {
    constexpr float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = kMag[c & 7u];
    return (c & 8u) != 0u ? -v : v;
}

// Byte of scale group kg (16 K elements) of row `row` in the SfAtom layout (128 rows x 4 groups = 512 B
// atoms).
__device__ __forceinline__ size_t ref_sfatom(int row, int kg, int n_k_tiles) {
    return (((static_cast<size_t>(row) / 128) * n_k_tiles) + (kg / 4)) * 512 + ((row % 32) * 16) +
           (((row % 128) / 32) * 4) + (kg % 4);
}

template <class Out>
__global__ void ref_nvfp4_gemm_kernel(const uint8_t* a, const uint8_t* sa, const uint8_t* w,
                                      const uint8_t* sw, float alpha, Out* d, int M, int N, int K) {
    const int n = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int m = static_cast<int>(blockIdx.y);
    if (n >= N || m >= M)
        return;
    const int n_k_tiles = (K + 63) / 64;
    const uint8_t* ar = a + (static_cast<size_t>(m) * (K / 2));
    const uint8_t* wr = w + (static_cast<size_t>(n) * (K / 2));
    float acc = 0.0f;
    for (int kg = 0; kg < K / 16; ++kg) {
        float s = 0.0f;
        for (int j = 0; j < 8; ++j) {
            const uint32_t ab = ar[(kg * 8) + j];
            const uint32_t wb = wr[(kg * 8) + j];
            s += (ref_e2m1(ab & 15u) * ref_e2m1(wb & 15u)) + (ref_e2m1(ab >> 4) * ref_e2m1(wb >> 4));
        }
        acc += s * fp8_e4m3_to_float_fast(sa[ref_sfatom(m, kg, n_k_tiles)]) *
               fp8_e4m3_to_float_fast(sw[ref_sfatom(n, kg, n_k_tiles)]);
    }
    if constexpr (std::is_same_v<Out, half>)
        d[(static_cast<size_t>(m) * N) + n] = __float2half_rn(alpha * acc);
    else
        d[(static_cast<size_t>(m) * N) + n] = alpha * acc;
}

// Launches the reference on `s`; false when the launch fails.
template <class Out>
inline bool ref_nvfp4_gemm(const void* a, const void* sa, const void* w, const void* sw, float alpha, Out* d,
                           int M, int N, int K, cudaStream_t s) {
    const dim3 grid((N + 127) / 128, M);
    ref_nvfp4_gemm_kernel<Out><<<grid, 128, 0, s>>>(static_cast<const uint8_t*>(a),
                                                    static_cast<const uint8_t*>(sa),
                                                    static_cast<const uint8_t*>(w),
                                                    static_cast<const uint8_t*>(sw), alpha, d, M, N, K);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace imp::test
