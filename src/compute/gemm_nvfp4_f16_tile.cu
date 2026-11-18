// Fused NVFP4 W4A16 prompt GEMM (ROADMAP B-85): the weight-decode tile (gemm_wdq_f16_tile.cuh) with the plain
// NVFP4 decoder (gemm_nvfp4_wdq_dec.cuh). The tensor scale stays in the fp32 epilogue (alpha).

#include "compute/gemm_nvfp4_f16_tile.h"

#include "compute/gemm_nvfp4_wdq_dec.cuh"

#include <cuda_fp16.h>
#include <cstdint>

namespace imp {

bool gemm_nvfp4_f16_tile(const void* a, const void* w_e2m1, const void* scales_e4m3, void* d, int M, int N,
                         int K, float alpha, float beta, bool acc_fp16, cudaStream_t stream, int variant) {
    using namespace wdq_tile;  // NOLINT(google-build-using-namespace): the shared decode-tile skeleton
    if (M <= 0 || N <= 0 || K <= 0 || K % 64 != 0 || N % 8 != 0 || a == nullptr || w_e2m1 == nullptr ||
        scales_e4m3 == nullptr || d == nullptr || reinterpret_cast<uintptr_t>(a) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(w_e2m1) % 16 != 0 || reinterpret_cast<uintptr_t>(scales_e4m3) % 4 != 0 ||
        reinterpret_cast<uintptr_t>(d) % 16 != 0)
        return false;
    const Nv4Args p{{static_cast<half*>(d), alpha, beta, M, N, K},
                    static_cast<const uint8_t*>(w_e2m1),
                    static_cast<const uint8_t*>(scales_e4m3)};
    return launch_auto<Nv4Dec>(p, a, acc_fp16, variant, stream);
}

}  // namespace imp
