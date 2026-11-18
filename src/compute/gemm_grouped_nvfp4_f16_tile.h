#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace imp {

// Grouped NVFP4 W4A16 GEMM over expert-sorted rows (MoE prompt rows, ROADMAP B-85): for every expert e,
// D[off[e] .. off[e+1]) = A[same rows] W_e^T * tensor_scales[e]. A [rows, K] fp16 (16 B aligned); W_e at
// w + e * w_stride (E2M1 [N, K/2]) and sc + e * sc_stride (E4M3 [N, K/16]), the NvFP4MoEQuantResult layout;
// d_offsets [ne + 1] prefix sums on device. Per expert bit-equal to gemm_nvfp4_f16_tile (fp32 accumulate)
// at alpha = tensor_scales[e] * 256. Graph-capturable (no host read of the offsets). K % 64 == 0,
// N % 8 == 0; false = refused.
struct GroupedNvfp4F16Args {
    const void* a = nullptr;
    int rows = 0;
    int N = 0;
    int K = 0;
    int ne = 0;
    const int32_t* d_offsets = nullptr;
    const void* w = nullptr;
    size_t w_stride = 0;
    const void* sc = nullptr;
    size_t sc_stride = 0;
    const float* d_tensor_scales = nullptr;
    void* d = nullptr;
};

[[nodiscard]] bool gemm_grouped_nvfp4_f16_tile(const GroupedNvfp4F16Args& g, cudaStream_t stream);

}  // namespace imp
