#pragma once

#include "core/qtype.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace imp {

struct block_q8_1;

// One weight of a gemv_dp4a_multi launch: raw GGUF blocks [rows, K], fp16 output [rows].
struct Dp4aGemvSeg {
    const void* w = nullptr;
    QType qtype = QType::NONE;
    half* y = nullptr;
    int rows = 0;
};
constexpr int kDp4aMultiMaxSegs = 4;

// y_s = W_s x for up to 4 weights of mixed GGUF types sharing one q8_1-quantized x [K]: one launch, one
// 128-thread block per row, the per-row math and reduction order of gemv_dp4a_kpar_kernel (bit-equal to
// it). Types: Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, Q2_K, Q3_K, IQ4_NL, IQ4_XS, IQ3_S. false = refused.
[[nodiscard]] bool gemv_dp4a_multi(const Dp4aGemvSeg* segs, int n_segs, const block_q8_1* q8_1,
                                   const float* d8, int K, cudaStream_t stream);

// Whether gemv_dp4a_multi serves qtype at this K.
[[nodiscard]] bool gemv_dp4a_multi_supports(QType qtype, int K);

}  // namespace imp
