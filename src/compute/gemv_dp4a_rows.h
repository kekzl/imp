#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "compute/gemm.h"
#include "core/qtype.h"

namespace imp {

// y[r][N] (+)= W[N][K] x[r] for 1..4 activation rows (speculative verify, short prompts) in one pass over
// the GGUF blocks: one 128-thread block per weight row, K split across 4 warps. x is Q8_1 (row r at
// q8_1 + r * K / 32). add: y += (beta 1, in place). False = qtype without dp4a traits or rows outside
// 1..4, nothing launched.
[[nodiscard]] bool gemv_dp4a_rows(QType qtype, const void* W, const block_q8_1* q8_1, const float* d8,
                                  half* y, int N, int K, int rows, bool add, cudaStream_t stream);

}  // namespace imp
