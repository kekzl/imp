#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace imp {

// One projection's LoRA delta over row-strided buffers (#522, B-111 S5):
// y[i][0, N) += scale * (x[i][0, K) . A^T) . B^T, A [r, K] and B [N, r] fp16 row-major.
struct LoraRowsArgs {
    const half* A = nullptr;
    const half* B = nullptr;
    int r = 0, K = 0, N = 0;
    float scale = 1.0f;
    const half* x = nullptr;  // [rows][ldx]
    int ldx = 0;
    half* y = nullptr;  // [rows][ldy], accumulated in fp32 and rounded once
    int ldy = 0;
    int rows = 0;
    float* t = nullptr;  // fp32 rank intermediate [rows][r]
};

// Two kernels, no cuBLAS and no allocation: safe under graph capture and on side streams.
void lora_delta_rows(const LoraRowsArgs& a, cudaStream_t s);

}  // namespace imp
