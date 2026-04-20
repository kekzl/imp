#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace imp {

// ggml-compatible MMVQ kernels for exact numerical parity with llama.cpp.
// Compute y[M, N] = x[M, K] @ W[N, K]^T
// where W is Q4_K or Q5_1 packed, x is FP16 or FP32.
// Internally quantizes x to Q8_1 using scratch buffer and uses ggml vec_dot.
// scratch: GPU buffer for Q8_1 temp data, must be at least M*K/32*36 bytes

void ggml_mmvq_q4k(
    const void* W,       // [N, K/256*144] raw Q4_K bytes
    const half* x,       // [M, K] FP16 input
    half* y,             // [M, N] FP16 output
    int M, int N, int K,
    void* scratch, size_t scratch_size,
    cudaStream_t stream);

// FP32 input variant: quantizes FP32→Q8_1 (higher precision, matches llama)
void ggml_mmvq_q4k_f32(
    const void* W,       // [N, K/256*144] raw Q4_K bytes
    const float* x,      // [M, K] FP32 input
    half* y,             // [M, N] FP16 output
    int M, int N, int K,
    void* scratch, size_t scratch_size,
    cudaStream_t stream);

void ggml_mmvq_q5_1(
    const void* W,       // [N, K/32*24] raw Q5_1 bytes
    const half* x,       // [M, K] FP16 input
    half* y,             // [M, N] FP16 output
    int M, int N, int K,
    void* scratch, size_t scratch_size,
    cudaStream_t stream);

void ggml_mmvq_q5k(
    const void* W,       // [N, K/256*176] raw Q5_K bytes
    const half* x,       // [M, K] FP16 input
    half* y,             // [M, N] FP16 output
    int M, int N, int K,
    void* scratch, size_t scratch_size,
    cudaStream_t stream);

void ggml_mmvq_q8_0(
    const void* W,       // [N, K/32*34] raw Q8_0 bytes
    const half* x,       // [M, K] FP16 input
    half* y,             // [M, N] FP16 output
    int M, int N, int K,
    void* scratch, size_t scratch_size,
    cudaStream_t stream);

} // namespace imp
