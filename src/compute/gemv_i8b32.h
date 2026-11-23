#pragma once
// int8 weights with one fp16 scale per 32 values (I8B32): w[n][k] = q[n][k] * scale[n][k / 32], q in
// [-127, 127]. Half the bytes of an fp16 matrix for a projection that reads at the memory bandwidth, and
// finer than E4M3 (Inkling-2-Mini dense projections: KLD +0.0001 against +0.0011 for per-row E4M3).

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace imp {

// rows x K fp16 -> q [rows, K] and scales [rows, K / 32] (scale = fp16(block absmax / 127)). K % 32 == 0.
void quantize_i8b32(const half* w, int8_t* q, half* scales, size_t rows, int K,
                    cudaStream_t stream = nullptr);

// y [n_rows, M] = x [n_rows, K] . W^T for 1 <= n_rows <= 16, fp32 accumulate on tensor cores, a row's bits
// independent of n_rows. K % 256 == 0, q rows 16-byte aligned; false = refused.
[[nodiscard]] bool gemv_i8b32_f16(const int8_t* q, const half* scales, const half* x, half* y, int M, int K,
                                  int n_rows, cudaStream_t stream = nullptr);

}  // namespace imp
