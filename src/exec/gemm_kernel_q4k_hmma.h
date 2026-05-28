#pragma once

#include <cuda_runtime.h>

namespace imp {

// Config-gated dispatch for Q4_K x FP16 HMMA GEMM (Phase 0 scaffold).
// Returns true if the kernel ran successfully, false if the shape was
// unsupported or the kernel failed. Caller falls through to the next
// dispatch option on false.
bool try_q4k_hmma_dispatch(const void* activations_fp16, const void* weight_q4k,
                           void* output_fp16, int M, int N, int K, cudaStream_t stream);

}  // namespace imp
