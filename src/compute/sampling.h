#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

// Greedy: argmax over logits
int32_t sample_greedy(const Tensor& logits, cudaStream_t stream = nullptr);

// Top-k + top-p + temperature sampling
int32_t sample_topk_topp(const Tensor& logits, int top_k, float top_p,
                         float temperature, unsigned int seed,
                         cudaStream_t stream = nullptr);

// Variants that use a pre-allocated device result buffer (avoids cudaMalloc per call).
// d_result must point to at least sizeof(int32_t) bytes of device memory.
int32_t sample_greedy(const Tensor& logits, int32_t* d_result,
                      cudaStream_t stream = nullptr);
int32_t sample_topk_topp(const Tensor& logits, int top_k, float top_p,
                         float temperature, unsigned int seed,
                         int32_t* d_result,
                         cudaStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// Async (device-side) sampling: writes result to device buffer AND mapped
// pinned memory. No cudaStreamSynchronize — GPU-side token stays on device.
// h_mapped: host-mapped pinned pointer (cudaHostAlloc with cudaHostAllocMapped).
// Returns immediately. Host polls *h_mapped for token readback.
// ---------------------------------------------------------------------------
void sample_greedy_device(const Tensor& logits, int32_t* d_result,
                          int32_t* h_mapped, cudaStream_t stream = nullptr);
void sample_topk_topp_device(const Tensor& logits, int top_k, float top_p,
                              float temperature, unsigned int seed,
                              int32_t* d_result, int32_t* h_mapped,
                              cudaStream_t stream = nullptr);

} // namespace imp
