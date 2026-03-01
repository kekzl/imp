#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

// Multi-block argmax scratch: the d_result buffer must be at least this many
// bytes to hold the partial reduction arrays used by the multi-block kernel.
static constexpr int ARGMAX_NBLOCKS = 64;
static constexpr size_t ARGMAX_SCRATCH_BYTES =
    sizeof(int32_t) + ARGMAX_NBLOCKS * (sizeof(float) + sizeof(int32_t));

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

// Apply repetition / frequency / presence penalties to logits in-place.
// token_ids: device array of previously generated tokens.
// n_tokens: number of tokens in history.
// Modifies logits before sampling to discourage repetition.
void apply_penalties(float* logits, int vocab_size,
                     const int32_t* token_ids, int n_tokens,
                     float repetition_penalty,
                     float frequency_penalty,
                     float presence_penalty,
                     cudaStream_t stream = nullptr);

// Apply min_p filtering to logits in-place: set logits below
// (min_p * max_logit_prob) to -inf after softmax. Works on raw logits
// by finding max and setting tokens whose exp(logit - max) < min_p to -inf.
void apply_min_p(float* logits, int vocab_size, float min_p,
                 cudaStream_t stream = nullptr);

} // namespace imp
