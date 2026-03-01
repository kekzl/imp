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

// DRY (Don't Repeat Yourself) penalty: penalizes tokens that would create
// repeated n-grams by scanning token history for suffix matches.
// host_token_ids: HOST-side array of previously generated tokens.
// multiplier: penalty scale (0 = disabled). base: exponential base (default 1.75).
// allowed_length: n-gram lengths ≤ this aren't penalized (default 2).
// penalty_last_n: how many recent tokens to scan (0 = all).
void apply_dry_penalty(float* d_logits, int vocab_size,
                       const int32_t* host_token_ids, int n_tokens,
                       float multiplier, float base,
                       int allowed_length, int penalty_last_n,
                       cudaStream_t stream = nullptr);

// Typical-P (locally typical) filtering: keeps tokens whose information
// content is closest to the distribution's entropy, up to cumulative
// probability >= typical_p.  Modifies logits in-place (sets filtered to -inf).
// typical_p in (0, 1): fraction of probability mass to keep. 1.0 = disabled.
void apply_typical_p(float* logits, int vocab_size, float typical_p,
                     cudaStream_t stream = nullptr);

// Mirostat v2 sampling: adaptively controls perplexity by maintaining a
// running target surprise (tau) and adapting mu.  Applies temperature
// internally (like topk_topp).  Updates *mu in-place after sampling.
// tau: target entropy (default 5.0), eta: learning rate (default 0.1).
int32_t sample_mirostat_v2(const Tensor& logits, float temperature,
                           float tau, float eta, float* mu,
                           unsigned int seed,
                           cudaStream_t stream = nullptr);

// Pre-allocated version (d_result must have at least 8 bytes: 4 for token + 4 for surprise).
int32_t sample_mirostat_v2(const Tensor& logits, float temperature,
                           float tau, float eta, float* mu,
                           unsigned int seed, int32_t* d_result,
                           cudaStream_t stream = nullptr);

} // namespace imp
