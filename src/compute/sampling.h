#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <vector>
#include <utility>
#include <cstdint>

namespace imp {

// Multi-block argmax scratch: the d_result buffer must be at least this many
// bytes to hold the partial reduction arrays used by the multi-block kernel.
static constexpr int ARGMAX_NBLOCKS = 64;
static constexpr size_t ARGMAX_SCRATCH_BYTES = sizeof(int32_t) +
                                               ARGMAX_NBLOCKS * (sizeof(float) + sizeof(int32_t));

// Multi-block top-k/top-p sampling scratch. The single-block sampler used 1 of
// 170 SMs (~737 us/call, the #1 GPU consumer in batched decode); the two-phase
// multi-block sampler spreads the full-vocab scans across SAMPLE_NBLOCKS blocks.
// Phase 1 writes, per block: block_max, block_sum, and the block's top_k logit
// candidates; phase 2 merges them, applies top-p and samples. Any buffer passed
// as d_result to the top-k/top-p path must be at least SAMPLE_SCRATCH_BYTES so
// the partials can live right after the int32 result (graph-capture safe — no
// allocation during capture). SAMPLE_SCRATCH_BYTES >= ARGMAX_SCRATCH_BYTES, so
// the same buffer also backs the greedy multi-block path.
static constexpr int SAMPLE_NBLOCKS = 64;
static constexpr int SAMPLE_MAX_TOP_K = 128;
static constexpr size_t SAMPLE_SCRATCH_BYTES =
    sizeof(int32_t) + SAMPLE_NBLOCKS * (2 * sizeof(float) +
                                        SAMPLE_MAX_TOP_K * (sizeof(float) + sizeof(int32_t)));

// Greedy: argmax over logits
int32_t sample_greedy(const Tensor& logits, cudaStream_t stream = nullptr);

// Top-k + top-p + temperature sampling
int32_t sample_topk_topp(const Tensor& logits, int top_k, float top_p, float temperature, unsigned int seed,
                         cudaStream_t stream = nullptr);

// Variants that use a pre-allocated device result buffer (avoids cudaMalloc per call).
// d_result must point to at least sizeof(int32_t) bytes of device memory.
int32_t sample_greedy(const Tensor& logits, int32_t* d_result, cudaStream_t stream = nullptr);
int32_t sample_topk_topp(const Tensor& logits, int top_k, float top_p, float temperature, unsigned int seed,
                         int32_t* d_result, cudaStream_t stream = nullptr);

// Enqueue-only variants for the batched decode path: launch the sampler
// writing the token to d_result[0] (multi-block scratch right after it, so
// d_result needs SAMPLE_SCRATCH_BYTES like the synchronous variants) with NO
// readback and NO sync. The caller enqueues one per sequence into its own
// scratch slot and performs a single pinned D2H + single stream sync for the
// whole batch — the per-sequence pageable readback serialized batched decode
// against ~200 us host round-trips each (29% GPU idle at n=16, 2026-07-12).
// Kernels and parameter normalization are identical to the synchronous
// variants, so tokens are bit-identical for the same logits and seed.
// sample_topk_topp_async returns false when top_k (after the <=0 -> vocab
// normalization) exceeds SAMPLE_MAX_TOP_K -- that regime needs the CUB path,
// which syncs internally; the caller falls back to the synchronous variant.
void sample_greedy_async(const Tensor& logits, int32_t* d_result, cudaStream_t stream = nullptr);
bool sample_topk_topp_async(const Tensor& logits, int top_k, float top_p, float temperature,
                            unsigned int seed, int32_t* d_result, cudaStream_t stream = nullptr);

// Row-parallel batched top-k/top-p: ONE partial launch (grid 64 x n_rows) +
// ONE finalize launch (grid n_rows) sample every row of a decode batch —
// replacing n_rows serialized <<<64>>> + <<<1>>> launch pairs whose finalize
// blocks ran one-at-a-time (~10% of GPU time at n=16, nsys 2026-07-12).
// Per-row reduction geometry (blockIdx.x / gridDim.x) is identical to the
// per-row variants, so tokens are bit-identical for the same logits/seed.
// All fields pre-normalized by the caller: top_k in [1, SAMPLE_MAX_TOP_K],
// temperature > 0 folded into inv_temperature, top_p in (0, 1].
struct TopkRowArgs {
    const float* logits;   // this row's [vocab] logits
    int32_t* d_result;     // this row's SAMPLE_SCRATCH_BYTES slot
    float inv_temperature;
    float top_p;
    unsigned int seed;
    int top_k;
};
void launch_topk_topp_rows(const TopkRowArgs* d_rows, int n_rows, int max_top_k, int vocab_size,
                           cudaStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// Async (device-side) sampling: writes result to device buffer AND mapped
// pinned memory. No cudaStreamSynchronize — GPU-side token stays on device.
// h_mapped: host-mapped pinned pointer (cudaHostAlloc with cudaHostAllocMapped).
// Returns immediately. Host polls *h_mapped for token readback.
// ---------------------------------------------------------------------------
void sample_greedy_device(const Tensor& logits, int32_t* d_result, int32_t* h_mapped,
                          cudaStream_t stream = nullptr);
void sample_topk_topp_device(const Tensor& logits, int top_k, float top_p, float temperature,
                             unsigned int seed, int32_t* d_result, int32_t* h_mapped,
                             cudaStream_t stream = nullptr);

// Apply repetition / frequency / presence penalties to logits in-place.
// token_ids: device array of previously generated tokens.
// Force a single token: set all logits to -inf except keep_token.
void force_single_token(float* logits, int vocab_size, int32_t keep_token, cudaStream_t stream);

// n_tokens: number of tokens in history.
// Modifies logits before sampling to discourage repetition.
void apply_penalties(float* logits, int vocab_size, const int32_t* token_ids, int n_tokens,
                     float repetition_penalty, float frequency_penalty, float presence_penalty,
                     cudaStream_t stream = nullptr);

// Variant for CUDA graph loop: reads token count from device pointer.
// d_n_tokens points to a device int that changes each graph iteration.
void apply_penalties_device_count(float* logits, int vocab_size, const int32_t* token_ids,
                                  const int* d_n_tokens, int repeat_last_n, float repetition_penalty,
                                  float frequency_penalty, float presence_penalty,
                                  cudaStream_t stream = nullptr);

// Apply min_p filtering to logits in-place: set logits below
// (min_p * max_logit_prob) to -inf after softmax. Works on raw logits
// by finding max and setting tokens whose exp(logit - max) < min_p to -inf.
void apply_min_p(float* logits, int vocab_size, float min_p, cudaStream_t stream = nullptr);

// DRY (Don't Repeat Yourself) penalty: penalizes tokens that would create
// repeated n-grams by scanning token history for suffix matches.
// host_token_ids: HOST-side array of previously generated tokens.
// multiplier: penalty scale (0 = disabled). base: exponential base (default 1.75).
// allowed_length: n-gram lengths ≤ this aren't penalized (default 2).
// penalty_last_n: how many recent tokens to scan (0 = all).
void apply_dry_penalty(float* d_logits, int vocab_size, const int32_t* host_token_ids, int n_tokens,
                       float multiplier, float base, int allowed_length, int penalty_last_n,
                       cudaStream_t stream = nullptr);

// Typical-P (locally typical) filtering: keeps tokens whose information
// content is closest to the distribution's entropy, up to cumulative
// probability >= typical_p.  Modifies logits in-place (sets filtered to -inf).
// typical_p in (0, 1): fraction of probability mass to keep. 1.0 = disabled.
void apply_typical_p(float* logits, int vocab_size, float typical_p, cudaStream_t stream = nullptr);

// Mirostat v2 sampling: adaptively controls perplexity by maintaining a
// running target surprise (tau) and adapting mu.  Applies temperature
// internally (like topk_topp).  Updates *mu in-place after sampling.
// tau: target entropy (default 5.0), eta: learning rate (default 0.1).
int32_t sample_mirostat_v2(const Tensor& logits, float temperature, float tau, float eta, float* mu,
                           unsigned int seed, cudaStream_t stream = nullptr);

// Pre-allocated version (d_result must have at least 8 bytes: 4 for token + 4 for surprise).
int32_t sample_mirostat_v2(const Tensor& logits, float temperature, float tau, float eta, float* mu,
                           unsigned int seed, int32_t* d_result, cudaStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// CPU-side logprob computation: log-softmax on host logits, extract sampled
// token logprob + top-N alternatives. Called after D2H copy of logits.
// ---------------------------------------------------------------------------
struct LogprobResult {
    float sampled_logprob{};                     // logprob of the sampled token
    std::vector<std::pair<int32_t, float>> top;  // (token_id, logprob) sorted desc
};

void compute_logprobs_cpu(const float* logits, int vocab_size, int32_t sampled_token, int top_n,
                          LogprobResult* out);

// Pre-allocate DRY penalty GPU buffers at engine init time to avoid
// cudaStreamSynchronize on first use during inference.
// max_seq_len: maximum sequence length (context + generation tokens).
void sampling_preallocate_dry(int max_seq_len, cudaStream_t stream = nullptr);

// logit_bias: add a per-token bias to the logits before sampling.
//
// host_pairs is a HOST array of (token_id, bias). Out-of-range ids are
// skipped, matching what the per-entry loop this replaces did. Entries are
// applied with atomicAdd because two JSON keys can name the same token ("1"
// and "01"), and the loop accumulated both.
//
// Why it is a kernel: the old code read each logit back with a BLOCKING
// cudaMemcpy D2H, added on the host, and wrote it back - one full device
// synchronisation per entry per decode step, on three separate copies of the
// same loop (#1617).
void apply_logit_bias(float* d_logits, int vocab_size, const std::pair<int32_t, float>* host_pairs, int n,
                      cudaStream_t stream = nullptr);

// Engine-lifetime slots for apply_logit_bias, from the T2 arena. When this has
// not run, or the arena had no room, apply_logit_bias falls back to the old
// per-entry copies rather than dropping the bias: a silently unapplied bias
// changes the output without saying so.
void sampling_preallocate_logit_bias(int max_entries);

// Free persistent CUB sort scratch (call at engine shutdown).
void sampling_cleanup();

}  // namespace imp
