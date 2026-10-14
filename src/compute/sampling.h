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

// Multi-block top-k/top-p scratch: single-block sampler used 1 of 170 SMs (the #1 GPU
// consumer in batched decode); two-phase multi-block spreads full-vocab scans across
// SAMPLE_NBLOCKS blocks. Phase 1 writes block_max/block_sum/top_k candidates per block;
// phase 2 merges, applies top-p, samples.
// d_result buffer must be >= SAMPLE_SCRATCH_BYTES (>= ARGMAX_SCRATCH_BYTES, backs greedy too).
static constexpr int SAMPLE_NBLOCKS = 64;
static constexpr int SAMPLE_MAX_TOP_K = 128;
static constexpr size_t SAMPLE_SCRATCH_BYTES =
    sizeof(int32_t) + SAMPLE_NBLOCKS * (2 * sizeof(float) +
                                        SAMPLE_MAX_TOP_K * (sizeof(float) + sizeof(int32_t)));

// Synchronous samplers (sample_greedy, sample_topk_topp, sample_mirostat_v2): sync the stream,
// never under graph capture. Failed scratch alloc / kernel launch / readback throws std::runtime_error
// (#2307, #2310).
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

// Enqueue-only sampler for batched decode: writes token to d_result[0] (scratch right
// after), no readback/sync. Caller enqueues per sequence then does one pinned D2H + one
// stream sync for the whole batch - avoids per-sequence pageable-readback serialization.
// Kernels/normalization identical to the synchronous variants: bit-identical tokens.
// false = CUB scratch (top_k > SAMPLE_MAX_TOP_K) unavailable or a kernel launch failed: the slot
// holds a stale token (#2307, #2310).
[[nodiscard]] bool sample_greedy_async(const Tensor& logits, int32_t* d_result,
                                       cudaStream_t stream = nullptr);
[[nodiscard]] bool sample_topk_topp_async(const Tensor& logits, int top_k, float top_p, float temperature,
                            unsigned int seed, int32_t* d_result, cudaStream_t stream = nullptr);

// Row-parallel batched top-k/top-p: one partial launch (grid 64 x n_rows) + one finalize
// launch (grid n_rows) sample every row of a decode batch, replacing n_rows serialized
// launch pairs. Per-row reduction geometry matches the per-row variants: bit-identical
// tokens for the same logits/seed.
// Caller pre-normalizes: top_k in [1,SAMPLE_MAX_TOP_K], temperature>0 folded into
// inv_temperature, top_p in (0,1].
struct TopkRowArgs {
    const float* logits;   // this row's [vocab] logits
    int32_t* d_result;     // this row's SAMPLE_SCRATCH_BYTES slot
    float inv_temperature;
    float top_p;
    unsigned int seed;
    int top_k;
};
// Returns the launch status; != cudaSuccess = the rows hold stale tokens (#2310).
[[nodiscard]] cudaError_t launch_topk_topp_rows(const TopkRowArgs* d_rows, int n_rows, int max_top_k,
                                                int vocab_size, cudaStream_t stream = nullptr);

// top_k the eager sampler runs for a request's top_k: 0 means 50, past the vocab means the vocab.
constexpr int sample_effective_top_k(int top_k, int vocab_size) {
    const int k = top_k > 0 ? top_k : 50;
    return k > vocab_size ? vocab_size : k;
}

// Sampled speculative verify: row r draws with seed (seed0 + r) & 0x7fffffff, the eager sampler's
// seed for that output index. Filter order as the eager chain: min_p, typical_p, top_k/top_p/temperature.
struct VerifySampling {
    float temperature = 1.0f;
    float top_p = 1.0f;
    float min_p = 0.0f;
    float typical_p = 0.0f;
    int top_k = 50;  // sample_effective_top_k, <= SAMPLE_MAX_TOP_K
    int seed0 = 0;   // engine_internal::compute_step_seed at the chunk's first row
};
// Rows [row0, row0 + n_rows) of the chunk: d_logits [n_rows x vocab] fp32 (filtered in place) ->
// d_out[n_rows]. h_args/d_args: n_rows TopkRowArgs (pinned / device); d_slots: n_rows x SAMPLE_SCRATCH_BYTES.
[[nodiscard]] cudaError_t sample_verify_rows(float* d_logits, int n_rows, int vocab_size,
                                             const VerifySampling& s, int row0, TopkRowArgs* h_args,
                                             TopkRowArgs* d_args, void* d_slots, int32_t* d_out,
                                             cudaStream_t stream = nullptr);

// Row-parallel batched GREEDY argmax (temperature-0 twin of the top-k stash): one partial
// launch (grid ARGMAX_NBLOCKS x n_rows) + one reduce launch (grid n_rows) replace n
// serialized launch pairs. Per-row geometry/scratch layout match sample_greedy_async:
// bit-identical tokens.
struct GreedyRowArgs {
    const float* logits;   // this row's [vocab] logits
    int32_t* d_result;     // this row's SAMPLE_SCRATCH_BYTES slot
};
// Returns the launch status; != cudaSuccess = the rows hold stale tokens (#2310).
[[nodiscard]] cudaError_t launch_greedy_rows(const GreedyRowArgs* d_rows, int n_rows, int vocab_size,
                                             cudaStream_t stream = nullptr);

// Row-parallel batched penalties: same per-thread math as apply_penalties (grid over
// vocab), one launch per stashed row. Caller pre-filters rows with an active penalty and
// n_tokens>0, and pre-applies the repeat_last_n window.
struct PenaltyRowArgs {
    float* logits = nullptr;
    const int32_t* token_ids = nullptr;  // device history, already windowed
    int n_tokens = 0;
    float repetition_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    // Banned ids (device list, may be null): applied in the same sweep by the thread owning
    // the vocab entry, so an id that is both banned and in history is written once (-1e30).
    // Lets penalties + engine-static ban list share one batched sweep instead of two launches.
    const int32_t* banned = nullptr;
    int n_banned = 0;
};
void launch_penalties_rows(const PenaltyRowArgs* d_rows, int n_rows, int vocab_size,
                           cudaStream_t stream = nullptr);
// The vocab-sweep form of the three penalty launchers: O(vocab x history),
// the reference the history-sized kernels are held bit-identical to (tests).
void launch_penalties_rows_sweep(const PenaltyRowArgs* d_rows, int n_rows, int vocab_size,
                                 cudaStream_t stream = nullptr);
void apply_penalties_sweep(float* logits, int vocab_size, const int32_t* token_ids, int n_tokens,
                           float repetition_penalty, float frequency_penalty, float presence_penalty,
                           cudaStream_t stream = nullptr);
void apply_penalties_device_count_sweep(float* logits, int vocab_size, const int32_t* token_ids,
                                        const int* d_n_tokens, int repeat_last_n, float repetition_penalty,
                                        float frequency_penalty, float presence_penalty,
                                        cudaStream_t stream = nullptr);
// Per-row 16-bit token counts for the history-sized penalty kernels: rows x
// vocab halves from the T2 arena, once per engine (charged as
// ExecT2Demand::penalty_counts). False leaves the penalties on the sweep.
[[nodiscard]] bool sampling_preallocate_penalty_counts(int rows, int vocab_size);
// Forget the count scratch (engine teardown, test arenas): the arena that
// backed it is gone, the launchers fall back to the sweep until the next
// sampling_preallocate_penalty_counts.
void sampling_reset_penalty_counts();

// Async device-side sampling: writes result to a device buffer AND mapped pinned memory,
// no cudaStreamSynchronize (GPU-side token stays on device). h_mapped: pinned host pointer
// (cudaHostAllocMapped). Returns immediately; host polls *h_mapped for readback.
// false = copy enqueue, CUB scratch or kernel launch failed: *h_mapped is stale, never read it
// (#2307, #2310). Capture-safe: no sync, no throw.
[[nodiscard]] bool sample_greedy_device(const Tensor& logits, int32_t* d_result, int32_t* h_mapped,
                                        cudaStream_t stream = nullptr);
// d_seed_salt: device int added to seed inside the kernel, so a captured graph draws a new
// quantile every step (nullptr = seed as is).
[[nodiscard]] bool sample_topk_topp_device(const Tensor& logits, int top_k, float top_p, float temperature,
                                           unsigned int seed, int32_t* d_result, int32_t* h_mapped,
                                           cudaStream_t stream = nullptr, const int* d_seed_salt = nullptr);

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

// DRY penalty: penalizes tokens that would create repeated n-grams (scans token history
// for suffix matches). multiplier: scale (0=disabled). base: exponential base (1.75
// default). allowed_length: n-gram lengths <= this unpenalized (2 default).
// penalty_last_n: recent tokens scanned (0=all).
void apply_dry_penalty(float* d_logits, int vocab_size, const int32_t* host_token_ids, int n_tokens,
                       float multiplier, float base, int allowed_length, int penalty_last_n,
                       cudaStream_t stream = nullptr);

// Typical-P (locally typical) filtering: keeps tokens whose information content is
// closest to the distribution's entropy, up to cumulative probability >= typical_p.
// typical_p in (0,1); 1.0 = disabled. Modifies logits in place (filtered -> -inf).
void apply_typical_p(float* logits, int vocab_size, float typical_p, cudaStream_t stream = nullptr);

// Mirostat v2: adaptively controls perplexity via a running target surprise (tau) and mu,
// applying temperature internally (like topk_topp). Updates *mu in place after sampling.
// tau: target entropy (default 5.0), eta: learning rate (default 0.1).
int32_t sample_mirostat_v2(const Tensor& logits, float temperature, float tau, float eta, float* mu,
                           unsigned int seed, cudaStream_t stream = nullptr);

// Pre-allocated version (d_result must have at least 8 bytes: 4 for token + 4 for surprise).
int32_t sample_mirostat_v2(const Tensor& logits, float temperature, float tau, float eta, float* mu,
                           unsigned int seed, int32_t* d_result, cudaStream_t stream = nullptr);

// CPU-side logprob: log-softmax on host logits, extract sampled token logprob + top-N
// alternatives. Called after D2H copy of logits.
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

// logit_bias: adds a per-token bias before sampling. host_pairs is a HOST array of
// (token_id, bias); out-of-range ids skipped. atomicAdd handles duplicate token ids
// (two JSON keys naming the same token, e.g. "1" and "01").
// Kernel instead of the old per-entry blocking D2H+add+writeback: that cost one device
// sync per entry per decode step, x3 copies of the same loop (#1617).
void apply_logit_bias(float* d_logits, int vocab_size, const std::pair<int32_t, float>* host_pairs, int n,
                      cudaStream_t stream = nullptr);

// Engine-lifetime slots for apply_logit_bias, from the T2 arena. If not run, or the arena
// had no room, apply_logit_bias falls back to per-entry copies rather than silently
// dropping the bias (which would change output without saying so).
void sampling_preallocate_logit_bias(int max_entries);

// Batched penalty-history append (n>1 decode): one launch appends row i's sampled token
// to its request's device-resident history at hist[slots[i]*cap+offs[i]], replacing the
// per-row pageable H2D re-upload of the whole history every step. Row/slot/offset tables
// ride in kernel args - no upload at all.
struct PenaltyAppendArgs {
    static constexpr int kMaxRows = 64;  // = Engine::kMaxGraphPoolSize
    int n = 0;
    int cap = 0;
    int slots[kMaxRows] = {};
    int offs[kMaxRows] = {};
};
void penalty_hist_append(const void* d_sample_base, size_t slot_stride_bytes,
                         const PenaltyAppendArgs& args, int32_t* d_hist,
                         cudaStream_t stream = nullptr);

// Free persistent CUB sort scratch (call at engine shutdown).
void sampling_cleanup();

}  // namespace imp
