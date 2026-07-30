#include "compute/sampling.h"
#include "compute/sampling_internal.cuh"
#include "compute/warp_reduce.cuh"
#include "core/logging.h"
#include "memory/engine_arena.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <vector>

namespace imp {

// ===========================================================================
// Repetition / frequency / presence penalties
// ===========================================================================

// Kernel: for each token in history, adjust its logit.
// Uses atomics to handle tokens appearing multiple times.
// Strategy: first count occurrences, then apply penalties.
// For simplicity with small history, we iterate the history per thread.
__global__ void apply_penalties_kernel(float* __restrict__ logits, const int32_t* __restrict__ token_ids,
                                       int n_tokens, int vocab_size, float repetition_penalty,
                                       float frequency_penalty, float presence_penalty) {
    // Each thread handles one vocab entry
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vocab_size)
        return;

    // Count occurrences of this token in history
    int count = 0;
    for (int i = 0; i < n_tokens; i++) {
        if (token_ids[i] == idx)
            count++;
    }
    if (count == 0)
        return;

    float logit = logits[idx];

    // Repetition penalty (multiplicative): divide positive, multiply negative
    if (repetition_penalty != 1.0f) {
        if (logit > 0.0f)
            logit /= repetition_penalty;
        else
            logit *= repetition_penalty;
    }

    // Frequency penalty (subtractive per-occurrence)
    logit -= frequency_penalty * static_cast<float>(count);

    // Presence penalty (subtractive binary)
    logit -= presence_penalty;

    logits[idx] = logit;
}

// Variant: reads n_tokens from a device pointer (for CUDA graph loop where count changes).
// repeat_last_n: when > 0, only scan the last N tokens in the history.
__global__ void apply_penalties_device_count_kernel(
    float* __restrict__ logits, const int32_t* __restrict__ token_ids,
    const int* __restrict__ d_n_tokens,  // [1] device-side token count
    int vocab_size, int repeat_last_n, float repetition_penalty, float frequency_penalty,
    float presence_penalty) {
    int n_tokens = *d_n_tokens;
    if (n_tokens <= 0)
        return;

    // Apply repeat_last_n window
    int start = 0;
    if (repeat_last_n > 0 && n_tokens > repeat_last_n) {
        start = n_tokens - repeat_last_n;
    }

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vocab_size)
        return;

    int count = 0;
    for (int i = start; i < n_tokens; i++) {
        if (token_ids[i] == idx)
            count++;
    }
    if (count == 0)
        return;

    float logit = logits[idx];

    if (repetition_penalty != 1.0f) {
        if (logit > 0.0f)
            logit /= repetition_penalty;
        else
            logit *= repetition_penalty;
    }

    logit -= frequency_penalty * static_cast<float>(count);
    logit -= presence_penalty;

    logits[idx] = logit;
}

// Force a single token: set all logits to -inf except the given token.
// Used by think-budget to force </think> generation via logit manipulation.
__global__ void force_single_token_kernel(float* logits, int vocab_size, int32_t keep_token) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vocab_size)
        return;
    logits[idx] = (idx == keep_token) ? 0.0f : -1e30f;
}

void force_single_token(float* logits, int vocab_size, int32_t keep_token, cudaStream_t stream) {
    int blocks = (vocab_size + 255) / 256;
    force_single_token_kernel<<<blocks, 256, 0, stream>>>(logits, vocab_size, keep_token);
    IMP_CUDA_CHECK_LAUNCH();
}

void apply_penalties(float* logits, int vocab_size, const int32_t* token_ids, int n_tokens,
                     float repetition_penalty, float frequency_penalty, float presence_penalty,
                     cudaStream_t stream) {
    if (n_tokens == 0)
        return;
    if (repetition_penalty == 1.0f && frequency_penalty == 0.0f && presence_penalty == 0.0f)
        return;

    int blocks = (vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_penalties_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(logits, token_ids, n_tokens, vocab_size,
                                                              repetition_penalty, frequency_penalty,
                                                              presence_penalty);
    IMP_CUDA_CHECK_LAUNCH();
}

void apply_penalties_device_count(float* logits, int vocab_size, const int32_t* token_ids,
                                  const int* d_n_tokens, int repeat_last_n, float repetition_penalty,
                                  float frequency_penalty, float presence_penalty, cudaStream_t stream) {
    if (repetition_penalty == 1.0f && frequency_penalty == 0.0f && presence_penalty == 0.0f)
        return;

    int blocks = (vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_penalties_device_count_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(logits, token_ids, d_n_tokens,
                                                                           vocab_size, repeat_last_n,
                                                                           repetition_penalty,
                                                                           frequency_penalty,
                                                                           presence_penalty);
    IMP_CUDA_CHECK_LAUNCH();
}

// ===========================================================================
// DRY (Don't Repeat Yourself) repetition penalty
// ===========================================================================

// File-scope persistent GPU buffers for DRY penalty application.
// Promoted from function-local statics so sampling_preallocate_dry() can
// pre-allocate them at engine init time and avoid cudaStreamSynchronize on
// first use during inference.
static int32_t* s_dry_tokens_buf = nullptr;
static float* s_dry_values_buf = nullptr;
static size_t s_dry_buf_cap = 0;

// Sparse penalty application kernel: subtracts penalty from each listed token.
__global__ void apply_dry_sparse_kernel(float* __restrict__ logits,
                                        const int32_t* __restrict__ penalty_tokens,
                                        const float* __restrict__ penalty_values, int n_penalties) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_penalties) {
        logits[penalty_tokens[idx]] -= penalty_values[idx];
    }
}

void apply_dry_penalty(float* d_logits, int vocab_size, const int32_t* host_token_ids, int n_tokens,
                       float multiplier, float base, int allowed_length, int penalty_last_n,
                       cudaStream_t stream) {
    if (multiplier <= 0.0f || n_tokens < 2)
        return;

    int search_start = (penalty_last_n > 0) ? std::max(0, n_tokens - penalty_last_n) : 0;

    // CPU: scan history for suffix matches, compute max match length per token.
    // Use a flat array indexed by token ID (no heap allocation) instead of unordered_map.
    // Reuse a static buffer to avoid per-call allocation for large vocab.
    static thread_local std::vector<int> match_buf;
    if (static_cast<int>(match_buf.size()) < vocab_size) {
        match_buf.assign(vocab_size, 0);
    }
    // Zero only entries we write (sparse clear is faster than memset for large vocab)
    std::vector<int32_t> touched_tokens;

    for (int pos = search_start; pos < n_tokens; pos++) {
        int match_len = 0;
        int a = pos - 1;
        int b = n_tokens - 1;
        while (a >= search_start && b >= 0 && host_token_ids[a] == host_token_ids[b]) {
            match_len++;
            a--;
            b--;
        }
        if (match_len > allowed_length) {
            int32_t token = host_token_ids[pos];
            if (token >= 0 && token < vocab_size) {
                if (match_buf[token] == 0)
                    touched_tokens.push_back(token);
                if (match_len > match_buf[token])
                    match_buf[token] = match_len;
            }
        }
    }

    if (touched_tokens.empty())
        return;

    // Build sparse penalty arrays
    int n = static_cast<int>(touched_tokens.size());
    std::vector<int32_t> h_tokens(n);
    std::vector<float> h_values(n);
    for (int i = 0; i < n; i++) {
        int32_t tok = touched_tokens[i];
        h_tokens[i] = tok;
        h_values[i] = multiplier * std::pow(base, static_cast<float>(match_buf[tok] - allowed_length));
        match_buf[tok] = 0;  // sparse clear
    }

    // Upload to GPU and apply — reuse persistent buffers to avoid per-call cudaMalloc
    // (buffers are file-scope; pre-allocated by sampling_preallocate_dry at engine init)
    size_t needed = static_cast<size_t>(n);
    if (!s_dry_tokens_buf || !s_dry_values_buf)
        return;  // preallocation did not run or the arena was closed
    if (needed > s_dry_buf_cap) {
        // Unreachable by construction, and that is why the grow-and-realloc path
        // that used to live here is gone: `n` counts DISTINCT tokens collected
        // from the penalty window, so it cannot exceed the token history, which
        // cannot exceed max_seq_len — the capacity sampling_preallocate_dry()
        // takes at engine init. The old path freed and re-cudaMalloc'd both
        // buffers on the sampling hot path, behind a cudaStreamSynchronize.
        // Clamping keeps the impossible case bounded instead of allocating for
        // it (A7 step 8, AUDIT B72).
        IMP_LOG_WARN("apply_dry_penalty: %zu tokens exceeds the %zu-slot capacity — applying the "
                     "first %zu. This should be impossible; please report the config.",
                     needed, s_dry_buf_cap, s_dry_buf_cap);
        n = static_cast<int>(s_dry_buf_cap);
    }

    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(s_dry_tokens_buf, h_tokens.data(), n * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(s_dry_values_buf, h_values.data(), n * sizeof(float),
                                       cudaMemcpyHostToDevice, stream));

    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_dry_sparse_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(d_logits, s_dry_tokens_buf, s_dry_values_buf, n);
    IMP_CUDA_CHECK_LAUNCH();
}

// ===========================================================================
// Mirostat v2 sampling
// ===========================================================================

// Single-block kernel: computes log-sum-exp, filters by surprise threshold,
// samples from filtered set, and outputs token + surprise.
__global__ void mirostat_v2_sample_kernel(const float* __restrict__ logits, int vocab_size, float mu,
                                          float inv_temperature, unsigned int seed,
                                          int32_t* __restrict__ d_result, float* __restrict__ d_surprise) {
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float s_warp[NUM_WARPS];
    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ float s_fsum;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // --- Step 1: Find max logit ---
    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += blockDim.x)
        local_max = fmaxf(local_max, logits[i]);

    local_max = warp_reduce_max(local_max);
    if (lane_id == 0)
        s_warp[warp_id] = local_max;
    __syncthreads();

    if (tid == 0) {
        float mx = -FLT_MAX;
        for (int w = 0; w < NUM_WARPS; w++)
            mx = fmaxf(mx, s_warp[w]);
        s_max = mx;
    }
    __syncthreads();
    float gmax = s_max;

    // --- Step 2: Compute sum of exp((logit - max) * inv_temperature) ---
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x)
        local_sum += expf((logits[i] - gmax) * inv_temperature);

    local_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0)
        s_warp[warp_id] = local_sum;
    __syncthreads();

    if (tid == 0) {
        float sm = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++)
            sm += s_warp[w];
        s_sum = sm;
    }
    __syncthreads();

    // Mirostat threshold: keep tokens with surprise ≤ mu
    // With temperature T, p_i = exp((l_i - max)/T) / sum_exp
    // surprise_i = -log2(p_i) ≤ mu
    // ⟺ (l_i - max)/T ≥ log(sum_exp) - mu * ln(2)
    // ⟺ l_i ≥ max + T * (log(sum_exp) - mu * ln(2))
    float temperature = (inv_temperature > 0.0f) ? (1.0f / inv_temperature) : 1.0f;
    float log_sum_exp = logf(s_sum);
    float threshold = gmax + temperature * (log_sum_exp - mu * 0.6931471805599453f);

    // --- Step 3: Compute filtered probability sum ---
    float local_fsum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        if (logits[i] >= threshold)
            local_fsum += expf((logits[i] - gmax) * inv_temperature);
    }

    local_fsum = warp_reduce_sum(local_fsum);
    if (lane_id == 0)
        s_warp[warp_id] = local_fsum;
    __syncthreads();

    if (tid == 0) {
        float fs = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++)
            fs += s_warp[w];
        // Fallback: if no tokens pass threshold, use entire distribution
        s_fsum = (fs > 0.0f) ? fs : s_sum;
    }
    __syncthreads();

    float fsum = s_fsum;
    bool use_threshold = (fsum < s_sum * 0.9999f);

    // --- Step 4: Sample from filtered distribution ---
    // Thread 0 scans through vocab, accumulating filtered probabilities.
    if (tid == 0) {
        float inv_fsum = 1.0f / fsum;
        unsigned int rng = seed;
        float r = lcg_rand_float(rng);

        float acc = 0.0f;
        int chosen = 0;
        bool found = false;

        for (int i = 0; i < vocab_size; i++) {
            if (!use_threshold || logits[i] >= threshold) {
                float p = expf((logits[i] - gmax) * inv_temperature) * inv_fsum;
                acc += p;
                if (r < acc) {
                    chosen = i;
                    found = true;
                    break;
                }
            }
        }

        // Fallback: pick highest-logit token
        if (!found) {
            float best = -FLT_MAX;
            for (int i = 0; i < vocab_size; i++) {
                if (logits[i] > best) {
                    best = logits[i];
                    chosen = i;
                }
            }
        }

        // Compute surprise using temperature-adjusted probability
        float chosen_prob = expf((logits[chosen] - gmax) * inv_temperature) / s_sum;
        float surprise = -log2f(fmaxf(chosen_prob, 1e-30f));

        d_result[0] = chosen;
        d_surprise[0] = surprise;
    }
}

static int32_t sample_mirostat_v2_impl(const Tensor& logits, float temperature, float tau, float eta,
                                       float* mu, unsigned int seed, int32_t* d_result, bool owns_result,
                                       cudaStream_t stream) {
    const int vocab_size = static_cast<int>(logits.shape[0]);
    const float* d_logits = static_cast<const float*>(logits.data);

    if (temperature <= 0.0f)
        temperature = 1.0f;
    float inv_temperature = 1.0f / temperature;

    // Surprise value stored right after the token result
    float* d_surprise = reinterpret_cast<float*>(d_result + 1);

    mirostat_v2_sample_kernel<<<1, BLOCK_SIZE, 0, stream>>>(d_logits, vocab_size, *mu, inv_temperature, seed,
                                                            d_result, d_surprise);
    IMP_CUDA_CHECK_LAUNCH();

    // Read results
    int32_t h_result = 0;
    float h_surprise = 0.0f;
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(&h_result, d_result, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    IMP_CUDA_CHECK_LOG(
        cudaMemcpyAsync(&h_surprise, d_surprise, sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    if (owns_result)
        IMP_CUDA_CHECK_LOG(cudaFree(d_result));

    // Update mu: mu = mu - eta * (surprise - tau)
    *mu = *mu - eta * (h_surprise - tau);

    return h_result;
}

int32_t sample_mirostat_v2(const Tensor& logits, float temperature, float tau, float eta, float* mu,
                           unsigned int seed, cudaStream_t stream) {
    // Allocate temp buffer: 4 bytes for token + 4 bytes for surprise
    int32_t* d_result = nullptr;
    if (cudaMalloc(&d_result, 2 * sizeof(int32_t)) != cudaSuccess) {
        IMP_LOG_ERROR("sample_mirostat_v2: cudaMalloc failed");
        return 0;
    }
    return sample_mirostat_v2_impl(logits, temperature, tau, eta, mu, seed, d_result, true, stream);
}

int32_t sample_mirostat_v2(const Tensor& logits, float temperature, float tau, float eta, float* mu,
                           unsigned int seed, int32_t* d_result, cudaStream_t stream) {
    return sample_mirostat_v2_impl(logits, temperature, tau, eta, mu, seed, d_result, false, stream);
}

// ============================================================================
// CPU-side logprob computation
// ============================================================================

void compute_logprobs_cpu(const float* logits, int vocab_size, int32_t sampled_token, int top_n,
                          LogprobResult* out) {
    // 1. Find max for numerical stability
    float max_val = -FLT_MAX;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > max_val)
            max_val = logits[i];
    }

    // 2. Compute log-sum-exp
    double sum_exp = 0.0;
    for (int i = 0; i < vocab_size; i++) {
        sum_exp += std::exp(static_cast<double>(logits[i]) - static_cast<double>(max_val));
    }
    float log_sum_exp = static_cast<float>(std::log(sum_exp)) + max_val;

    // 3. Extract sampled token's logprob
    out->sampled_logprob = logits[sampled_token] - log_sum_exp;

    // 4. Top-N via partial sort with min-heap
    out->top.clear();
    if (top_n <= 0)
        return;

    // Use a simple approach: collect all (logprob, token) and partial sort
    // For vocab ~150K and top_n <= 20, this is fast enough (~0.3ms)
    struct Entry {
        float logprob;
        int32_t token;
        bool operator<(const Entry& o) const { return logprob > o.logprob; }  // max-heap order
    };

    // Min-heap of size top_n to track the top-N largest
    std::vector<Entry> heap;
    heap.reserve(top_n + 1);

    for (int i = 0; i < vocab_size; i++) {
        float lp = logits[i] - log_sum_exp;
        if (static_cast<int>(heap.size()) < top_n) {
            heap.push_back({lp, i});
            std::push_heap(heap.begin(), heap.end(), [](const Entry& a, const Entry& b) {
                return a.logprob > b.logprob;  // min-heap: smallest logprob at top
            });
        } else if (lp > heap[0].logprob) {
            std::pop_heap(heap.begin(), heap.end(),
                          [](const Entry& a, const Entry& b) { return a.logprob > b.logprob; });
            heap.back() = {lp, i};
            std::push_heap(heap.begin(), heap.end(),
                           [](const Entry& a, const Entry& b) { return a.logprob > b.logprob; });
        }
    }

    // Sort descending by logprob
    std::sort(heap.begin(), heap.end(), [](const Entry& a, const Entry& b) { return a.logprob > b.logprob; });

    out->top.reserve(heap.size());
    for (const auto& e : heap) {
        out->top.push_back({e.token, e.logprob});
    }
}

void sampling_preallocate_dry(int max_seq_len, cudaStream_t /*stream*/) {
    if (max_seq_len <= 0)
        return;
    size_t cap = static_cast<size_t>(max_seq_len);
    if (cap <= s_dry_buf_cap)
        return;  // already large enough

    // T2 (A7 step 8). Engine-lifetime, sized once from max_seq_len, and the
    // caller already treats a null buffer as "DRY penalty off" — so no
    // direct-allocation fallback and the sites leave the I1 allowlist rather
    // than moving (AUDIT B47).
    s_dry_tokens_buf = nullptr;
    s_dry_values_buf = nullptr;
    s_dry_buf_cap = 0;

    auto tokens = engine_arena().take_bytes(cap * sizeof(int32_t));
    auto values = engine_arena().take_bytes(cap * sizeof(float));
    if (tokens.empty() || values.empty()) {
        IMP_LOG_WARN("sampling_preallocate_dry: %zu slots unavailable from the T2 arena — the DRY "
                     "penalty will be skipped",
                     cap);
        return;
    }
    s_dry_tokens_buf = reinterpret_cast<int32_t*>(tokens.data());
    s_dry_values_buf = reinterpret_cast<float*>(values.data());
    s_dry_buf_cap = cap;
    IMP_LOG_DEBUG("sampling_preallocate_dry: pre-allocated %zu DRY penalty slots", cap);
}

// Free persistent DRY penalty buffers. Called by sampling_cleanup().
void sampling_cleanup_dry() {
    // Arena-owned since A7 step 8 — ~Engine closes the arena after every
    // executor teardown, so only the pointer nulling remains here.
    s_dry_tokens_buf = nullptr;
    s_dry_values_buf = nullptr;
    s_dry_buf_cap = 0;
}

void sampling_cleanup() {
    sampling_cleanup_cub();
    sampling_cleanup_dry();
}

}  // namespace imp
