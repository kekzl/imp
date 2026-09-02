// History-sized repetition / frequency / presence penalties and the dispatch
// in front of the vocab sweeps (sampling_penalties.cu). Own translation unit:
// the file-size gate holds sampling_penalties.cu at the kernel ceiling.
#include "compute/sampling.h"
#include "compute/sampling_internal.cuh"
#include "core/cuda_static_reset.h"
#include "core/logging.h"
#include "memory/engine_arena.h"
#include <cuda_runtime.h>
#include "compute/pdl_device.cuh"
#include "runtime/pdl.h"

namespace imp {

// ===========================================================================
// History-sized penalties: the path the engine runs
// ===========================================================================
// The vocab sweeps above cost every vocab entry a pass over the history,
// O(vocab x n_tokens) per row: 192 us per decode step at 32 rows and ~300
// tokens of history on Qwen3.8-27B (nsys, 2026-09-02), growing linearly with
// the generation. Only tokens IN the history change a logit, so this form
// walks the history twice: count every token (16-bit halves of 32-bit words,
// word atomics), then the thread that claims a token's count with a CAS
// applies the penalty once and leaves the count at zero for the next step.
// Same per-token arithmetic as apply_penalties_body: bit-identical logits.
// Scratch: one 16-bit count per vocab entry per row, taken from the T2 arena
// once (sampling_preallocate_penalty_counts, charged as ExecT2Demand::
// penalty_counts); a row count above 65535 wraps, which no window reaches.
// Without the scratch the sweeps run.
static uint32_t* s_pen_counts = nullptr;
static int s_pen_counts_rows = 0;
static int s_pen_counts_vocab = 0;
static constexpr int kPenHistThreads = 512;

static inline int pen_words_per_row(int vocab_size) { return (vocab_size + 1) / 2; }

__device__ __forceinline__ void pen_count_add(uint32_t* words, int tok) {
    atomicAdd(words + (tok >> 1), (tok & 1) ? 0x10000u : 1u);
}
// Claims tok's count: returns it and zeroes that half-word, or 0 when another
// thread of the row claimed it first. The CAS keeps the neighbour's half.
__device__ __forceinline__ int pen_count_claim(uint32_t* words, int tok) {
    uint32_t* w = words + (tok >> 1);
    const uint32_t mask = (tok & 1) ? 0xFFFF0000u : 0x0000FFFFu;
    uint32_t old = __ldcg(w);
    while (true) {
        const uint32_t cnt = old & mask;
        if (cnt == 0)
            return 0;
        const uint32_t seen = atomicCAS(w, old, old & ~mask);
        if (seen == old)
            return static_cast<int>((tok & 1) ? (cnt >> 16) : cnt);
        old = seen;
    }
}
__device__ __forceinline__ void pen_apply_once(float* __restrict__ logits, int tok, int count,
                                               float repetition_penalty, float frequency_penalty,
                                               float presence_penalty) {
    float logit = logits[tok];
    if (repetition_penalty != 1.0f) {
        if (logit > 0.0f)
            logit /= repetition_penalty;
        else
            logit *= repetition_penalty;
    }
    logit -= frequency_penalty * static_cast<float>(count);
    logit -= presence_penalty;
    logits[tok] = logit;
}
// One block per row; the threads stride the history. Block-uniform control
// flow around the barrier: n_tokens is the same for every thread of the row.
__device__ __forceinline__ void penalties_hist_body(float* __restrict__ logits,
                                                    const int32_t* __restrict__ token_ids, int n_tokens,
                                                    int vocab_size, float repetition_penalty,
                                                    float frequency_penalty, float presence_penalty,
                                                    uint32_t* __restrict__ words) {
    for (int i = threadIdx.x; i < n_tokens; i += blockDim.x) {
        const int tok = token_ids[i];
        if (tok >= 0 && tok < vocab_size)
            pen_count_add(words, tok);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_tokens; i += blockDim.x) {
        const int tok = token_ids[i];
        if (tok < 0 || tok >= vocab_size)
            continue;
        const int c = pen_count_claim(words, tok);
        if (c > 0)
            pen_apply_once(logits, tok, c, repetition_penalty, frequency_penalty, presence_penalty);
    }
}
__global__ void penalties_hist_rows_kernel(const PenaltyRowArgs* __restrict__ rows, int vocab_size,
                                           uint32_t* __restrict__ counts, int words_per_row) {
    pdl_wait();
    const PenaltyRowArgs r = rows[blockIdx.y];
    uint32_t* words = counts + static_cast<size_t>(blockIdx.y) * words_per_row;
    if (r.n_tokens > 0)
        penalties_hist_body(r.logits, r.token_ids, r.n_tokens, vocab_size, r.repetition_penalty,
                            r.frequency_penalty, r.presence_penalty, words);
    // Bans after the penalties, ordered by the barrier: a banned id that also
    // sits in the history ends at -1e30, as the sweep leaves it.
    __syncthreads();
    for (int i = threadIdx.x; i < r.n_banned; i += blockDim.x) {
        const int tok = r.banned[i];
        if (tok >= 0 && tok < vocab_size)
            r.logits[tok] = -1e30f;
    }
}
// Single row (M=1): n_tokens from the argument, or from d_n_tokens inside a
// captured graph, with the repeat_last_n window applied here.
__global__ void penalties_hist_kernel(float* __restrict__ logits, const int32_t* __restrict__ token_ids,
                                      int n_tokens, const int* __restrict__ d_n_tokens, int repeat_last_n,
                                      int vocab_size, float repetition_penalty, float frequency_penalty,
                                      float presence_penalty, uint32_t* __restrict__ words) {
    if (d_n_tokens)
        n_tokens = *d_n_tokens;
    int start = 0;
    if (repeat_last_n > 0 && n_tokens > repeat_last_n)
        start = n_tokens - repeat_last_n;
    if (n_tokens <= start)
        return;
    penalties_hist_body(logits, token_ids + start, n_tokens - start, vocab_size, repetition_penalty,
                        frequency_penalty, presence_penalty, words);
}

bool sampling_preallocate_penalty_counts(int rows, int vocab_size) {
    if (rows <= 0 || vocab_size <= 0)
        return false;
    // Always re-taken: the caller re-opens the arena between engine loads.
    const size_t words = static_cast<size_t>(rows) * pen_words_per_row(vocab_size);
    auto slab = engine_arena().take_bytes(words * sizeof(uint32_t));
    if (slab.empty()) {
        IMP_LOG_WARN("sampling_preallocate_penalty_counts: %zu bytes unavailable from the T2 arena, "
                     "penalties stay on the vocab sweep",
                     words * sizeof(uint32_t));
        return false;
    }
    if (cudaMemset(slab.data(), 0, words * sizeof(uint32_t)) != cudaSuccess)
        return false;
    s_pen_counts = reinterpret_cast<uint32_t*>(slab.data());
    s_pen_counts_rows = rows;
    s_pen_counts_vocab = vocab_size;
    return true;
}

void sampling_reset_penalty_counts() {
    s_pen_counts = nullptr;
    s_pen_counts_rows = 0;
    s_pen_counts_vocab = 0;
}
// The arena that backed the counts closes with the engine (~Engine ->
// reset_static_cuda_state): re-arm the guard there, like every other tenant.
IMP_REGISTER_CUDA_STATIC_RESET(sampling_reset_penalty_counts);

static inline bool pen_hist_ready(int rows, int vocab_size) {
    return s_pen_counts != nullptr && rows <= s_pen_counts_rows && vocab_size <= s_pen_counts_vocab;
}

void launch_penalties_rows(const PenaltyRowArgs* d_rows, int n_rows, int vocab_size, cudaStream_t stream) {
    if (!pen_hist_ready(n_rows, vocab_size)) {
        launch_penalties_rows_sweep(d_rows, n_rows, vocab_size, stream);
        return;
    }
    dim3 grid(1, n_rows);
    pdl::enable_kernel(penalties_hist_rows_kernel);
    pdl::launch(penalties_hist_rows_kernel, grid, dim3(kPenHistThreads), size_t(0), stream, d_rows, vocab_size,
                s_pen_counts, pen_words_per_row(s_pen_counts_vocab));
    IMP_CUDA_CHECK_LAUNCH();
}

void apply_penalties(float* logits, int vocab_size, const int32_t* token_ids, int n_tokens,
                     float repetition_penalty, float frequency_penalty, float presence_penalty,
                     cudaStream_t stream) {
    if (n_tokens == 0)
        return;
    if (repetition_penalty == 1.0f && frequency_penalty == 0.0f && presence_penalty == 0.0f)
        return;
    if (!pen_hist_ready(1, vocab_size)) {
        apply_penalties_sweep(logits, vocab_size, token_ids, n_tokens, repetition_penalty, frequency_penalty,
                              presence_penalty, stream);
        return;
    }
    penalties_hist_kernel<<<1, kPenHistThreads, 0, stream>>>(logits, token_ids, n_tokens, nullptr, 0, vocab_size,
                                                             repetition_penalty, frequency_penalty,
                                                             presence_penalty, s_pen_counts);
    IMP_CUDA_CHECK_LAUNCH();
}

void apply_penalties_device_count(float* logits, int vocab_size, const int32_t* token_ids,
                                  const int* d_n_tokens, int repeat_last_n, float repetition_penalty,
                                  float frequency_penalty, float presence_penalty, cudaStream_t stream) {
    if (repetition_penalty == 1.0f && frequency_penalty == 0.0f && presence_penalty == 0.0f)
        return;
    if (!pen_hist_ready(1, vocab_size)) {
        apply_penalties_device_count_sweep(logits, vocab_size, token_ids, d_n_tokens, repeat_last_n,
                                           repetition_penalty, frequency_penalty, presence_penalty, stream);
        return;
    }
    penalties_hist_kernel<<<1, kPenHistThreads, 0, stream>>>(logits, token_ids, 0, d_n_tokens, repeat_last_n,
                                                             vocab_size, repetition_penalty, frequency_penalty,
                                                             presence_penalty, s_pen_counts);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
