#include "exec/executor.h"
#include "exec/executor_kernels.h"
#include "compute/embedding.h"
#include "compute/layernorm.h"
#include "compute/gemm.h"
#include "compute/gemm_q6k.h"
#include "compute/sampling.h"
#include "quant/quant_gemm.h"
#include "quant/nvfp4_gemm.h"
#include "compute/json_constrain.h"
#include "compute/schema_constrain.h"
#include "compute/regex_constrain.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Column-wise partial sum of the hidden-state buffer: out[d] = sum_t h[t][d].
// Embedding requests (#1005) mean-pool every token's hidden state; this runs
// once per prefill chunk so chunked inputs pool correctly (the old
// /v1/embeddings path could only pool single-pass prefills).
__global__ void hidden_pool_sum_kernel(const __half* __restrict__ hidden, int n_tokens, int d_model,
                                       float* __restrict__ out) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= d_model)
        return;
    float acc = 0.0f;
    for (int t = 0; t < n_tokens; t++)
        acc += __half2float(hidden[(size_t)t * d_model + d]);
    out[d] = acc;
}

// Ban specific token IDs by setting their logits to -inf.
// Used in the CUDA graph decode path where host-side logit manipulation
// (cudaMemcpyAsync per token) is not possible during graph replay.
__global__ __launch_bounds__(256) void ban_logits_kernel(float* __restrict__ logits,
                                                         const int32_t* __restrict__ banned_ids, int n_banned,
                                                         int vocab_size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_banned; i += gridDim.x * blockDim.x) {
        int32_t tid = banned_ids[i];
        if (tid >= 0 && tid < vocab_size)
            logits[tid] = -1e30f;
    }
}

namespace imp {

void GraphExecutor::pool_hidden_sum(int n_tokens, float* d_out, cudaStream_t stream) {
    Tensor hidden = view_hidden(n_tokens);
    const int d_model = static_cast<int>(hidden.shape[hidden.ndim - 1]);
    const int threads = 256;
    const int blocks = (d_model + threads - 1) / threads;
    hidden_pool_sum_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __half*>(hidden.data), n_tokens, d_model, d_out);
    IMP_CUDA_CHECK_LAUNCH();
}

int32_t GraphExecutor::forward(const InferenceState& state, cudaStream_t stream) {
    Tensor logits;
    forward_logits(state, logits, stream);

#ifdef IMP_DEBUG
    // Check for CUDA errors after the forward pass (debug only)
    {
        cudaError_t err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("CUDA error after forward: %s", cudaGetErrorString(err));
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("CUDA last error: %s", cudaGetErrorString(err));
        }
    }
#endif

    // Sample from the last token's logits.
    // forward_logits returns [1, V] for prefill, [n, V] for decode.
    // For single-token forward, always use row 0.
    Tensor last_logits = logits.slice(0, 1);
    int64_t vocab_shape[1] = {last_logits.shape[1]};
    last_logits = last_logits.reshape(1, vocab_shape);

    // Apply penalties before sampling (modifies logits in-place)
    float* logits_ptr = static_cast<float*>(last_logits.data);
    int vocab_size = static_cast<int>(last_logits.shape[0]);

    if (state.penalty_tokens != nullptr && state.n_penalty_tokens > 0) {
        // Apply repeat_last_n window: only scan the last N tokens
        const int32_t* pen_ptr = state.penalty_tokens;
        int pen_n = state.n_penalty_tokens;
        if (state.repeat_last_n > 0 && pen_n > state.repeat_last_n) {
            pen_ptr += (pen_n - state.repeat_last_n);
            pen_n = state.repeat_last_n;
        }
        apply_penalties(logits_ptr, vocab_size, pen_ptr, pen_n, state.repetition_penalty,
                        state.frequency_penalty, state.presence_penalty, stream);
    }

    if (state.dry_multiplier > 0.0f && state.host_penalty_tokens != nullptr && state.n_penalty_tokens > 0) {
        apply_dry_penalty(logits_ptr, vocab_size, state.host_penalty_tokens, state.n_penalty_tokens,
                          state.dry_multiplier, state.dry_base, state.dry_allowed_length,
                          state.dry_penalty_last_n, stream);
    }

    // Ban special tokens (e.g. <|im_start|>, <|im_end|>) from generation.
    // These are chat template delimiters that should never appear in model output.
    // Without this, the model can emit <|im_start|> mid-generation, which starts
    // a phantom new turn and causes output degeneration (Qwen3, Llama3, etc.).
    if (state.banned_tokens != nullptr && state.n_banned_tokens > 0) {
        // Small list (typically 2-5 tokens) — copy to device and set to -inf.
        // Use a small stack-allocated device buffer via cudaMemcpyAsync.
        float neg_inf = -1e30f;
        for (int bi = 0; bi < state.n_banned_tokens; bi++) {
            int32_t tid = state.banned_tokens[bi];
            if (tid >= 0 && tid < vocab_size) {
                IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(logits_ptr + tid, &neg_inf, sizeof(float),
                                                   cudaMemcpyHostToDevice, stream));
            }
        }
    }

    // Apply logit bias
    if (state.n_logit_bias > 0 && state.logit_bias != nullptr) {
        for (int i = 0; i < state.n_logit_bias; i++) {
            int32_t tid = state.logit_bias[i].first;
            float bias = state.logit_bias[i].second;
            if (tid >= 0 && tid < vocab_size) {
                float logit;
                IMP_CUDA_CHECK_LOG(
                    cudaMemcpy(&logit, logits_ptr + tid, sizeof(float), cudaMemcpyDeviceToHost));
                logit += bias;
                IMP_CUDA_CHECK_LOG(
                    cudaMemcpyAsync(logits_ptr + tid, &logit, sizeof(float), cudaMemcpyHostToDevice, stream));
            }
        }
    }

    // Force token: set all logits except force_token to -inf.
    // Used by think-budget to force </think> via logit manipulation
    // so the model generates it naturally into the KV cache (NVIDIA NIM approach).
    if (state.force_token >= 0 && state.force_token < vocab_size) {
        force_single_token(logits_ptr, vocab_size, state.force_token, stream);
    }

    // JSON/Schema mode: apply logit mask to constrain output
    if (state.force_token < 0) {
        // Only apply constraints when not forcing a token
        if (state.regex_constrainer) {
            state.regex_constrainer->apply_mask(logits_ptr, vocab_size, stream);
        } else if (state.schema_constrainer) {
            state.schema_constrainer->apply_mask(logits_ptr, vocab_size, stream);
        } else if (state.json_constrainer) {
            state.json_constrainer->apply_mask(logits_ptr, vocab_size, stream);
        }
    }

    int32_t token;
    if (state.mirostat == 2) {
        // Mirostat v2: handles temperature + filtering internally, skip min_p
        unsigned int seed = state.seed >= 0 ? static_cast<unsigned int>(state.seed) : 42u;
        token = d_sample_result_ ? sample_mirostat_v2(last_logits, state.temperature, state.mirostat_tau,
                                                      state.mirostat_eta, &state.mirostat_mu, seed,
                                                      d_sample_result_, stream)
                                 : sample_mirostat_v2(last_logits, state.temperature, state.mirostat_tau,
                                                      state.mirostat_eta, &state.mirostat_mu, seed, stream);
    } else {
        if (state.min_p > 0.0f) {
            apply_min_p(logits_ptr, vocab_size, state.min_p, stream);
        }
        if (state.typical_p > 0.0f && state.typical_p < 1.0f) {
            apply_typical_p(logits_ptr, vocab_size, state.typical_p, stream);
        }

        if (state.temperature <= 0.0f || state.top_k == 1) {
            if (d_sample_result_ && h_sample_pinned_) {
                sample_greedy_device(last_logits, d_sample_result_, h_sample_pinned_, stream);
                cudaStreamSynchronize(stream);
                token = *h_sample_pinned_;
            } else if (d_sample_result_) {
                token = sample_greedy(last_logits, d_sample_result_, stream);
            } else {
                token = sample_greedy(last_logits, stream);
            }
        } else {
            int top_k = state.top_k > 0 ? state.top_k : 50;
            float top_p = state.top_p > 0.0f ? state.top_p : 1.0f;
            unsigned int seed = state.seed >= 0 ? static_cast<unsigned int>(state.seed) : 42u;
            if (d_sample_result_ && h_sample_pinned_) {
                sample_topk_topp_device(last_logits, top_k, top_p, state.temperature, seed, d_sample_result_,
                                        h_sample_pinned_, stream);
                cudaStreamSynchronize(stream);
                token = *h_sample_pinned_;
            } else if (d_sample_result_) {
                token = sample_topk_topp(last_logits, top_k, top_p, state.temperature, seed, d_sample_result_,
                                         stream);
            } else {
                token = sample_topk_topp(last_logits, top_k, top_p, state.temperature, seed, stream);
            }
        }
    }

    return token;
}

std::vector<int32_t> GraphExecutor::sample_from_logits(const Tensor& logits, const InferenceState& state,
                                                       cudaStream_t stream) {
    int n_seq = state.n_sequences;
    std::vector<int32_t> tokens(n_seq);

    // Helper: flatten [1, V] to [V] for sampling
    auto flatten_logits = [](Tensor t) -> Tensor {
        int64_t vocab_shape[1] = {t.shape[t.ndim - 1]};
        return t.reshape(1, vocab_shape);
    };

    // Helper: apply penalties + filters to logits before sampling
    auto apply_pre_sample = [&](Tensor& seq_logits, const InferenceState& st) {
        float* lp = static_cast<float*>(seq_logits.data);
        int vocab = static_cast<int>(seq_logits.shape[0]);

        if (st.penalty_tokens != nullptr && st.n_penalty_tokens > 0) {
            const int32_t* pen_ptr = st.penalty_tokens;
            int pen_n = st.n_penalty_tokens;
            if (st.repeat_last_n > 0 && pen_n > st.repeat_last_n) {
                pen_ptr += (pen_n - st.repeat_last_n);
                pen_n = st.repeat_last_n;
            }
            apply_penalties(lp, vocab, pen_ptr, pen_n, st.repetition_penalty, st.frequency_penalty,
                            st.presence_penalty, stream);
        }
        if (st.dry_multiplier > 0.0f && st.host_penalty_tokens != nullptr && st.n_penalty_tokens > 0) {
            apply_dry_penalty(lp, vocab, st.host_penalty_tokens, st.n_penalty_tokens, st.dry_multiplier,
                              st.dry_base, st.dry_allowed_length, st.dry_penalty_last_n, stream);
        }
        if (st.n_logit_bias > 0 && st.logit_bias != nullptr) {
            for (int i = 0; i < st.n_logit_bias; i++) {
                int32_t tid = st.logit_bias[i].first;
                float bias = st.logit_bias[i].second;
                if (tid >= 0 && tid < vocab) {
                    float logit;
                    IMP_CUDA_CHECK_LOG(cudaMemcpy(&logit, lp + tid, sizeof(float), cudaMemcpyDeviceToHost));
                    logit += bias;
                    IMP_CUDA_CHECK_LOG(
                        cudaMemcpyAsync(lp + tid, &logit, sizeof(float), cudaMemcpyHostToDevice, stream));
                }
            }
        }
        if (st.regex_constrainer) {
            st.regex_constrainer->apply_mask(lp, vocab, stream);
        } else if (st.schema_constrainer) {
            st.schema_constrainer->apply_mask(lp, vocab, stream);
        } else if (st.json_constrainer) {
            st.json_constrainer->apply_mask(lp, vocab, stream);
        }
        // Ban special tokens (chat template delimiters etc.). MUST happen
        // before sampling — without this, greedy can pick a banned token
        // (e.g. Gemma-4 NVFP4 picks `<|channel>` as the natural argmax) and
        // the request finishes immediately because is_stop_token treats banned
        // tokens as stop tokens. forward() (line 88) already does this; the
        // sample_from_logits / sample_single_from_logits / use_event_sync
        // prefill paths historically forgot to. Match forward()'s impl.
        if (st.banned_tokens != nullptr && st.n_banned_tokens > 0) {
            float neg_inf = -1e30f;
            for (int bi = 0; bi < st.n_banned_tokens; bi++) {
                int32_t tid = st.banned_tokens[bi];
                if (tid >= 0 && tid < vocab) {
                    IMP_CUDA_CHECK_LOG(
                        cudaMemcpyAsync(lp + tid, &neg_inf, sizeof(float), cudaMemcpyHostToDevice, stream));
                }
            }
        }
        if (st.min_p > 0.0f) {
            apply_min_p(lp, vocab, st.min_p, stream);
        }
        if (st.typical_p > 0.0f && st.typical_p < 1.0f) {
            apply_typical_p(lp, vocab, st.typical_p, stream);
        }
    };

    if (state.is_prefill || n_seq <= 1) {
        // Single sequence or prefill: logits is [1, V] (forward_logits already sliced)
        Tensor last_logits = flatten_logits(logits.slice(0, 1));
        apply_pre_sample(last_logits, state);

        if (state.mirostat == 2) {
            unsigned int seed = state.seed >= 0 ? static_cast<unsigned int>(state.seed) : 42u;
            tokens[0] = d_sample_result_
                            ? sample_mirostat_v2(last_logits, state.temperature, state.mirostat_tau,
                                                 state.mirostat_eta, &state.mirostat_mu, seed,
                                                 d_sample_result_, stream)
                            : sample_mirostat_v2(last_logits, state.temperature, state.mirostat_tau,
                                                 state.mirostat_eta, &state.mirostat_mu, seed, stream);
        } else {
            tokens[0] =
                (state.temperature <= 0.0f || state.top_k == 1)
                    ? (d_sample_result_ ? sample_greedy(last_logits, d_sample_result_, stream)
                                        : sample_greedy(last_logits, stream))
                    : (d_sample_result_
                           ? sample_topk_topp(last_logits, state.top_k > 0 ? state.top_k : 50,
                                              state.top_p > 0.0f ? state.top_p : 1.0f, state.temperature,
                                              state.seed >= 0 ? static_cast<unsigned int>(state.seed) : 42u,
                                              d_sample_result_, stream)
                           : sample_topk_topp(last_logits, state.top_k > 0 ? state.top_k : 50,
                                              state.top_p > 0.0f ? state.top_p : 1.0f, state.temperature,
                                              state.seed >= 0 ? static_cast<unsigned int>(state.seed) : 42u,
                                              stream));
        }
    } else {
        // Batched decode: n_tokens == n_sequences, each row is one sequence's
        // logits. Enqueue every sequence's penalty+sampler chain back-to-back
        // into per-sequence scratch slots, then gather ALL tokens with ONE
        // pinned D2H + ONE stream sync. The previous per-sequence readback
        // (pageable 4-byte D2H + cudaStreamSynchronize each) serialized the
        // batch against ~200 us host round-trips: at n=16 sustained load that
        // was ~3 ms host time per decode step = 29% GPU idle (nsys,
        // 2026-07-12). Kernels, parameter normalization, and per-sequence
        // seeds are identical to the fallback loop, so tokens are
        // bit-identical.
        const bool greedy = (state.temperature <= 0.0f || state.top_k == 1);
        const int top_k = state.top_k > 0 ? state.top_k : 50;
        const float top_p = state.top_p > 0.0f ? state.top_p : 1.0f;
        // Eligibility is batch-uniform (depends only on shared sampling params
        // and vocab), so decide BEFORE applying any penalties — no sequence is
        // ever half-processed across the two paths. top_k <= 0 / > vocab
        // normalizes to vocab inside the samplers, which lands in the CUB
        // regime (> SAMPLE_MAX_TOP_K) that syncs internally.
        const int vocab = static_cast<int>(logits.shape[logits.ndim - 1]);
        const int eff_top_k = (top_k <= 0 || top_k > vocab) ? vocab : top_k;
        const bool can_batch = d_sample_result_ && h_sample_pinned_ && n_seq <= sample_slots_ &&
                               (greedy || eff_top_k <= SAMPLE_MAX_TOP_K);
        if (can_batch) {
            for (int i = 0; i < n_seq; i++) {
                Tensor seq_logits = flatten_logits(logits.slice(i, i + 1));
                apply_pre_sample(seq_logits, state);
                auto* slot = reinterpret_cast<int32_t*>(reinterpret_cast<char*>(d_sample_result_) +
                                                        static_cast<size_t>(i) * SAMPLE_SCRATCH_BYTES);
                if (greedy) {
                    sample_greedy_async(seq_logits, slot, stream);
                } else {
                    unsigned int seed =
                        state.seed >= 0 ? static_cast<unsigned int>(state.seed + i) : (42u + i);
                    bool ok = sample_topk_topp_async(seq_logits, top_k, top_p, state.temperature, seed,
                                                     slot, stream);
                    (void)ok;  // eligibility pre-checked above; cannot decline here
                }
            }
            // Slots are SAMPLE_SCRATCH_BYTES apart; the token is the first
            // int32 of each slot — one strided D2H gathers the whole batch.
            IMP_CUDA_CHECK_LOG(cudaMemcpy2DAsync(h_sample_pinned_, sizeof(int32_t), d_sample_result_,
                                                 SAMPLE_SCRATCH_BYTES, sizeof(int32_t), n_seq,
                                                 cudaMemcpyDeviceToHost, stream));
            IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
            for (int i = 0; i < n_seq; i++)
                tokens[i] = h_sample_pinned_[i];
        } else {
            // Fallback: per-sequence synchronous sampling (no slot buffers, or
            // top_k in the CUB regime).
            for (int i = 0; i < n_seq; i++) {
                Tensor seq_logits = flatten_logits(logits.slice(i, i + 1));
                apply_pre_sample(seq_logits, state);
                tokens[i] =
                    greedy ? (d_sample_result_ ? sample_greedy(seq_logits, d_sample_result_, stream)
                                               : sample_greedy(seq_logits, stream))
                           : (d_sample_result_
                                  ? sample_topk_topp(seq_logits, top_k, top_p, state.temperature,
                                                     state.seed >= 0
                                                         ? static_cast<unsigned int>(state.seed + i)
                                                         : (42u + i),
                                                     d_sample_result_, stream)
                                  : sample_topk_topp(seq_logits, top_k, top_p, state.temperature,
                                                     state.seed >= 0
                                                         ? static_cast<unsigned int>(state.seed + i)
                                                         : (42u + i),
                                                     stream));
            }
        }
    }

    return tokens;
}

// Shared per-row logits filter chain (penalties, DRY, token bans, logit
// bias, forced token, schema/json masks, min-p, typical-p) — used by both
// the synchronous sample_single_from_logits and the enqueue-only
// sample_single_from_logits_async (which declines the host-blocking
// logit-bias mode before calling this).
void GraphExecutor::apply_row_filters_(float* lp, int vocab, const InferenceState& state,
                                       cudaStream_t stream) {
    if (state.penalty_tokens != nullptr && state.n_penalty_tokens > 0) {
        const int32_t* pen_ptr = state.penalty_tokens;
        int pen_n = state.n_penalty_tokens;
        if (state.repeat_last_n > 0 && pen_n > state.repeat_last_n) {
            pen_ptr += (pen_n - state.repeat_last_n);
            pen_n = state.repeat_last_n;
        }
        apply_penalties(lp, vocab, pen_ptr, pen_n, state.repetition_penalty, state.frequency_penalty,
                        state.presence_penalty, stream);
    }
    if (state.dry_multiplier > 0.0f && state.host_penalty_tokens != nullptr && state.n_penalty_tokens > 0) {
        apply_dry_penalty(lp, vocab, state.host_penalty_tokens, state.n_penalty_tokens, state.dry_multiplier,
                          state.dry_base, state.dry_allowed_length, state.dry_penalty_last_n, stream);
    }
    // Ban special tokens. The list is engine-static (same host array every
    // step), so cache the device copy instead of re-allocating + re-uploading
    // it per row per step — at n=16 that was 16 cudaMallocAsync/H2D/FreeAsync
    // triplets per decode step for identical bytes.
    if (state.banned_tokens != nullptr && state.n_banned_tokens > 0) {
        const bool cache_hit = d_banned_cache_ != nullptr &&
                               banned_cache_src_ == state.banned_tokens &&
                               banned_cache_n_ == state.n_banned_tokens;
        if (!cache_hit) {
            size_t ban_bytes = static_cast<size_t>(state.n_banned_tokens) * sizeof(int32_t);
            if (banned_cache_capacity_ < static_cast<size_t>(state.n_banned_tokens)) {
                if (d_banned_cache_)
                    IMP_CUDA_CHECK_LOG(cudaFree(d_banned_cache_));
                d_banned_cache_ = nullptr;
                if (cudaMalloc(&d_banned_cache_, ban_bytes) != cudaSuccess) {
                    d_banned_cache_ = nullptr;
                    banned_cache_capacity_ = 0;
                }
                banned_cache_capacity_ = d_banned_cache_ ? state.n_banned_tokens : 0;
            }
            if (d_banned_cache_) {
                cudaMemcpyAsync(d_banned_cache_, state.banned_tokens, ban_bytes,
                                cudaMemcpyHostToDevice, stream);
                banned_cache_src_ = state.banned_tokens;
                banned_cache_n_ = state.n_banned_tokens;
            }
        }
        if (d_banned_cache_) {
            constexpr int kBanThreads = 256;
            int blocks = (state.n_banned_tokens + kBanThreads - 1) / kBanThreads;
            ban_logits_kernel<<<blocks, kBanThreads, 0, stream>>>(lp, d_banned_cache_,
                                                                  state.n_banned_tokens, vocab);
            IMP_CUDA_CHECK_LAUNCH();
        }
    }
    // Apply logit bias
    if (state.n_logit_bias > 0 && state.logit_bias != nullptr) {
        for (int i = 0; i < state.n_logit_bias; i++) {
            int32_t tid = state.logit_bias[i].first;
            float bias = state.logit_bias[i].second;
            if (tid >= 0 && tid < vocab) {
                float logit;
                IMP_CUDA_CHECK_LOG(cudaMemcpy(&logit, lp + tid, sizeof(float), cudaMemcpyDeviceToHost));
                logit += bias;
                IMP_CUDA_CHECK_LOG(
                    cudaMemcpyAsync(lp + tid, &logit, sizeof(float), cudaMemcpyHostToDevice, stream));
            }
        }
    }
    // Force token (think-budget)
    if (state.force_token >= 0 && state.force_token < vocab) {
        force_single_token(lp, vocab, state.force_token, stream);
    }
    if (state.force_token < 0) {
        if (state.regex_constrainer)
            state.regex_constrainer->apply_mask(lp, vocab, stream);
        else if (state.schema_constrainer)
            state.schema_constrainer->apply_mask(lp, vocab, stream);
        else if (state.json_constrainer)
            state.json_constrainer->apply_mask(lp, vocab, stream);
    }
    if (state.min_p > 0.0f)
        apply_min_p(lp, vocab, state.min_p, stream);
    if (state.typical_p > 0.0f && state.typical_p < 1.0f)
        apply_typical_p(lp, vocab, state.typical_p, stream);
}

int32_t GraphExecutor::sample_single_from_logits(const Tensor& logits, const InferenceState& state,
                                                 cudaStream_t stream) {
    // Flatten [1, V] to [V]
    int64_t vocab_shape[1] = {logits.shape[logits.ndim - 1]};
    Tensor flat = logits.slice(0, 1).reshape(1, vocab_shape);

    float* lp = static_cast<float*>(flat.data);
    int vocab = static_cast<int>(flat.shape[0]);
    apply_row_filters_(lp, vocab, state, stream);

    // Sample
    if (state.mirostat == 2) {
        unsigned int seed = state.seed >= 0 ? static_cast<unsigned int>(state.seed) : 42u;
        return d_sample_result_
                   ? sample_mirostat_v2(flat, state.temperature, state.mirostat_tau, state.mirostat_eta,
                                        &state.mirostat_mu, seed, d_sample_result_, stream)
                   : sample_mirostat_v2(flat, state.temperature, state.mirostat_tau, state.mirostat_eta,
                                        &state.mirostat_mu, seed, stream);
    }
    return (state.temperature <= 0.0f || state.top_k == 1)
               ? (d_sample_result_ ? sample_greedy(flat, d_sample_result_, stream)
                                   : sample_greedy(flat, stream))
               : (d_sample_result_
                      ? sample_topk_topp(flat, state.top_k > 0 ? state.top_k : 50,
                                         state.top_p > 0.0f ? state.top_p : 1.0f, state.temperature,
                                         state.seed >= 0 ? static_cast<unsigned int>(state.seed) : 42u,
                                         d_sample_result_, stream)
                      : sample_topk_topp(flat, state.top_k > 0 ? state.top_k : 50,
                                         state.top_p > 0.0f ? state.top_p : 1.0f, state.temperature,
                                         state.seed >= 0 ? static_cast<unsigned int>(state.seed) : 42u,
                                         stream));
}

bool GraphExecutor::sample_single_from_logits_async(const Tensor& logits, const InferenceState& state,
                                                    int slot_idx, cudaStream_t stream) {
    // Enqueue-only per-row sampling for the batched decode loop: filters +
    // sampler land on the stream writing into scratch slot `slot_idx`; the
    // caller gathers ALL rows' tokens with collect_sampled_tokens (one pinned
    // D2H + one sync). The synchronous per-row readback cost ~850 us of
    // blocked host time per sequence per step at n=16 sustained serving
    // (pageable 4-byte D2H + stream sync each, nsys 2026-07-12).
    if (!d_sample_result_ || !h_sample_pinned_ || slot_idx < 0 || slot_idx >= sample_slots_)
        return false;
    // Parity offset: the pipelined decode enqueues into the half selected by
    // set_sample_parity while the other half's gather is still in flight.
    const int abs_slot = sample_parity_ * sample_slots_ + slot_idx;
    // Sync-only sampling modes decline BEFORE any filter is applied, so the
    // caller can re-run this row through sample_single_from_logits untouched:
    // mirostat mutates host-side mu every step; logit_bias does per-entry
    // host read-modify-write on the logits.
    if (state.mirostat == 2)
        return false;
    if (state.n_logit_bias > 0 && state.logit_bias != nullptr)
        return false;

    int64_t vocab_shape[1] = {logits.shape[logits.ndim - 1]};
    Tensor flat = logits.slice(0, 1).reshape(1, vocab_shape);
    float* lp = static_cast<float*>(flat.data);
    const int vocab = static_cast<int>(flat.shape[0]);

    const bool greedy = (state.temperature <= 0.0f || state.top_k == 1);
    const int top_k = state.top_k > 0 ? state.top_k : 50;
    const int eff_top_k = (top_k <= 0 || top_k > vocab) ? vocab : top_k;
    if (!greedy && eff_top_k > SAMPLE_MAX_TOP_K)
        return false;  // CUB regime syncs internally

    apply_row_filters_(lp, vocab, state, stream);

    auto* slot = reinterpret_cast<int32_t*>(reinterpret_cast<char*>(d_sample_result_) +
                                            static_cast<size_t>(abs_slot) * SAMPLE_SCRATCH_BYTES);
    if (greedy) {
        sample_greedy_async(flat, slot, stream);
        return true;
    }
    unsigned int seed = state.seed >= 0 ? static_cast<unsigned int>(state.seed) : 42u;
    float temperature = state.temperature <= 0.0f ? 1.0f : state.temperature;
    if (h_row_args_ && d_row_args_) {
        // STASH the row for the row-parallel batched launch in
        // collect_sampled_tokens — n serialized <<<64>>>+<<<1>>> launch pairs
        // become ONE partial + ONE finalize launch for the whole batch.
        TopkRowArgs& a = h_row_args_[sample_parity_ * sample_slots_ + n_pending_topk_rows_++];
        a.logits = lp;
        a.d_result = slot;
        a.inv_temperature = 1.0f / temperature;
        a.top_p = state.top_p > 0.0f ? state.top_p : 1.0f;
        a.seed = seed;
        a.top_k = eff_top_k;
        pending_topk_max_k_ = std::max(pending_topk_max_k_, eff_top_k);
        pending_topk_vocab_ = vocab;
        return true;
    }
    bool ok = sample_topk_topp_async(flat, top_k, state.top_p > 0.0f ? state.top_p : 1.0f,
                                     state.temperature, seed, slot, stream);
    (void)ok;  // eligibility pre-checked above
    return true;
}

// Flush the stashed top-k rows of the ACTIVE parity half: one pinned H2D of
// the args, one partial + one finalize launch covering every row.
void GraphExecutor::flush_pending_topk_rows_(cudaStream_t stream) {
    if (n_pending_topk_rows_ <= 0)
        return;
    TopkRowArgs* h_base = h_row_args_ + sample_parity_ * sample_slots_;
    TopkRowArgs* d_base = d_row_args_ + sample_parity_ * sample_slots_;
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_base, h_base, sizeof(TopkRowArgs) * n_pending_topk_rows_,
                                       cudaMemcpyHostToDevice, stream));
    launch_topk_topp_rows(d_base, n_pending_topk_rows_, pending_topk_max_k_, pending_topk_vocab_,
                          stream);
    n_pending_topk_rows_ = 0;
    pending_topk_max_k_ = 0;
}

const int32_t* GraphExecutor::collect_sampled_tokens(int n_slots, cudaStream_t stream) {
    if (!d_sample_result_ || !h_sample_pinned_ || n_slots <= 0 || n_slots > sample_slots_) {
        n_pending_topk_rows_ = 0;
        pending_topk_max_k_ = 0;
        return nullptr;
    }
    flush_pending_topk_rows_(stream);
    // Slots are SAMPLE_SCRATCH_BYTES apart; the token is the first int32 of
    // each slot — one strided D2H gathers the whole batch.
    const size_t base = static_cast<size_t>(sample_parity_) * sample_slots_;
    IMP_CUDA_CHECK_LOG(cudaMemcpy2DAsync(h_sample_pinned_ + base, sizeof(int32_t),
                                         reinterpret_cast<char*>(d_sample_result_) +
                                             base * SAMPLE_SCRATCH_BYTES,
                                         SAMPLE_SCRATCH_BYTES, sizeof(int32_t), n_slots,
                                         cudaMemcpyDeviceToHost, stream));
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
    return h_sample_pinned_ + base;
}

// Event-based split of collect_sampled_tokens for the pipelined decode:
// flush + strided D2H + event record, NO stream sync. The engine enqueues
// the NEXT step's work after this and only waits on the event — so the wait
// covers exactly this gather, not the freshly enqueued step.
bool GraphExecutor::gather_sampled_tokens_async(int n_slots, cudaStream_t stream) {
    if (!sample_pipeline_ready() || n_slots <= 0 || n_slots > sample_slots_) {
        n_pending_topk_rows_ = 0;
        pending_topk_max_k_ = 0;
        return false;
    }
    flush_pending_topk_rows_(stream);
    const size_t base = static_cast<size_t>(sample_parity_) * sample_slots_;
    IMP_CUDA_CHECK_LOG(cudaMemcpy2DAsync(h_sample_pinned_ + base, sizeof(int32_t),
                                         reinterpret_cast<char*>(d_sample_result_) +
                                             base * SAMPLE_SCRATCH_BYTES,
                                         SAMPLE_SCRATCH_BYTES, sizeof(int32_t), n_slots,
                                         cudaMemcpyDeviceToHost, stream));
    IMP_CUDA_CHECK_LOG(cudaEventRecord(sample_gather_evt_[sample_parity_], stream));
    return true;
}

const int32_t* GraphExecutor::wait_gathered_tokens(int parity) {
    parity &= 1;
    if (!h_sample_pinned_ || !sample_gather_evt_[parity])
        return nullptr;
    IMP_CUDA_CHECK_LOG(cudaEventSynchronize(sample_gather_evt_[parity]));
    return h_sample_pinned_ + static_cast<size_t>(parity) * sample_slots_;
}

const int32_t* GraphExecutor::sample_slot_base(int parity) const {
    if (!d_sample_result_)
        return nullptr;
    return reinterpret_cast<const int32_t*>(reinterpret_cast<const char*>(d_sample_result_) +
                                            static_cast<size_t>(parity & 1) * sample_slots_ *
                                                SAMPLE_SCRATCH_BYTES);
}

std::vector<int32_t> GraphExecutor::forward_batch(const InferenceState& state, cudaStream_t stream) {
    Tensor logits;
    forward_logits(state, logits, stream);
    return sample_from_logits(logits, state, stream);
}

// ---------------------------------------------------------------------------
// Async decode: embedding from device token → forward → sample to device
// ---------------------------------------------------------------------------

void GraphExecutor::masked_sample_async(const InferenceState& state, const Tensor& logits, int32_t* d_result,
                                        int32_t* h_pinned, cudaStream_t stream) {
    int vocab = static_cast<int>(logits.shape[logits.ndim - 1]);
    float* lp = static_cast<float*>(logits.data);

    // Forced token (think-budget </think> injection): mirrors the eager
    // sampler — the force overrides bans and constraint masks.
    if (state.force_token >= 0 && state.force_token < vocab) {
        force_single_token(lp, vocab, state.force_token, stream);
        Tensor last_f = logits.slice(0, 1);
        int64_t vshape_f[1] = {last_f.shape[1]};
        last_f = last_f.reshape(1, vshape_f);
        sample_greedy_device(last_f, d_result, h_pinned, stream);
        return;
    }

    // Repetition / frequency / presence penalties — same order as the eager
    // sampler (penalties before bans and constraint mask). The engine uploads
    // the token history per tick (upload_penalties), exactly like eager.
    if (state.penalty_tokens != nullptr && state.n_penalty_tokens > 0) {
        const int32_t* pen_ptr = state.penalty_tokens;
        int pen_n = state.n_penalty_tokens;
        if (state.repeat_last_n > 0 && pen_n > state.repeat_last_n) {
            pen_ptr += (pen_n - state.repeat_last_n);
            pen_n = state.repeat_last_n;
        }
        apply_penalties(lp, vocab, pen_ptr, pen_n, state.repetition_penalty, state.frequency_penalty,
                        state.presence_penalty, stream);
    }

    // Banned special tokens — device list, graph-/replay-safe.
    if (state.d_banned_tokens && state.n_d_banned_tokens > 0) {
        constexpr int kBanThreads = 256;
        int blocks = (state.n_d_banned_tokens + kBanThreads - 1) / kBanThreads;
        ban_logits_kernel<<<blocks, kBanThreads, 0, stream>>>(lp, state.d_banned_tokens,
                                                              state.n_d_banned_tokens, vocab);
        IMP_CUDA_CHECK_LAUNCH();
    }

    // Grammar constraint mask (host-computed this step, uploaded stream-ordered).
    if (state.regex_constrainer)
        state.regex_constrainer->apply_mask(lp, vocab, stream);
    else if (state.schema_constrainer)
        state.schema_constrainer->apply_mask(lp, vocab, stream);
    else if (state.json_constrainer)
        state.json_constrainer->apply_mask(lp, vocab, stream);

    // Device-side sampling → d_result, async copy to h_pinned.
    Tensor last = logits.slice(0, 1);
    int64_t vocab_shape[1] = {last.shape[1]};
    last = last.reshape(1, vocab_shape);
    if (state.temperature <= 0.0f || state.top_k == 1) {
        sample_greedy_device(last, d_result, h_pinned, stream);
    } else {
        int top_k = state.top_k > 0 ? state.top_k : 50;
        float top_p = state.top_p > 0.0f ? state.top_p : 1.0f;
        unsigned int seed = state.seed >= 0 ? static_cast<unsigned int>(state.seed) : 42u;
        sample_topk_topp_device(last, top_k, top_p, state.temperature, seed, d_result, h_pinned, stream);
    }
}

void GraphExecutor::forward_decode_async(const InferenceState& state, int32_t* d_token_id, int32_t* h_mapped,
                                         cudaStream_t stream) {
    if (!initialized_) {
        IMP_LOG_ERROR("GraphExecutor::forward_decode_async called before init()");
        return;
    }

    // Delegate the heavy lifting (embedding → layers → final norm → LM head →
    // softcap) to the canonical forward_logits path. Caller must pre-set
    // state.token_ids = d_token_id so embedding_lookup reads the freshly
    // sampled token from device memory (CudaGraphConditionalRunner::setup
    // does this). Unifying here eliminates the parallel reimplementation
    // that previously diverged on Gemma-4 (samples <eos> at step 0).
    Tensor logits;
    forward_logits(state, logits, stream);

    // ---- Device-side post-processing (graph-safe, runs each iteration) ----
    int vocab = static_cast<int>(logits.shape[logits.ndim - 1]);
    float* lp = static_cast<float*>(logits.data);

    // Ban special tokens — device-side kernel (graph-safe).
    if (state.d_banned_tokens && state.n_d_banned_tokens > 0) {
        constexpr int kBanThreads = 256;
        int blocks = (state.n_d_banned_tokens + kBanThreads - 1) / kBanThreads;
        ban_logits_kernel<<<blocks, kBanThreads, 0, stream>>>(
            lp, state.d_banned_tokens, state.n_d_banned_tokens, vocab);
        IMP_CUDA_CHECK_LAUNCH();
    }

    // Repetition / frequency / presence penalties — device counter grows each
    // iteration as tokens are appended to the penalty ring.
    if (state.penalty_tokens != nullptr && state.d_n_penalty_tokens != nullptr) {
        apply_penalties_device_count(lp, vocab, state.penalty_tokens, state.d_n_penalty_tokens,
                                     state.repeat_last_n, state.repetition_penalty, state.frequency_penalty,
                                     state.presence_penalty, stream);
    }

    // ---- Async sampling → write to d_token_id + h_mapped (mapped pinned) ----
    Tensor last_logits = logits.slice(0, 1);
    int64_t vocab_shape[1] = {last_logits.shape[1]};
    last_logits = last_logits.reshape(1, vocab_shape);

    if (state.temperature <= 0.0f || state.top_k == 1) {
        sample_greedy_device(last_logits, d_token_id, h_mapped, stream);
    } else {
        int top_k = state.top_k > 0 ? state.top_k : 50;
        float top_p = state.top_p > 0.0f ? state.top_p : 1.0f;
        unsigned int seed = state.seed >= 0 ? static_cast<unsigned int>(state.seed) : 42u;
        sample_topk_topp_device(last_logits, top_k, top_p, state.temperature, seed, d_token_id, h_mapped,
                                stream);
    }
    // No cudaStreamSynchronize — host polls h_mapped asynchronously.
}

}  // namespace imp
