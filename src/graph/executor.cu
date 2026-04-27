#include "graph/executor.h"
#include "graph/executor_kernels.h"
#include "compute/embedding.h"
#include "compute/layernorm.h"
#include "compute/gemm.h"
#include "compute/gemm_q6k.h"
#include "compute/sampling.h"
#include "quant/quant_gemm.h"
#include "quant/nvfp4_gemm.h"
#include "compute/json_constrain.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Ban specific token IDs by setting their logits to -inf.
// Used in the CUDA graph decode path where host-side logit manipulation
// (cudaMemcpyAsync per token) is not possible during graph replay.
__global__ __launch_bounds__(256) void ban_logits_kernel(float* __restrict__ logits,
                                   const int32_t* __restrict__ banned_ids,
                                   int n_banned, int vocab_size) {
    int i = threadIdx.x;
    if (i < n_banned) {
        int32_t tid = banned_ids[i];
        if (tid >= 0 && tid < vocab_size) logits[tid] = -1e30f;
    }
}

namespace imp {

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
        apply_penalties(logits_ptr, vocab_size,
                        pen_ptr, pen_n,
                        state.repetition_penalty,
                        state.frequency_penalty,
                        state.presence_penalty, stream);
    }

    if (state.dry_multiplier > 0.0f && state.host_penalty_tokens != nullptr &&
        state.n_penalty_tokens > 0) {
        apply_dry_penalty(logits_ptr, vocab_size,
                          state.host_penalty_tokens, state.n_penalty_tokens,
                          state.dry_multiplier, state.dry_base,
                          state.dry_allowed_length, state.dry_penalty_last_n,
                          stream);
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
                IMP_CUDA_CHECK_LOG(cudaMemcpy(&logit, logits_ptr + tid, sizeof(float), cudaMemcpyDeviceToHost));
                logit += bias;
                IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(logits_ptr + tid, &logit, sizeof(float),
                                cudaMemcpyHostToDevice, stream));
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
        if (state.schema_constrainer) {
            state.schema_constrainer->apply_mask(logits_ptr, vocab_size, stream);
        } else if (state.json_constrainer) {
            state.json_constrainer->apply_mask(logits_ptr, vocab_size, stream);
        }
    }

    int32_t token;
    if (state.mirostat == 2) {
        // Mirostat v2: handles temperature + filtering internally, skip min_p
        unsigned int seed = state.seed >= 0
                                ? static_cast<unsigned int>(state.seed)
                                : 42u;
        token = d_sample_result_
            ? sample_mirostat_v2(last_logits, state.temperature,
                                 state.mirostat_tau, state.mirostat_eta,
                                 &state.mirostat_mu, seed, d_sample_result_, stream)
            : sample_mirostat_v2(last_logits, state.temperature,
                                 state.mirostat_tau, state.mirostat_eta,
                                 &state.mirostat_mu, seed, stream);
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
            int top_k  = state.top_k > 0  ? state.top_k  : 50;
            float top_p = state.top_p > 0.0f ? state.top_p : 1.0f;
            unsigned int seed = state.seed >= 0
                                    ? static_cast<unsigned int>(state.seed)
                                    : 42u;
            if (d_sample_result_ && h_sample_pinned_) {
                sample_topk_topp_device(last_logits, top_k, top_p,
                                         state.temperature, seed,
                                         d_sample_result_, h_sample_pinned_, stream);
                cudaStreamSynchronize(stream);
                token = *h_sample_pinned_;
            } else if (d_sample_result_) {
                token = sample_topk_topp(last_logits, top_k, top_p,
                                   state.temperature, seed, d_sample_result_, stream);
            } else {
                token = sample_topk_topp(last_logits, top_k, top_p,
                                   state.temperature, seed, stream);
            }
        }
    }

    return token;
}

std::vector<int32_t> GraphExecutor::sample_from_logits(const Tensor& logits,
                                                        const InferenceState& state,
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
            apply_penalties(lp, vocab,
                            pen_ptr, pen_n,
                            st.repetition_penalty,
                            st.frequency_penalty,
                            st.presence_penalty, stream);
        }
        if (st.dry_multiplier > 0.0f && st.host_penalty_tokens != nullptr &&
            st.n_penalty_tokens > 0) {
            apply_dry_penalty(lp, vocab,
                              st.host_penalty_tokens, st.n_penalty_tokens,
                              st.dry_multiplier, st.dry_base,
                              st.dry_allowed_length, st.dry_penalty_last_n,
                              stream);
        }
        if (st.n_logit_bias > 0 && st.logit_bias != nullptr) {
            for (int i = 0; i < st.n_logit_bias; i++) {
                int32_t tid = st.logit_bias[i].first;
                float bias = st.logit_bias[i].second;
                if (tid >= 0 && tid < vocab) {
                    float logit;
                    IMP_CUDA_CHECK_LOG(cudaMemcpy(&logit, lp + tid, sizeof(float), cudaMemcpyDeviceToHost));
                    logit += bias;
                    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(lp + tid, &logit, sizeof(float),
                                    cudaMemcpyHostToDevice, stream));
                }
            }
        }
        if (st.schema_constrainer) {
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
                    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(
                        lp + tid, &neg_inf, sizeof(float),
                        cudaMemcpyHostToDevice, stream));
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
            unsigned int seed = state.seed >= 0
                                    ? static_cast<unsigned int>(state.seed) : 42u;
            tokens[0] = d_sample_result_
                ? sample_mirostat_v2(last_logits, state.temperature,
                                     state.mirostat_tau, state.mirostat_eta,
                                     &state.mirostat_mu, seed, d_sample_result_, stream)
                : sample_mirostat_v2(last_logits, state.temperature,
                                     state.mirostat_tau, state.mirostat_eta,
                                     &state.mirostat_mu, seed, stream);
        } else {
            tokens[0] = (state.temperature <= 0.0f || state.top_k == 1)
                ? (d_sample_result_ ? sample_greedy(last_logits, d_sample_result_, stream)
                                    : sample_greedy(last_logits, stream))
                : (d_sample_result_
                    ? sample_topk_topp(last_logits,
                                       state.top_k > 0 ? state.top_k : 50,
                                       state.top_p > 0.0f ? state.top_p : 1.0f,
                                       state.temperature,
                                       state.seed >= 0 ? static_cast<unsigned int>(state.seed) : 42u,
                                       d_sample_result_, stream)
                    : sample_topk_topp(last_logits,
                                       state.top_k > 0 ? state.top_k : 50,
                                       state.top_p > 0.0f ? state.top_p : 1.0f,
                                       state.temperature,
                                       state.seed >= 0 ? static_cast<unsigned int>(state.seed) : 42u,
                                       stream));
        }
    } else {
        // Batched decode: n_tokens == n_sequences, each row is one sequence's logits
        for (int i = 0; i < n_seq; i++) {
            Tensor seq_logits = flatten_logits(logits.slice(i, i + 1));
            apply_pre_sample(seq_logits, state);
            tokens[i] = (state.temperature <= 0.0f || state.top_k == 1)
                ? (d_sample_result_ ? sample_greedy(seq_logits, d_sample_result_, stream)
                                    : sample_greedy(seq_logits, stream))
                : (d_sample_result_
                    ? sample_topk_topp(seq_logits,
                                       state.top_k > 0 ? state.top_k : 50,
                                       state.top_p > 0.0f ? state.top_p : 1.0f,
                                       state.temperature,
                                       state.seed >= 0 ? static_cast<unsigned int>(state.seed + i) : (42u + i),
                                       d_sample_result_, stream)
                    : sample_topk_topp(seq_logits,
                                       state.top_k > 0 ? state.top_k : 50,
                                       state.top_p > 0.0f ? state.top_p : 1.0f,
                                       state.temperature,
                                       state.seed >= 0 ? static_cast<unsigned int>(state.seed + i) : (42u + i),
                                       stream));
        }
    }

    return tokens;
}

int32_t GraphExecutor::sample_single_from_logits(const Tensor& logits,
                                                   const InferenceState& state,
                                                   cudaStream_t stream) {
    // Flatten [1, V] to [V]
    int64_t vocab_shape[1] = {logits.shape[logits.ndim - 1]};
    Tensor flat = logits.slice(0, 1).reshape(1, vocab_shape);

    // Apply penalties + filters
    float* lp = static_cast<float*>(flat.data);
    int vocab = static_cast<int>(flat.shape[0]);

    if (state.penalty_tokens != nullptr && state.n_penalty_tokens > 0) {
        const int32_t* pen_ptr = state.penalty_tokens;
        int pen_n = state.n_penalty_tokens;
        if (state.repeat_last_n > 0 && pen_n > state.repeat_last_n) {
            pen_ptr += (pen_n - state.repeat_last_n);
            pen_n = state.repeat_last_n;
        }
        apply_penalties(lp, vocab, pen_ptr, pen_n,
                        state.repetition_penalty, state.frequency_penalty,
                        state.presence_penalty, stream);
    }
    if (state.dry_multiplier > 0.0f && state.host_penalty_tokens != nullptr &&
        state.n_penalty_tokens > 0) {
        apply_dry_penalty(lp, vocab, state.host_penalty_tokens, state.n_penalty_tokens,
                          state.dry_multiplier, state.dry_base,
                          state.dry_allowed_length, state.dry_penalty_last_n, stream);
    }
    // Ban special tokens
    if (state.banned_tokens != nullptr && state.n_banned_tokens > 0) {
        float neg_inf = -1e30f;
        for (int bi = 0; bi < state.n_banned_tokens; bi++) {
            int32_t tid = state.banned_tokens[bi];
            if (tid >= 0 && tid < vocab)
                IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(lp + tid, &neg_inf, sizeof(float),
                                cudaMemcpyHostToDevice, stream));
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
                IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(lp + tid, &logit, sizeof(float),
                                cudaMemcpyHostToDevice, stream));
            }
        }
    }
    // Force token (think-budget)
    if (state.force_token >= 0 && state.force_token < vocab) {
        force_single_token(lp, vocab, state.force_token, stream);
    }
    if (state.force_token < 0) {
        if (state.schema_constrainer)
            state.schema_constrainer->apply_mask(lp, vocab, stream);
        else if (state.json_constrainer)
            state.json_constrainer->apply_mask(lp, vocab, stream);
    }
    if (state.min_p > 0.0f) apply_min_p(lp, vocab, state.min_p, stream);
    if (state.typical_p > 0.0f && state.typical_p < 1.0f)
        apply_typical_p(lp, vocab, state.typical_p, stream);

    // Sample
    if (state.mirostat == 2) {
        unsigned int seed = state.seed >= 0 ? static_cast<unsigned int>(state.seed) : 42u;
        return d_sample_result_
            ? sample_mirostat_v2(flat, state.temperature, state.mirostat_tau,
                                 state.mirostat_eta, &state.mirostat_mu, seed,
                                 d_sample_result_, stream)
            : sample_mirostat_v2(flat, state.temperature, state.mirostat_tau,
                                 state.mirostat_eta, &state.mirostat_mu, seed, stream);
    }
    return (state.temperature <= 0.0f || state.top_k == 1)
        ? (d_sample_result_ ? sample_greedy(flat, d_sample_result_, stream)
                            : sample_greedy(flat, stream))
        : (d_sample_result_
            ? sample_topk_topp(flat, state.top_k > 0 ? state.top_k : 50,
                               state.top_p > 0.0f ? state.top_p : 1.0f,
                               state.temperature,
                               state.seed >= 0 ? static_cast<unsigned int>(state.seed) : 42u,
                               d_sample_result_, stream)
            : sample_topk_topp(flat, state.top_k > 0 ? state.top_k : 50,
                               state.top_p > 0.0f ? state.top_p : 1.0f,
                               state.temperature,
                               state.seed >= 0 ? static_cast<unsigned int>(state.seed) : 42u,
                               stream));
}

std::vector<int32_t> GraphExecutor::forward_batch(const InferenceState& state,
                                                  cudaStream_t stream) {
    Tensor logits;
    forward_logits(state, logits, stream);
    return sample_from_logits(logits, state, stream);
}

// ---------------------------------------------------------------------------
// Async decode: embedding from device token → forward → sample to device
// ---------------------------------------------------------------------------

void GraphExecutor::forward_decode_async(const InferenceState& state,
                                          int32_t* d_token_id, int32_t* h_mapped,
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

    // Ban special tokens — device-side variant for graph capture.
    if (state.d_banned_tokens && state.n_d_banned_tokens > 0) {
        int threads = ((state.n_d_banned_tokens + 31) / 32) * 32;
        if (threads > 256) threads = 256;
        ban_logits_kernel<<<1, threads, 0, stream>>>(
            lp, state.d_banned_tokens, state.n_d_banned_tokens, vocab);
    }

    // Repetition / frequency / presence penalties — device counter grows each
    // iteration as tokens are appended to the penalty ring.
    if (state.penalty_tokens != nullptr && state.d_n_penalty_tokens != nullptr) {
        apply_penalties_device_count(lp, vocab, state.penalty_tokens,
                                     state.d_n_penalty_tokens,
                                     state.repeat_last_n,
                                     state.repetition_penalty,
                                     state.frequency_penalty,
                                     state.presence_penalty, stream);
    }

    // ---- Async sampling → write to d_token_id + h_mapped (mapped pinned) ----
    Tensor last_logits = logits.slice(0, 1);
    int64_t vocab_shape[1] = {last_logits.shape[1]};
    last_logits = last_logits.reshape(1, vocab_shape);

    if (state.temperature <= 0.0f || state.top_k == 1) {
        sample_greedy_device(last_logits, d_token_id, h_mapped, stream);
    } else {
        int top_k  = state.top_k > 0  ? state.top_k  : 50;
        float top_p = state.top_p > 0.0f ? state.top_p : 1.0f;
        unsigned int seed = state.seed >= 0
                                ? static_cast<unsigned int>(state.seed)
                                : 42u;
        sample_topk_topp_device(last_logits, top_k, top_p,
                                 state.temperature, seed,
                                 d_token_id, h_mapped, stream);
    }
    // No cudaStreamSynchronize — host polls h_mapped asynchronously.
}

} // namespace imp
