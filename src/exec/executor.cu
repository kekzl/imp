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
#include "compute/grammar_constrain.h"
#include "core/logging.h"
#include "exec/executor_sampling_internal.h"

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

void launch_ban_logits(float* logits, const int32_t* banned_ids, int n_banned, int vocab_size,
                       cudaStream_t stream) {
    constexpr int kBanThreads = 256;
    int blocks = (n_banned + kBanThreads - 1) / kBanThreads;
    ban_logits_kernel<<<blocks, kBanThreads, 0, stream>>>(logits, banned_ids, n_banned, vocab_size);
    IMP_CUDA_CHECK_LAUNCH();
}

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
    if (state.n_logit_bias > 0 && state.logit_bias != nullptr)
        apply_logit_bias(logits_ptr, vocab_size, state.logit_bias, state.n_logit_bias, stream);

    // Force token: set all logits except force_token to -inf.
    // Used by think-budget to force </think> via logit manipulation
    // so the model generates it naturally into the KV cache (NVIDIA NIM approach).
    if (state.force_token >= 0 && state.force_token < vocab_size) {
        force_single_token(logits_ptr, vocab_size, state.force_token, stream);
    }

    // JSON/Schema/regex/grammar mode: mask before sampling, unless a token is
    // being forced (think-budget) — then the mask must not fight it.
    if (state.force_token < 0)
        apply_constraint_mask(state, logits_ptr, vocab_size, stream);

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
            if (d_sample_result_ && h_sample_pinned_.as<int32_t>()) {
                sample_greedy_device(last_logits, d_sample_result_, h_sample_pinned_.as<int32_t>(), stream);
                cudaStreamSynchronize(stream);
                token = *h_sample_pinned_.as<int32_t>();
            } else if (d_sample_result_) {
                token = sample_greedy(last_logits, d_sample_result_, stream);
            } else {
                token = sample_greedy(last_logits, stream);
            }
        } else {
            int top_k = state.top_k > 0 ? state.top_k : 50;
            float top_p = state.top_p > 0.0f ? state.top_p : 1.0f;
            unsigned int seed = state.seed >= 0 ? static_cast<unsigned int>(state.seed) : 42u;
            if (d_sample_result_ && h_sample_pinned_.as<int32_t>()) {
                sample_topk_topp_device(last_logits, top_k, top_p, state.temperature, seed, d_sample_result_,
                                        h_sample_pinned_.as<int32_t>(), stream);
                cudaStreamSynchronize(stream);
                token = *h_sample_pinned_.as<int32_t>();
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

// The per-row / row-batched sampling family (sample_from_logits,
// sample_single_from_logits{,_async}, apply_row_filters_, the pending-row
// flushes and the token collectors) lives in executor_sampling.cu since
// 2026-08-27 — see the header comment there.


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

    // Constraint mask (host-computed this step, uploaded stream-ordered).
    apply_constraint_mask(state, lp, vocab, stream);

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
