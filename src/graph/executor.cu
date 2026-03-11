#include "graph/executor.h"
#include "graph/executor_kernels.h"
#include "compute/embedding.h"
#include "compute/layernorm.h"
#include "compute/gemm.h"
#include "compute/gemm_q6k.h"
#include "compute/sampling.h"
#include "quant/quant_gemm.h"
#include "compute/json_constrain.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

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

    // JSON mode: apply logit mask to constrain output to valid JSON
    if (state.json_constrainer) {
        state.json_constrainer->apply_mask(logits_ptr, vocab_size, stream);
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
            token = d_sample_result_
                ? sample_greedy(last_logits, d_sample_result_, stream)
                : sample_greedy(last_logits, stream);
        } else {
            int top_k  = state.top_k > 0  ? state.top_k  : 50;
            float top_p = state.top_p > 0.0f ? state.top_p : 1.0f;
            unsigned int seed = state.seed >= 0
                                    ? static_cast<unsigned int>(state.seed)
                                    : 42u;
            token = d_sample_result_
                ? sample_topk_topp(last_logits, top_k, top_p,
                                   state.temperature, seed, d_sample_result_, stream)
                : sample_topk_topp(last_logits, top_k, top_p,
                                   state.temperature, seed, stream);
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
        if (st.json_constrainer) {
            st.json_constrainer->apply_mask(lp, vocab, stream);
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

    const auto& cfg = model_->config();
    int n = state.n_tokens;  // should be 1 for decode
    cur_n_tokens_ = n;
    cur_force_fp16_ = false;  // async decode path never forces FP16
    cur_per_row_lm_ = false;
    // Clear any stale CUDA error state before starting the decode pass.
    { cudaError_t e_ = cudaGetLastError();
      if (e_ != cudaSuccess) IMP_LOG_DEBUG("Cleared stale error before decode: %s", cudaGetErrorString(e_)); }

    // ---- Step 1: Embedding lookup from device-side token ID ----
    Tensor h = view_tokens(hidden_, n);
    embedding_lookup_from_device(model_->token_embedding(), d_token_id, h,
                                  model_->tok_emb_qtype_, stream);

    // Gemma: scale embeddings by sqrt(d_model)
    if (cfg.embed_scale > 0.0f && h.dtype == DType::FP16) {
        int64_t total = static_cast<int64_t>(n) * cfg.d_model;
        int threads = 256;
        int blocks = static_cast<int>((total / 2 + threads - 1) / threads);
        scale_fp16_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<half*>(h.data), __float2half(cfg.embed_scale), total);
    }

    // Initialize FP32 residual accumulator from FP16 embedding (post-norm models).
    // Without this, the graph loop would accumulate on stale FP32 state from
    // the previous iteration instead of starting fresh from the embedding.
    if (fp32_accum_buf_) {
        int64_t total = static_cast<int64_t>(n) * cfg.d_model;
        int threads = 256;
        int blocks = static_cast<int>((total + threads - 1) / threads);
        fp16_to_fp32_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const half*>(h.data),
            static_cast<float*>(view_tokens(fp32_hidden_, n).data), total);
    }

    // ---- Step 2: Transformer layers ----
    for (int i = 0; i < cfg.n_layers; ++i) {
        if (offload_mgr_) {
            offload_mgr_->ensure_layer(i, stream);
            if (i + 1 < cfg.n_layers) offload_mgr_->prefetch_layer(i + 1);
        }

        if (layer_has_attention(i)) run_attention(i, state, stream);
        else if (layer_has_ssm(i))  run_ssm(i, state, stream);

        if (layer_has_moe(i))            run_moe_ffn(i, stream);
        else if (layer_has_dense_ffn(i)) run_ffn(i, stream);

        if (offload_mgr_) offload_mgr_->release_layer(i);
    }

    // ---- Step 3: Final RMSNorm + LM head ----
    Tensor h_final = view_tokens(hidden_, n);
    Tensor lg = view_tokens(logits_, n);

    const auto out_qtype = model_->out_proj_qtype_;
    if (q8_1_buf_ && compute_dtype_ == DType::FP16 &&
        (out_qtype == GGMLQuantType::Q6_K || out_qtype == GGMLQuantType::Q8_0 ||
         out_qtype == GGMLQuantType::Q4_0 || out_qtype == GGMLQuantType::Q4_K ||
         out_qtype == GGMLQuantType::Q5_K || out_qtype == GGMLQuantType::Q2_K ||
         out_qtype == GGMLQuantType::Q3_K)) {
        auto* q8 = static_cast<block_q8_1*>(q8_1_buf_);
        rmsnorm_quantize_q8_1(
            static_cast<const half*>(h_final.data),
            static_cast<const half*>(model_->output_norm().data),
            q8, d8_buf_, nullptr, cfg.d_model, cfg.rms_norm_eps, stream, norm_w_off_);
        if (out_qtype == GGMLQuantType::Q6_K)
            gemv_q6k_q8_1_fp32(model_->output_proj().data, q8, d8_buf_,
                               static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
        else if (out_qtype == GGMLQuantType::Q8_0)
            gemv_q8_0_q8_1_fp32(model_->output_proj().data, q8, d8_buf_,
                                static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
        else if (out_qtype == GGMLQuantType::Q4_K)
            gemv_q4_k_q8_1_fp32(model_->output_proj().data, q8, d8_buf_,
                                static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
        else if (out_qtype == GGMLQuantType::Q5_K)
            gemv_q5_k_q8_1_fp32(model_->output_proj().data, q8, d8_buf_,
                                static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
        else if (out_qtype == GGMLQuantType::Q2_K)
            gemv_q2_k_q8_1_fp32(model_->output_proj().data, q8, d8_buf_,
                                static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
        else if (out_qtype == GGMLQuantType::Q3_K)
            gemv_q3_k_q8_1_fp32(model_->output_proj().data, q8, d8_buf_,
                                static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
        else
            gemv_q4_0_q8_1_fp32(model_->output_proj().data, q8, d8_buf_,
                                static_cast<float*>(lg.data), cfg.vocab_size, cfg.d_model, stream);
    } else {
        Tensor no_final = view_tokens(norm_out_, n);
        rmsnorm(h_final, model_->output_norm(), no_final, cfg.rms_norm_eps, stream, norm_w_off_);
        gemm(no_final, model_->output_proj(), lg, 1.0f, 0.0f, stream);
    }

    // ---- Step 4: Async sampling → write to d_token_id + h_mapped ----
    Tensor last_logits = lg.slice(0, 1);
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
    // No cudaStreamSynchronize — host polls h_mapped asynchronously
}

} // namespace imp
