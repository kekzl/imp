#include "runtime/speculative.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace imp {

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

// ─── Init ────────────────────────────────────────────────────────────────────

bool SpeculativeDecoder::init(GraphExecutor* target_executor,
                              std::shared_ptr<Model> draft_model,
                              std::unique_ptr<GraphExecutor> draft_executor,
                              KVCacheManager* target_kv_manager,
                              KVCacheManager* draft_kv_manager,
                              const SpeculativeConfig& config) {
    if (!target_executor) {
        IMP_LOG_ERROR("speculative: target_executor is null");
        return false;
    }
    if (!draft_model) {
        IMP_LOG_ERROR("speculative: draft_model is null");
        return false;
    }
    if (!draft_executor) {
        IMP_LOG_ERROR("speculative: draft_executor is null");
        return false;
    }
    if (!target_kv_manager) {
        IMP_LOG_ERROR("speculative: target_kv_manager is null");
        return false;
    }
    if (!draft_kv_manager) {
        IMP_LOG_ERROR("speculative: draft_kv_manager is null");
        return false;
    }
    if (config.spec_k < 1 || config.spec_k > 32) {
        IMP_LOG_ERROR("speculative: spec_k must be in [1, 32], got %d", config.spec_k);
        return false;
    }

    config_ = config;
    target_executor_ = target_executor;
    draft_model_ = std::move(draft_model);
    draft_executor_ = std::move(draft_executor);
    target_kv_manager_ = target_kv_manager;
    draft_kv_manager_ = draft_kv_manager;
    initialized_ = true;

    IMP_LOG_INFO("speculative: initialized with spec_k=%d", config_.spec_k);
    return true;
}

// ─── Draft token generation ──────────────────────────────────────────────────

std::vector<int32_t> SpeculativeDecoder::draft_tokens(int32_t last_token,
                                                       int position,
                                                       int seq_id,
                                                       cudaStream_t stream) {
    if (!draft_executor_ || !draft_kv_manager_) {
        IMP_LOG_ERROR("speculative: draft_tokens called without initialized draft model");
        return {};
    }

    std::vector<int32_t> drafts;
    drafts.reserve(config_.spec_k);

    // Small device buffers for single-token decode (1 element each).
    int32_t* d_token = nullptr;
    int*     d_pos   = nullptr;
    check_cuda(cudaMalloc(&d_token, sizeof(int32_t)), "draft cudaMalloc d_token");
    check_cuda(cudaMalloc(&d_pos, sizeof(int)), "draft cudaMalloc d_pos");

    const auto& draft_blocks = draft_kv_manager_->block_table(seq_id);
    int max_blocks = static_cast<int>(draft_blocks.size());

    // Upload block table for this sequence (may change as we append blocks).
    int* d_block_table = nullptr;
    if (max_blocks > 0) {
        check_cuda(cudaMalloc(&d_block_table, max_blocks * sizeof(int)), "draft cudaMalloc d_block_table");
        check_cuda(cudaMemcpy(d_block_table, draft_blocks.data(),
                   max_blocks * sizeof(int), cudaMemcpyHostToDevice), "draft memcpy block_table");
    }

    int* d_ctx_len = nullptr;
    check_cuda(cudaMalloc(&d_ctx_len, sizeof(int)), "draft cudaMalloc d_ctx_len");

    int32_t cur_token = last_token;
    int cur_pos = position;

    for (int k = 0; k < config_.spec_k; ++k) {
        // Upload current token and position.
        check_cuda(cudaMemcpy(d_token, &cur_token, sizeof(int32_t), cudaMemcpyHostToDevice), "draft memcpy token");
        check_cuda(cudaMemcpy(d_pos, &cur_pos, sizeof(int), cudaMemcpyHostToDevice), "draft memcpy pos");

        // Context length is cur_pos + 1 (all tokens seen so far including this one).
        int ctx_len = cur_pos + 1;
        check_cuda(cudaMemcpy(d_ctx_len, &ctx_len, sizeof(int), cudaMemcpyHostToDevice), "draft memcpy ctx_len");

        // Re-upload block table in case we appended a new block.
        const auto& cur_blocks = draft_kv_manager_->block_table(seq_id);
        int cur_max_blocks = static_cast<int>(cur_blocks.size());
        if (cur_max_blocks > max_blocks) {
            cudaFree(d_block_table);
            max_blocks = cur_max_blocks;
            check_cuda(cudaMalloc(&d_block_table, max_blocks * sizeof(int)), "draft cudaMalloc d_block_table realloc");
        }
        if (max_blocks > 0) {
            check_cuda(cudaMemcpy(d_block_table, cur_blocks.data(),
                       max_blocks * sizeof(int), cudaMemcpyHostToDevice), "draft memcpy block_table");
        }

        InferenceState state;
        state.token_ids = d_token;
        state.positions = d_pos;
        state.n_tokens = 1;
        state.kv_cache = nullptr;  // set by executor from model
        state.block_tables = d_block_table;
        state.context_lens = d_ctx_len;
        state.max_context_len = ctx_len;
        state.n_sequences = 1;
        state.max_blocks_per_seq = max_blocks;
        state.is_prefill = false;
        // Greedy sampling for draft (temperature=0 equivalent via argmax).
        state.temperature = 0.0f;
        state.top_p = 1.0f;
        state.top_k = 1;
        state.seed = -1;

        int32_t sampled = draft_executor_->forward(state, stream);
        drafts.push_back(sampled);

        cur_token = sampled;
        cur_pos += 1;

        // Ensure the draft KV cache has enough blocks for the next position.
        int block_size = kKVBlockSize;
        int needed_blocks = (cur_pos + block_size) / block_size;
        int have_blocks = static_cast<int>(draft_kv_manager_->block_table(seq_id).size());
        if (needed_blocks > have_blocks) {
            draft_kv_manager_->append_block(seq_id);
        }
    }

    cudaFree(d_token);
    cudaFree(d_pos);
    cudaFree(d_block_table);
    cudaFree(d_ctx_len);

    return drafts;
}

// ─── Verification ────────────────────────────────────────────────────────────

SpeculativeDecoder::VerifyResult
SpeculativeDecoder::verify(const std::vector<int32_t>& draft,
                           int32_t prompt_last_token,
                           int position,
                           int seq_id,
                           float temperature, float top_p_val, int top_k_val,
                           int seed,
                           cudaStream_t stream) {
    VerifyResult result;
    if (!target_executor_ || !target_kv_manager_) {
        IMP_LOG_ERROR("speculative: verify called without initialized target model");
        return result;
    }

    const int K = static_cast<int>(draft.size());
    if (K == 0) {
        IMP_LOG_WARN("speculative: verify called with empty draft");
        return result;
    }

    const int n_verify = K + 1;  // last_token + K draft tokens

    // Build token array: [prompt_last_token, draft[0], draft[1], ..., draft[K-1]]
    std::vector<int32_t> h_tokens(n_verify);
    h_tokens[0] = prompt_last_token;
    for (int i = 0; i < K; ++i) {
        h_tokens[i + 1] = draft[i];
    }

    // Build position array: [position, position+1, ..., position+K]
    std::vector<int> h_positions(n_verify);
    for (int i = 0; i < n_verify; ++i) {
        h_positions[i] = position + i;
    }

    // Allocate device buffers for the verification pass.
    int32_t* d_tokens = nullptr;
    int*     d_positions = nullptr;
    check_cuda(cudaMalloc(&d_tokens, n_verify * sizeof(int32_t)), "verify cudaMalloc d_tokens");
    check_cuda(cudaMalloc(&d_positions, n_verify * sizeof(int)), "verify cudaMalloc d_positions");
    check_cuda(cudaMemcpy(d_tokens, h_tokens.data(), n_verify * sizeof(int32_t), cudaMemcpyHostToDevice), "verify memcpy tokens");
    check_cuda(cudaMemcpy(d_positions, h_positions.data(), n_verify * sizeof(int), cudaMemcpyHostToDevice), "verify memcpy positions");

    // Ensure the target KV cache has enough blocks.
    int final_pos = position + K;
    int block_size = kKVBlockSize;
    int needed_blocks = (final_pos + block_size) / block_size;
    int have_blocks = static_cast<int>(target_kv_manager_->block_table(seq_id).size());
    while (needed_blocks > have_blocks) {
        target_kv_manager_->append_block(seq_id);
        have_blocks++;
    }

    const auto& target_blocks = target_kv_manager_->block_table(seq_id);
    int max_blocks = static_cast<int>(target_blocks.size());

    int* d_block_table = nullptr;
    int* d_ctx_len = nullptr;
    if (max_blocks > 0) {
        check_cuda(cudaMalloc(&d_block_table, max_blocks * sizeof(int)), "verify cudaMalloc d_block_table");
        check_cuda(cudaMemcpy(d_block_table, target_blocks.data(),
                   max_blocks * sizeof(int), cudaMemcpyHostToDevice), "verify memcpy block_table");
    }
    check_cuda(cudaMalloc(&d_ctx_len, sizeof(int)), "verify cudaMalloc d_ctx_len");
    int ctx_len = position + n_verify;
    check_cuda(cudaMemcpy(d_ctx_len, &ctx_len, sizeof(int), cudaMemcpyHostToDevice), "verify memcpy ctx_len");

    // Build InferenceState for pseudo-prefill (all n_verify tokens at once).
    InferenceState state;
    state.token_ids = d_tokens;
    state.positions = d_positions;
    state.n_tokens = n_verify;
    state.kv_cache = nullptr;
    state.block_tables = d_block_table;
    state.context_lens = d_ctx_len;
    state.max_context_len = ctx_len;
    state.n_sequences = 1;
    state.max_blocks_per_seq = max_blocks;
    state.is_prefill = true;  // pseudo-prefill: process all tokens at once
    state.temperature = temperature;
    state.top_p = top_p_val;
    state.top_k = top_k_val;
    state.seed = seed;

    // Run forward to get logits for all n_verify positions.
    Tensor logits;
    target_executor_->forward_logits(state, logits, stream);

    // logits shape: [n_verify, vocab_size]
    // logits[i] gives the distribution after seeing tokens[0..i]
    // So logits[i] predicts token at position (position + i + 1)
    // We compare logits[i] against draft[i] for i in 0..K-1
    int vocab_size = static_cast<int>(logits.shape[logits.ndim - 1]);

    // Download logits to host for acceptance checking.
    size_t logits_bytes = n_verify * vocab_size * dtype_size(logits.dtype);
    std::vector<float> h_logits(n_verify * vocab_size);

    if (logits.dtype == DType::FP32) {
        check_cuda(cudaMemcpy(h_logits.data(), logits.data, logits_bytes, cudaMemcpyDeviceToHost), "verify memcpy logits");
    } else {
        // For FP16/BF16 logits, we need a conversion -- but for now, the
        // executor's logits buffer is already FP32 (as used by sampling).
        // If not FP32, fall back to copying raw bytes and interpreting.
        // This path should not normally be hit since logits are computed in FP32.
        IMP_LOG_WARN("speculative: logits dtype is %s, expected FP32",
                     dtype_name(logits.dtype));
        check_cuda(cudaMemcpy(h_logits.data(), logits.data,
                   n_verify * vocab_size * sizeof(float), cudaMemcpyDeviceToHost), "verify memcpy logits fallback");
    }

    cudaStreamSynchronize(stream);

    // Greedy mode: temperature <= 0 or very small.
    bool greedy = (temperature <= 1e-6f);

    // RNG state for stochastic acceptance.
    unsigned int rng_state = (seed >= 0) ? static_cast<unsigned int>(seed) : 42u;

    // Convert logits to probabilities (softmax) per position.
    // For efficiency, only compute softmax for the rows we need.
    auto softmax_row = [&](int row, std::vector<float>& probs) {
        const float* row_logits = h_logits.data() + row * vocab_size;
        probs.resize(vocab_size);

        // Find max for numerical stability.
        float max_val = row_logits[0];
        for (int v = 1; v < vocab_size; ++v) {
            max_val = std::max(max_val, row_logits[v]);
        }

        // Apply temperature before softmax.
        float inv_temp = greedy ? 1.0f : (1.0f / temperature);
        float sum = 0.0f;
        for (int v = 0; v < vocab_size; ++v) {
            probs[v] = std::exp((row_logits[v] - max_val) * inv_temp);
            sum += probs[v];
        }
        float inv_sum = 1.0f / (sum + 1e-10f);
        for (int v = 0; v < vocab_size; ++v) {
            probs[v] *= inv_sum;
        }
    };

    // Argmax helper.
    auto argmax = [&](const std::vector<float>& probs) -> int32_t {
        return static_cast<int32_t>(
            std::distance(probs.begin(), std::max_element(probs.begin(), probs.end())));
    };

    // Weighted random sample helper (for stochastic acceptance resampling).
    auto sample_from = [&](const std::vector<float>& probs) -> int32_t {
        // LCG step for random float in [0, 1).
        rng_state = rng_state * 1664525u + 1013904223u;
        float r = static_cast<float>(rng_state & 0x00FFFFFFu) / static_cast<float>(0x01000000u);

        float cumsum = 0.0f;
        for (int v = 0; v < static_cast<int>(probs.size()); ++v) {
            cumsum += probs[v];
            if (cumsum >= r) {
                return static_cast<int32_t>(v);
            }
        }
        return static_cast<int32_t>(probs.size() - 1);
    };

    std::vector<float> target_probs;

    // Verify each draft token.
    int n_accepted = 0;
    for (int i = 0; i < K; ++i) {
        softmax_row(i, target_probs);

        if (greedy) {
            // Greedy: accept if target argmax matches draft token.
            int32_t target_choice = argmax(target_probs);
            if (target_choice == draft[i]) {
                result.accepted.push_back(draft[i]);
                n_accepted++;
            } else {
                // Reject: use target's argmax as the corrected token.
                result.n_accepted = n_accepted;
                result.next_token = target_choice;
                goto cleanup;
            }
        } else {
            // Stochastic acceptance: accept with prob min(1, p_target / p_draft).
            // We don't have the draft model's probabilities stored, so we
            // re-run a simple heuristic: accept if the target probability
            // for the draft token is above the acceptance threshold, or
            // use the standard ratio test.
            //
            // In a full implementation, the draft model's per-token
            // probabilities would be cached during draft_tokens(). Here we
            // use a simplified acceptance: accept if target_prob > threshold.
            float p_target = target_probs[draft[i]];

            // Without cached draft probabilities, use a simplified test:
            // Accept if p_target exceeds a minimum floor, otherwise reject.
            // When acceptance_threshold == 0, always accept (optimistic).
            if (config_.acceptance_threshold > 0.0f && p_target < config_.acceptance_threshold) {
                // Reject: resample from the target distribution.
                result.n_accepted = n_accepted;
                result.next_token = sample_from(target_probs);
                goto cleanup;
            }

            // Stochastic acceptance with a uniform draft assumption:
            // Assume p_draft ~ 1/top_k or a flat approximation.
            // This is a simplification; a production implementation would
            // cache draft logits for exact ratio computation.
            float p_draft_approx = 1.0f / static_cast<float>(std::max(top_k_val, 1));
            if (stochastic_accept(p_target, p_draft_approx, rng_state)) {
                result.accepted.push_back(draft[i]);
                n_accepted++;
            } else {
                // Reject: resample from adjusted distribution
                // p_adjusted(x) = max(0, p_target(x) - p_draft(x)) / Z
                std::vector<float> adjusted(vocab_size);
                float adj_sum = 0.0f;
                for (int v = 0; v < vocab_size; ++v) {
                    adjusted[v] = std::max(0.0f, target_probs[v] - p_draft_approx);
                    adj_sum += adjusted[v];
                }
                if (adj_sum > 1e-10f) {
                    float inv = 1.0f / adj_sum;
                    for (int v = 0; v < vocab_size; ++v) {
                        adjusted[v] *= inv;
                    }
                    result.next_token = sample_from(adjusted);
                } else {
                    result.next_token = sample_from(target_probs);
                }
                result.n_accepted = n_accepted;
                goto cleanup;
            }
        }
    }

    // All K draft tokens accepted. Sample token K+1 from the last logit row.
    {
        softmax_row(K, target_probs);
        if (greedy) {
            result.next_token = argmax(target_probs);
        } else {
            result.next_token = sample_from(target_probs);
        }
        result.n_accepted = K;
    }

cleanup:
    cudaFree(d_tokens);
    cudaFree(d_positions);
    cudaFree(d_block_table);
    cudaFree(d_ctx_len);

    return result;
}

// ─── Full speculative step ───────────────────────────────────────────────────

std::vector<int32_t> SpeculativeDecoder::step(int32_t last_token, int position,
                                               int seq_id,
                                               float temperature, float top_p_val,
                                               int top_k_val,
                                               int seed,
                                               cudaStream_t stream) {
    if (!initialized_) {
        IMP_LOG_ERROR("speculative: not initialized");
        return {};
    }

    // 1. Generate K draft tokens with the draft model.
    std::vector<int32_t> draft = draft_tokens(last_token, position, seq_id, stream);

    IMP_LOG_DEBUG("speculative: drafted %zu tokens from position %d",
                  draft.size(), position);

    // 2. Verify draft tokens against the target model.
    VerifyResult vr = verify(draft, last_token, position, seq_id,
                             temperature, top_p_val, top_k_val, seed, stream);

    IMP_LOG_DEBUG("speculative: accepted %d / %d draft tokens",
                  vr.n_accepted, static_cast<int>(draft.size()));

    // 3. Combine results: accepted draft tokens + the next corrected/sampled token.
    std::vector<int32_t> output;
    output.reserve(vr.n_accepted + 1);
    for (int i = 0; i < vr.n_accepted; ++i) {
        output.push_back(vr.accepted[i]);
    }
    if (vr.next_token >= 0) {
        output.push_back(vr.next_token);
    }

    // 4. Roll back KV caches to discard stale entries from rejected tokens.
    //    The draft model wrote K entries starting at `position`, and the target
    //    model wrote K+1 entries in pseudo-prefill. Only the first n_accepted+1
    //    positions are valid. Without rollback, rejected positions waste blocks
    //    and leave stale KV data that could be read by paged attention if the
    //    block is not fully overwritten on the next step.
    int rollback_pos = position + vr.n_accepted + 1;
    target_kv_manager_->rollback(seq_id, rollback_pos);
    draft_kv_manager_->rollback(seq_id, rollback_pos);

    return output;
}

// ─── Stochastic acceptance ───────────────────────────────────────────────────

bool SpeculativeDecoder::stochastic_accept(float p_target, float p_draft,
                                           unsigned int& rng_state) {
    if (p_draft <= 0.0f) {
        // Draft assigned zero probability -- always accept the target's choice.
        return false;
    }

    float ratio = p_target / p_draft;
    if (ratio >= 1.0f) {
        return true;
    }

    // LCG random number in [0, 1).
    rng_state = rng_state * 1664525u + 1013904223u;
    float r = static_cast<float>(rng_state & 0x00FFFFFFu) / static_cast<float>(0x01000000u);

    return r < ratio;
}

} // namespace imp
