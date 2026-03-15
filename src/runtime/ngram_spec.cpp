#include "runtime/ngram_spec.h"
#include "core/logging.h"
#include "compute/sampling.h"
#include <cuda_runtime.h>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace imp {

NgramSpecDecoder::~NgramSpecDecoder() {
    if (d_tokens_) cudaFree(d_tokens_);
    if (d_positions_) cudaFree(d_positions_);
    if (d_block_table_) cudaFree(d_block_table_);
    if (d_ctx_len_) cudaFree(d_ctx_len_);
}

bool NgramSpecDecoder::init(GraphExecutor* executor, KVCacheManager* kv_manager,
                            KVCache* kv_cache, int n_layers, int spec_k, int ngram_n) {
    executor_ = executor;
    kv_manager_ = kv_manager;
    kv_cache_ = kv_cache;
    n_layers_ = n_layers;
    config_.spec_k = spec_k;
    config_.ngram_n = ngram_n;

    // Pre-allocate device buffers for verify (max spec_k + 1 tokens)
    int max_tokens = config_.spec_k + 1;
    if (cudaMalloc(&d_tokens_, max_tokens * sizeof(int32_t)) != cudaSuccess ||
        cudaMalloc(&d_positions_, max_tokens * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&d_ctx_len_, sizeof(int)) != cudaSuccess) {
        IMP_LOG_ERROR("NgramSpecDecoder: failed to allocate device buffers");
        return false;
    }

    IMP_LOG_INFO("N-gram speculative decoder: k=%d, n=%d", config_.spec_k, config_.ngram_n);
    return true;
}

std::vector<int32_t> NgramSpecDecoder::draft_tokens(
    const std::vector<int32_t>& input_tokens,
    const std::vector<int32_t>& output_tokens)
{
    const int n = config_.ngram_n;
    const int k = config_.spec_k;

    // Build full history: input + output
    // Search key: last n tokens of output
    if (static_cast<int>(output_tokens.size()) < n) return {};

    const int out_sz = static_cast<int>(output_tokens.size());
    const int in_sz = static_cast<int>(input_tokens.size());

    // The search key is the last n tokens of output
    const int32_t* key = output_tokens.data() + out_sz - n;

    // Search backwards through output (excluding the key itself) for matching n-gram.
    // Prefer recent matches (more likely to be relevant context).
    for (int i = out_sz - n - 1; i >= 0; i--) {
        bool match = true;
        for (int j = 0; j < n; j++) {
            if (output_tokens[i + j] != key[j]) {
                match = false;
                break;
            }
        }
        if (match) {
            int after = i + n;
            std::vector<int32_t> draft;
            for (int d = 0; d < k && after + d < out_sz - n; d++) {
                draft.push_back(output_tokens[after + d]);
            }
            if (!draft.empty()) return draft;
        }
    }

    // Search input tokens for matching n-gram (prompt patterns → output)
    for (int i = in_sz - n; i >= 0; i--) {
        bool match = true;
        for (int j = 0; j < n; j++) {
            if (input_tokens[i + j] != key[j]) {
                match = false;
                break;
            }
        }
        if (match) {
            int after = i + n;
            std::vector<int32_t> draft;
            // Continue from input into output if match is near the boundary
            for (int d = 0; d < k; d++) {
                int pos = after + d;
                if (pos < in_sz) {
                    draft.push_back(input_tokens[pos]);
                } else {
                    int out_pos = pos - in_sz;
                    if (out_pos < out_sz - n) {
                        draft.push_back(output_tokens[out_pos]);
                    } else {
                        break;
                    }
                }
            }
            if (!draft.empty()) return draft;
        }
    }

    return {};
}

NgramSpecDecoder::VerifyResult NgramSpecDecoder::verify(
    const std::vector<int32_t>& draft,
    std::shared_ptr<Request> req,
    int position, int seq_id, cudaStream_t stream)
{
    int n_verify = static_cast<int>(draft.size()) + 1;
    int kv_bs = kv_cache_->block_size();

    // Ensure KV cache has enough blocks for the verify pass
    int final_ctx = position + n_verify;
    int blocks_needed = (final_ctx + kv_bs - 1) / kv_bs;
    auto& bt = kv_manager_->block_table(seq_id);
    for (int b = static_cast<int>(bt.size()); b < blocks_needed; b++) {
        int new_block = kv_manager_->append_block(seq_id);
        if (new_block < 0) {
            // Can't allocate — fall back to single-token decode
            return {0, -1};
        }
    }

    // Build token array: [last_accepted_token, draft[0], draft[1], ..., draft[K-1]]
    std::vector<int32_t> h_tokens(n_verify);
    std::vector<int> h_positions(n_verify);
    h_tokens[0] = req->output_tokens.back();
    h_positions[0] = position;
    for (int i = 0; i < static_cast<int>(draft.size()); i++) {
        h_tokens[i + 1] = draft[i];
        h_positions[i + 1] = position + i + 1;
    }

    // Upload to device
    cudaMemcpyAsync(d_tokens_, h_tokens.data(), n_verify * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_positions_, h_positions.data(), n_verify * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    // Block table
    const auto& block_table = kv_manager_->block_table(seq_id);
    int n_blocks = static_cast<int>(block_table.size());
    if (n_blocks > d_block_table_cap_) {
        if (d_block_table_) cudaFree(d_block_table_);
        d_block_table_cap_ = n_blocks * 2;
        cudaMalloc(&d_block_table_, d_block_table_cap_ * sizeof(int));
    }
    cudaMemcpyAsync(d_block_table_, block_table.data(), n_blocks * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    int ctx_len = position + n_verify;
    cudaMemcpyAsync(d_ctx_len_, &ctx_len, sizeof(int), cudaMemcpyHostToDevice, stream);

    // Ensure workspace can handle n_verify tokens (may be in decode-only mode)
    if (n_verify > executor_->max_tokens()) {
        executor_->resize_workspace(n_verify, stream);
    }

    // Build InferenceState for batched verify (pseudo-prefill)
    InferenceState state;
    state.token_ids = d_tokens_;
    state.positions = d_positions_;
    state.n_tokens = n_verify;
    state.kv_cache = kv_cache_;
    state.block_tables = d_block_table_;
    state.context_lens = d_ctx_len_;
    state.max_context_len = ctx_len;
    state.n_sequences = 1;
    state.max_blocks_per_seq = n_blocks;
    state.is_prefill = true;
    state.all_logits = true;
    state.temperature = req->temperature;
    state.top_p = req->top_p;
    state.top_k = req->top_k;
    state.seed = req->seed;
    state.min_p = req->min_p;

    // Forward pass: get logits for all positions
    Tensor logits_out;
    executor_->forward_logits(state, logits_out, stream);
    cudaStreamSynchronize(stream);

    // Download logits to host for acceptance check
    int vocab_size = static_cast<int>(logits_out.shape[1]);
    std::vector<float> h_logits(static_cast<size_t>(n_verify) * vocab_size);
    cudaMemcpy(h_logits.data(), logits_out.data,
               h_logits.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Greedy acceptance: check if argmax(target_logits[i]) == draft[i]
    VerifyResult result;
    result.n_accepted = 0;
    result.corrected_token = -1;

    bool greedy = (req->temperature <= 0.0f || req->temperature == 1e-7f);

    for (int i = 0; i < static_cast<int>(draft.size()); i++) {
        const float* row = h_logits.data() + static_cast<size_t>(i) * vocab_size;

        if (greedy) {
            // Greedy: accept if argmax matches draft
            int argmax_id = 0;
            float argmax_val = row[0];
            for (int v = 1; v < vocab_size; v++) {
                if (row[v] > argmax_val) {
                    argmax_val = row[v];
                    argmax_id = v;
                }
            }
            if (argmax_id == draft[i]) {
                result.n_accepted++;
            } else {
                result.corrected_token = argmax_id;
                break;
            }
        } else {
            // Stochastic: accept with probability based on target distribution
            // Simple approach: accept if draft token is in top-p of target
            float max_logit = *std::max_element(row, row + vocab_size);
            float sum_exp = 0.0f;
            for (int v = 0; v < vocab_size; v++)
                sum_exp += std::exp((row[v] - max_logit) / req->temperature);
            float draft_prob = std::exp((row[draft[i]] - max_logit) / req->temperature) / sum_exp;

            // Accept if probability is reasonable (> 10% of uniform)
            if (draft_prob > 0.1f / vocab_size) {
                result.n_accepted++;
            } else {
                // Sample from target distribution for correction
                int best = 0;
                float best_val = row[0];
                for (int v = 1; v < vocab_size; v++) {
                    if (row[v] > best_val) { best_val = row[v]; best = v; }
                }
                result.corrected_token = best;
                break;
            }
        }
    }

    // If all drafts accepted, sample the next token from the last logits row
    if (result.n_accepted == static_cast<int>(draft.size())) {
        const float* last_row = h_logits.data() + static_cast<size_t>(n_verify - 1) * vocab_size;
        int best = 0;
        float best_val = last_row[0];
        for (int v = 1; v < vocab_size; v++) {
            if (last_row[v] > best_val) { best_val = last_row[v]; best = v; }
        }
        result.corrected_token = best;
    }

    return result;
}

NgramSpecDecoder::StepResult NgramSpecDecoder::step(
    std::shared_ptr<Request> req, int32_t last_token,
    int position, int seq_id, cudaStream_t stream)
{
    total_steps_++;

    // Try to draft tokens from n-gram history
    auto draft = draft_tokens(req->input_tokens, req->output_tokens);

    if (draft.empty()) {
        // No n-gram match — fall back to normal single-token decode
        return {{}, 0, 0};
    }

    IMP_LOG_DEBUG("N-gram draft: %d tokens from history (output=%zu, input=%zu)",
                  static_cast<int>(draft.size()), req->output_tokens.size(),
                  req->input_tokens.size());

    // Limit draft size
    if (static_cast<int>(draft.size()) > config_.spec_k) {
        draft.resize(config_.spec_k);
    }

    total_drafted_ += static_cast<int>(draft.size());

    // Verify draft tokens via batched forward pass
    auto vr = verify(draft, req, position, seq_id, stream);

    if (vr.n_accepted == 0 && vr.corrected_token < 0) {
        // Verify failed (e.g., KV allocation) — fall back
        return {{}, static_cast<int>(draft.size()), 0};
    }

    total_accepted_ += vr.n_accepted;

    // Build result: accepted draft tokens + corrected/next token
    StepResult result;
    result.n_drafted = static_cast<int>(draft.size());
    result.n_accepted = vr.n_accepted;

    for (int i = 0; i < vr.n_accepted; i++) {
        result.tokens.push_back(draft[i]);
    }
    if (vr.corrected_token >= 0) {
        result.tokens.push_back(vr.corrected_token);
    }

    // Rollback KV cache: keep only position + n_accepted + 1 entries
    int keep = position + vr.n_accepted + 1;
    kv_manager_->rollback(seq_id, keep);

    return result;
}

} // namespace imp
