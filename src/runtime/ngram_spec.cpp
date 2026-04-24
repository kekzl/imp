#include "runtime/ngram_spec.h"
#include "core/logging.h"
#include "compute/sampling.h"
#include <cuda_runtime.h>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace imp {

NgramSpecDecoder::~NgramSpecDecoder() {
    if (d_tokens_) IMP_CUDA_CHECK_LOG(cudaFree(d_tokens_));
    if (d_positions_) IMP_CUDA_CHECK_LOG(cudaFree(d_positions_));
    if (d_block_table_) IMP_CUDA_CHECK_LOG(cudaFree(d_block_table_));
    if (d_ctx_len_) IMP_CUDA_CHECK_LOG(cudaFree(d_ctx_len_));
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
        cudaMalloc(&d_ctx_len_, max_tokens * sizeof(int)) != cudaSuccess) {
        IMP_LOG_ERROR("NgramSpecDecoder: failed to allocate device buffers");
        return false;
    }

    // Graph pool for verify forward pass (one per n_verify ∈ [2, spec_k+1]).
    // Sampling stays eager after replay — same pattern as self_speculative.
    // Honors IMP_NO_CUDA_GRAPH: empty pool forces eager forward in verify().
    if (!getenv("IMP_NO_CUDA_GRAPH")) {
        verify_graphs_.resize(max_tokens + 1);
        verify_graph_max_blocks_.assign(max_tokens + 1, -1);
        for (int i = 2; i <= max_tokens; ++i) {
            verify_graphs_[i] = std::make_unique<CudaGraphRunner>();
        }
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
    // Multi-sequence decode verify: each draft position is a separate single-token
    // decode sequence. Uses the same paged attention kernel as normal decode,
    // avoiding numerical divergence from prefill-mode attention (CUTLASS FMHA).
    int K = static_cast<int>(draft.size());
    int n_verify = K + 1;
    int kv_bs = kv_cache_->block_size();
    int max_ctx = position + K + 1;

    int blocks_needed = (max_ctx + kv_bs - 1) / kv_bs;
    auto& bt = kv_manager_->block_table(seq_id);
    for (int b = static_cast<int>(bt.size()); b < blocks_needed; b++) {
        int new_block = kv_manager_->append_block(seq_id);
        if (new_block < 0) {
            return {0, -1};
        }
    }

    // Tokens: [last_token, draft[0], ..., draft[K-1]]
    std::vector<int32_t> h_tokens(n_verify);
    h_tokens[0] = req->output_tokens.back();
    for (int i = 0; i < K; i++)
        h_tokens[i + 1] = draft[i];

    // Positions: [pos, pos+1, ..., pos+K]
    std::vector<int> h_positions(n_verify);
    for (int i = 0; i < n_verify; i++)
        h_positions[i] = position + i;

    // Context lens: [pos+1, pos+2, ..., pos+K+1]
    std::vector<int> h_ctx_lens(n_verify);
    for (int i = 0; i < n_verify; i++)
        h_ctx_lens[i] = position + i + 1;

    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_tokens_, h_tokens.data(), n_verify * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_positions_, h_positions.data(), n_verify * sizeof(int),
                    cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_ctx_len_, h_ctx_lens.data(), n_verify * sizeof(int),
                    cudaMemcpyHostToDevice, stream));

    // Replicated block table: n_verify identical copies
    const auto& block_table = kv_manager_->block_table(seq_id);
    int n_blocks = static_cast<int>(block_table.size());
    int bt_total = n_verify * n_blocks;
    if (bt_total > d_block_table_cap_) {
        if (d_block_table_) IMP_CUDA_CHECK_LOG(cudaFree(d_block_table_));
        d_block_table_cap_ = bt_total * 2;
        IMP_CUDA_CHECK_LOG(cudaMalloc(&d_block_table_, d_block_table_cap_ * sizeof(int)));
        // Pointer changed → captured graphs hold stale addresses
        for (auto& g : verify_graphs_) if (g) g->invalidate();
    }
    std::vector<int> h_bt(bt_total, 0);
    for (int c = 0; c < n_verify; c++)
        for (int b = 0; b < n_blocks; b++)
            h_bt[c * n_blocks + b] = block_table[b];
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_block_table_, h_bt.data(), bt_total * sizeof(int),
                    cudaMemcpyHostToDevice, stream));

    // Pre-size workspace to spec_k+1 once and keep it — toggling back to 1
    // between verify steps breaks graph capture (host-side offset recompute).
    (void)executor_->resize_workspace(config_.spec_k + 1, stream);

    InferenceState state;
    state.token_ids = d_tokens_;
    state.positions = d_positions_;
    state.n_tokens = n_verify;
    state.kv_cache = kv_cache_;
    state.block_tables = d_block_table_;
    state.context_lens = d_ctx_len_;
    state.max_context_len = max_ctx;
    state.n_sequences = n_verify;
    state.max_blocks_per_seq = n_blocks;
    state.is_prefill = false;
    state.all_logits = false;
    state.temperature = 0.0f;
    state.top_k = 1;

    // Forward is graph-captured; sampling stays eager so per-row argmax + D2H
    // readback runs after replay.
    Tensor logits;
    bool use_graph = (n_verify >= 2 && n_verify < static_cast<int>(verify_graphs_.size()) &&
                      verify_graphs_[n_verify]);
    if (use_graph) {
        if (verify_graph_max_blocks_[n_verify] != n_blocks) {
            verify_graphs_[n_verify]->invalidate();
            verify_graph_max_blocks_[n_verify] = n_blocks;
        }
        verify_graphs_[n_verify]->set_decode_fn(
            [this, &state, &logits](cudaStream_t s) {
                executor_->forward_logits(state, logits, s);
            });
        verify_graphs_[n_verify]->execute(stream);
        if (logits.data == nullptr) {
            logits = executor_->get_logits_view(n_verify);
        }
    } else {
        executor_->forward_logits(state, logits, stream);
    }

    std::vector<int32_t> targets = executor_->sample_from_logits(logits, state, stream);

    // Acceptance: targets[i] vs draft[i]
    VerifyResult result;
    result.n_accepted = 0;
    result.corrected_token = -1;

    for (int i = 0; i < K; i++) {
        if (targets[i] == draft[i]) {
            result.n_accepted++;
        } else {
            result.corrected_token = targets[i];
            break;
        }
    }

    if (result.n_accepted == K) {
        result.corrected_token = targets[K];
    }

    return result;
}

NgramSpecDecoder::StepResult NgramSpecDecoder::step(
    std::shared_ptr<Request> req, int32_t last_token,
    int position, int seq_id, cudaStream_t stream)
{
    (void)last_token;
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
