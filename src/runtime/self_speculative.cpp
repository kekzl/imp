#include "runtime/self_speculative.h"
#include "compute/sampling.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace imp {

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("self_spec %s: %s", msg, cudaGetErrorString(err));
    }
}

SelfSpeculativeDecoder::~SelfSpeculativeDecoder() {
    if (d_tokens_) cudaFree(d_tokens_);
    if (d_positions_) cudaFree(d_positions_);
    if (d_block_table_) cudaFree(d_block_table_);
    if (d_ctx_len_) cudaFree(d_ctx_len_);
    if (d_draft_scratch_) cudaFree(d_draft_scratch_);
    if (h_draft_results_) cudaFreeHost(h_draft_results_);
    if (d_position_array_) cudaFree(d_position_array_);
    if (d_ctx_len_array_) cudaFree(d_ctx_len_array_);
}

bool SelfSpeculativeDecoder::init(GraphExecutor* executor,
                                   KVCacheManager* kv_manager,
                                   KVCache* kv_cache,
                                   int n_layers,
                                   const SelfSpecConfig& config) {
    if (!executor || !kv_manager || !kv_cache) {
        IMP_LOG_ERROR("self_spec: null executor/kv_manager/kv_cache");
        return false;
    }
    if (config.spec_k < 1 || config.spec_k > 32) {
        IMP_LOG_ERROR("self_spec: spec_k must be in [1, 32], got %d", config.spec_k);
        return false;
    }

    config_ = config;
    executor_ = executor;
    kv_manager_ = kv_manager;
    kv_cache_ = kv_cache;
    n_layers_ = n_layers;

    if (config.layer_skip) {
        // Layer skipping: skip middle layers, keep first and last
        // exit_layer controls how many layers to actually run (skip the rest)
        int run_n = (config.exit_layer > 0) ? config.exit_layer : std::max(n_layers - 8, n_layers / 2);
        int skip_n = (config.skip_n > 0) ? config.skip_n : (n_layers - run_n);
        skip_start_ = (n_layers - skip_n) / 2;  // centered
        skip_end_ = skip_start_ + skip_n;
        exit_layer_ = -1;  // not used
    } else {
        exit_layer_ = (config.exit_layer > 0) ? config.exit_layer : (n_layers / 2);
        skip_start_ = -1;
        skip_end_ = -1;
    }

    // Pre-allocate device buffers for max K+1 tokens
    int max_n = config_.spec_k + 1;
    int K = config_.spec_k;
    check_cuda(cudaMalloc(&d_tokens_, max_n * sizeof(int32_t)), "malloc d_tokens");
    check_cuda(cudaMalloc(&d_positions_, max_n * sizeof(int)), "malloc d_positions");
    check_cuda(cudaMalloc(&d_ctx_len_, max_n * sizeof(int)), "malloc d_ctx_len");

    // Argmax scratch buffer (needs ARGMAX_SCRATCH_BYTES for multi-block reduction)
    check_cuda(cudaMalloc(&d_draft_scratch_, ARGMAX_SCRATCH_BYTES), "malloc d_draft_scratch");

    // Mapped pinned memory for K draft tokens (zero-copy readback after graph)
    check_cuda(cudaHostAlloc(&h_draft_results_, K * sizeof(int32_t),
               cudaHostAllocMapped), "hostalloc draft_results");
    check_cuda(cudaHostGetDevicePointer(&d_draft_results_, h_draft_results_, 0),
               "getdevptr draft_results");

    // Pre-computed position/ctx_len arrays for K iterations (uploaded before graph)
    check_cuda(cudaMalloc(&d_position_array_, K * sizeof(int)), "malloc d_position_array");
    check_cuda(cudaMalloc(&d_ctx_len_array_, K * sizeof(int)), "malloc d_ctx_len_array");

    initialized_ = true;
    if (config.layer_skip) {
        IMP_LOG_INFO("self_spec: layer-skip mode, spec_k=%d, skip layers [%d,%d) of %d (runs %d/%d layers)",
                     config_.spec_k, skip_start_, skip_end_, n_layers,
                     n_layers - (skip_end_ - skip_start_), n_layers);
    } else {
        IMP_LOG_INFO("self_spec: early-exit mode, spec_k=%d, exit_layer=%d/%d",
                     config_.spec_k, exit_layer_, n_layers);
    }
    return true;
}

// ─── Helper: upload block table (single copy) ────────────────────────────────

void SelfSpeculativeDecoder::upload_block_table(int seq_id, cudaStream_t stream) {
    const auto& bt = kv_manager_->block_table(seq_id);
    int n_blocks = static_cast<int>(bt.size());
    if (n_blocks > d_block_table_cap_) {
        if (d_block_table_) cudaFree(d_block_table_);
        d_block_table_cap_ = n_blocks + 16;
        check_cuda(cudaMalloc(&d_block_table_, d_block_table_cap_ * sizeof(int)),
                   "malloc d_block_table");
        draft_graph_.invalidate();  // device pointer changed
    }
    check_cuda(cudaMemcpyAsync(d_block_table_, bt.data(),
               n_blocks * sizeof(int), cudaMemcpyHostToDevice, stream),
               "memcpy block_table");
}

// ─── Helper: upload replicated block table for batched verify ────────────────

void SelfSpeculativeDecoder::upload_block_table_replicated(
        int seq_id, int n_copies, int max_blocks_per_seq, cudaStream_t stream) {
    const auto& bt = kv_manager_->block_table(seq_id);
    int n_blocks = static_cast<int>(bt.size());

    int total_entries = n_copies * max_blocks_per_seq;
    if (total_entries > d_block_table_cap_) {
        if (d_block_table_) cudaFree(d_block_table_);
        d_block_table_cap_ = total_entries + 16;
        check_cuda(cudaMalloc(&d_block_table_, d_block_table_cap_ * sizeof(int)),
                   "malloc d_block_table replicated");
        draft_graph_.invalidate();  // device pointer changed
    }

    // Build replicated layout on host
    std::vector<int> replicated(total_entries, 0);
    for (int c = 0; c < n_copies; ++c) {
        for (int b = 0; b < n_blocks && b < max_blocks_per_seq; ++b) {
            replicated[c * max_blocks_per_seq + b] = bt[b];
        }
    }
    check_cuda(cudaMemcpyAsync(d_block_table_, replicated.data(),
               total_entries * sizeof(int), cudaMemcpyHostToDevice, stream),
               "memcpy block_table replicated");
}

// ─── Helper: ensure KV blocks ────────────────────────────────────────────────

void SelfSpeculativeDecoder::ensure_kv_blocks(int seq_id, int ctx_len) {
    const int bs = kv_cache_->block_size();
    int needed = (ctx_len + bs - 1) / bs;
    int have = static_cast<int>(kv_manager_->block_table(seq_id).size());
    while (needed > have) {
        if (kv_manager_->append_block(seq_id) < 0) break;
        have++;
    }
}

// ─── Draft token generation (GPU-autonomous K-iteration CUDA graph) ──────────
//
// All K draft iterations are captured in a single CUDA graph. Between iterations,
// D2D memcpy stages position/ctx_len from pre-computed arrays, and the sampled
// token feeds back to d_tokens_[0] for the next forward pass. Only ONE host sync
// is needed after all K iterations complete.

std::vector<int32_t> SelfSpeculativeDecoder::draft_tokens(
        int32_t last_token, int position, int seq_id, cudaStream_t stream) {
    const int K = config_.spec_k;

    // Pre-allocate KV blocks for all K draft steps
    int max_ctx = position + K + 1;
    ensure_kv_blocks(seq_id, max_ctx);
    upload_block_table(seq_id, stream);

    int n_blocks = static_cast<int>(kv_manager_->block_table(seq_id).size());

    // Pad max_blocks_per_seq to reduce graph re-captures.
    int max_blocks_per_seq = (n_blocks + 7) & ~7;
    if (max_blocks_per_seq < 8) max_blocks_per_seq = 8;

    // Invalidate graph if padded block count changed (grid size depends on it)
    if (max_blocks_per_seq != draft_graph_max_blocks_) {
        draft_graph_.invalidate();
        draft_graph_max_blocks_ = max_blocks_per_seq;
    }

    // Upload initial token BEFORE the graph
    check_cuda(cudaMemcpyAsync(d_tokens_, &last_token, sizeof(int32_t),
               cudaMemcpyHostToDevice, stream), "memcpy token");

    // Upload pre-computed position and ctx_len arrays for K iterations
    {
        std::vector<int> positions(K), ctx_lens(K);
        for (int k = 0; k < K; ++k) {
            positions[k] = position + k;
            ctx_lens[k] = position + k + 1;
        }
        check_cuda(cudaMemcpyAsync(d_position_array_, positions.data(),
                   K * sizeof(int), cudaMemcpyHostToDevice, stream),
                   "memcpy position_array");
        check_cuda(cudaMemcpyAsync(d_ctx_len_array_, ctx_lens.data(),
                   K * sizeof(int), cudaMemcpyHostToDevice, stream),
                   "memcpy ctx_len_array");
    }

    // Build a fixed InferenceState for graph capture/replay.
    // max_context_len is set to maximum (position+K+1) so split-K grid is stable.
    InferenceState state;
    state.token_ids = d_tokens_;
    state.positions = d_positions_;
    state.n_tokens = 1;
    state.kv_cache = kv_cache_;
    state.block_tables = d_block_table_;
    state.context_lens = d_ctx_len_;
    state.max_context_len = max_ctx;  // fixed for stable grid
    state.n_sequences = 1;
    state.max_blocks_per_seq = max_blocks_per_seq;
    state.is_prefill = false;
    state.temperature = 0.0f;  // greedy draft
    state.top_k = 1;
    state.exit_layer = exit_layer_;  // -1 when using layer skip
    state.skip_layer_start = skip_start_;
    state.skip_layer_end = skip_end_;

    // Set up the graph decode function: ALL K iterations captured in one graph.
    // Each iteration:
    //   1. D2D memcpy: d_position_array_[k] → d_positions_[0]
    //   2. D2D memcpy: d_ctx_len_array_[k] → d_ctx_len_[0]
    //   3. forward_logits (reads d_tokens_[0], d_positions_[0], d_ctx_len_[0])
    //   4. sample_greedy_device → d_draft_scratch_ (scratch) + D2H to h_draft_results_[k]
    //   5. D2D memcpy: d_draft_scratch_[0] → d_tokens_[0] (feed token to next iter)
    draft_graph_.set_decode_fn(
        [this, &state, K](cudaStream_t s) {
            for (int k = 0; k < K; ++k) {
                // Stage position and ctx_len from pre-computed arrays
                cudaMemcpyAsync(d_positions_, d_position_array_ + k,
                                sizeof(int), cudaMemcpyDeviceToDevice, s);
                cudaMemcpyAsync(d_ctx_len_, d_ctx_len_array_ + k,
                                sizeof(int), cudaMemcpyDeviceToDevice, s);

                // Forward pass (layer-skip/early-exit)
                Tensor logits_out;
                executor_->forward_logits(state, logits_out, s);
                if (logits_out.data == nullptr)
                    logits_out = executor_->get_logits_view(1);
                int64_t vshape[1] = {logits_out.shape[logits_out.ndim - 1]};
                Tensor flat = logits_out.slice(0, 1).reshape(1, vshape);

                // Greedy sample → scratch buffer + mapped readback slot
                sample_greedy_device(flat, d_draft_scratch_,
                                     h_draft_results_ + k, s);

                // Feed sampled token back to d_tokens_[0] for next iteration
                cudaMemcpyAsync(d_tokens_, d_draft_scratch_,
                                sizeof(int32_t), cudaMemcpyDeviceToDevice, s);
            }
        });

    // Execute graph (warmup → capture → replay)
    draft_graph_.execute(stream);

    // Single sync after all K iterations — read all draft tokens
    cudaEvent_t draft_done;
    cudaEventCreateWithFlags(&draft_done, cudaEventDisableTiming);
    cudaEventRecord(draft_done, stream);
    while (cudaEventQuery(draft_done) == cudaErrorNotReady) {}
    cudaEventDestroy(draft_done);

    std::vector<int32_t> drafts(K);
    for (int k = 0; k < K; ++k)
        drafts[k] = h_draft_results_[k];

    return drafts;
}

// ─── Batched verification (all K+1 tokens through full model in one pass) ───

SelfSpeculativeDecoder::VerifyResult
SelfSpeculativeDecoder::verify(const std::vector<int32_t>& draft,
                                int32_t last_token, int position, int seq_id,
                                float temperature, float top_p_val, int top_k_val,
                                int seed, cudaStream_t stream) {
    VerifyResult result;
    const int K = static_cast<int>(draft.size());
    if (K == 0) return result;

    const int n_verify = K + 1;

    // Ensure KV blocks for the maximum context length
    int max_ctx = position + K + 1;
    ensure_kv_blocks(seq_id, max_ctx);

    // max_blocks_per_seq MUST match paged_attention's stride:
    //   max_num_blocks = (max_context_len + block_size - 1) / block_size
    int max_blocks_per_seq = (max_ctx + kv_cache_->block_size() - 1) / kv_cache_->block_size();

    // Upload replicated block table: n_verify copies, each max_blocks_per_seq entries
    upload_block_table_replicated(seq_id, n_verify, max_blocks_per_seq, stream);

    // Build token array: [last_token, draft[0], ..., draft[K-1]]
    std::vector<int32_t> verify_tokens(n_verify);
    verify_tokens[0] = last_token;
    for (int i = 0; i < K; ++i)
        verify_tokens[i + 1] = draft[i];

    // Build positions: [position, position+1, ..., position+K]
    std::vector<int> positions(n_verify);
    for (int i = 0; i < n_verify; ++i)
        positions[i] = position + i;

    // Build context lens: [position+1, position+2, ..., position+K+1]
    std::vector<int> ctx_lens(n_verify);
    for (int i = 0; i < n_verify; ++i)
        ctx_lens[i] = position + i + 1;

    // Upload to device
    check_cuda(cudaMemcpyAsync(d_tokens_, verify_tokens.data(),
               n_verify * sizeof(int32_t), cudaMemcpyHostToDevice, stream),
               "verify memcpy tokens");
    check_cuda(cudaMemcpyAsync(d_positions_, positions.data(),
               n_verify * sizeof(int), cudaMemcpyHostToDevice, stream),
               "verify memcpy positions");
    check_cuda(cudaMemcpyAsync(d_ctx_len_, ctx_lens.data(),
               n_verify * sizeof(int), cudaMemcpyHostToDevice, stream),
               "verify memcpy ctx_lens");

    // Resize workspace for K+1 tokens BEFORE the batched forward
    executor_->resize_workspace(n_verify, stream);

    // Build inference state for batched decode
    InferenceState state;
    state.token_ids = d_tokens_;
    state.positions = d_positions_;
    state.n_tokens = n_verify;
    state.kv_cache = kv_cache_;
    state.block_tables = d_block_table_;
    state.context_lens = d_ctx_len_;
    state.max_context_len = max_ctx;
    state.n_sequences = n_verify;
    state.max_blocks_per_seq = max_blocks_per_seq;
    state.is_prefill = false;
    state.temperature = 0.0f;  // greedy for target selection
    state.top_k = 1;
    state.exit_layer = -1;  // full model
    state.per_row_lm_head = true;  // per-row Q8_1 GEMV avoids FP8 per-tensor quantization artifacts

    // forward_batch returns one sampled token per sequence
    std::vector<int32_t> targets = executor_->forward_batch(state, stream);

    // Resize workspace back to 1 for subsequent draft passes
    executor_->resize_workspace(1, stream);

    // Acceptance: compare target[i] with draft[i]
    int n_accepted = 0;
    for (int i = 0; i < K; ++i) {
        if (targets[i] == draft[i]) {
            result.accepted.push_back(draft[i]);
            n_accepted++;
        } else {
            result.n_accepted = n_accepted;
            result.next_token = targets[i];
            return result;
        }
    }

    // All K accepted — target[K] is the bonus token
    result.n_accepted = K;
    result.next_token = targets[K];
    return result;
}

// ─── Full step ───────────────────────────────────────────────────────────────

std::vector<int32_t> SelfSpeculativeDecoder::step(
        int32_t last_token, int position, int seq_id,
        float temperature, float top_p, int top_k, int seed,
        cudaStream_t stream) {
    if (!initialized_) {
        IMP_LOG_ERROR("self_spec: not initialized");
        return {};
    }

    // Resize workspace for single-token draft passes
    executor_->resize_workspace(1, stream);

    // 1. Draft K tokens with layer skip/early exit
    std::vector<int32_t> draft = draft_tokens(last_token, position, seq_id, stream);
    total_drafted_ += draft.size();

    // 2. Verify with full model (batched: all K+1 tokens in one pass)
    VerifyResult vr = verify(draft, last_token, position, seq_id,
                              temperature, top_p, top_k, seed, stream);
    total_accepted_ += vr.n_accepted;

    IMP_LOG_DEBUG("self_spec: accepted %d/%d (cumulative: %lld/%lld = %.1f%%)",
                  vr.n_accepted, static_cast<int>(draft.size()),
                  static_cast<long long>(total_accepted_),
                  static_cast<long long>(total_drafted_),
                  total_drafted_ > 0 ? 100.0 * total_accepted_ / total_drafted_ : 0.0);

    // 3. Combine: accepted + next token
    std::vector<int32_t> output;
    output.reserve(vr.n_accepted + 1);
    for (int i = 0; i < vr.n_accepted; ++i)
        output.push_back(vr.accepted[i]);
    if (vr.next_token >= 0)
        output.push_back(vr.next_token);

    return output;
}

} // namespace imp
